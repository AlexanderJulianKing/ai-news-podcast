# Editorial decisions benchmark

Measures how well a model makes the pipeline's editorial calls. It replays real
logged inputs through a candidate model and scores the answers against a reference.

[CALL_INVENTORY.md](CALL_INVENTORY.md) lists every LLM call by tier and marks which
ones are decisions.

Status (2026-09-20): the lead-pick label set and its replay scorer are built.
12 mornings are labeled (2 real, 10 invented). No model has been replayed yet.
The other decisions (dedup tagger, arc match, edit vet) are not started.

## Lead pick: the hand-labeled set

The lead pick is the Tier 3 "most important story" call (`topic_finder.py`, Opus).
It is a taste call, so the reference is Alex's own label, not another model's answer.

### 1. Harvest cases

The audit log stores the exact prompt Opus saw each morning. The production log
lives on the Pi:

```bash
scp alex@raspberrypi.tail470879.ts.net:/home/alex/ai-news-podcast/logs/llm_audit.jsonl benchmarks/editorial_decisions/data/pi_llm_audit.jsonl
```

```bash
rsync -a --include='log_*.txt' --exclude='*' alex@raspberrypi.tail470879.ts.net:/home/alex/ai-news-podcast/logs/ benchmarks/editorial_decisions/data/pi_logs/
```

```bash
python3 -m benchmarks.editorial_decisions.harvest_lead_pick --audit benchmarks/editorial_decisions/data/pi_llm_audit.jsonl --logs-dir benchmarks/editorial_decisions/data/pi_logs
```

The audit log does not store responses, but the pick is still in it. The
pipeline feeds Opus's selection essay straight into `headline_extractor`, so the
next logged call carries the essay as its user prompt. The harvester reads the
pick from there. The daily log (first `Answer:` after `TIER 3: Selecting
stories`) is the fallback.

A case is keyed by date plus its tag-stripped headlines. A replay of the same
briefs under different tags or a different prompt folds into one case. Each run
is kept in `runs` with its own tags, eligible briefs, prompt hash, essay, and
pick, so one labeled morning can score several rule sets. The latest run
supplies the tags shown on the labeling page.
The case date comes from the log timestamp. For a dev replay that is the replay
day, not the news day; the DATE line inside each brief shows the real day.

### 2. Label

```bash
python3 -m benchmarks.editorial_decisions.label_server
```

Then open http://127.0.0.1:8917. Each save appends to
`data/lead_pick_labels.jsonl`. The latest line per case is the label of record.

What the page does to keep the labels clean:

- Briefs appear in a shuffled order that is fixed per case. The order Opus saw
  is hidden, so position cannot lead the labeler. The shown order is saved.
- The show's pick stays hidden until after you save.
- Tags are visible, and a story the rules barred (`[UPDATE]`) can still be
  picked. `lead_was_eligible` records that. This separates "the model has
  different taste" from "the rules blocked the right story".

### Label fields

| Field | Meaning |
|---|---|
| `lead_index` | Brief number (as Opus saw it) that should have led. Null with `none_deserved` |
| `second_index` | The other full segment. Null with `no_second`. Labels saved before 2026-09-20 lack this key, and the page lists them as needing a second story |
| `acceptable_indexes` | Other briefs that would also have been fine leads |
| `lead_was_eligible` | False when the chosen lead carried a barring tag |
| `remembers_outcome` | `no`, `vaguely`, `yes`: does the labeler know how these stories turned out |
| `hindsight_mode`, `hindsight_index` | The pick knowing what happened since: `same`, `different` (with index), or `unsure` |
| `confidence` | `low`, `medium`, `high` |
| `seconds_spent`, `display_order`, `labeled_at`, `note` | Bookkeeping |

### Hindsight

Labels are made days or weeks after the morning in question, so the labeler
knows things the model could not. The set records that instead of pretending
it away. Two picks are kept: the pick judged from the briefs alone
(`lead_index`) and the pick with hindsight (`hindsight_index`).

Comparisons this allows later:

- Score a model against `lead_index`. That is the fair test, since the model
  only had the briefs.
- Count mornings where the two picks differ. Those are stories whose
  importance was not visible in the briefs. No model could be blamed for them,
  and they point at research or ingestion gaps.
- Compare agreement on `remembers_outcome = no` mornings against `yes`
  mornings. A gap suggests memory is leaking into the "briefs alone" pick.
- Compare agreement by label age (`labeled_at` minus case date).

## Invented mornings

[hypothetical_cases.json](hypothetical_cases.json) holds 10 made-up mornings of
5 briefs each. Every event in it is fictional. The labeling page serves them
after the real mornings and marks them "invented".

Why they exist:

- They carry no hindsight, because nothing in them happened. The page drops the
  memory and hindsight questions for them.
- Each one isolates a single axis of taste (listed in the file as `axis`, with
  `probe` mapping each pole to a brief). The page shows the axis only after the
  label is saved. Do not read the JSON before labeling; knowing the axis would
  steer the pick.
- A real morning mixes many pulls at once. A controlled morning shows which
  pull wins when only one is in play.

Limits: the briefs are about 70 words, where real ones run to a few hundred.
A model that does well here has not been shown to do well on real mornings.
Score and report the two sets apart (`synthetic` is set on every case and label).

### The hard set

The first 10 mornings turned out too easy: on 2026-09-20 every tier landed in
the acceptable set on all 10, so they cannot separate models.
[hypothetical_cases_hard.json](hypothetical_cases_hard.json) adds 8 mornings
built to be close calls:

- at least three strong contenders per morning, no filler leads;
- briefs of 90 to 145 words in the research layer's neutral memo style (what
  happened, key facts, context, sourcing), with no wording that says which
  story matters;
- realistic tags, with the coverage-notes trailer a real morning carries, so
  `[DEVELOPMENT]` stories come with what the audience already knows;
- one morning sets a huge story with a thin, partly unconfirmed brief against
  smaller stories with rich briefs, to test whether brief quality sways the pick;
- stored brief order shuffled once with a fixed seed, because models see that
  order and the first draft listed the strongest contender first.

They report as their own set, `invented_hard`. Run them alone with
`--set invented_hard`, and use `--repeats 3` so a small gap can be told from
run-to-run variation.

### The third set

[hypothetical_cases_hard2.json](hypothetical_cases_hard2.json), set
`invented_hard2`, holds 10 mornings written by an Opus subagent that never saw
the labels or any model output. A second author breaks the framing habits the
first two sets share. Four of its mornings carry 9 to 11 briefs, closer to a
real morning's 11 to 13. Outlet counts are deliberately decoupled from
importance, and two mornings hinge on an UNVERIFIED brief. The candidate
prompts were written before this set was labeled, so it is the first
out-of-sample test of them.

`hypotheticals.py` renders each case into the production research-document
format and parses it back with the harvester's parser, so a replay sees the
layout a real morning has.

## Replay and score

```bash
python3 -m benchmarks.editorial_decisions.run_lead_pick --modes light standard advanced adversary heavy --dry-run
```

Drop `--dry-run` to make the calls. Each mode hits whatever model `config.py`
maps it to, and the calls cost real money. Three things differ from the pipeline
on purpose:

- No GPT-5.5 fallback. The router falls back when a tier fails, which would
  score GPT-5.5's pick under another model's name. A failed call is an error row.
- No audit logging. A replayed Tier 3 prompt in `logs/llm_audit.jsonl` would
  look like a real morning to the harvester. The harvester also skips any row
  with `phase: benchmark`.
- No LLM extractor. The pick is parsed from the essay: an `Answer` marker
  first, else the essay's first line. `parse_method` records which. The small
  models skip the marker and open with the headline.

Rows are appended as they finish, so a killed run keeps what it paid for. The
run is slow (minutes per tier); launch one process per mode to run tiers side by side.
`--repeats N` calls each case N times to show run-to-run spread. `--only real`
or `--only invented` limits the set.

Every case runs under the current `TIER3_IMPORTANT_STORY_PROMPT`. Real mornings
keep the tags from their latest logged run.

Reported per mode, with real and invented mornings apart:

- **exact**: the model picked the labeled lead.
- **acceptable**: it picked the lead or an "also fine" brief. This is the main number.
- **barred**: it picked a brief its own input tagged ineligible. A rules failure.
- **unparsed**: no brief matched the essay's answer.

Two scoring rules worth knowing:

- **Same story, two briefs.** Real mornings can carry one story twice. Briefs
  whose headlines are identical, or where one opens the other, are merged
  automatically. [story_groups.json](story_groups.json) holds hand-checked
  groups for same-event briefs worded differently. Picking either copy scores
  the same.
- **Paraphrased answers.** Opus often rewrites the headline in its Answer line.
  `match_pick` falls back to the share of the headline's core words that the
  answer repeats (at least 0.6, at least three words, and 0.15 clear of any
  different story). After any matcher change, re-score the saved essays instead
  of paying for new calls; every row keeps its essay.

Results land in the gitignored `outputs/`.

### The second slot

```bash
python3 -m benchmarks.editorial_decisions.run_second_pick --from-run benchmarks/editorial_decisions/outputs/lead_pick_X.jsonl --dry-run
```

The show has two full segments. The second comes from
`TIER3_EVERYMAN_STORY_PROMPT`, which is told which headline already leads. This
script replays that call on top of a saved lead-pick run, using the lead that
same model chose, so no lead call is paid for twice. It reports how often the
second slot matches, and how many of the labeler's two stories the model's two
stories cover regardless of slot.

## Testing a prompt change

[candidate_prompts.py](candidate_prompts.py) holds revisions to the two Tier 3
prompts as named edits applied on top of the production text. The show never
imports it. An edit that no longer matches the production prompt raises, so the
candidate cannot drift silently.

```bash
python3 -m benchmarks.editorial_decisions.run_lead_pick --modes standard heavy --variant candidate --dry-run
```

`run_second_pick` takes the same `--variant`. Compare saved runs, with no new
calls, using:

```bash
python3 -m benchmarks.editorial_decisions.compare_variants --lead <lead files> --second <second files>
```

It re-scores every row against the current labels and lists the mornings whose
majority lead pick moved into or out of the labeled set.

A prompt written after reading the labels is graded in-sample on those
mornings. Look for regressions there, and judge gains on mornings labeled later.

This is not a formality. On 2026-09-20 the revised lead prompt raised Opus from
62% to 88% on the 8 hard mornings it was written against, then scored 5 of 10
against production's 6 of 10 on 10 fresh mornings labeled afterwards. The gain
was overfitting to two mornings.

## How consistent are the labels?

A model cannot be expected to agree with the labeler more often than he agrees
with himself. On 2026-09-20 three models across a 100x price range (Gemma,
Opus 4.8, Fable 5.1) all matched 22 of 30 mornings, and almost every miss fell
on a morning labeled with medium or low confidence. That points at the labels
as the ceiling, so measure it:

```bash
python3 -m benchmarks.editorial_decisions.label_server --relabel --port 8918
```

Relabel mode serves only mornings that already have a label, hides the old
label, shuffles the order, and saves to `data/lead_pick_relabels.jsonl`. Add
`--only-confidence medium low` to focus on the uncertain ones. Then:

```bash
python3 -m benchmarks.editorial_decisions.self_agreement
```

It scores the second pass against the first exactly as a model is scored.
Leave at least a day between passes so memory of the first pick fades.

## Other models

`run_lead_pick --modes fable` or `opus5` runs a model that is not a pipeline
tier (see `EXTRA_TIERS`). `run_jev.py` scores TypeSafe's Jev decision model
through OpenRouter's System One endpoint; it returns a probability per brief,
which the chat models do not.

## Tests

```bash
python3 -m pytest benchmarks/editorial_decisions -q
```
