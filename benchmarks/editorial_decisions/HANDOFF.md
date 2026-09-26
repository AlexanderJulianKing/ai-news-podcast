# Handoff: editorial-decisions benchmark and commute weather

Written 2026-09-22 for the next session. Repo: `/Users/alexanderking/Desktop/random_stuff/newscaster3.5`, branch `story-selection-eligibility`. Nothing from this work is committed. Read `/Users/alexanderking/CLAUDE.md` first. The project memory index has a fuller history under "Editorial decisions benchmark" and "Move and commute weather".

## Where things stand

**Holding pattern by Alex's choice.** New Opus, GPT, and Gemini models are expected this week (from about 2026-09-22). Alex wants model testing held until all three are out. Do not run the holdout test, ask him to label, or edit the frozen prompts before then. The Pi is packed in a box for his move to Temecula; it being offline is expected, not a fault.

**Alex's plan, in his words:** tune prompts freely on the 30 mornings he has already labeled (contamination accepted), get a good model to parity with him, then he labels fresh mornings to see whether it generalizes. That plan is at the "labels next" step.

## The benchmark (all under `benchmarks/editorial_decisions/`)

- Purpose: score how well a model and a prompt reproduce Alex's picks for the show's two main story slots (lead, then second). Reference = Alex's own labels.
- Data: 30 labeled mornings (2 real, 28 invented; sets `real`, `invented`, `invented_hard`, `invented_hard2`) with two labeling passes each (`data/lead_pick_labels.jsonl`, `data/lead_pick_relabels.jsonl`). Plus 20 UNLABELED holdout mornings (`hypothetical_cases_hold1.json`, set `holdout1`), written by an Opus subagent blind to labels.
- Alex's self-consistency (second pass vs first): lead in his set 25/30, exact lead 22/30, two stories covered 48/60 (80%), same second with lead fixed 14/22 (64%). No model should be expected to beat these.
- Best prompts: variant `v_kind4_s2` in `candidate_prompts.py` (lead = `kind4_important`, second = a "THE SECOND SLOT" block). On the 30 dev mornings with Opus 4.8: exact lead 22/30 (parity), lead in set 24/30, second exact with Alex's lead supplied 15/22 (parity), pair coverage 42/60 (70%) vs production 53%. The rule behind it: he leads with CHANGES (binding acts of government in effect, verified firsts, frontier AI with substance, shifts in the international order) and never with incidents or anticipation; the second slot re-admits incidents, ranked by how much lands on people and how soon, with disasters only when they hit California. Alex confirmed both readings. All these scores are in-sample.
- What did NOT work, so do not retry blind: eight instruction-style lead variants in rounds 1-3 (`candidate`, `v_lives`, `v_taste`, `v_fewshot`, `v_ai`, ...); a 12-dimension rating-and-regression fit (`rate_briefs.py`, `fit_preferences.py`) scored below the models; Gemma applies any stated rule too literally to be the tuning model, use Opus.
- Models compared under the production prompt: Opus 4.8, Fable 5.1, Gemma 4 31B all 22/30 lead in set; Jev 1.13 (TypeSafe decision model, `run_jev.py`) 19/30 for under a cent, with usable confidence signal.

## The frozen holdout test (ready to score, do not touch until models are in)

- `frozen/holdout1_20260920.json`: sha256 of the production and `v_kind4_s2` prompts, plus hashes of every prediction file, recorded when 0 holdout labels existed.
- Predictions already locked for Opus 4.8 (x1) and Gemma (x3), both slots, all 20 holdout mornings: `outputs/lead_pick_{production,v_kind4_s2}_{heavy,standard}_20260920_2305*.jsonl` and `outputs/second_pick_*_20260920_23*.jsonl`. Nobody has looked at the picks.
- When the new models arrive: add each as a named tier in `EXTRA_TIERS` in `run_lead_pick.py` (Anthropic models go through the pipeline's adapter; OpenAI/Gemini need a spec the router's `_call_with_retry` accepts, check `newscaster/llm/router.py`), then run `run_lead_pick --modes <tier> --variant production --unlabeled --set holdout1` and the same with `--variant v_kind4_s2`, then `run_second_pick --from-run <that file> --variant <same> --modes <tier> --unlabeled --shard i/4`. Verify the prompt hashes still match before running. Record new file hashes in the frozen JSON.
- Then Alex labels the 20 holdout mornings at http://127.0.0.1:8917 (launch config `lead-pick-labeler`; `.claude/launch.json` is gitignored and may need recreating: `python3 -m benchmarks.editorial_decisions.label_server --port 8917`). Score with `compare_variants --lead <holdout lead files> --second <holdout second files> --set holdout1 --new v_kind4_s2 --modes heavy standard <new tiers>`.
- Judge on margins: 20 mornings cannot resolve a 2-3 morning gap.

## Costs and mechanics learned

- Opus 4.8 is about $0.05-0.08 per Tier 3 call; a 30-morning single-slot run is $1.50-2.50. Fable 5.1 about $0.10-0.13 per call. Gemma cents. Session total was roughly $75.
- Benchmark calls bypass the router's GPT-5.5 fallback and turn off the audit log (`phase: benchmark` rows are skipped by the harvester). Keys load via `config.init()`.
- One process per tier; five in one process exceeds the 10-minute background limit. `--shard i/n` splits a tier.
- `parse_pick` handles "Answer:", headline-first openings, "Brief N" announcements, and paraphrased answers; re-score saved essays after any matcher change (`compare_variants` does this without new calls).
- Real mornings must eventually come from the Pi's `logs/llm_audit.jsonl` (README has the scp/rsync commands). The harvester recovers Opus's pick from the `headline_extractor` call that follows each Tier 3 call.

## Weather (uncommitted, in `newscaster/weather.py`, `newscaster/prompts.py`, `tests/test_weather.py`)

Alex will commute Temecula -> La Jolla. The intro weather line now reports La Jolla's high plus midday and 5 p.m. temperatures, Temecula's evening temperature and overnight low, and rain only when the chance is 30 percent or more (dry is the default and unmentioned). It groups by America/Los_Angeles time (the old code grouped by UTC, a bug). `get_daily_temp()` keeps its signature. The intro prompt's Riverside example was replaced. Full test suite passes (459). Riverside-oriented news sources were left alone on purpose; ask before retargeting them.

## Also uncommitted from earlier sessions

`main2.bash`, `newscaster/upload.py`, `tests/test_upload.py` (upload retry work), and `benchmarks/rag_recall/*` edits. Not touched here; do not fold them into a commit without checking what they are.

## Open questions for Alex

1. Once the new models are in: label the 20 holdout mornings, then score.
2. Should the Riverside local sources (Press-Enterprise scrape, California triage) shift toward San Diego?
3. Deployment: when the Pi is set up again, the weather change and (if the holdout says so) the new prompts go to `main` and then `git pull --ff-only` on the Pi. Never run `main2.bash` to test; it publishes.
