# Opinion review loop: replay and blind judging (2026-11-15)

`replay.py KEY` runs `opinion.reviewed()` (fresh Opus reviewer -> writer with lookups, up to 3 rounds,
then the faithfulness check on revised sentences) on an aired Sunday piece. Copy
`segment_summaries/KEY_opinion_state.json` and `output_scripts/KEY_segment_0.txt` from the Pi into
`$REVIEW_REPLAY_DIR/data/` first. It writes `out/KEY/{before,after}.txt` and `audit.json`.

Judging: put the two versions in random order as `A.txt`/`B.txt` beside `evidence.txt` (all lookups
included) and run `codex exec -m gpt-6-astra -c 'model_reasoning_effort="medium"' --sandbox read-only
"$(cat judge_prompt.txt)"` in that folder. Keep the A/B key out of the judge's folder. Run the judges as
one tracked job (`( ... ) & ... wait`): launched detached, they were killed a minute in.

## Results on the four aired Sundays

| Sunday | rounds | blocking notes per round | words | cost | judge | factual problems old -> new | unnamed sources old -> new |
|---|---|---|---|---|---|---|---|
| 2026-10-25 | 3 (backstop) | 4, 2, 3 | 2060 -> 2135 | $2.71 | new, high | 5 -> 2 | 2 -> 0 |
| 2026-11-01 | 3 (backstop) | 3, 4, 2 | 2181 -> 1987 | $2.39 | new, high | 4 -> 2 | 2 -> 1 |
| 2026-11-08 | 3 (backstop) | 6, 6, 1 | 1805 -> 1871 | $2.94 | new, high | 3 -> 0 | 3 -> 1 |
| 2026-11-15 | 3 (backstop) | 6, 2, 3 | 2215 -> 2514 | $3.06 | new, high | 3 -> 2 | 4 -> 0 |

Errors that aired and the loop fixed: a quote credited to Mullin that the evidence doesn't give him
(10-25); "almost nine to one" for 88% (11-08); Kavanaugh's question about Congress read backwards, and two
quotes from "one observer" / "someone quoted" (Mary Hampton Mason, ex-DOJ, found by the writer's lookup)
(11-15). The reviewer left the correct body-camera sentence (11-15) alone.

Open question: no piece converged. Round 3's notes turn to "a critic could argue..." and the writer
restructures (11-15 grew 300 words). Each round's piece is now saved in the audit (`piece_after`), so
judge round 2 against round 3 on live Sundays before raising or lowering the cap. One judge per
Sunday, N=4; the counts are the judge's, not checked one by one.
