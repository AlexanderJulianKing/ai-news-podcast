# Source Hunter Resilience V2

## Goal

Improve URL discovery and source validation without turning the benchmark or production research path into a complicated multi-agent system.

The core shape stays simple:

```text
question
  -> generated evidence contract
  -> web URL discovery
  -> controlled fetch / PDF extraction
  -> source validation
  -> answer from accepted evidence only
```

The only "agentic" behavior is a bounded retry loop that already exists in the source-hunter strategy. A failed or weak attempt feeds rejection memory into the next search prompt.

## What Changed

### 1. Generated Contracts Stay Soft

Generated contracts still list the facts the answer should try to recover:

- date, entity, item, or proceeding constraints
- numeric fields such as counts, amounts, votes, ranges, rates, and percentages
- text fields such as rationale, mechanism, decision language, or caveats
- source preferences such as official agency page, docket filing, agenda page, or a domain directly implied by the question
- generic rejection traps such as announcement-without-results

But generated slots are not all hard gates. A strong official source can be accepted even if it only covers part of the contract. The final answer prompt still sees the evidence checklist and must say when a requested fact is unsupported.

Hard rejections remain:

- wrong date or year
- wrong entity
- wrong agenda item or proceeding
- wrong document type, such as a calendar page when the question asks for a statement
- generated source-preference mismatch, such as a news liveblog when the contract asks for the official Federal Reserve page
- true trap pages, such as webinar announcements that do not contain the released numbers

### 2. Source Preference Is Enforced

Each fetched URL is classified as primary-ish or secondary-ish:

- primary-ish: `.gov`, `.mil`, official public-record systems, Legistar-style agenda systems, CPUC docs/apps, or local-government branded sites that match the named entity
- secondary-ish: news, commentary, social, blogs, copied releases, Reddit, and similar sources

If the generated contract asks for a primary or official source, secondary pages do not pass just because they contain plausible text. They become rejection memory for the next search round.

This is meant to prevent the Fed failure mode where a liveblog had enough facts to look useful but was not the official statement.

### 3. `reject_if` Is Less Brittle

Generated `reject_if` rules are still useful, but they are no longer allowed to nuke good official pages just because a generated slot is missing.

Examples:

- "news summary without the full official text" should reject a news article, not a CPUC decision PDF.
- "date other than June 16, 2026" should not be reinterpreted loosely by string matching; deterministic date validation already handles dates.
- "announcement without results" still rejects pages that say a report will be released later.

### 4. Excerpt Rescue

Sometimes the right page is fetched, but the first excerpt pass grabs navigation, menus, or boilerplate. The v2 selector now checks whether the full page supports a field that the excerpt missed. If so, it appends a nearby answer-bearing chunk for that field within the same source-character budget.

This keeps the final answer model focused on controlled excerpts, but makes those excerpts less likely to omit the key paragraph.

### 5. Nearby Official Source Expansion

If global search finds the right official source family but not the exact evidence page, the hunter now does a small same-domain expansion before asking global search again.

The expansion is intentionally bounded:

- only starts from fetched official/primary sources
- only follows same-domain links
- scores links using question terms, dates, times, document words, and generated slot search terms
- fetches only the top few candidates
- defaults to four nearby hops

This handles source systems where the useful page is a few official links away from the discovered URL:

- current advisory page -> storm archive -> exact advisory
- meeting page -> agenda attachment
- proceeding page -> decision PDF
- press release -> linked report

It is not a general crawl and does not add site-specific adapters.

## What Did Not Change

- No new external API.
- No new model dependency.
- No recursive autonomous browsing agent.
- No benchmark-specific source resolver enabled by default.
- No hardcoded answer values.
- No final answer call when there is zero accepted evidence.

## Expected Tradeoff

This should improve cases where the system found the right source but rejected it too aggressively. It may also increase safe `NO_EVIDENCE` outcomes when only secondary sources are available and the contract asks for official evidence. That is the intended behavior: better to retry or refuse than answer from a weak copy.
