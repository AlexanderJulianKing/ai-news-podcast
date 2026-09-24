# Archive: June and July 2026 design notes

These pages describe the system as it was designed in June and July 2026. They are kept
as a record of the reasoning, not as a description of the current code. Several
details are out of date:

- **Models.** They name Gemma, GLM and GPT-5.5, which were replaced in September 2026.
- **Sources.** They describe headline gathering through Gemini grounding.
- **Pipeline.** The diagram predates the structured tagger, coverage-depth
  eligibility and the browser reader.

For how a run works today, see [`../PIPELINE.md`](../PIPELINE.md).

| File | What it covered |
|---|---|
| `pipeline_diagram.html` | The whole pipeline as of 2026-06-17 |
| `source_hunter_agent_flow.html` | The source hunter's fetch, validate and answer flow |
| `source_hunter_resilience_v2.md` | Softer evidence contracts and retry memory for the source hunter |
| `research_controller_prompt_caching.html` | Prompt caching for the research controller |
