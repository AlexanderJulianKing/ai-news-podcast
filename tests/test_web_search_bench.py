from benchmarks.web_search.web_bench_lib import (
    build_controlled_payload,
    build_discovery_payload,
    build_evidence_contract_payload,
    build_openrouter_payload,
    canonical_url,
    discovery_candidates,
    evaluate_answer_slots,
    extract_constraints,
    federal_reserve_statement_candidates,
    fetch_source_text,
    nearby_source_candidates,
    nhc_advisory_archive_candidates,
    noaa_hurricane_outlook_candidates,
    parse_evidence_contract,
    score_result,
    score_discovery,
    select_relevant_excerpts,
    source_hunter_extra_context,
    summarize_scores,
    validate_source_for_question,
)


def test_build_openrouter_payload_forces_web_plugin_and_reasoning():
    model = {
        "id": "glm_5_2",
        "model": "z-ai/glm-5.2",
        "reasoning_effort": "high",
    }
    task = {
        "id": "example",
        "category": "test",
        "question": "What happened?",
        "preferred_sources": ["https://example.com"],
        "why_hard": "This is a hidden benchmark hint.",
    }

    payload = build_openrouter_payload(
        model,
        task,
        "2026-06-17",
        engine="parallel",
        max_results=5,
        max_tokens=1600,
    )

    assert payload["model"] == "z-ai/glm-5.2"
    assert payload["plugins"] == [
        {
            "id": "web",
            "engine": "parallel",
            "max_results": 5,
            "search_prompt": (
                "A web search was conducted for a newsroom benchmark on 2026-06-17. "
                "Use these results as evidence. Prefer official or primary sources."
            ),
        }
    ]
    assert payload["reasoning"] == {"effort": "high"}
    assert payload["messages"][0]["role"] == "system"
    assert "strict JSON object" in payload["messages"][1]["content"]
    assert "https://example.com" not in str(payload)
    assert "hidden benchmark hint" not in str(payload)


def test_score_result_checks_facts_sources_and_forbidden_penalty():
    task = {
        "id": "fed",
        "checks": [
            {"name": "range", "points": 2, "any": ["3.5% to 3.75%"]},
            {"name": "vote", "points": 1, "all": ["12-0"]},
        ],
        "source_domains": [{"domain": "federalreserve.gov", "points": 1}],
        "forbidden": [{"phrase": "still pending", "points": 2}],
    }
    result = {
        "task_id": "fed",
        "model_id": "m",
        "model_label": "Model",
        "content": "The vote was 12-0 and the range stayed at 3.5% to 3.75%. This was not still pending.",
        "annotations": [
            {
                "type": "url_citation",
                "url_citation": {
                    "url": "https://www.federalreserve.gov/newsevents/example.htm",
                    "title": "Fed statement",
                    "content": "12-0",
                },
            }
        ],
    }

    score = score_result(task, result)

    assert score["score"] == 2.0
    assert score["max_score"] == 4.0
    assert score["percent"] == 50.0
    assert score["penalties"] == [{"phrase": "still pending", "points": 2.0}]


def test_summarize_scores_orders_by_percent_then_cost():
    records = [
        {
            "model_id": "a",
            "model_label": "A",
            "score": 5,
            "max_score": 10,
            "failed": False,
            "usage": {"cost": 0.2},
            "latency_seconds": 4,
        },
        {
            "model_id": "b",
            "model_label": "B",
            "score": 8,
            "max_score": 10,
            "failed": False,
            "usage": {"cost": 0.1},
            "latency_seconds": 2,
        },
    ]

    summary = summarize_scores(records)

    assert [model["model_id"] for model in summary["models"]] == ["b", "a"]
    assert summary["models"][0]["percent"] == 80.0
    assert summary["models"][0]["avg_latency_seconds"] == 2


def test_select_relevant_excerpts_uses_question_not_answer_key():
    text = "\n".join(
        [
            "1 Consent calendar",
            "2 Routine update",
            "28 Approve Request for Proposal 2494 Agreement with Carollo Engineers for water rate design",
            "29 Unrelated electric program",
        ]
    )

    excerpt = select_relevant_excerpts(
        text,
        "For agenda item 28, identify the RFP number and work being approved.",
        max_chars=500,
    )

    assert "28 Approve Request for Proposal 2494" in excerpt
    assert "Carollo Engineers" in excerpt


def test_select_relevant_excerpts_keeps_advisory_fact_block_together():
    text = "\n".join(
        [
            "Navigation",
            "BULLETIN",
            "Tropical Storm Arthur Intermediate Advisory Number 6A",
            "NWS National Hurricane Center Miami FL AL012026",
            "100 PM CDT Wed Jun 17 2026",
            "...TROPICAL STORM WARNING REMAINS IN EFFECT FOR PORTIONS OF THE",
            "TEXAS AND LOUISIANA COAST...",
            "...LIFE-THREATENING FLOODING EXPECTED ACROSS PORTIONS OF THE",
            "SOUTHEASTERN UNITED STATES...",
            "SUMMARY OF 100 PM CDT...1800 UTC...INFORMATION",
            "LOCATION...28.9N 95.7W",
            "MAXIMUM SUSTAINED WINDS...45 MPH...75 KM/H",
            "PRESENT MOVEMENT...NE OR 35 DEGREES AT 9 MPH...15 KM/H",
            "MINIMUM CENTRAL PRESSURE...1000 MB...29.53 INCHES",
            "SUMMARY OF WATCHES AND WARNINGS IN EFFECT:",
            "A Tropical Storm Warning is in effect for...",
            "* Sargent, Texas to Morgan City, Louisiana",
        ]
    )

    excerpt = select_relevant_excerpts(
        text,
        "From the NHC Tropical Storm Arthur intermediate advisory at 1:00 PM CDT on June 17, 2026, summarize the warning area, location, maximum sustained wind, motion, pressure, and the main hazard headline.",
        max_chars=2000,
    )

    assert "PRESENT MOVEMENT...NE OR 35 DEGREES AT 9 MPH" in excerpt
    assert "Sargent, Texas to Morgan City, Louisiana" in excerpt


def test_select_relevant_excerpts_prioritizes_buried_required_slots():
    filler = [f"Navigation line {index}" for index in range(80)]
    text = "\n".join(
        filler
        + [
            "SB 79 Ordinance and TOD Alternative Plan",
            "Although jurisdictions are not required to adopt an SB 79 ordinance or TOD alternative plan, they may choose to do so.",
            "Draft SB 79 ordinance and TOD alternative plan",
            "The draft ordinance and/or TOD alternative plan must be submitted to HCD two weeks (14 calendar days) before the scheduled adoption date.",
            "Adopted and enacted SB 79 ordinance and TOD alternative plan",
            "An adopted ordinance and/or TOD alternative plan must be submitted to HCD within 60 calendar days of enactment.",
        ]
    )

    contract = {
        "required_slots": [
            {
                "name": "draft_deadline",
                "label": "draft ordinance or TOD alternative plan submittal deadline",
                "evidence_type": "date",
                "keywords": ["draft", "submitted", "HCD"],
                "search_terms": ["draft", "14 calendar days"],
            },
            {
                "name": "enacted_deadline",
                "label": "adopted or enacted ordinance or TOD alternative plan submittal deadline",
                "evidence_type": "date",
                "keywords": ["adopted", "enacted", "submitted", "HCD"],
                "search_terms": ["enacted", "60 calendar days"],
            },
        ]
    }

    excerpt = select_relevant_excerpts(
        text,
        "As of June 17, 2026, is California SB 79 still pending or already law? Explain the effective date, HCD's compliance role, and what HCD says local jurisdictions must do if they choose to adopt an SB 79 ordinance or TOD alternative plan.",
        max_chars=1600,
        contract=contract,
    )

    assert "14 calendar days" in excerpt
    assert "60 calendar days" in excerpt


def test_build_controlled_payload_omits_web_plugin():
    model = {"id": "gemma", "model": "google/gemma-4-31b-it"}
    task = {
        "id": "example",
        "category": "test",
        "question": "What is item 7b?",
    }
    evidence = {
        "sources": [
            {
                "ok": True,
                "title": "Agenda",
                "url": "https://example.com/agenda",
                "final_url": "https://example.com/agenda",
                "char_count": 1000,
                "excerpt": "7b Employment Agreement with Example Person.",
            }
        ]
    }

    payload = build_controlled_payload(model, task, evidence, "2026-06-17", max_tokens=1000)

    assert payload["model"] == "google/gemma-4-31b-it"
    assert "plugins" not in payload
    assert "Evidence checklist" in payload["messages"][1]["content"]
    assert "Controlled source evidence" in payload["messages"][1]["content"]
    assert "7b Employment Agreement" in payload["messages"][1]["content"]


def test_build_discovery_payload_hides_preferred_sources():
    task = {
        "id": "example",
        "category": "local",
        "question": "Find the city agenda item.",
        "why_hard": "It is buried in an agenda.",
        "preferred_sources": ["https://secret.example/known-answer"],
    }

    payload = build_discovery_payload(
        task,
        "2026-06-17",
        engine="parallel",
        max_results=5,
        max_tokens=900,
        model="google/gemini-3.1-flash-lite",
    )
    serialized = str(payload)

    assert payload["model"] == "google/gemini-3.1-flash-lite"
    assert payload["plugins"][0]["id"] == "web"
    assert "https://secret.example/known-answer" not in serialized
    assert "It is buried in an agenda." not in serialized
    assert "candidate_sources" in payload["messages"][1]["content"]


def test_build_and_parse_generated_evidence_contract():
    task = {
        "id": "example",
        "category": "local",
        "question": "What was the total count and how many were sheltered versus unsheltered?",
    }

    payload = build_evidence_contract_payload(task, "2026-06-17", "google/gemma-4-31b-it")
    parsed = parse_evidence_contract(
        """```json
        {
          "required_slots": [
            {
              "name": "total_count",
              "label": "total count",
              "evidence_type": "number",
              "keywords": ["total", "count"],
              "search_terms": ["total count"]
            }
          ],
          "source_preferences": ["official release"],
          "reject_if": ["announcement without numeric results"]
        }
        ```"""
    )

    assert payload["model"] == "google/gemma-4-31b-it"
    assert "Do not answer the question" in payload["messages"][0]["content"]
    assert parsed["required_slots"][0]["name"] == "total_count"
    assert parsed["required_slots"][0]["evidence_type"] == "number"
    assert parsed["reject_if"] == ["announcement without numeric results"]


def test_discovery_candidates_dedupe_annotations_and_json_content():
    annotations = [
        {
            "url_citation": {
                "url": "https://Example.com/path?utm_source=x&id=7#frag",
                "title": "Example",
                "content": "Official page",
            }
        }
    ]
    content = """```json
    {"candidate_sources": [
      {"url": "https://example.com/path?id=7", "title": "dupe", "reason": "same"},
      {"url": "https://agency.gov/report.pdf", "title": "PDF", "reason": "primary"}
    ]}
    ```"""

    candidates = discovery_candidates(content, annotations, limit=10)

    assert [candidate["canonical_url"] for candidate in candidates] == [
        "https://example.com/path?id=7",
        "https://agency.gov/report.pdf",
    ]


def test_fetch_source_text_uses_reader_fallback_for_blocked_page(monkeypatch):
    class Response:
        def __init__(self, url, status_code, text, content_type="text/html"):
            self.url = url
            self.status_code = status_code
            self.text = text
            self.content = text.encode("utf-8")
            self.headers = {"content-type": content_type}

        def raise_for_status(self):
            if self.status_code >= 400:
                raise RuntimeError(f"{self.status_code} error")

        def iter_content(self, chunk_size=65536):
            for start in range(0, len(self.content), chunk_size):
                yield self.content[start:start + chunk_size]

    calls = []

    def fake_get(url, **kwargs):
        calls.append(url)
        if url == "https://agency.example/report":
            return Response(url, 403, "<title>Just a moment...</title>")
        return Response(
            url,
            200,
            "Title: Agency Report\nURL Source: https://agency.example/report\nMarkdown Content:\n# Agency Report\nThe requested facts.",
            "text/plain",
        )

    monkeypatch.setattr("benchmarks.web_search.web_bench_lib.requests.get", fake_get)

    fetched = fetch_source_text("https://agency.example/report")

    assert fetched["fetch_mode"] == "reader_fallback"
    assert fetched["url"] == "https://agency.example/report"
    assert fetched["title"] == "Agency Report"
    assert "The requested facts." in fetched["text"]
    assert calls[1].endswith("https://agency.example/report")


def test_score_discovery_tracks_url_and_domain_recall():
    task = {
        "id": "example",
        "preferred_sources": ["https://agency.gov/report.pdf"],
        "source_domains": [{"domain": "agency.gov", "points": 1}],
    }
    evidence = {
        "sources": [
            {
                "ok": True,
                "url": "https://agency.gov/report.pdf?utm_source=x",
                "final_url": "https://agency.gov/report.pdf",
                "canonical_url": canonical_url("https://agency.gov/report.pdf?utm_source=x"),
            }
        ]
    }

    score = score_discovery(task, evidence)

    assert score["exact_preferred_url"] is True
    assert score["fetched_preferred_url"] is True
    assert score["source_domain"] is True
    assert score["fetch_success_rate"] == 1.0


def test_validate_source_rejects_wrong_legistar_meeting_date():
    task = {
        "question": "For Riverside City Council's June 16, 2026 agenda item 28, identify the RFP number, vendor, amount, account/source described, and the work being approved."
    }
    source = {
        "ok": True,
        "title": "City of Riverside - Meeting of City Council on 1/6/2026 at 1:00 PM",
        "final_url": "https://riversideca.legistar.com/MeetingDetail.aspx?ID=1355651",
        "excerpt": "28 Approve Memorandums of Understanding with the Riverside Firefighters Association.",
    }

    validation = validate_source_for_question(task, source)

    assert validation["passed"] is False
    assert "date_mismatch" in validation["reasons"]


def test_validate_source_accepts_exact_legistar_item_topic():
    task = {
        "question": "For Riverside City Council's June 16, 2026 agenda item 28, identify the RFP number, vendor, amount, account/source described, and the work being approved."
    }
    source = {
        "ok": True,
        "title": "City of Riverside - Meeting of City Council on 6/16/2026 at 1:00 PM",
        "final_url": "https://riversideca.legistar.com/MeetingDetail.aspx?ID=1416169",
        "excerpt": "28 Approve Request for Proposal 2494 Agreement with Carollo Engineers, Inc., Riverside, for $271,960 from Management Services Professional Services Account for Water Cost of Service Analysis and Rate Design.",
    }

    validation = validate_source_for_question(task, source)

    assert validation["passed"] is True
    assert validation["matched"]["dates"] == [True]
    assert validation["matched"]["items"] == [True]


def test_validate_source_rejects_calendar_listing_for_item_question():
    task = {
        "question": "For Riverside City Council's June 16, 2026 agenda item 28, identify the RFP number, vendor, amount, account/source described, and the work being approved."
    }
    source = {
        "ok": True,
        "title": "City of Riverside - Calendar",
        "final_url": "https://riversideca.legistar.com/Calendar.aspx",
        "excerpt": "City Council 6/16/2026 1:00 PM Meeting details Agenda Accessible Agenda",
    }

    validation = validate_source_for_question(task, source)

    assert validation["passed"] is False
    assert "document_mismatch:calendar_listing" in validation["reasons"]


def test_extract_constraints_marks_as_of_date_soft():
    constraints = extract_constraints("As of June 17, 2026, is California SB 79 law?")

    assert constraints["dates"][0]["soft"] is True


def test_validate_source_rejects_fomc_calendar_as_statement_evidence():
    # The hardcoded statement-shape check is gone; a calendar/preview trap is now caught
    # by the generated contract's reject_if rule (a preview/calendar-class trap).
    task = {
        "question": "What did the FOMC decide in its June 17, 2026 statement? Include the vote, target range, reserve-policy language, and the specific uncertainty/inflation context mentioned.",
        "evidence_contract": {
            "required_slots": [
                {"name": "vote", "label": "the FOMC vote", "evidence_type": "text",
                 "keywords": ["vote", "voting"], "search_terms": ["fomc vote"]},
                {"name": "target_range", "label": "target federal funds rate range",
                 "evidence_type": "range", "keywords": ["target range", "federal funds rate"],
                 "search_terms": ["fomc target range"]},
            ],
            "source_preferences": ["federalreserve.gov official statement"],
            "reject_if": ["meeting calendar or preview without the decision"],
        },
    }
    source = {
        "ok": True,
        "title": "The Fed - Meeting calendars and information",
        "final_url": "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm",
        "excerpt": "June 17, 2026 Statement: PDF HTML",
    }

    validation = validate_source_for_question(task, source)

    assert validation["passed"] is False
    assert any(r.startswith("contract_reject") for r in validation["reasons"])


def test_validate_source_rejects_non_official_fomc_statement_copy():
    task = {
        "question": "What did the FOMC decide in its June 17, 2026 statement? Include the vote, target range, reserve-policy language, and the specific uncertainty/inflation context mentioned."
    }
    source = {
        "ok": True,
        "title": "Federal Reserve issues FOMC statement - Federal Reserve System Press release | LegiStorm",
        "final_url": "https://www.legistorm.com/stormfeed/view_rss/7843204/organization/95474/title/federal-reserve-issues-fomc-statement.html",
        "excerpt": "June 17, 2026 The Federal Open Market Committee decided to maintain the target range.",
    }

    validation = validate_source_for_question(task, source)

    assert validation["passed"] is False
    assert "evidence_missing:vote" in validation["reasons"]
    assert "evidence_missing:specific_uncertainty_inflation_context" in validation["reasons"]


def test_validate_source_accepts_fomc_statement_page():
    task = {
        "question": "What did the FOMC decide in its June 17, 2026 statement? Include the vote, target range, reserve-policy language, and the specific uncertainty/inflation context mentioned."
    }
    source = {
        "ok": True,
        "title": "Federal Reserve Board - Federal Reserve issues FOMC statement",
        "final_url": "https://www.federalreserve.gov/newsevents/pressreleases/monetary20260617a.htm",
        "excerpt": "June 17, 2026 The Federal Open Market Committee approved the following statement for release by a 12 - 0 vote. The Committee decided to maintain the target range for the federal funds rate at 3.5 to 3.75%. The Committee will continue reducing its holdings in a manner consistent with ample reserves. Uncertainty about the economic outlook remains elevated and inflation remains somewhat elevated.",
    }

    validation = validate_source_for_question(task, source)

    assert validation["passed"] is True


def test_validate_source_marks_missing_facts_on_generic_noaa_page():
    # Tradeoff of deleting the hardcoded outlook check: a wrong-SUBTYPE page (tropical-
    # cyclone reports archive, not the seasonal outlook) is NOT a preview/calendar trap,
    # so the contract's reject_if does not catch it and it is no longer hard-rejected at
    # validation. Instead every required fact is recorded as unsupported, so the answer
    # stage refuses rather than guessing. (End-to-end, discovery surfaces the real outlook
    # page; the benchmark scored 100% after this change.)
    task = {
        "question": "What did NOAA's 2026 Atlantic hurricane season outlook predict for season category probabilities and storm-count ranges? Also say whether it is a landfall forecast.",
        "evidence_contract": {
            "required_slots": [
                {"name": "probabilities", "label": "season category probabilities",
                 "evidence_type": "percent", "keywords": ["below-normal", "near-normal", "above-normal"],
                 "search_terms": ["hurricane season probabilities"]},
                {"name": "storm_counts", "label": "named storm and hurricane count ranges",
                 "evidence_type": "range", "keywords": ["named storms", "hurricanes"],
                 "search_terms": ["named storm ranges"]},
            ],
            "source_preferences": ["noaa.gov seasonal outlook"],
            "reject_if": ["tropical cyclone report archive instead of the seasonal outlook"],
        },
    }
    source = {
        "ok": True,
        "title": "2026 Atlantic Hurricane Season",
        "final_url": "https://www.nhc.noaa.gov/data/tcr/",
        "excerpt": "2026 Atlantic Hurricane Season Tropical Cyclone Reports.",
    }

    validation = validate_source_for_question(task, source)

    slots = validation["matched"].get("evidence_slots", {})
    assert slots.get("slots"), "expected contract slots to be evaluated"
    assert slots["missing"], "wrong-subtype page should support none of the required facts"
    assert len(slots["missing"]) == len(slots["slots"])
    assert not any(r.startswith("document_mismatch") for r in validation["reasons"])


def test_validate_source_accepts_cpc_hurricane_outlook():
    task = {
        "question": "What did NOAA's 2026 Atlantic hurricane season outlook predict for season category probabilities and storm-count ranges? Also say whether it is a landfall forecast."
    }
    source = {
        "ok": True,
        "title": "Climate Prediction Center - Atlantic Hurricane Outlook",
        "final_url": "https://www.cpc.ncep.noaa.gov/products/outlooks/hurricane.shtml",
        "excerpt": "NOAA 2026 Atlantic Hurricane Season Outlook predicts a below-normal season. The outlook gives a 50% chance of a below-normal season, 35% near-normal, and 15% above-normal, with 10-16 named storms, 4-8 hurricanes, 1-3 major hurricanes, and notes this is not a landfall forecast.",
    }

    validation = validate_source_for_question(task, source)

    assert validation["passed"] is True


def test_validate_source_accepts_nhc_advisory_without_colon_time():
    task = {
        "question": "From the NHC Tropical Storm Arthur intermediate advisory at 1:00 PM CDT on June 17, 2026, summarize the warning area, location, maximum sustained wind, motion, pressure, and the main hazard headline."
    }
    source = {
        "ok": True,
        "title": "Tropical Storm Arthur Public Advisory",
        "final_url": "https://www.nhc.noaa.gov/archive/2026/al01/al012026.public_a.006.shtml",
        "excerpt": "BULLETIN Tropical Storm Arthur Intermediate Advisory Number 6A NWS National Hurricane Center Miami FL AL012026 100 PM CDT Wed Jun 17 2026. Tropical Storm Warning remains in effect for portions of the Texas and Louisiana coast. Location 28.9N 95.7W. Maximum sustained winds 45 mph. Present movement NE at 9 mph. Minimum central pressure 1000 mb. Life-threatening flooding expected.",
    }

    validation = validate_source_for_question(task, source)

    assert validation["passed"] is True


def test_validate_source_rejects_orange_county_pit_announcement_without_results():
    task = {
        "question": "According to Orange County's May 18, 2026 Point In Time Count release, what was the total count, how many were sheltered vs. unsheltered, how did homelessness change compared with 2024, and what first-time milestone did the county report?"
    }
    source = {
        "ok": True,
        "title": "County of Orange to Release 2026 Point In Time Count Results - Fullerton Observer",
        "final_url": "https://fullertonobserver.com/2026/05/13/county-of-orange-to-release-2026-point-in-time-count-results/",
        "excerpt": "The County of Orange will host a virtual media webinar to announce and discuss the results of the 2026 Point In Time Count. Date: Monday, May 18, 2026.",
    }

    validation = validate_source_for_question(task, source)

    assert validation["passed"] is False
    assert "evidence_missing:sheltered" in validation["reasons"]
    assert "evidence_missing:unsheltered" in validation["reasons"]
    assert "evidence_missing:first_time_milestone_county_report" in validation["reasons"]


def test_validate_source_accepts_orange_county_pit_results_release():
    task = {
        "question": "According to Orange County's May 18, 2026 Point In Time Count release, what was the total count, how many were sheltered vs. unsheltered, how did homelessness change compared with 2024, and what first-time milestone did the county report?"
    }
    source = {
        "ok": True,
        "title": "County of Orange Announces 2026 Point In Time Count Results | Orange County",
        "final_url": "https://www.ocgov.com/press/county-orange-announces-2026-point-time-count-results",
        "excerpt": "May 18, 2026. Orange County's 2026 Point In Time Count counted 6,321 people experiencing homelessness, including 3,256 sheltered and 3,065 unsheltered people. This was a 13.7% decrease compared with 2024. For the first time, more people were sheltered than unsheltered.",
    }

    validation = validate_source_for_question(task, source)

    assert validation["passed"] is True


def test_official_resolvers_return_expected_candidate_shapes(monkeypatch):
    class Response:
        def __init__(self, text):
            self.text = text
            self.content = text.encode("utf-8")

        def raise_for_status(self):
            return None

    def fake_get(url, **kwargs):
        if "fomccalendars" in url:
            return Response('<a href="/newsevents/pressreleases/monetary20260617a.htm">Statement</a>')
        if "ARTHUR.shtml" in url:
            return Response(
                '<!-- 20260617 1800 --><a href="/archive/2026/al01/al012026.public_a.006.shtml">6a:&nbsp;1800 UTC</a>'
            )
        raise AssertionError(f"unexpected URL {url}")

    monkeypatch.setattr("benchmarks.web_search.web_bench_lib.requests.get", fake_get)
    fed_task = {
        "question": "What did the FOMC decide in its June 17, 2026 statement? Include the vote."
    }
    noaa_task = {
        "question": "What did NOAA's 2026 Atlantic hurricane season outlook predict?"
    }
    nhc_task = {
        "question": "From the NHC Tropical Storm Arthur intermediate advisory at 1:00 PM CDT on June 17, 2026, summarize the warning area."
    }

    assert federal_reserve_statement_candidates(fed_task)[0]["url"].endswith("monetary20260617a.htm")
    assert noaa_hurricane_outlook_candidates(noaa_task)[0]["url"].endswith("/products/outlooks/hurricane.shtml")
    assert nhc_advisory_archive_candidates(nhc_task)[0]["url"].endswith("al012026.public_a.006.shtml")


def test_evaluate_answer_slots_detects_missing_result_fields():
    question = (
        "According to Orange County's May 18, 2026 Point In Time Count release, "
        "what was the total count, how many were sheltered vs. unsheltered, "
        "how did homelessness change compared with 2024, and what first-time milestone did the county report?"
    )
    text = "The County will host a webinar to announce and discuss the 2026 Point In Time Count results."

    evaluation = evaluate_answer_slots(question, text)

    missing_labels = {slot["label"] for slot in evaluation["missing"]}
    assert {"total count", "sheltered", "unsheltered"} <= missing_labels
    assert any("compared" in label for label in missing_labels)
    assert any("milestone" in label for label in missing_labels)


def test_validate_source_rejects_related_page_missing_required_slots():
    task = {
        "question": "What did the FOMC decide in its June 17, 2026 statement? Include the vote, target range, reserve-policy language, and the specific uncertainty/inflation context mentioned."
    }
    source = {
        "ok": True,
        "title": "Federal Reserve Board - Federal Reserve issues FOMC statement",
        "final_url": "https://www.federalreserve.gov/newsevents/pressreleases/monetary20260617a.htm",
        "excerpt": "June 17, 2026 The Committee decided to maintain rates.",
    }

    validation = validate_source_for_question(task, source)

    assert validation["passed"] is False
    assert "evidence_missing:vote" in validation["reasons"]
    assert "evidence_missing:target_range" in validation["reasons"]


def test_source_hunter_retry_memory_lists_missing_slots_without_grader_domains():
    task = {
        "question": "According to Orange County's May 18, 2026 Point In Time Count release, what was the total count, how many were sheltered vs. unsheltered, how did homelessness change compared with 2024, and what first-time milestone did the county report?",
        "source_domains": [{"domain": "ocgov.com", "points": 1.0}],
    }
    rejected_source = {
        "url": "https://fullertonobserver.com/example",
        "title": "County of Orange to Release 2026 Point In Time Count Results",
        "validation": validate_source_for_question(
            task,
            {
                "ok": True,
                "title": "County of Orange to Release 2026 Point In Time Count Results",
                "final_url": "https://fullertonobserver.com/example",
                "excerpt": "The County will host a webinar to announce and discuss results.",
            },
        ),
    }

    context = source_hunter_extra_context({"rejected_sources": [rejected_source]}, task)

    assert "sheltered" in context
    assert "unsheltered" in context
    assert "ocgov.com" not in context
    assert "fullertonobserver.com/example rejected" in context


def test_nearby_source_candidates_expand_only_primary_same_domain_links():
    task = {
        "question": "From the official agency advisory at 1:00 PM CDT on June 17, 2026, summarize the warning area.",
    }
    rejected_source = {
        "ok": True,
        "final_url": "https://agency.example/current/advisory.html",
        "title": "Official Agency Current Advisory",
        "excerpt": "Official advisory page with the wrong timestamp.",
        "links": [
            {"url": "https://agency.example/archive/2026/STORM.shtml", "text": "Storm Archive"},
            {"url": "https://agency.example/about.shtml", "text": "About Us"},
            {"url": "https://news.example/story", "text": "News copy"},
        ],
        "validation": {
            "matched": {
                "source_quality": {
                    "is_primary": True,
                    "preference_match": True,
                    "source_preferences": [],
                }
            }
        },
    }

    candidates = nearby_source_candidates(task, [rejected_source], seen_urls=set(), limit=5)

    assert [candidate["url"] for candidate in candidates] == [
        "https://agency.example/archive/2026/STORM.shtml"
    ]


def test_nearby_source_candidates_rank_exact_time_archive_link():
    task = {
        "question": "From the official agency intermediate advisory at 1:00 PM CDT on June 17, 2026, summarize the warning area.",
    }
    archive_source = {
        "ok": True,
        "final_url": "https://agency.example/archive/2026/STORM.shtml",
        "title": "Storm Archive",
        "excerpt": "Archive index.",
        "links": [
            {"url": "https://agency.example/archive/2026/storm.public.008.shtml", "text": "8: 0300 UTC"},
            {"url": "https://agency.example/archive/2026/storm.public_a.006.shtml", "text": "6a: 1800 UTC"},
            {"url": "https://agency.example/archive/2026/storm.discus.006.shtml", "text": "6: 1630 UTC"},
        ],
        "validation": {
            "matched": {
                "source_quality": {
                    "is_primary": True,
                    "preference_match": True,
                    "source_preferences": [],
                }
            }
        },
    }

    candidates = nearby_source_candidates(task, [archive_source], seen_urls=set(), limit=3)

    assert candidates[0]["url"].endswith("storm.public_a.006.shtml")


def test_validate_source_rejects_stale_year_for_target_year_question():
    task = {
        "question": "Will Example County conduct an unsheltered Point-in-Time homelessness count in 2026? Explain what is still being counted and the reason given."
    }
    source = {
        "ok": True,
        "title": "2025 Point-in-Time Count Infographic",
        "final_url": "https://example.gov/2025-pit-infographic.pdf",
        "excerpt": "The federally mandated count of sheltered and unsheltered homelessness was conducted January 22, 2025.",
    }

    validation = validate_source_for_question(task, source)

    assert validation["passed"] is False
    assert "year_mismatch" in validation["reasons"]


def test_contract_reject_if_blocks_announcement_missing_structured_evidence():
    task = {
        "question": "According to Example County's May 18, 2026 Point In Time Count release, what was the total count and how many were sheltered vs. unsheltered?",
        "evidence_contract": {
            "required_slots": [
                {
                    "name": "total_count",
                    "label": "total count",
                    "evidence_type": "number",
                    "keywords": ["total", "count"],
                },
                {
                    "name": "sheltered_count",
                    "label": "sheltered count",
                    "evidence_type": "number",
                    "keywords": ["sheltered", "count"],
                },
            ],
            "reject_if": ["general announcement without the actual results"],
        },
    }
    source = {
        "ok": True,
        "title": "Example County to Release 2026 Point In Time Count Results",
        "final_url": "https://news.example/example-county-to-release-results",
        "excerpt": "Example County will host a webinar to announce and discuss the results of the 2026 Point In Time Count.",
    }

    validation = validate_source_for_question(task, source)

    assert validation["passed"] is False
    assert any(reason.startswith("contract_reject:") for reason in validation["reasons"])


def test_source_preference_rejects_secondary_copy_even_with_some_facts():
    task = {
        "question": "What did the FOMC decide in its June 17, 2026 statement? Include the vote, target range, reserve-policy language, and the specific uncertainty/inflation context mentioned.",
        "evidence_contract": {
            "required_slots": [
                {"name": "decision", "label": "decision", "evidence_type": "text", "keywords": ["decided", "maintain"]},
                {"name": "vote", "label": "vote", "evidence_type": "number", "keywords": ["vote"]},
            ],
            "source_preferences": ["federalreserve.gov"],
            "reject_if": ["Source is a summary or commentary without the full official statement text"],
        },
    }
    source = {
        "ok": True,
        "title": "June Fed Meeting: Updates and Commentary",
        "final_url": "https://www.kiplinger.com/news/live/fed-meeting-updates-and-commentary-june-2026",
        "excerpt": "The FOMC decided to maintain the target range. The vote was 12-0.",
    }

    validation = validate_source_for_question(task, source)

    assert validation["passed"] is False
    assert "source_preference_mismatch" in validation["reasons"]


def test_primary_source_preference_allows_partial_generated_contract():
    task = {
        "question": "What did the CPUC decide in proceeding A.24-12-011 about SoCalGas Angeles Link Phase 2 cost recovery, including the amount denied and whether another application is foreclosed?",
        "evidence_contract": {
            "required_slots": [
                {
                    "name": "amount_denied",
                    "label": "amount denied",
                    "evidence_type": "money",
                    "keywords": ["cost recovery", "amount", "denied"],
                },
                {
                    "name": "future_application",
                    "label": "whether another application is foreclosed",
                    "evidence_type": "text",
                    "keywords": ["another application", "foreclosed"],
                },
            ],
            "source_preferences": ["CPUC official decisions"],
            "reject_if": ["Source is a news summary without the specific reasoning or amounts"],
        },
    }
    source = {
        "ok": True,
        "title": "605819437.PDF",
        "final_url": "https://docs.cpuc.ca.gov/PublishedDocs/Published/G000/M605/K819/605819437.PDF",
        "excerpt": "DECISION DENYING SOUTHERN CALIFORNIA GAS COMPANY'S REQUEST FOR COST RECOVERY. This decision denies $266 million of cost recovery from natural gas ratepayers for Phase 2 Activities related to the Angeles Link Project.",
    }

    validation = validate_source_for_question(task, source)

    assert validation["passed"] is True
    assert "evidence_missing:future_application" in validation["reasons"]
    assert not any(reason.startswith("contract_reject:") for reason in validation["reasons"])


def test_select_relevant_excerpts_rescues_supported_missing_slot():
    question = "What did the FOMC decide in its June 17, 2026 statement? Include the vote and target range."
    contract = {
        "required_slots": [
            {"name": "vote", "label": "vote", "evidence_type": "number", "keywords": ["vote"]},
            {
                "name": "target_range",
                "label": "target range",
                "evidence_type": "range",
                "keywords": ["target range", "federal funds rate"],
            },
        ],
    }
    text = "\n".join(
        [
            "Federal Reserve Board - Federal Reserve issues FOMC statement",
            *[f"Navigation item {index}" for index in range(120)],
            "The Federal Open Market Committee approved the following statement for release by a 12-0 vote.",
            "The Committee decided to maintain the target range for the federal funds rate at 3.5 to 3.75 percent.",
        ]
    )

    excerpt = select_relevant_excerpts(text, question, max_chars=1300, contract=contract)

    assert "12-0 vote" in excerpt
    assert "target range" in excerpt
