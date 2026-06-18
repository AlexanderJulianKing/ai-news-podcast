"""Regression tests for contract-driven document validation.

The source hunter no longer carries hardcoded FOMC/NOAA/NHC document-shape checks.
Source validation is driven by the generated evidence contract:
  * generic "statement"/"advisory"/"outlook" news questions are NOT hard-rejected
    for lacking any particular family's phrasing, and
  * trap pages (a meeting calendar/preview standing in for the real document) are
    still rejected when the contract supplies a sensible ``reject_if`` rule.

No network or API keys: ``validate_source_for_question`` is pure.
"""
import newscaster.source_hunter_primitives as P


def _validate(question, excerpt, url="https://example.gov/page", contract=None):
    task = {"question": question}
    if contract is not None:
        task["evidence_contract"] = contract
    source = {"title": "", "url": url, "excerpt": excerpt, "ok": True, "links": []}
    return P.validate_source_for_question(task, source)


# --- generic questions must not be hard-rejected by any family-specific document check ---

def test_generic_statement_not_document_rejected():
    res = _validate(
        "What did the mayor's June 2026 budget statement propose for libraries and police?",
        "Mayor Lopez announced June 9, 2026 that the budget protects all 12 library branches "
        "and adds 15 police officer positions.",
        url="https://www.cityexample.gov/news/budget-2026",
    )
    assert not any(r.startswith("document_mismatch") for r in res["reasons"])


def test_generic_advisory_not_document_rejected():
    res = _validate(
        "What does the CDC travel advisory for Country Y recommend in June 2026?",
        "CDC, June 2026: travel notice for Country Y raised to Level 2. Travelers should "
        "practice enhanced precautions due to a measles outbreak.",
        url="https://www.cdc.gov/travel/notices/country-y",
    )
    assert not any(r.startswith("document_mismatch") for r in res["reasons"])


def test_generic_outlook_not_document_rejected():
    res = _validate(
        "What is Acme Corporation's revenue outlook for fiscal year 2026?",
        "Acme Corporation raised its full-year fiscal 2026 revenue guidance to a range of "
        "$4.2 billion to $4.4 billion, citing strong cloud demand.",
        url="https://investors.acme.com/guidance",
    )
    assert not any(r.startswith("document_mismatch") for r in res["reasons"])


# --- the contract's reject_if is now the trap-catcher (replaces the hardcoded checks) ---

def test_contract_reject_if_catches_calendar_trap():
    contract = {
        "required_slots": [
            {"name": "rate", "label": "target federal funds rate range", "evidence_type": "range",
             "keywords": ["target range", "federal funds rate"], "search_terms": ["fomc rate"]},
        ],
        "source_preferences": ["federalreserve.gov official statement"],
        "reject_if": ["meeting calendar or preview without the decision"],
    }
    res = _validate(
        "What did the FOMC decide in its June 17, 2026 statement? Include the vote and target range.",
        "Federal Reserve FOMC. The June 17, 2026 meeting appears on the FOMC meeting calendar. "
        "Statements and minutes are posted after each meeting.",
        url="https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm",
        contract=contract,
    )
    assert res["passed"] is False
    assert any(r.startswith("contract_reject") for r in res["reasons"])


def test_good_official_source_passes_with_contract():
    contract = {
        "required_slots": [
            {"name": "rate", "label": "target federal funds rate range", "evidence_type": "range",
             "keywords": ["target range", "federal funds rate"], "search_terms": ["fomc rate"]},
        ],
        "source_preferences": ["federalreserve.gov official statement"],
        "reject_if": ["meeting calendar or preview without the decision"],
    }
    res = _validate(
        "What did the FOMC decide in its June 17, 2026 statement? Include the vote and target range.",
        "Federal Reserve FOMC statement, June 17, 2026. The Committee decided to maintain the "
        "target range for the federal funds rate at 4-1/4 to 4-1/2 percent. Voting for the action "
        "were the Chair and six members.",
        url="https://www.federalreserve.gov/newsevents/pressreleases/monetary20260617a.htm",
        contract=contract,
    )
    assert res["passed"] is True
