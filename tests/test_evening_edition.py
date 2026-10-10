"""Evening edition (2026-10-09): a second weekday show for the drive home.

The morning show's files, ledger entries and history must stay exactly as they were,
and the evening show must know what the morning show already aired.
"""
from datetime import date, datetime

from newscaster import dedup, edition, schedule
from newscaster.scrapers import topic_finder as tf

OCT9, OCT10 = date(2026, 10, 9), date(2026, 10, 10)


def test_evening_gets_its_own_key_and_the_morning_key_is_unchanged():
    assert edition.episode_key(OCT9, "morning") == "2026_10_09"
    assert edition.episode_key(OCT9, "evening") == "2026_10_09_pm"
    assert edition.key_date("2026_10_09_pm") == OCT9 == edition.key_date("2026_10_09")
    assert "2026_10_09" < "2026_10_09_pm" < "2026_10_10"           # ledger order = air order
    # the fact check globs a key's scripts; the morning key must not pick up the evening's
    assert not edition.belongs_to("2026_10_09_pm_segment_0.txt", "2026_10_09")
    assert edition.belongs_to("2026_10_09_segment_0.txt", "2026_10_09")


def _arc(*episodes):
    return {"arcs": {"quake": {"last_covered": episodes[-1]["date"], "audience_state": "A 6.1 quake hit.",
                               "episodes": list(episodes)}}}


def test_ledger_keeps_and_counts_evening_coverage(monkeypatch):
    ledger = _arc({"date": "2026_10_09_pm", "coverage": "main", "coverage_slot": 0, "headline": "Quake hits"})
    monkeypatch.setattr(dedup, "date", type("D", (), {"today": staticmethod(lambda: OCT10)}))
    assert "quake" in dedup.prune_ledger(ledger)["arcs"]            # an unparseable key used to delete the arc
    assert dedup.last_main_date(ledger["arcs"]["quake"]) == OCT9
    # led this morning -> still barred from leading tonight unless it escalates
    morning = _arc({"date": "2026_10_09", "coverage": "main", "coverage_slot": 0, "headline": "Quake hits"})
    text, _ = dedup.apply_coverage_depth("[UPDATE: quake] Aftershock", morning, today=OCT9)
    assert text.startswith("[UPDATE: quake]")


def test_followups_say_this_morning_and_yesterday_evening():
    morning = _arc({"date": "2026_10_09", "coverage": "main", "coverage_slot": 0, "headline": "Quake hits"})
    note = tf._followup_note(("UPDATE", "quake"), morning, today=OCT9, current_key="2026_10_09_pm")
    assert note and "this morning" in note
    evening = _arc({"date": "2026_10_09_pm", "coverage": "side", "coverage_slot": 2, "headline": "Quake hits"})
    note = tf._followup_note(("UPDATE", "quake"), evening, today=OCT10, current_key="2026_10_10")
    assert note and "yesterday evening" in note
    # the evening's own episode is not "earlier coverage"
    assert tf._followup_note(("UPDATE", "quake"), evening, today=OCT9, current_key="2026_10_09_pm") is None


def test_schedule_runs_morning_daily_and_evening_on_weekdays_only():
    fri, sat = datetime(2026, 10, 9, 15, 50), datetime(2026, 10, 10, 15, 50)
    assert schedule.due_edition(datetime(2026, 10, 10, 4, 5), done=set()) == "morning"
    assert schedule.due_edition(fri, done={"2026_10_09"}) == "evening"
    assert schedule.due_edition(fri, done={"2026_10_09", "2026_10_09_pm"}) is None
    assert schedule.due_edition(sat, done={"2026_10_10"}) is None
    assert schedule.due_edition(datetime(2026, 10, 9, 15, 40), done=set()) is None
    assert schedule.due_edition(datetime(2026, 10, 9, 17, 5), done=set()) is None   # too late for the drive


def test_evening_show_speaks_like_an_evening_show():
    from newscaster import weather
    from newscaster.script.intro import greeting
    from newscaster.prompts import outro_for
    assert greeting("Friday, October ninth", "evening").startswith("Good evening")
    assert greeting("Friday, October ninth", "morning").startswith("Good Morning")
    assert "evening" in outro_for("evening") and "tomorrow morning" in outro_for("evening")
    now = datetime(2026, 10, 9, 15, 45, tzinfo=weather.LOCAL_TZ)
    slot = lambda h, temp: {"dt": int(datetime(2026, 10, 9 if h < 24 else 10, h % 24, tzinfo=weather.LOCAL_TZ).timestamp()),
                            "main": {"temp": temp, "temp_min": temp - 2, "temp_max": temp + 2}, "pop": 0}
    forecast = {"list": [slot(h, 70 - (h - 12)) for h in range(12, 33, 3)]}
    line = weather.describe({"La Jolla": forecast, "Temecula": forecast}, now=now, edition="evening")
    assert "La Jolla today" not in line and "Temecula" in line
