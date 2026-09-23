from datetime import date

from newscaster.scrapers import riverside

CARD = """<figure class="pressCard card-1"><a href="{href}"><img src="x.png"/></a>
  <div class="date"><span class="day">{day}</span><span class="month">{month}</span></div>
  <figcaption><h4 class="departmentCard">{dept}</h4><h4><p>{title}</p></h4></figcaption></figure>"""

PAGE = "<html><body>" + "".join(CARD.format(**c) for c in [
    dict(href="/press/traffic-stop", day="23", month="Sept", dept="Police", title="TRAFFIC STOP UNCOVERS IMPERSONATION OF POLICE"),
    dict(href="/press/more-arrests", day="22", month="Sept", dept="Police", title="Continued Enforcement Efforts Lead to Additional Arrests"),
    dict(href="/press/old", day="21", month="Sept", dept="Police", title="Suspect Arrested for Document Fraud"),
]) + "</body></html>"


def test_keeps_today_and_yesterday_only():
    items = riverside.parse_press_releases(PAGE, date(2026, 9, 23))
    assert [(i[0], i[1], i[2]) for i in items] == [
        (date(2026, 9, 23), "Police", "TRAFFIC STOP UNCOVERS IMPERSONATION OF POLICE"),
        (date(2026, 9, 22), "Police", "Continued Enforcement Efforts Lead to Additional Arrests"),
    ]
    assert items[0][3] == "https://www.riversideca.gov/press/traffic-stop"


def test_december_card_read_in_january_is_last_year():
    page = CARD.format(href="/press/x", day="31", month="Dec", dept="Citywide", title="New Year notice")
    items = riverside.parse_press_releases(page, date(2027, 1, 1))
    assert items and items[0][0] == date(2026, 12, 31)


def test_scraper_says_when_nothing_is_new(monkeypatch):
    class Resp:
        content = PAGE.encode()
        def raise_for_status(self):
            pass
    monkeypatch.setattr(riverside.requests, "get", lambda *a, **k: Resp())
    out = riverside.riverside_scraper(today=date(2026, 10, 5))
    assert "released no press releases today or yesterday" in out
