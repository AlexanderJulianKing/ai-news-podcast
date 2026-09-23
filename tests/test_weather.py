"""The weather line: local-time grouping, the commute's two places, and rain."""
from datetime import datetime, timedelta

from newscaster import weather
from newscaster.weather import LOCAL_TZ


def _forecast(start, temps, pops=None):
    """Three-hour steps from `start` (local). temps: list of temperatures."""
    pops = pops or [0.0] * len(temps)
    return {"list": [
        {"dt": int((start + timedelta(hours=3 * i)).timestamp()),
         "main": {"temp": t, "temp_min": t - 1, "temp_max": t + 1}, "pop": p}
        for i, (t, p) in enumerate(zip(temps, pops))
    ]}


NOW = datetime(2026, 9, 21, 4, 0, tzinfo=LOCAL_TZ)          # the show runs at 4 a.m.
START = datetime(2026, 9, 21, 5, 0, tzinfo=LOCAL_TZ)         # steps at 5, 8, 11, 14, 17, 20, 23, 02, 05, 08
SD = [60, 66, 72, 77, 74, 68, 64, 62, 61, 65]
TEM = [52, 62, 78, 88, 84, 70, 62, 57, 54, 60]


def test_line_gives_la_jolla_by_day_and_temecula_by_night():
    line = weather.describe({"La Jolla": _forecast(START, SD), "Temecula": _forecast(START, TEM)}, now=NOW)
    assert "in La Jolla today the high will be 78 degrees" in line      # max temp_max today
    assert "around 72 at midday and 74 at five" in line                  # steps nearest 12:00 and 17:00
    assert "back in Temecula it will be about 70 this evening" in line   # step nearest 20:00
    assert "overnight low of 53 degrees" in line                         # min temp_min from 18:00 to 08:00
    assert "rain" not in line                                            # dry is the default and is not mentioned
    assert line.endswith("overnight low of 53 degrees.")
    assert "Riverside" not in line


def test_today_means_the_local_day_not_the_utc_day():
    # 17:00 local on the 21st is already the 22nd in UTC; it must still count as today.
    line = weather.describe({"La Jolla": _forecast(START, [60, 60, 60, 60, 95, 60, 60, 60, 60, 60])}, now=NOW)
    assert "the high will be 96 degrees" in line


def test_rain_is_named_by_place_and_time_when_likely():
    pops = [0, 0, 0.1, 0.62, 0.2, 0, 0, 0, 0, 0]
    line = weather.describe({"La Jolla": _forecast(START, SD, pops), "Temecula": _forecast(START, TEM)}, now=NOW)
    assert "a 60 percent chance of rain in La Jolla this afternoon" in line
    assert "Temecula this" not in line.split("chance of rain")[1]        # Temecula stays dry, so it is not named


def test_one_place_failing_still_gives_the_other_and_both_failing_says_so():
    line = weather.describe({"La Jolla": None, "Temecula": _forecast(START, TEM)}, now=NOW)
    assert line.startswith("back in Temecula")
    assert weather.describe({"La Jolla": None, "Temecula": None}, now=NOW) == weather.UNAVAILABLE
