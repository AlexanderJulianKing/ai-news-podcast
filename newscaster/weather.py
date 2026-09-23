"""Weather line for the intro, built around a Temecula to La Jolla commute.

The listener wakes in Temecula, hears the show on the drive south, spends the day
in La Jolla, and is back in Temecula at night. So the line gives La Jolla's daytime
numbers and Temecula's evening and overnight numbers. Rain is mentioned only when it
is likely; dry is the default here and not worth airtime.
"""
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

import requests

import newscaster.config as _config

FORECAST_URL = "https://api.openweathermap.org/data/2.5/forecast"
LOCAL_TZ = ZoneInfo("America/Los_Angeles")
DAY_PLACE, NIGHT_PLACE = "La Jolla", "Temecula"
PLACES = {
    DAY_PLACE: (32.8328, -117.2713),
    NIGHT_PLACE: (33.4936, -117.1484),
}
RAIN_CHANCE_TO_MENTION = 0.3   # below this, rain is not worth airtime
UNAVAILABLE = "the weather is not available."


def fetch_forecast(lat, lon):
    """Five-day forecast in three-hour steps, Fahrenheit. Returns the parsed JSON or None."""
    try:
        response = requests.get(
            FORECAST_URL,
            params={"lat": lat, "lon": lon, "units": "imperial", "appid": _config.OPENWEATHERMAP_API_KEY},
            timeout=20,
        )
    except requests.RequestException:
        return None
    if response.status_code != 200:
        return None
    return response.json()


def _slots(data):
    """(local datetime, temp, low, high, chance of rain) for each forecast step."""
    out = []
    for item in (data or {}).get("list", []):
        when = datetime.fromtimestamp(item["dt"], tz=LOCAL_TZ)
        main = item["main"]
        out.append((when, main["temp"], main.get("temp_min", main["temp"]), main.get("temp_max", main["temp"]), item.get("pop", 0.0)))
    return out


def _nearest(slots, target):
    """The forecast step closest to `target`, or None when nothing is within two hours."""
    best = min(slots, key=lambda s: abs(s[0] - target), default=None)
    if best is None or abs(best[0] - target) > timedelta(hours=2):
        return None
    return best


def _part_of_day(when):
    hour = when.hour
    if hour < 12:
        return "this morning"
    if hour < 17:
        return "this afternoon"
    if hour < 21:
        return "this evening"
    return "tonight"


def rain_phrase(slots_by_place, today):
    """One clause on rain across every place, for today from 6 a.m. to midnight.
    Returns "" when rain is unlikely everywhere."""
    mentions = []
    for place, slots in slots_by_place.items():
        window = [s for s in slots if s[0].date() == today and s[0].hour >= 6]
        if not window:
            continue
        wettest = max(window, key=lambda s: s[4])
        if wettest[4] >= RAIN_CHANCE_TO_MENTION:
            mentions.append("a {} percent chance of rain in {} {}".format(int(round(wettest[4] * 100, -1)), place, _part_of_day(wettest[0])))
    if not mentions:
        return ""
    return "there is " + " and ".join(mentions)


def describe(forecasts, now=None):
    """Build the weather clause from {place: forecast JSON}. `now` is injectable for tests."""
    now = now or datetime.now(LOCAL_TZ)
    today = now.date()
    slots_by_place = {place: _slots(data) for place, data in forecasts.items() if data}
    parts = []

    day_slots = slots_by_place.get(DAY_PLACE)
    if day_slots:
        daytime = [s for s in day_slots if s[0].date() == today]
        if daytime:
            text = "in {} today the high will be {} degrees".format(DAY_PLACE, round(max(s[3] for s in daytime)))
            midday = _nearest(daytime, datetime.combine(today, time(12), tzinfo=LOCAL_TZ))
            five = _nearest(daytime, datetime.combine(today, time(17), tzinfo=LOCAL_TZ))
            if midday and five:
                text += ", around {} at midday and {} at five".format(round(midday[1]), round(five[1]))
            parts.append(text)

    night_slots = slots_by_place.get(NIGHT_PLACE)
    if night_slots:
        evening_start = datetime.combine(today, time(18), tzinfo=LOCAL_TZ)
        overnight = [s for s in night_slots if evening_start <= s[0] <= evening_start + timedelta(hours=14)]
        if overnight:
            text = "back in {} the overnight low will be {} degrees".format(NIGHT_PLACE, round(min(s[2] for s in overnight)))
            evening = _nearest(overnight, datetime.combine(today, time(20), tzinfo=LOCAL_TZ))
            if evening:
                text = "back in {} it will be about {} this evening, with an overnight low of {} degrees".format(
                    NIGHT_PLACE, round(evening[1]), round(min(s[2] for s in overnight)))
            parts.append(text)

    if not parts:
        return UNAVAILABLE
    rain = rain_phrase(slots_by_place, today)
    if rain:
        parts.append(rain)
    return "; ".join(parts[:-1]) + ("; and " if len(parts) > 1 else "") + parts[-1] + "."


def get_daily_temp():
    """The weather clause the intro prompt is told to open with."""
    return describe({place: fetch_forecast(lat, lon) for place, (lat, lon) in PLACES.items()})
