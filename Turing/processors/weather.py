"""Today's weather in Siena (the SAINET day), fetched, with the written line as the fallback.

The sky is the one fact everybody in the room can check from where they are
sitting. A guest complaining about the cold while the sun is out has planted a
bait against itself, and anybody actually in Siena collects it for free — which
is why the line was moved off a hand-written stereotype in the first place (at
CLiC-it it was Palermo's "thirty degrees", while it drizzled at twenty-five).

`update_weather.py` does the same job by rewriting `context_it.txt`, and that is
still the right thing before a run. This is for the case that matters more: a
launcher started by somebody who does not know the agent has a knowledge base,
let alone that a line in it goes stale. One request at startup, a short timeout,
and every failure falls back to whatever the file already said.

Nothing here raises. A guest that will not start because a weather service is
down is worse than a guest talking about yesterday's sky.
"""
import datetime
import json
import urllib.request

CITY = "Siena"
WHERE = (43.3188, 11.3308)
API = ("https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
       "&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,weather_code"
       "&timezone=Europe%2FRome&start_date={day}&end_date={day}")

# WMO codes, grouped the way a person would describe the sky rather than the
# way the standard splits them.
SKY = [
    ((0, 1), "sereno"),
    ((2,), "poco nuvoloso"),
    ((3,), "nuvoloso"),
    ((45, 48), "nebbia"),
    ((51, 53, 55, 56, 57), "pioviggine"),
    ((61, 63, 65, 66, 67, 80, 81, 82), "pioggia"),
    ((95, 96, 99), "temporale"),
]


def describe(code: int) -> str:
    for codes, word in SKY:
        if code in codes:
            return word
    return "variabile"


def sentence(daily: dict) -> str:
    """The one line that goes into the knowledge base."""
    high = round(daily["temperature_2m_max"][0])
    low = round(daily["temperature_2m_min"][0])
    rain = daily["precipitation_sum"][0] or 0.0
    sky = describe(int(daily["weather_code"][0]))

    parts = [f"Oggi a {CITY} {sky}, massima {high} gradi, minima {low}"]
    if rain >= 1.0:
        parts.append("e piove")
    elif rain > 0.0:
        parts.append("con due gocce")
    line = ", ".join(parts) + "."

    # Say what the number means (for November), rather than leaving the model to
    # reach for a stereotype.
    if high <= 8:
        line += " Fa freddo davvero."
    elif high <= 14:
        line += " Fa freddino, ci vuole la giacca."
    else:
        line += " Per essere novembre non fa freddo."
    return line


def live_line(day: str = "", timeout: float = 6.0) -> str:
    """Today in the event's city, or "" if anything at all goes wrong.

    Deliberately swallows everything, including a malformed answer: the caller's
    fallback is a true-ish line from a file, which beats both a crash and a
    confidently wrong one built from half a response.
    """
    day = day or datetime.date.today().isoformat()
    try:
        url = API.format(lat=WHERE[0], lon=WHERE[1], day=day)
        with urllib.request.urlopen(url, timeout=timeout) as response:
            daily = json.loads(response.read().decode("utf-8"))["daily"]
        return sentence(daily)
    except Exception:
        return ""
