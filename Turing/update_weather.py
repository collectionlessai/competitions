"""Put the real Palermo weather into the knowledge base.

The context file used to assert "fa caldo, sui trenta gradi" — written by hand
and never checked. The forecast for the conference days is 26.7, 27.3 and 25.4
degrees, with drizzle on two of the three. That gap matters more than it looks:
the weather is the single most checkable fact between people standing in the
same city. An agent complaining about thirty degrees while it drizzles at
twenty-five has planted a bait against itself, and anyone actually in Palermo
collects it for free.

So the line is filled from a forecast instead of from a stereotype. Run it the
morning of each conference day — a forecast fetched nine days early is worth
very little, and this costs one request:

    python update_weather.py            # today in Palermo
    python update_weather.py 2026-09-15 # a specific day

It rewrites only the ## METEO block of `processors/context_it.txt` and leaves
everything else alone. With no network it changes nothing and says so, which is
the right failure: a stale true-ish line beats a wrong fresh one.
"""
import datetime
import io
import json
import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
CONTEXT = os.path.join(HERE, "processors", "context_it.txt")

PALERMO = (38.1157, 13.3615)
API = ("https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
       "&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,weather_code"
       "&timezone=Europe%2FRome&start_date={day}&end_date={day}")

# WMO codes, grouped the way a person would actually describe the sky rather
# than the way the standard splits them.
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


def fetch(day: str) -> dict:
    url = API.format(lat=PALERMO[0], lon=PALERMO[1], day=day)
    with urllib.request.urlopen(url, timeout=20) as response:
        return json.loads(response.read().decode("utf-8"))


def sentence(daily: dict) -> str:
    high = round(daily["temperature_2m_max"][0])
    low = round(daily["temperature_2m_min"][0])
    rain = daily["precipitation_sum"][0] or 0.0
    sky = describe(int(daily["weather_code"][0]))

    parts = [f"Oggi a Palermo {sky}, massima {high} gradi, minima {low}"]
    if rain >= 1.0:
        parts.append("e piove")
    elif rain > 0.0:
        parts.append("con due gocce")
    line = ", ".join(parts) + "."
    # The old line claimed heat that was not there. Say what the number means
    # rather than leaving the model to reach for the stereotype again.
    if high >= 30:
        line += " Fa caldo davvero."
    elif high >= 27:
        line += " Si sta bene, caldo ma non da morire."
    else:
        line += " Non fa il caldo che uno si aspetta a Palermo."
    return line


def rewrite(line: str) -> bool:
    text = io.open(CONTEXT, encoding="utf-8", newline="").read()
    eol = "\r\n" if "\r\n" in text else "\n"
    block = f"## METEO{eol}{line}{eol}{eol}"

    start = text.find("## METEO")
    if start >= 0:
        end = text.find("## ", start + 3)
        text = text[:start] + block + (text[end:] if end > 0 else "")
    else:
        marker = "## BLOCCO sede"
        at = text.find(marker)
        if at < 0:
            return False
        text = text[:at] + block + eol + text[at:]

    io.open(CONTEXT, "w", encoding="utf-8", newline="").write(text)
    return True


def main() -> int:
    day = sys.argv[1] if len(sys.argv) > 1 else datetime.date.today().isoformat()
    try:
        daily = fetch(day)["daily"]
    except Exception as problem:
        print(f"niente meteo ({problem}); il contesto resta com'era")
        return 1
    line = sentence(daily)
    if not rewrite(line):
        print("non ho trovato dove scriverlo in context_it.txt")
        return 1
    print(f"{day}: {line}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
