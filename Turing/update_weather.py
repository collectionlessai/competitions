"""Put the real weather of the event's city into the knowledge base.

The context file used to assert "fa caldo, sui trenta gradi" — written by hand
and never checked (CLiC-it, Palermo: the forecast said 25-27 with drizzle). The
weather is the single most checkable fact between people standing in the same
city, so the line is filled from a forecast instead of from a stereotype. Run it
the morning of the event — a forecast fetched days early is worth little, and
this costs one request:

    python update_weather.py            # today (SAINET day: Siena, 2026-11-11)
    python update_weather.py 2026-11-11 # a specific day

It rewrites only the ## METEO block of `processors/context_it.txt` and leaves
everything else alone. With no network it changes nothing and says so, which is
the right failure: a stale true-ish line beats a wrong fresh one.

The city, the request and the wording live in `processors/weather.py`, which the
launcher also uses at startup: one place to change for the next event.
"""
import datetime
import io
import json
import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
CONTEXT = os.path.join(HERE, "processors", "context_it.txt")
sys.path.insert(0, HERE)

from processors.weather import API, WHERE, sentence  # noqa: E402


def fetch(day: str) -> dict:
    url = API.format(lat=WHERE[0], lon=WHERE[1], day=day)
    with urllib.request.urlopen(url, timeout=20) as response:
        return json.loads(response.read().decode("utf-8"))


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
