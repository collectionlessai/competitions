"""Recompute the two scores over recorded transcripts, and compare.

Changing a term in `Speaker.read` changes who we stop listening to: a guest
over `bot >= 0.60` with `human < 0.35` is dropped from the conversation. That
is cheap to get wrong and expensive to notice, so before shipping a new term,
rebuild every speaker from the journals and look at where the distribution
lands.

The delivery batches are recoverable from the journal because every `heard`
event carries the elapsed second it arrived at, and the world hands us the
conversation in chunks — messages sharing a timestamp shared a window.

    python -m bench.rescore <journal folder> [more folders...]
"""
import glob
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from processors.room import RoomSense, Speaker  # noqa: E402

# Read the live thresholds rather than copying them: a tool that measures who
# gets ignored must not be able to disagree with the code that ignores them.
BLATANT_BOT = RoomSense.BLATANT_BOT
BLATANT_NOT_HUMAN = RoomSense.BLATANT_NOT_HUMAN


def rooms(folder: str):
    """Every room in the folder, as (speaker -> messages, window count)."""
    for path in glob.glob(os.path.join(folder, "*.jsonl")):
        current, window, last_t = None, 0, None
        for raw in open(path, encoding="utf-8"):
            try:
                event = json.loads(raw)
            except ValueError:
                continue
            kind = event.get("kind")
            if kind == "room":
                if current:
                    yield current, window
                current, window, last_t = {}, 0, None
            elif kind == "heard" and current is not None:
                speaker = event.get("speaker")
                if not speaker or speaker == "MANAGER":
                    continue
                when = event.get("t")
                if when != last_t:
                    window += 1
                    last_t = when
                current.setdefault(speaker, []).append((event.get("text") or "", window))
        if current:
            yield current, window


def score(folder: str, markers=()):
    out = []
    for room, windows in rooms(folder):
        for name, said in room.items():
            speaker = Speaker(name)
            for text, window in said:
                speaker.add(text, None)
                speaker.windows.add(window)
            if speaker.count < 2:
                continue
            human, bot = speaker.read(None, markers, windows)
            out.append((human, bot, name, speaker.count))
    return out


def report(label: str, rows: list) -> None:
    if not rows:
        print(f"{label}: nessuna lettura")
        return
    bots = sorted(b for _, b, _, _ in rows)
    ignored = [r for r in rows if r[1] >= BLATANT_BOT and r[0] < BLATANT_NOT_HUMAN]
    print(f"{label}: {len(rows)} letture")
    print(f"   bot score  mediana {statistics.median(bots):.2f}  "
          f"90° {bots[int(0.9 * (len(bots) - 1))]:.2f}  massimo {max(bots):.2f}")
    for edge in (0.4, 0.5, 0.6, 0.7):
        n = sum(1 for b in bots if b >= edge)
        print(f"   >= {edge:.1f}: {n:4d} ({100 * n / len(bots):4.1f}%)")
    print(f"   verrebbero ignorati: {len(ignored)} ({100 * len(ignored) / len(rows):.1f}%)")


def main() -> int:
    folders = sys.argv[1:]
    if not folders:
        print(__doc__)
        return 1
    everything = []
    for folder in folders:
        rows = score(folder)
        report(os.path.basename(folder.rstrip("/\\")), rows)
        everything += rows
    if len(folders) > 1:
        print()
        report("TUTTO INSIEME", everything)
    return 0


if __name__ == "__main__":
    sys.exit(main())
