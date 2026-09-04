"""Replay the real competition transcript through RoomSense.

The reminder bug — the manager's "resterai in questa stanza per N secondi"
being read as a new briefing — never reproduced on the local bench, because
the local manager does not send reminders on that cadence. It only ever
appeared in the real hotel, which made the fix unverifiable by running the
stack again.

But the transcript of the run that produced the bug is on disk, so the input
that caused it is reproducible even though the conditions are not. This feeds
those messages back in, in order, and counts how many rooms RoomSense believes
it has been in. Ground truth is in the transcript itself: the manager names
the guest once per room, so the number of distinct names is the number of
rooms.

Usage:  python -m bench.replay_general [path-to-journal.jsonl]
"""
import json
import os
import re
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from processors.room import RoomSense                        # noqa: E402

DEFAULT = os.path.join(
    os.environ.get("TEMP", "/tmp"), "claude", "c--Users-Edoardo-UNaIVERSE",
    "11035f95-8681-4aaa-a1a5-7892b1827499", "scratchpad", "generale",
    "TuringBoss.jsonl",
)

# The manager names the guest it is talking to in both the briefing ("ti
# chiami X") and the reminder ("il tuo nome è X"). Distinct names across both
# is the true room count, independent of anything we classify. Counting only
# reminders undercounts: a room opened near the end of the run gets a briefing
# but no reminder before the transcript stops.
NAMED = re.compile(r"(?:il\s+tuo\s+nome\s+è|ti\s+chiami)\s+\*{0,2}(\w+)", re.IGNORECASE)


def load(path):
    heard = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if event.get("kind") == "heard":
                heard.append(event)
    return heard


def replay(heard):
    sense = RoomSense()
    counts = {"start": 0, "vote": 0, "chat": 0, "roster": 0, "reminder": 0, "quiet": 0}
    for event in heard:
        message = SimpleNamespace(speaker=event.get("speaker") or "", text=event.get("text") or "")
        turn = sense.read(event.get("text") or "", [message])
        counts[turn.kind] = counts.get(turn.kind, 0) + 1
    return counts


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    if not os.path.exists(path):
        print(f"no transcript at {path}")
        return 1

    heard = load(path)
    manager = [e for e in heard if (e.get("speaker") or "").upper() == "MANAGER"]
    truth = {m.group(1) for m in (NAMED.search(e.get("text") or "") for e in manager) if m}
    counts = replay(heard)

    rooms, votes = counts["start"], counts["vote"]
    print(f"  {len(heard)} messaggi rigiocati, {len(truth)} stanze vere: {', '.join(sorted(truth))}")
    print()
    for kind in ("start", "vote", "chat", "roster", "reminder", "quiet"):
        print(f"  {counts.get(kind, 0):4d}  {kind}")
    print()
    print(f"  stanze rilevate: {rooms}   (vere: {len(truth)}, in gara ne contò 46)")
    # The vote count here is text-only. In a live room most vote requests
    # arrive as the `can_vote` state rather than as words, and a replay has
    # no state machine, so this number is not the one the world would ask.
    print(f"  (i {votes} voti sono solo quelli riconoscibili dal testo: "
          f"in gara la richiesta arriva dallo stato, che il replay non ha)")

    if rooms == len(truth):
        print(f"  OK — il promemoria non risveglia più la stanza")
        return 0
    print(f"  FALLITO — scarto di {rooms - len(truth)}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
