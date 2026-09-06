"""Render a room journal into something a person can read.

    BOSS_JOURNAL=/some/dir python my_agent.py     # write it
    python -m bench.room_report /some/dir         # read it

The transcript on its own only shows what the room saw. What is worth reading
back is the part the room could not see: what the agent made of each guest,
what it was holding in mind when it spoke, what the analyst concluded before
the vote, and which of its own sentences it threw away. Those are interleaved
here in the order they happened, indented, so a room reads top to bottom.
"""

import os
import sys
import json
import glob


def rooms(path: str):
    """Every journal line, grouped by (agent, room number)."""
    out = {}
    for name in sorted(glob.glob(os.path.join(path, "*.jsonl"))):
        for raw in open(name, encoding="utf-8"):
            try:
                event = json.loads(raw)
            except ValueError:
                continue
            out.setdefault((event.get("who", "?"), event.get("room", 0)), []).append(event)
    return out


def persona_line(text: str) -> str:
    """The first real line of the persona, without the style directives."""
    for line in (text or "").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            return line[:90]
    return "?"


def render(events: list) -> str:
    lines = []
    for e in events:
        kind, t = e.get("kind"), e.get("t", 0.0)
        stamp = f"{t:6.1f}s "
        if kind == "room":
            lines.append(f"\n{'=' * 72}\nSTANZA — mi chiamo {e.get('me')}, "
                         f"con {', '.join(e.get('roster') or []) or '(ancora nessuno)'}\n"
                         f"persona: {persona_line(e.get('persona'))}\n{'=' * 72}")
        elif kind == "heard":
            text = e.get("text") or ""
            # The briefing is a page long and identical every room; it is
            # context, not conversation, and printing it whole buries the room.
            if e.get("speaker", "").upper().startswith("MANAGER") and len(text) > 110:
                text = text[:110] + " [...]"
            lines.append(f"{stamp}   {e.get('speaker')}: {text}")
        elif kind == "said":
            extra = f"[{e.get('style')}, gen {e.get('gen')}s, ritmo {e.get('pace')}s]"
            lines.append(f"{stamp}>> IO: {e.get('text')}   {extra}")
            if e.get("settled"):
                lines.append(f"{' ' * 8}   (gia risolti: {', '.join(e['settled'])} | "
                             f"aperti: {', '.join(e.get('open') or []) or '-'})")
        elif kind == "dropped":
            lines.append(f"{stamp}xx SCARTATO ({e.get('why')}): {e.get('text')}")
        elif kind == "read":
            lines.append(f"{stamp}?? come li legge:")
            for who, text in (e.get("profiles") or {}).items():
                lines.append(f"{' ' * 11}{who}: {text}")
        elif kind == "pad":
            notes = "; ".join(f"[{n['kind']}] {n['text']}"
                              + (f" (su {n['about']})" if n.get("about") else "")
                              for n in e.get("notes") or [])
            lines.append(f"{stamp}.. in mente: {notes}")
        elif kind == "analyst":
            lines.append(f"{stamp}AN l'analista, prima del voto:")
            for line in (e.get("text") or "").splitlines():
                if line.strip():
                    lines.append(f"{' ' * 11}{line.strip()}")
        elif kind == "vote":
            lines.append(f"{stamp}!! VOTO: {e.get('vote')}")
            lines.append(f"{' ' * 11}messaggi visti: {e.get('counts')}")
            lines.append(f"{' ' * 11}indice artificialita: {e.get('scores')}")
            if e.get("settled"):
                lines.append(f"{' ' * 11}dati per bot senza dubbio: {', '.join(e['settled'])}")
    return "\n".join(lines)


if __name__ == "__main__":
    where = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("BOSS_JOURNAL", "")
    if not where or not os.path.isdir(where):
        sys.exit("uso: python -m bench.room_report <cartella del journal>")
    found = rooms(where)
    if not found:
        sys.exit(f"nessun journal in {where}")
    for (who, number), events in sorted(found.items()):
        print(f"\n\n######## {who}, stanza {number} ########")
        print(render(events))
