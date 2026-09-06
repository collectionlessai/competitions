"""Render one room's whole life as a page you can read top to bottom.

The journal already holds everything — both halves of every model call, the
persona, the tactic, the scores as they move, the notepad, the analyst, the
vote. What it did not have was a way to look at it: reading a 400-line JSONL by
hand is how three separate wrong conclusions got drawn this week, each from a
number that looked fine in isolation.

So this puts the room back together in order. Every event on one timeline, every
prompt reachable, nothing summarised away. It answers the only question that
matters when a verdict looks wrong — what did the agent know, and when.

    python -m bench.timeline journal_v17 --rooms 2 --out timeline.html
"""
import argparse
import glob
import html
import json
import os
import re
import sys

KINDS_SHOWN = ("room", "tattica", "heard", "said", "punteggi", "pad",
               "bait", "call", "analyst", "vote", "dropped")


def esc(value) -> str:
    return html.escape(str(value if value is not None else ""))


def load(folder: str) -> dict:
    """Every agent's journal in `folder`, split into rooms.

    Split on the journal's own `room` counter rather than on the appearance of
    a `room` event. The two are not the same at the seam: `t` is seconds
    elapsed inside the room, and it is reset when the world moves us on —
    *before* the next `room` event gets written. Grouping on the events
    therefore drops the last handful of the old room, and the tail of every
    room read as a run of messages at 0s.
    """
    agents = {}
    for path in sorted(glob.glob(os.path.join(folder, "*.jsonl"))):
        name = os.path.basename(path).replace(".jsonl", "")
        rooms, current, number = [], None, None
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                where = event.get("room")
                if where != number or current is None:
                    number = where
                    current = {"head": {}, "events": []}
                    rooms.append(current)
                if event.get("kind") == "room":
                    current["head"] = event
                else:
                    current["events"].append(event)
        agents[name] = [r for r in rooms if r["events"]]
    return agents


def when(event, zero: float) -> str:
    return f"{event.get('t', zero) - zero:.0f}s"


def fold(summary: str, body: str, cls: str = "") -> str:
    """A collapsed block. Prompts are long; the point is that they are all here."""
    return (f'<details class="fold {cls}"><summary>{summary}</summary>'
            f'<pre>{body}</pre></details>')


def scores_table(scores: dict) -> str:
    if not scores:
        return ""
    cells = []
    for name, (human, bot) in sorted(scores.items()):
        cells.append(f'<span class="sc"><b>{esc(name)}</b>'
                     f'<i class="h">{human:.2f}</i><i class="b">{bot:.2f}</i></span>')
    return '<div class="scores">' + "".join(cells) + "</div>"


def render_event(event, zero: float, tactics: bool = True, bait_words=frozenset()) -> str:
    """One row. `tactics` is False for an agent running without them, whose

    `said` events still carry the drawn-but-unused tactic in older journals;
    `bait_words` lets a bait-carrying message be marked in runs recorded
    before the flag for it existed.
    """
    kind = event.get("kind")
    stamp = f'<span class="ts">{when(event, zero)}</span>'

    if kind == "tattica":
        return (f'<div class="ev tattica">{stamp}'
                f'<span class="what">tattica</span>'
                f'<span class="body"><b>{esc(event.get("nome"))}</b> '
                f'&mdash; {esc(event.get("motivo"))}, proattivit&agrave; '
                f'{event.get("spinta", 0):.2f}</span></div>')

    if kind == "heard":
        return (f'<div class="ev heard">{stamp}'
                f'<span class="what">{esc(event.get("speaker"))}</span>'
                f'<span class="body">{esc(event.get("text"))}</span></div>')

    if kind == "said":
        extra = []
        if event.get("style"):
            extra.append(f'stile <b>{esc(event["style"])}</b>')
        if tactics and event.get("tattica"):
            extra.append(f'tattica <b>{esc(event["tattica"])}</b>')
        if event.get("gen") is not None:
            extra.append(f'generato in {event["gen"]}s')
        meta = f'<div class="meta">{" &middot; ".join(extra)}</div>' if extra else ""
        words = set(re.findall(r"[a-zà-ÿ']+", (event.get("text") or "").lower()))
        carried = event.get("esca") or len(words & bait_words) >= 2
        tag = ' <span class="baittag">esca</span>' if carried else ""
        return (f'<div class="ev said{" carries" if carried else ""}">{stamp}'
                f'<span class="what">noi</span>'
                f'<span class="body">{esc(event.get("text"))}{tag}{meta}</span></div>')

    if kind == "punteggi":
        return (f'<div class="ev punteggi">{stamp}'
                f'<span class="what">punteggi</span>'
                f'<span class="body">{scores_table(event.get("tutti") or {})}</span></div>')

    if kind == "bait":
        return (f'<div class="ev bait">{stamp}'
                f'<span class="what">esca</span>'
                f'<span class="body">&laquo;{esc(event.get("say"))}&raquo;'
                f'<div class="meta">in realt&agrave;: {esc(event.get("truth"))}</div>'
                f'</span></div>')

    if kind == "pad":
        notes = event.get("notes") or []
        items = "".join(f'<div class="note">[{esc(n.get("kind"))}] {esc(n.get("text"))}'
                        + (f' &rarr; {esc(n.get("about"))}' if n.get("about") else "")
                        + "</div>" for n in notes)
        return (f'<div class="ev pad">{stamp}<span class="what">taccuino</span>'
                f'<span class="body">{items}</span></div>')

    if kind == "dropped":
        return (f'<div class="ev dropped">{stamp}<span class="what">scartato</span>'
                f'<span class="body">{esc(event.get("text"))}'
                f'<div class="meta">motivo: {esc(event.get("why"))}</div></span></div>')

    if kind == "call":
        situation = event.get("situation") or "—"
        seconds = event.get("seconds")
        head = (f'chiamata al modello &middot; {esc(situation)}'
                + (f' &middot; {seconds}s' if seconds is not None else ""))
        body = (f'<b>SISTEMA</b>\n{esc(event.get("system"))}\n\n'
                f'<b>PROMPT</b>\n{esc(event.get("prompt"))}\n\n'
                f'<b>RISPOSTA</b>\n{esc(event.get("answer"))}')
        cls = "vote" if situation == "vote" else ""
        return (f'<div class="ev call">{stamp}<span class="what">modello</span>'
                f'<span class="body">{fold(head, body, cls)}</span></div>')

    if kind == "analyst":
        rows = []
        for line in (event.get("text") or "").splitlines():
            line = line.strip()
            if not line:
                continue
            label = "z"
            for mark, css in (("[PERSONA]", "p"), ("[MACCHINA]", "m"), ("[NIENTE]", "z")):
                if mark in line:
                    label = css
                    break
            rows.append(f'<div class="verdict {label}">{esc(line)}</div>')
        return (f'<div class="ev analyst">{stamp}<span class="what">analista</span>'
                f'<span class="body">{"".join(rows) or "(nessuna analisi)"}</span></div>')

    if kind == "vote":
        vote = event.get("vote")
        shown = esc(vote) if vote and vote != "nessuno" else "nessuno"
        return (f'<div class="ev vote">{stamp}<span class="what">voto</span>'
                f'<span class="body"><b>{shown}</b>'
                f'<div class="meta">candidati: '
                f'{esc(", ".join(event.get("candidates") or []))}</div></span></div>')

    return ""


def render_room(agent: str, room: dict, index: int) -> str:
    head = room["head"]
    # `t` is already seconds inside the room, so there is nothing to subtract.
    zero = 0.0
    persona = head.get("persona") or ""
    style = ""
    if persona.startswith("# STILE:"):
        style, _, persona = persona.partition("\n")
        style = style[len("# STILE:"):].strip()
    roster = ", ".join(head.get("roster") or []) or "(nessuno ancora)"

    events = [e for e in room["events"] if e.get("kind") in KINDS_SHOWN]
    tactics = any(e.get("kind") == "tattica" for e in room["events"])
    bait_words = set()
    for e in room["events"]:
        if e.get("kind") == "bait":
            bait_words |= set(re.findall(r"[a-zà-ÿ']+", (e.get("say") or "").lower()))
    body = "".join(render_event(e, zero, tactics, bait_words) for e in events)

    return f'''
<section class="room">
  <h3>{esc(agent)} &middot; stanza {index} &middot; in scena come <b>{esc(head.get("me"))}</b></h3>
  <div class="roomhead">
    <div><span class="lbl">persona</span>{esc(persona.strip())}</div>
    {f'<div><span class="lbl">grafia</span>{esc(style)}</div>' if style else ""}
    <div><span class="lbl">presenti all'apertura</span>{esc(roster)}</div>
  </div>
  <div class="timeline">{body}</div>
</section>'''


CSS = """
:root{
  --ground:#F2F5F4; --surface:#FFF; --sunk:#E8EDEB; --ink:#15201E; --body:#33413E;
  --muted:#65746F; --line:#D5DEDB; --us:#0E6E63; --us-soft:#DEEFEC;
  --bait:#7E6210; --bait-soft:#FAF0D4; --bad:#A33417; --pla:#8A2E5D; --pla-soft:#F7E5EF;
  --mono:"IBM Plex Mono",ui-monospace,Consolas,monospace;
  --sans:"IBM Plex Sans",system-ui,"Segoe UI",sans-serif;
  --serif:"Instrument Serif",Georgia,serif;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --ground:#0F1514; --surface:#161E1C; --sunk:#0B1110; --ink:#E9EFED; --body:#BDC9C6;
  --muted:#849690; --line:#27332F; --us:#5AC8B7; --us-soft:#123029;
  --bait:#D8B25F; --bait-soft:#2E2614; --bad:#E8846A; --pla:#E68CB6; --pla-soft:#341E2A;}}
:root[data-theme="dark"]{
  --ground:#0F1514; --surface:#161E1C; --sunk:#0B1110; --ink:#E9EFED; --body:#BDC9C6;
  --muted:#849690; --line:#27332F; --us:#5AC8B7; --us-soft:#123029;
  --bait:#D8B25F; --bait-soft:#2E2614; --bad:#E8846A; --pla:#E68CB6; --pla-soft:#341E2A;}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--body);font-family:var(--sans);
  font-size:16px;line-height:1.6}
.wrap{max-width:1000px;margin:0 auto;padding:clamp(24px,4vw,56px) clamp(14px,3vw,32px) 80px}
h1{font-family:var(--serif);font-weight:400;color:var(--ink);
  font-size:clamp(2.2rem,5vw,3.2rem);line-height:1.06;margin:0}
h2{font-family:var(--serif);font-weight:400;color:var(--ink);
  font-size:clamp(1.5rem,3vw,2rem);margin:0 0 .4em}
h3{font-family:var(--sans);font-size:.95rem;font-weight:600;color:var(--ink);margin:0}
p{margin:0 0 1em;max-width:66ch}
.eyebrow{font-family:var(--mono);font-size:.7rem;letter-spacing:.14em;
  text-transform:uppercase;color:var(--muted);margin:0 0 1em}
.arm{margin-top:56px}
.room{background:var(--surface);border:1px solid var(--line);border-radius:3px;
  margin:20px 0;overflow:hidden}
.room>h3{padding:14px 18px;background:var(--sunk);border-bottom:1px solid var(--line)}
.roomhead{padding:14px 18px;border-bottom:1px solid var(--line);font-size:.87rem}
.roomhead>div{margin-bottom:.5em}
.lbl{font-family:var(--mono);font-size:.66rem;letter-spacing:.1em;text-transform:uppercase;
  color:var(--muted);display:block;margin-bottom:.2em}
.timeline{padding:6px 0}
.ev{display:grid;grid-template-columns:52px 78px 1fr;gap:0 12px;padding:6px 18px;
  align-items:baseline;border-left:3px solid transparent}
.ts{font-family:var(--mono);font-size:.7rem;color:var(--muted);text-align:right;
  font-variant-numeric:tabular-nums}
.what{font-family:var(--mono);font-size:.7rem;color:var(--muted);text-transform:uppercase;
  letter-spacing:.05em}
.body{min-width:0;font-size:.93rem;overflow-wrap:anywhere}
.meta{font-family:var(--mono);font-size:.7rem;color:var(--muted);margin-top:.25em}
.ev.said{background:var(--us-soft);border-left-color:var(--us)}
.ev.said.carries{border-left-color:var(--bait)}
.baittag{font-family:var(--mono);font-size:.6rem;letter-spacing:.1em;text-transform:uppercase;color:var(--bait);background:var(--bait-soft);border:1px solid var(--bait);border-radius:2px;padding:1px 5px;margin-left:7px}
.ev.said .what,.ev.said .body{color:var(--ink)}
.ev.heard .body{color:var(--muted)}
.ev.tattica{background:var(--sunk)}
.ev.tattica .body{color:var(--ink)}
.ev.bait{background:var(--bait-soft);border-left-color:var(--bait)}
.ev.bait .body{color:var(--ink)}
.ev.dropped .body{color:var(--bad);text-decoration:line-through}
.ev.dropped .meta{text-decoration:none}
.ev.vote{background:var(--pla-soft);border-left-color:var(--pla)}
.ev.vote .body b{color:var(--pla);font-size:1.05rem}
.scores{display:flex;flex-wrap:wrap;gap:5px}
.sc{font-family:var(--mono);font-size:.7rem;border:1px solid var(--line);
  border-radius:2px;padding:1px 6px;display:inline-flex;gap:6px;align-items:baseline}
.sc b{color:var(--ink);font-weight:500}
.sc i{font-style:normal}
.sc .h{color:var(--us)} .sc .b{color:var(--pla)}
.note{font-family:var(--mono);font-size:.74rem;color:var(--body)}
.verdict{font-size:.86rem;margin-bottom:.3em;padding-left:9px;border-left:2px solid var(--line)}
.verdict.p{border-left-color:var(--us);color:var(--ink)}
.verdict.m{border-left-color:var(--pla)}
.verdict.z{border-left-color:var(--line);color:var(--muted)}
.fold{border:1px solid var(--line);border-radius:2px;background:var(--ground)}
.fold summary{cursor:pointer;padding:5px 10px;font-family:var(--mono);font-size:.72rem;
  color:var(--muted);user-select:none}
.fold summary:hover{color:var(--ink)}
.fold.vote summary{color:var(--pla)}
.fold pre{margin:0;padding:12px;border-top:1px solid var(--line);overflow-x:auto;
  font-family:var(--mono);font-size:.72rem;line-height:1.55;white-space:pre-wrap;
  overflow-wrap:anywhere;color:var(--body);max-height:520px;overflow-y:auto}
.fold pre b{color:var(--ink)}
footer{margin-top:64px;padding-top:20px;border-top:1px solid var(--line);
  font-family:var(--mono);font-size:.72rem;color:var(--muted)}
@media (max-width:620px){
  .ev{grid-template-columns:44px 1fr;padding:6px 12px}
  .what{grid-column:2;font-size:.64rem}
  .body{grid-column:2}
}
"""


def build(folder: str, per_agent: int, labels: dict) -> str:
    agents = load(folder)
    blocks = []
    for agent, rooms in agents.items():
        chosen = [r for r in rooms if len(r["events"]) > 8][:per_agent]
        label = labels.get(agent, agent)
        inner = "".join(render_room(agent, r, i) for i, r in enumerate(chosen))
        blocks.append(f'<div class="arm"><h2>{esc(label)}</h2>{inner}</div>')
    return "".join(blocks)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("folder")
    parser.add_argument("--rooms", type=int, default=2)
    parser.add_argument("--out", default="timeline.html")
    parser.add_argument("--title", default="Timeline dell'agente")
    args = parser.parse_args()

    if not os.path.isdir(args.folder):
        print(f"non trovo {args.folder}")
        return 1

    labels = {"TuringBoss": "TuringBoss — CON le tattiche",
              "TuringBoss2": "TuringBoss2 — SENZA tattiche"}
    body = build(args.folder, args.rooms, labels)
    page = (f"<title>{esc(args.title)}</title>\n"
            '<link rel="preconnect" href="https://fonts.googleapis.com">\n'
            '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
            'family=Instrument+Serif&family=IBM+Plex+Sans:wght@400;500;600'
            '&family=IBM+Plex+Mono:wght@400;500&display=swap">\n'
            f"<style>{CSS}</style>\n"
            f'<div class="wrap">{body}</div>')
    with open(args.out, "w", encoding="utf-8") as handle:
        handle.write(page)
    print(f"scritto {args.out} ({len(page) // 1024} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
