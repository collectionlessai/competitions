"""Come si distribuiscono i tempi di generazione, da un journal.

    python -m bench.latency <cartella del journal>

La mediana non dice niente di utile qui: il problema è la coda. Misurato nella
prima sessione in albergo vero, con un solo nostro agente in gara, la mediana
era 3,6s e il novantesimo percentile 59,8s. Quello che conta è quante STANZE
vengono toccate, perché una generazione lenta manda a monte un turno, non una
statistica.
"""

import os
import sys
import json
import glob
import statistics


def load(path):
    calls, spoke, slow_rooms = [], set(), set()
    for name in glob.glob(os.path.join(path, "*.jsonl")):
        for raw in open(name, encoding="utf-8"):
            try:
                e = json.loads(raw)
            except ValueError:
                continue
            if e.get("kind") == "call" and e.get("seconds"):
                calls.append((e["seconds"], e.get("room"), e.get("situation") or "-"))
                if e["seconds"] > 20:
                    slow_rooms.add(e.get("room"))
            if e.get("kind") in ("said", "dropped"):
                spoke.add(e.get("room"))
    return calls, spoke, slow_rooms


def main(path):
    calls, spoke, slow_rooms = load(path)
    if not calls:
        sys.exit(f"nessuna chiamata registrata in {path}")

    times = sorted(c[0] for c in calls)
    n = len(times)

    def pct(p):
        return times[min(n - 1, int(n * p))]

    print(f"\n  chiamate            {n}")
    print(f"  mediana             {statistics.median(times):6.1f}s")
    print(f"  75°                 {pct(0.75):6.1f}s")
    print(f"  90°                 {pct(0.90):6.1f}s")
    print(f"  peggiore            {max(times):6.1f}s")
    print()
    for soglia in (10, 20, 60):
        over = sum(1 for t in times if t > soglia)
        print(f"  oltre {soglia:2d}s            {over:3d}/{n}  ({100 * over / n:4.1f}%)")

    hit = slow_rooms & spoke
    if spoke:
        print()
        print(f"  stanze con parola   {len(spoke)}")
        print(f"  toccate da lentezza {len(hit)}"
              + (f"   -> 1 ogni {len(spoke) / len(hit):.1f}" if hit else "   -> nessuna"))

    worst = sorted(calls, reverse=True)[:5]
    print("\n  le peggiori:")
    for secs, room, situation in worst:
        print(f"    {secs:6.1f}s  stanza {room}  ({situation})")
    print()


if __name__ == "__main__":
    where = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("BOSS_JOURNAL", "")
    if not where or not os.path.isdir(where):
        sys.exit("uso: python -m bench.latency <cartella del journal>")
    main(where)
