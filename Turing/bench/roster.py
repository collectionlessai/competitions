"""The lab's characters, from every kit file, with the model each one wants.

The bench has been running three characters that all speak with the same
voice: the factory forces one model because Gemma rejects a system role, and
the override was applied to everybody rather than to Gemma. So every habit we
measured off "the guests" — the 25.7% of messages opening with "Ahah", say —
could be one model's tic counted three times, and we had no way to tell.

The kit already holds the cure. `agents_characters.csv` has eight Italian
characters with pacing numbers; `oldies/agents_characters copy.csv` has twenty
across three model families. This merges them and hands out a random cast, so
a run can be read per model instead of as an average that hides everything.

    python -m bench.roster --pick 3
    python -m bench.roster --list
"""
import argparse
import csv
import glob
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KIT = os.path.join(os.path.dirname(HERE), "..", "AppData")  # replaced below

# The kit lives beside the bench scripts, not in the repo.
DEFAULT_KIT = os.environ.get(
    "HOTEL_KIT",
    os.path.join(os.environ.get("TEMP", "/tmp"), "claude",
                 "c--Users-Edoardo-UNaIVERSE",
                 "11035f95-8681-4aaa-a1a5-7892b1827499", "scratchpad",
                 "hotel_kit", "ita", "basic_factory"))

# What the gateway will actually return, measured with `bench.model_probe`.
#
# Gemma answers a system role fine (1.4s, in Italian) — the blanket override
# onto Qwen was inherited from a problem that does not reproduce, and it is
# why three characters spoke in one voice. The two reasoning models cannot be
# used at all through this gateway: Qwen3.5-9B burns even a 600-token budget
# thinking and returns finish_reason=length, and R1-distill finishes cleanly
# with an empty content field, its reasoning never reaching us.
USABLE = (
    "Qwen/Qwen2.5-7B-Instruct",      # 1 unit, clean Italian
    "google/gemma-4-31B-it",         # 2 units
    "Qwen/Qwen2.5-32B-Instruct",     # 2 units
)
UNUSABLE = {
    "Qwen/Qwen3.5-9B": "esaurisce i token pensando",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B": "torna con contenuto vuoto",
    "meta-llama/Llama-3.1-8B-Instruct": "gated, serve collegare HuggingFace",
    "mistralai/Mistral-7B-Instruct-v0.3": "italiano illeggibile",
}
FALLBACK = "Qwen/Qwen2.5-7B-Instruct"

# Featherless does not bill tokens, it sells concurrency: this account is on
# `feather_claw_standard`, 8 units, and a request over the ceiling comes back
# 429 instead of costing money. A 72B request reserves 4 of those 8 on its own,
# so a room of guests on big models queues behind our own agent and slows the
# very thing we are measuring — 224 gateway failures in the logs, and the 56
# and 74 second generations we had been blaming on the network.
#
# So a cast is drawn to a budget, not just to a size.
PLAN_UNITS = 8
BOSS_UNITS = 4          # Qwen2.5-72B, the model we actually compete with


def _rows(path: str) -> list:
    try:
        with open(path, encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    except OSError:
        return []


def load(kit: str = "") -> list:
    """Every distinct character across the kit's CSVs, newest file winning."""
    kit = kit or DEFAULT_KIT
    paths = [os.path.join(kit, "agents_characters.csv")]
    paths += sorted(glob.glob(os.path.join(kit, "oldies", "*.csv")))

    seen, out = set(), []
    for path in paths:
        for row in _rows(path):
            name = (row.get("agent_name") or "").strip()
            if not name or name in seen:
                continue
            seen.add(name)
            wanted = (row.get("llm") or "").strip() or FALLBACK
            out.append({
                "character": name,
                "wanted": wanted,
                "model": wanted if wanted not in UNUSABLE else "",
                "why": UNUSABLE.get(wanted, ""),
                "csv": os.path.basename(path),
                "reply_prob": (row.get("reply_prob") or "").strip(),
            })

    # Characters whose model we cannot run are spread across the ones we can,
    # round-robin, rather than all landing on the same fallback — the whole
    # point of the exercise is more than one voice in the room.
    spare = [m for m in USABLE]
    turn = 0
    for person in out:
        if not person["model"]:
            person["model"] = spare[turn % len(spare)]
            person["swapped"] = True
            turn += 1
        else:
            person["swapped"] = False
    return out


def pick(how_many: int, kit: str = "", seed=None, budget: int = 0) -> list:
    """A random cast, spread across model families, inside a concurrency budget.

    `budget` is how many units the guests may hold in flight between them. Zero
    means do not care, which is what produced a room asking for fourteen units
    against a plan of eight.
    """
    from processors.featherless import cost_for

    people = load(kit)
    if not people:
        return []
    rng = random.Random(seed)

    by_model = {}
    for person in people:
        by_model.setdefault(person["model"], []).append(person)
    for group in by_model.values():
        rng.shuffle(group)

    # Take one from each family before taking a second from any, so three
    # guests are three voices whenever the kit can supply them — but stop when
    # the next voice would not fit, and try a cheaper family instead.
    cast, spent = [], 0
    families = sorted(by_model, key=lambda m: cost_for(m))
    while len(cast) < how_many:
        took = False
        for model in families:
            if not by_model[model] or len(cast) >= how_many:
                continue
            price = cost_for(model)
            if budget and spent + price > budget:
                continue
            person = by_model[model].pop()
            person["units"] = price
            cast.append(person)
            spent += price
            took = True
        if not took:
            break
    return cast


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pick", type=int, default=0)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--kit", default="")
    parser.add_argument("--seed", default=None)
    parser.add_argument("--budget", type=int, default=0,
                        help="unita di concorrenza spendibili dagli ospiti")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.list or not args.pick:
        people = load(args.kit)
        if not people:
            print(f"nessun personaggio trovato in {args.kit or DEFAULT_KIT}")
            return 1
        wanted, running = {}, {}
        for person in people:
            wanted[person["wanted"]] = wanted.get(person["wanted"], 0) + 1
            running[person["model"]] = running.get(person["model"], 0) + 1
        print(f"{len(people)} personaggi")
        print("\n  modello chiesto dalla scheda:")
        for model, count in sorted(wanted.items(), key=lambda kv: -kv[1]):
            note = f"   -> {UNUSABLE[model]}, si ripiega" if model in UNUSABLE else ""
            print(f"   {count:3d}  {model}{note}")
        print("\n  modello su cui girano davvero:")
        for model, count in sorted(running.items(), key=lambda kv: -kv[1]):
            print(f"   {count:3d}  {model}")
        return 0

    cast = pick(args.pick, args.kit, args.seed, args.budget)
    if args.json:
        print(json.dumps(cast, ensure_ascii=False))
        return 0
    for person in cast:
        print(f"{person['character']}\t{person['model']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
