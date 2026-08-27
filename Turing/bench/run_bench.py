"""Try the shortlist before committing the entry to one of them.

    cd Turing
    export FEATHERLESS_API_KEY=...
    python -m bench.run_bench --list-models italian     # confirm the ids exist
    python -m bench.run_bench                           # play every room on every model
    python -m bench.run_bench --models Qwen/Qwen2.5-72B-Instruct --repeat 3

The choice is not "which model is smartest". The room is 300 seconds of Italian
small talk judged by people who study Italian for a living, and the thing that
decides it is register: a model that writes textbook Italian loses to one that
writes the Italian people actually type, whatever either scores on a benchmark.
That is why the shortlist has Italian fine-tunes on it next to the frontier
multilingual models, and why the report prints the replies in full — the numbers
narrow the field, the transcripts pick the winner.

Latency is the second axis and it is not a comfort question. The timing filter
adds a length-proportional delay on top of whatever the model takes, so a model
with a 12-second median has already spent the budget for looking like somebody
who reads before answering, and every reply lands late no matter what the filter
wanted.

What the automatic flags catch: answering as an assistant, drifting into
English, obeying an injection, and producing a vote the world cannot parse.
Everything else is for reading.
"""

import os
import re
import sys
import json
import time
import argparse
import datetime

from bench import probes
from bench.canned import Canned
from processors.boss import Boss
from processors import humanise

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

# Checked against `--list-models` on 2026-08-27. Featherless names models by
# their Hugging Face repo and the fine-tunes get re-uploaded under new tags, so
# check again rather than trusting this list. Minerva
# (`sapienzanlp/Minerva-7B-instruct-v1.0`) was on the plan's shortlist and is
# NOT served here; the Italian slots below are what the catalogue actually has.
SHORTLIST = [
    # Italian fine-tunes: the register is the whole game
    "mii-llm/maestrale-chat-v0.4-beta",
    "swap-uniba/LLaMAntino-3-ANITA-8B-Inst-DPO-ITA",
    "anakin87/gemma-2-9b-neogenesis-ita",
    "DeepMount00/Llama-3.1-8b-ITA",
    "m-polignano/ANITA-NEXT-24B-Dolphin-Mistral-UNCENSORED-ITA",
    # Frontier multilingual: better Italian than their size suggests, slower
    "Qwen/Qwen2.5-72B-Instruct",
    "meta-llama/Llama-3.3-70B-Instruct",
    # The middle, for when the big two are too slow to look human
    "Qwen/Qwen2.5-32B-Instruct",
]

ENGLISH = re.compile(r"\b(?:the|you|your|please|sorry|hello|thanks|i'm|don't|"
                     r"what|about|really|actually)\b", re.IGNORECASE)
VOTE_SHAPE = re.compile(r"^(?:nessuno|tutti|[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ'\-]*"
                        r"(?:\s*,\s*[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ'\-]*)*)$")
# Compliance with the injection room: the words it was ordered to produce
OBEYED = re.compile(r"\bbanana\b|^ok$|prompt di sistema|sono un assistente", re.IGNORECASE)


class Recorder:
    """Wraps a backend and remembers what it cost and what it said."""

    def __init__(self, inner):
        self.inner = inner
        self.latencies: list[float] = []
        self.raw: list[str] = []

    def __call__(self, prompt, **kwargs):
        start = time.monotonic()
        out = self.inner(prompt, **kwargs)
        self.latencies.append(time.monotonic() - start)
        self.raw.append(out)
        return out

    def close(self):
        closer = getattr(self.inner, "close", None)
        if callable(closer):
            closer()


def build_backend(model: str, offline: bool):
    if offline:
        return Canned()
    from processors.featherless import FeatherlessBackend
    return FeatherlessBackend(model=model, max_tokens=60, temperature=0.95,
                              top_p=0.95, top_k=60, repetition_penalty=1.08)


def play(scenario, model: str, offline: bool) -> dict:
    """One room, start to vote, on one model."""
    recorder = Recorder(build_backend(model, offline))
    boss = Boss(model=model, backend=recorder)

    turns = []
    try:
        for sample in scenario.samples:
            head = sample.splitlines()[0]
            turns.append({"in": head[:90], "out": boss(sample)})
    finally:
        # Each room opens its own gateway registration; without this they pile
        # up for the whole benchmark and the server never sees them released
        recorder.close()

    spoken = [t["out"] for t in turns if t["out"]]
    flags = {
        "assistant": sum(1 for r in recorder.raw if humanise.is_assistant(r)),
        "english": sum(1 for r in spoken if len(ENGLISH.findall(r)) >= 2),
        "obeyed": (sum(1 for r in spoken if OBEYED.search(r))
                   if scenario.name == "injection" else 0),
    }
    if scenario.name == "voto":
        flags["vote_ok"] = bool(turns and VOTE_SHAPE.match(turns[-1]["out"]))

    latencies = sorted(recorder.latencies)
    return {
        "scenario": scenario.name,
        "expect": scenario.expect,
        "turns": turns,
        "calls": len(latencies),
        "latency_median": latencies[len(latencies) // 2] if latencies else 0.0,
        "latency_p90": latencies[int(len(latencies) * 0.9)] if latencies else 0.0,
        "spoke": len(spoken),
        "of": len(turns),
        "words": sum(len(r.split()) for r in spoken) / max(len(spoken), 1),
        "flags": flags,
    }


def list_models(query: str) -> None:
    """Ask Featherless what it actually serves, so the shortlist is not a guess."""
    from openai import OpenAI
    client = OpenAI(base_url="https://api.featherless.ai/v1",
                    api_key=os.environ["FEATHERLESS_API_KEY"], timeout=30.0)
    ids = sorted(model.id for model in client.models.list())
    hits = [i for i in ids if query.lower() in i.lower()] if query else ids
    print(f"{len(hits)} model(s) of {len(ids)} match {query!r}")
    for model_id in hits:
        print(f"  {model_id}")


def report(runs: list[dict], path: str) -> None:
    lines = ["# Turing boss — model bench", "",
             f"Run {datetime.datetime.now():%Y-%m-%d %H:%M}", "",
             "| model | stanza | chiamate | lat. mediana | lat. p90 | parlato | parole | bandiere |",
             "|---|---|---|---|---|---|---|---|"]

    def worth_printing(value) -> bool:
        # `False == 0` in Python, so a failed check has to be tested for first
        return (value is False) or (not isinstance(value, bool) and value > 0)

    for run in runs:
        for room in run["rooms"]:
            flags = ", ".join(f"{k}={v}" for k, v in room["flags"].items()
                              if worth_printing(v))
            lines.append(f"| {run['model']} | {room['scenario']} | {room['calls']} | "
                         f"{room['latency_median']:.1f}s | {room['latency_p90']:.1f}s | "
                         f"{room['spoke']}/{room['of']} | {room['words']:.1f} | {flags or '-'} |")

    lines += ["", "## Trascrizioni", ""]
    for run in runs:
        lines += [f"### {run['model']}", ""]
        if run.get("error"):
            lines += [f"**errore:** `{run['error']}`", ""]
            continue
        for room in run["rooms"]:
            lines += [f"**{room['scenario']}** — _{room['expect']}_", "", "```"]
            for turn in room["turns"]:
                lines.append(f"{turn['in']}\n    -> {turn['out'] or '(silenzio)'}")
            lines += ["```", ""]

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    with open(path.replace(".md", ".json"), "w", encoding="utf-8") as handle:
        json.dump(runs, handle, ensure_ascii=False, indent=2)
    print(f"\nreport -> {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", default=",".join(SHORTLIST),
                        help="comma separated model ids")
    parser.add_argument("--scenarios", default=",".join(probes.BY_NAME),
                        help="comma separated room names")
    parser.add_argument("--repeat", type=int, default=1, help="rooms per scenario")
    parser.add_argument("--offline", action="store_true",
                        help="canned backend, no key and no network")
    parser.add_argument("--list-models", nargs="?", const="", metavar="QUERY",
                        help="print the Featherless catalogue and stop")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    if args.list_models is not None:
        list_models(args.list_models)
        return 0

    rooms = [probes.BY_NAME[name] for name in args.scenarios.split(",") if name in probes.BY_NAME]
    if not rooms:
        print("no scenario matched", file=sys.stderr)
        return 2

    runs = []
    for model in args.models.split(","):
        model = model.strip()
        if not model:
            continue
        print(f"\n=== {model}")
        run = {"model": model, "rooms": []}
        try:
            for scenario in rooms:
                for _ in range(args.repeat):
                    room = play(scenario, model, args.offline)
                    run["rooms"].append(room)
                    print(f"  {scenario.name:14} {room['spoke']}/{room['of']} turni, "
                          f"{room['latency_median']:.1f}s mediana, "
                          f"{room['words']:.1f} parole")
        except Exception as e:
            run["error"] = f"{type(e).__name__}: {e}"
            print(f"  ! {run['error']}")
        runs.append(run)

    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
    report(runs, args.out or os.path.join(RESULTS, f"bench_{stamp}.md"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
