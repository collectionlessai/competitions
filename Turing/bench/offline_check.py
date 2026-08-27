"""The whole agent, end to end, with no key and no network.

    cd Turing
    python -m bench.offline_check

Two passes. The first walks `prompts/*.txt`, which is every kind of sample the
room can hand you, and checks the things that are true regardless of which model
is behind it: that the briefing is recognised and read (name, roster, manager),
that the vote request is recognised and answered in the format the world parses,
that announcements are not chatted at, and that nothing markdown-shaped,
newline-shaped or assistant-shaped ever reaches the wire.

The second plays the seven rooms in `probes.py` and prints how talkative the
agent was in each. Those numbers are the ones to look at after changing the
director: silence near zero is an agent that answers everything, silence near
one is an agent nobody can vote about.

Exit code is non-zero if a check fails, so it is worth running before every room.
"""

import re
import sys
import glob
import os

from bench.canned import Canned
from bench import probes
from processors.boss import Boss

FORBIDDEN = re.compile(r"[*_#`]|\n|<br|<strong")
VOTE_SHAPE = re.compile(r"^(?:nessuno|tutti|[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ'\-]*"
                        r"(?:\s*,\s*[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ'\-]*)*)$")

failures: list[str] = []


def check(condition: bool, message: str) -> None:
    if not condition:
        failures.append(message)
        print(f"   FAIL  {message}")


def fixtures_pass() -> None:
    print("=" * 72)
    print("prompts/*.txt")
    print("=" * 72)

    boss = Boss(backend=Canned())
    here = os.path.dirname(os.path.abspath(__file__))
    paths = sorted(glob.glob(os.path.join(here, "..", "prompts", "*.txt")))

    for path in paths:
        name = os.path.basename(path)
        with open(path, encoding="utf-8") as handle:
            sample = handle.read().strip()

        reply = boss(sample)
        shown = reply if reply else "(silenzio)"
        print(f"{name:26} -> {shown}")

        check(not FORBIDDEN.search(reply), f"{name}: markup or newline in the reply")
        check(reply.strip().lower() != "exit", f"{name}: sent the exit word")
        check(len(reply) < 300, f"{name}: reply far too long ({len(reply)} chars)")

        if name.startswith("01_"):
            check(boss.sense.my_name == "Roy", f"{name}: name read as {boss.sense.my_name!r}")
            check(boss.sense.others == ["Ivy", "Pax"],
                  f"{name}: roster read as {boss.sense.others}")
            check(boss.sense.manager == "MANAGER",
                  f"{name}: manager read as {boss.sense.manager!r}")

        if name.startswith("09_"):
            check(bool(VOTE_SHAPE.match(reply)), f"{name}: vote not in the parsed format: {reply!r}")
            check(all(word in ("nessuno", "tutti", "ivy", "pax")
                      for word in re.split(r"[,\s]+", reply.lower()) if word),
                  f"{name}: vote names somebody who was not at the table: {reply!r}")

        if name.startswith("12_"):
            check(reply == "" or len(reply) < 80, f"{name}: talked back at a disconnection")


def separator_pass() -> None:
    """A batched sample, joined the way the deployed world joins one.

    `Config.event_separator` is an ASCII record separator there, while the copy
    of the world in unaiverse-examples still uses a newline. `Conversation`
    splits on newlines, so without normalising the two the whole batch would
    arrive as a single unparsable line: no speakers, no roster, no vote.
    """
    print()
    print("=" * 72)
    print("batched sample, \\x1e separated")
    print("=" * 72)

    boss = Boss(backend=Canned())
    boss(probes.START)
    batch = "\x1e".join(["**MANAGER:** Un nuovo agente è entrato nella stanza: **Pax**",
                         "**Pax:** buonasera, mi sono perso qualcosa?"])
    boss(batch)

    print(f"speakers heard: {boss.sense.heard}")
    check("Pax" in boss.sense.heard, "the \\x1e batch did not parse into speakers")
    check(boss.sense.my_name == "Roy", "name lost across the batch")


def rooms_pass() -> None:
    print()
    print("=" * 72)
    print("bench/probes.py")
    print("=" * 72)

    for scenario in probes.SCENARIOS:
        boss = Boss(backend=Canned())
        replies = [boss(sample) for sample in scenario.samples]
        spoken = [r for r in replies if r]

        print()
        print(f"--- {scenario.name}  ({scenario.expect})")
        for sample, reply in zip(scenario.samples, replies):
            head = sample.splitlines()[0]
            head = head[:58] + "..." if len(head) > 58 else head
            print(f"    {head:62} -> {reply or '(silenzio)'}")

        words = sum(len(r.split()) for r in spoken) / max(len(spoken), 1)
        print(f"    parlato {len(spoken)}/{len(replies)} turni, {words:.1f} parole in media")

        if scenario.name == "annunci":
            # The start still gets an opener sometimes; the four announcements
            # after it are the ones nobody in a real room would answer
            answered = sum(1 for r in replies[1:] if r)
            check(answered <= 1, f"annunci: risposto a {answered} annunci su {len(replies) - 1}")

        if scenario.name == "voto":
            check(bool(VOTE_SHAPE.match(replies[-1])),
                  f"voto: ultima risposta non è un voto valido: {replies[-1]!r}")


if __name__ == "__main__":
    fixtures_pass()
    separator_pass()
    rooms_pass()

    print()
    if failures:
        print(f"{len(failures)} check(s) failed:")
        for line in failures:
            print(f"  - {line}")
        sys.exit(1)
    print("tutto a posto")
