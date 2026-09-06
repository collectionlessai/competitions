"""Which game this room is being played, and how hard.

The strategy never changes — find the people, expose the machines, do not get
caught, leave the analyst something to work with. What changes is the tactic,
and it changes for two reasons.

The first is that messages need a purpose. Measured in the real hotel, the agent
produced grammatical Italian that went nowhere: "quindi voi cosa ne pensate di
quel talk sui modelli linguistici?" straight after two guests had ignored the
same question, because the turn had to be filled and nothing said what for.

The second is that a fixed way of playing is a signature. The persona rotates
every room and the writing style rotates with it, but an agent that always
probes, or always allies, is recognisable across rooms regardless of its name.

A tactic also carries a proactivity, which the director multiplies into its
speaking chance: being bored is not only a tactic, it is a different amount of
talking.
"""

import os
import re
import random

HERE = os.path.dirname(os.path.abspath(__file__))
TACTICS_FILE = os.path.join(HERE, "tactics_it.txt")


class Tactic:
    """One way of playing a room."""

    def __init__(self, name: str, drive: float, lines: list):
        self.name = name
        self.drive = drive          # 0 quiet, 1 talkative
        self.lines = lines

    def block(self) -> str:
        return "COME STAI GIOCANDO QUESTA STANZA\n" + "\n".join(self.lines)

    def __repr__(self):
        return f"<{self.name} drive={self.drive}>"


def load(path: str = TACTICS_FILE) -> list:
    """Every tactic in the file, in order."""
    out, name, drive, lines = [], "", 0.6, []

    def close():
        if name and lines:
            out.append(Tactic(name, drive, list(lines)))

    try:
        text = open(path, encoding="utf-8").read()
    except OSError:
        return out

    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("## TATTICA"):
            close()
            name, drive, lines = line[10:].strip().lower(), 0.6, []
        elif line.startswith("@") and name:
            try:
                drive = float(line[1:].strip())
            except ValueError:
                drive = 0.6
        elif line and not line.startswith("#") and name:
            lines.append(line)
    close()
    return out


class Playbook:
    """Picks a tactic per room and swaps it when the room stops fitting it.

    Switching is deliberately rare. A guest who changes how they play every
    minute is not reacting, they are flailing, and the point of holding one is
    that the messages add up to something.
    """

    def __init__(self, path: str = TACTICS_FILE):
        self.tactics = load(path)
        self.current = None
        self.since = 0.0
        self.switched = 0

    def new_room(self, when: float = 0.0) -> None:
        self.current = random.choice(self.tactics) if self.tactics else None
        self.since = when
        self.switched = 0

    def pick(self, name: str) -> None:
        for tactic in self.tactics:
            if tactic.name == name:
                self.current, self.switched = tactic, self.switched + 1
                return

    @property
    def drive(self) -> float:
        return self.current.drive if self.current else 0.6

    def block(self) -> str:
        return self.current.block() if self.current else ""

    def reconsider(self, elapsed: float, open_guests: list, settled: list,
                   addressed_us: bool) -> str:
        """Swap tactics when the room has plainly changed, and say why.

        Three situations earn a change, and nothing else does:

        * everybody left to judge has been settled as a machine, so probing is
          spent effort — go quiet instead;
        * we are being drawn into conversation by somebody unresolved, which is
          the moment allying is worth more than interrogating;
        * half the room has gone by and we are still holding a tactic that has
          produced nothing, which usually means the room will not play along.
        """
        if not self.current or self.switched >= 2 or elapsed - self.since < 90.0:
            return ""

        if settled and not open_guests and self.current.name != "annoiato":
            self.since = elapsed
            self.pick("annoiato")
            return "annoiato"

        if addressed_us and len(open_guests) >= 1 and self.current.name in ("sonda", "specchio"):
            self.since = elapsed
            self.pick("alleato")
            return "alleato"

        if elapsed > 150.0 and self.current.name in ("annoiato", "distratto") and open_guests:
            self.since = elapsed
            self.pick("sonda")
            return "sonda"
        return ""
