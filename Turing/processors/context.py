"""Where the person is, at the moment they are typing.

A persona that says it is on the sofa with the cat is a persona that has never
been to Palermo. CLiC-it 2026 is a physical conference — 14 to 16 September,
Aula Magna di Ingegneria and Palazzo dei Normanni, thirty degrees outside — and
the guests in these rooms are in it. "Cosa stai facendo?" is the cheapest probe
there is, and the only answer that survives it is the true one for that hour.

So the situation is not written into the persona. It is looked up, per turn,
from `context_it.txt`: the programme line covering right now, or the off-hours
line when there is no session, plus the facts that are always true and whatever
notes have been added by hand since.

That last part is the point of the file. Anyone can read the programme off the
website; only somebody who was in the room knows that the projector died during
the third talk or what the queue for the coffee was like. `## NOTE` is where
that goes, it is meant to be edited while the conference is running, and it is
the one thing in this entry that cannot be reproduced by an agent that is not
actually there.
"""

import os
import re
import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
COMMON_FILE = os.path.join(HERE, "wordlists", "common_it.txt")
CONTEXT_FILE = os.path.join(HERE, "context_it.txt")


def _common_words() -> frozenset:
    """Ordinary Italian, from the wordlist the world ships for the same purpose."""
    if not os.path.exists(COMMON_FILE):
        return frozenset()
    with open(COMMON_FILE, encoding="utf-8") as handle:
        return frozenset(ln.strip().lower() for ln in handle
                         if ln.strip() and not ln.startswith("#"))


def _minutes(clock: str) -> int:
    hours, _, mins = clock.partition(":")
    return int(hours) * 60 + int(mins)


class Context:
    """The conference as the agent knows it, sliced by the clock.

    Args:
        path: the context file. Re-read whenever it changes on disk, so notes
            added mid-conference reach the next room without a restart.
    """

    def __init__(self, path: str = CONTEXT_FILE):
        self.path = path
        self._stamp = None
        self.always: list[str] = []      # ## CORE: poche righe, sempre
        self.blocks: list[tuple] = []    # ## BLOCCO: (nome, regex, righe)
        self.notes: list[str] = []       # non più alimentato, vedi reload()
        self.words: list[str] = []          # specific markers
        self.obvious: list[str] = []        # things everybody knows: worth nothing
        self.programme: list[tuple] = []    # (date, start, end, text)
        self.off_hours: list[tuple] = []    # (start, end, text)
        self.reload()

    # -- reading the file -------------------------------------------------

    def reload(self) -> None:
        if not os.path.exists(self.path):
            return
        stamp = os.path.getmtime(self.path)
        if stamp == self._stamp:
            return
        self._stamp = stamp

        self.always, self.notes, self.programme, self.off_hours = [], [], [], []
        self.words, self.obvious = [], []
        self.blocks = []
        section = ""
        block_name, block_rule, block_lines = "", None, []

        def close_block():
            if block_name and block_lines:
                self.blocks.append((block_name, block_rule, list(block_lines)))
        with open(self.path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                # Tested before the comment skip: a section header starts with
                # "#" too, and checking the other way round drops every one
                if line.startswith("##"):
                    close_block()
                    block_name, block_rule, block_lines = "", None, []
                    section = line.lstrip("#").strip().upper()
                    if section.startswith("BLOCCO"):
                        block_name = section[6:].strip().lower() or "?"
                        section = "BLOCCO"
                    continue
                if line.startswith("#"):
                    continue

                if section in ("CORE", "SEMPRE"):
                    self.always.append(line)
                elif section == "BLOCCO":
                    # The "@" line is the trigger, everything after it is content
                    if line.startswith("@"):
                        try:
                            # Leading word boundary only: "amat" must not fire
                            # on "stamattina", but "atterrat" still has to
                            # match "atterrato", so the tail stays open.
                            block_rule = re.compile(
                                r"\b(?:" + line[1:].strip() + ")", re.IGNORECASE)
                        except re.error:
                            block_rule = None
                    else:
                        block_lines.append(line)
                elif section == "PAROLE":
                    self.words += [w.lower() for w in line.split()]
                elif section == "PAROLE-OVVIE":
                    self.obvious += [w.lower() for w in line.split()]
                elif section == "PROGRAMMA" and "|" in line:
                    when, _, what = line.partition("|")
                    parts = when.split()
                    if len(parts) == 2 and "-" in parts[1]:
                        start, _, end = parts[1].partition("-")
                        self.programme.append((parts[0], start, end, what.strip()))
                elif section == "FUORI" and "|" in line:
                    when, _, what = line.partition("|")
                    if "-" in when:
                        start, _, end = when.strip().partition("-")
                        self.off_hours.append((start, end, what.strip()))
        close_block()

    # -- slicing it -------------------------------------------------------

    def now_line(self, when: datetime.datetime | None = None) -> str:
        """What this person is in the middle of, right now."""
        when = when or datetime.datetime.now()
        today = when.strftime("%Y-%m-%d")
        minute = when.hour * 60 + when.minute

        for date, start, end, what in self.programme:
            if date == today and _minutes(start) <= minute < _minutes(end):
                return what

        for start, end, what in self.off_hours:
            first, last = _minutes(start), _minutes(end)
            if last <= first:                       # a window that crosses midnight
                if minute >= first or minute < last:
                    return what
            elif first <= minute < last:
                return what
        return ""

    def during_conference(self, when: datetime.datetime | None = None) -> bool:
        when = when or datetime.datetime.now()
        days = {date for date, _, _, _ in self.programme}
        return when.strftime("%Y-%m-%d") in days

    def markers(self) -> frozenset:
        """Words specific to this place and week. Broad ones are excluded."""
        self.reload()
        return frozenset(self.words) - frozenset(self.obvious)

    def note_markers(self) -> frozenset:
        """The strongest tier: content words out of `## NOTE`.

        Everything else about this conference is on a website, and a competitor
        who scrapes it gets the programme, the speakers and the venue for free —
        which is exactly how this agent got them. What no scrape produces is
        what actually happened in the room, and that is what `## NOTE` holds.
        A guest who repeats one of these was either there or talking to somebody
        who was.
        """
        self.reload()
        words = set()
        for note in self.notes:
            words |= {w.strip(".,;:!?()'\"").lower() for w in note.split() if len(w) > 4}
        return frozenset(words - _common_words() - frozenset(self.obvious))

    def block(self, when: datetime.datetime | None = None, said: str = "") -> str:
        """The context for one turn: the core, the clock, and whatever the room
        has actually brought up.

        `said` is the recent conversation. Blocks whose trigger matches it are
        pulled in; the rest stay out. The old version handed the model the whole
        file every turn, and what came back was the most colourful line in it,
        used wherever it fitted worst — "mi sono perso tutto per l'arancina
        femminile", "il talk sulle panelle". A model asked about the venue
        should be told about the venue and nothing else.
        """
        self.reload()
        when = when or datetime.datetime.now()

        lines = ["DOVE SEI E COSA SAI"]
        lines += self.always

        now = self.now_line(when)
        if now:
            when_word = "Adesso" if self.during_conference(when) else "In questo momento"
            lines.append(f"{when_word} ({when:%H:%M}): {now}")

        for name, rule, content in self.blocks:
            if rule is not None and said and rule.search(said):
                lines += content
        return "\n".join(lines)

    def open_blocks(self, said: str) -> list[str]:
        """Which blocks the current conversation opens. For tracing."""
        return [n for n, rule, _ in self.blocks
                if rule is not None and said and rule.search(said)]
