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
import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
CONTEXT_FILE = os.path.join(HERE, "context_it.txt")


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
        self.always: list[str] = []
        self.notes: list[str] = []
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
        section = ""
        with open(self.path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                # Tested before the comment skip: a section header starts with
                # "#" too, and checking the other way round drops every one
                if line.startswith("##"):
                    section = line.lstrip("#").strip().upper()
                    continue
                if line.startswith("#"):
                    continue

                if section == "SEMPRE":
                    self.always.append(line)
                elif section == "NOTE":
                    self.notes.append(line)
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

    def block(self, when: datetime.datetime | None = None) -> str:
        """The whole context for one turn, ready to drop into the prompt."""
        self.reload()
        when = when or datetime.datetime.now()

        lines = ["DOVE SEI E COSA SAI"]
        lines += self.always

        now = self.now_line(when)
        if now:
            when_word = "Adesso" if self.during_conference(when) else "In questo momento"
            lines.append(f"{when_word} ({when:%H:%M}): {now}")

        if self.notes:
            lines.append("Cose che sai perché c'eri:")
            lines += [f"- {note}" for note in self.notes]
        return "\n".join(lines)
