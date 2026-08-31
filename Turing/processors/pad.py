"""What the agent is keeping in the back of its mind.

The transcript says what was said. It does not say what any of it *meant to us*,
and three different failures come from the same missing memory:

**Contradicting ourselves.** Over five minutes a model will happily say it came
down on the Thursday and later that it flew in this morning. Nothing in a
transcript stops that; a person simply remembers what they claimed.

**Throwing the stone and hiding the hand.** The director draws a fresh beat every
turn, so it can propose a gambit and then, next turn, roll "change the subject"
and start talking about the coffee. That is worse than never proposing anything:
a person who suggests a joke sticks with it, defends it when somebody sneers,
folds in the objection and tries again. Following through needs an intention
that outlives the turn that had it.

**Baits with no bookkeeping.** A probe is only worth anything if we remember who
was in the room when it went out and what each of them did about it. Otherwise
it is theatre — enjoyable, but not evidence.

All three are the same object: a few things we are holding on to, each with a
shelf life. Kept deliberately small. This is a notepad, not a knowledge base,
and everything on it is thrown away when the room ends.
"""

import time


class Note:
    """One thing being held in mind."""

    def __init__(self, kind: str, text: str, about: str = "", ttl: float = 120.0):
        self.kind = kind          # claim | plan | read | bait
        self.text = text
        self.about = about        # which guest it concerns, when it concerns one
        self.born = time.monotonic()
        self.ttl = ttl
        self.closed = False

    @property
    def age(self) -> float:
        return time.monotonic() - self.born

    @property
    def live(self) -> bool:
        return not self.closed and self.age < self.ttl

    def __repr__(self):
        return f"<{self.kind} {self.text[:30]!r} about={self.about or '-'}>"


class Pad:
    """A handful of live notes, oldest dropped first.

    Args:
        keep: how many notes to hold at once. Small on purpose — the point is
            the two or three things actually on your mind, not a record.
    """

    KINDS = ("claim", "plan", "read", "bait")

    def __init__(self, keep: int = 8):
        self.keep = keep
        self.notes: list[Note] = []
        self.profiles: dict[str, str] = {}

    def clear(self) -> None:
        self.notes = []
        self.profiles = {}

    def add(self, kind: str, text: str, about: str = "", ttl: float = 120.0) -> Note:
        note = Note(kind, text, about, ttl)
        self.notes.append(note)
        if len(self.notes) > self.keep:
            del self.notes[:-self.keep]
        return note

    def live(self, kind: str = "") -> list:
        return [n for n in self.notes if n.live and (not kind or n.kind == kind)]

    def close(self, note: Note) -> None:
        note.closed = True

    # -- the three things it is for ---------------------------------------

    def open_plan(self):
        """The gambit we are in the middle of, if any."""
        plans = self.live("plan")
        return plans[-1] if plans else None

    def open_bait(self):
        baits = self.live("bait")
        return baits[-1] if baits else None

    def profile(self, name: str, text: str = "") -> str:
        """What we make of one guest: written to when given text, read otherwise.

        This is the actor's scratchpad, not the judge's. It is here to socialise
        with: what this person seems to be, what they said they were up to, how
        they write. Whether they are a machine is a separate question asked at
        the end, from scratch, by somebody with a different job.
        """
        if text:
            self.profiles[name] = text
        return self.profiles.get(name, "")

    def read_of(self, name: str) -> str:
        """What we last decided about one guest."""
        for note in reversed(self.live("read")):
            if note.about == name:
                return note.text
        return ""

    # -- what goes into the prompt ----------------------------------------

    def game_state(self, settled: list, open_names: list) -> str:
        """Where the game stands and what there is left to do about it.

        Without this the actor treats every guest as an equally open question
        for the whole room, which is both a waste of turns and, when one of them
        is plainly a spammer, visibly odd.
        """
        lines = []
        if settled:
            lines.append(f"Su {', '.join(settled)} non c'è più niente da capire: sono bot, "
                         "punto. Non discuterne come se fosse un'ipotesi, non stargli dietro, "
                         "al massimo una battuta e via.")
        if open_names:
            lines.append(f"Quelli su cui non hai ancora capito: {', '.join(open_names)}. "
                         "È lì che vale la pena spendere i messaggi: falli parlare di oggi, "
                         "di qui, di una cosa che uno che c'era saprebbe.")
        elif settled:
            lines.append("Non è rimasto nessuno di interessante: puoi anche startene zitto.")
        return "\n".join(lines)

    def block(self) -> str:
        """The notepad as the model should see it, or "" when there is nothing.

        Written as things already decided rather than as instructions, because
        it is describing the person's own state of mind, not directing them.
        """
        claims = [n.text for n in self.live("claim")]
        plan = self.open_plan()
        reads = [f"{n.about}: {n.text}" for n in self.live("read") if n.about]
        people = [f"{who}: {what}" for who, what in self.profiles.items() if what]

        lines = []
        if claims:
            lines.append("Cose che hai già detto di te (non contraddirti): "
                         + "; ".join(claims[-3:]))
        if plan:
            lines.append(f"Hai proposto tu una cosa poco fa: \"{plan.text}\". "
                         "Stai sul pezzo: portala avanti, difendila se ti sfottono, "
                         "prendi in giro chi non ci sta, o rilanciala cambiandola. "
                         "Non cambiare argomento come se non l'avessi mai detta.")
        if people:
            lines.append("Che tipi ti sembrano: " + " | ".join(people))
        if reads:
            lines.append("Cosa pensi degli altri finora: " + "; ".join(reads[-3:]))

        return "QUELLO CHE HAI IN TESTA\n" + "\n".join(lines) if lines else ""
