"""What the room has told us about itself, so far.

TuringBoss opened `context_it.txt` and knew the venue, the programme and the
weather before anybody spoke. This agent is not allowed to: it goes into rooms
it knows nothing about, and whatever it ends up knowing has to arrive through
the conversation.

So this takes the place `Context` had in the prompt. Same slot, different
source: written before, learned during.

Two rules run through it.

**Nothing without a source.** Every slot keeps the sentence it came from, and
who said it. A deduction with no provenance is indistinguishable from an
invention, and inventing is exactly the failure this agent is most exposed to —
it has no file to check itself against.

**Empty is a legitimate answer.** A room where nobody says where they are
leaves `luogo` empty for the whole five minutes, and the prompt has to say so
out loud rather than leave a gap the model will fill on its own. Being vague is
a tell; being concrete about the little you know is not.
"""


class Finding:
    """One thing we believe, and the line that made us believe it."""

    __slots__ = ("value", "quote", "who", "at")

    def __init__(self, value: str, quote: str = "", who: str = "", at: float = 0.0):
        self.value = value
        self.quote = quote
        self.who = who
        self.at = at

    def __repr__(self):
        return f"<{self.value!r} da {self.who or '?'}>"


class Findings:
    """The slots, in the order they are worth filling.

    `lingua` and `situazione` are the two that matter: they are what the rest of
    the agent needs to stop guessing. `luogo` unlocks the weather. `argomento`
    and `chi` are colour — useful in the prompt, not worth a turn of their own.
    """

    CASELLE = ("lingua", "situazione", "luogo", "argomento")

    # What the room has to have told us before the agent stops asking and
    # starts playing. Deliberately two, not four: a guest still taking a census
    # at minute three is doing something no person does.
    ESSENZIALI = ("lingua", "situazione")

    def __init__(self):
        self.caselle: dict = {}
        self.su_ciascuno: dict = {}      # nome -> quello che ha detto di sé

    def clear(self) -> None:
        self.caselle = {}
        self.su_ciascuno = {}

    # -- writing ----------------------------------------------------------

    def note(self, casella: str, value: str, quote: str = "",
             who: str = "", at: float = 0.0) -> bool:
        """Record a slot, if it is not already filled. Returns whether it took.

        First answer wins. A room that keeps revising where it is would be a
        room worth doubting, and overwriting would make the quote a lie.
        """
        if casella not in self.CASELLE or not value:
            return False
        if casella in self.caselle:
            return False
        self.caselle[casella] = Finding(value.strip(), quote.strip(), who, at)
        return True

    def note_person(self, who: str, said: str) -> None:
        """Something a guest asserted about themselves."""
        if not who or not said:
            return
        self.su_ciascuno.setdefault(who, []).append(said.strip())

    # -- reading ----------------------------------------------------------

    def get(self, casella: str) -> str:
        found = self.caselle.get(casella)
        return found.value if found else ""

    def missing(self) -> list:
        return [c for c in self.ESSENZIALI if c not in self.caselle]

    def enough(self) -> bool:
        """Whether the essentials are in and the agent can stop gathering."""
        return not self.missing()

    def block(self) -> str:
        """What the model is shown. Says the empty parts out loud.

        The absence has to be visible. Left as a gap, a model fills it in — and
        an agent that invents a city it was never told is doing the one thing
        this whole version exists to avoid.
        """
        righe = []
        for casella in self.CASELLE:
            found = self.caselle.get(casella)
            if found:
                fonte = f'  (l\'ha detto {found.who}: "{found.quote}")' if found.quote else ""
                righe.append(f"- {casella}: {found.value}{fonte}")
            else:
                righe.append(f"- {casella}: NON LO SAI ANCORA")

        if self.su_ciascuno:
            for who, detti in list(self.su_ciascuno.items())[:5]:
                righe.append(f"- {who} ha detto di sé: {'; '.join(detti[-2:])}")

        return "QUELLO CHE HAI CAPITO DI QUESTA STANZA\n" + "\n".join(righe)
