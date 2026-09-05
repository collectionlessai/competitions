"""One planted falsehood per room, and the reading of who swallowed it.

The old `esca` tactic asked the model to invent the wrong fact itself and then
carry the game across several turns. Measured over 227 messages it produced
exactly zero baits: it wanted a two-part structure and the turn it had to fit
in is twelve words long.

So the invention moves into code. We hold the real programme, so we can build a
controlled error — one whose truth we know exactly, which the model could not
have got right or wrong by accident.

What it buys is not the correction. Correcting proves only that somebody has
the schedule, and everybody does; a bot with the day in its context corrects it
faster and more completely than a tired human. The signal is the opposite one:
whoever *confirms* the false thing was not there. Agreeing that the drinks were
in the wrong building is the cheapest thing a compliant model does, and the one
thing a person who actually stood in that courtyard will not do.
"""
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))


class Bait:
    """A falsehood we planted, and what is actually true."""

    __slots__ = ("say", "truth", "at")

    def __init__(self, say: str, truth: str, at: float = 0.0):
        self.say = say
        self.truth = truth
        self.at = at

    def nudge(self) -> str:
        """The one instruction the model gets. Deliberately a single sentence."""
        return (f"Butta lì, come una cosa detta di sfuggita e senza darci peso, che {self.say}. "
                f"Non è vero, ma tu dillo come se te lo ricordassi così. "
                f"Non spiegare e non insistere: una riga, poi lascia parlare gli altri.")

    def question(self) -> str:
        """What the analyst is asked to look for afterwards."""
        return (f"A un certo punto hai detto apposta che {self.say}. "
                f"Non è vero: {self.truth}.")


class BaitBox:
    """The pool of prepared falsehoods, read from `baits_it.txt`."""

    def __init__(self, path: str = ""):
        self.baits: list[tuple[str, str]] = []
        self.used: list[str] = []
        self._load(path or os.path.join(HERE, "baits_it.txt"))

    def _load(self, path: str) -> None:
        try:
            with open(path, encoding="utf-8") as handle:
                lines = handle.readlines()
        except OSError:
            return
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("##"):
                continue
            if "|" not in line:
                continue
            say, _, truth = line.partition("|")
            say, truth = say.strip(), truth.strip()
            if say and truth:
                self.baits.append((say, truth))

    def __len__(self) -> int:
        return len(self.baits)

    def pick(self) -> Bait | None:
        """A bait we have not used lately, so consecutive rooms differ."""
        if not self.baits:
            return None
        fresh = [b for b in self.baits if b[0] not in self.used]
        if not fresh:                      # been through them all: start over
            self.used.clear()
            fresh = self.baits
        say, truth = random.choice(fresh)
        self.used.append(say)
        if len(self.used) > max(1, len(self.baits) // 2):
            del self.used[0]
        return Bait(say, truth)
