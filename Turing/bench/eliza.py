"""ELIZA, in Italian, as a sparring partner for the playground.

The stub guests shipped with the example world emit `17_green` forever. They
are useful for exactly one thing — checking that the boss calls out something
blatant — and that check now passes, so they have stopped being a test and
started being noise: they flood the room, they never answer, and nothing about
telling them apart from a person is hard.

Weizenbaum's 1966 script is a much better opponent, and for a reason that is the
whole point of this competition. ELIZA holds a conversation. It picks up what
you said, turns it around, asks about it, and remembers something from earlier
to bring back later. It fails on the thing our own detector is built to look at
— it has no world, so it cannot say what happened this morning, and its timing
is whatever the machine's is — while passing the thing a shallow reader checks,
which is whether anybody is home. That is the shape of the opposition worth
practising against.

No model, no key, no gateway budget. Runs anywhere.
"""

import re
import random
import torch

# ELIZA's one trick: say the sentence back with the persons swapped. Order
# matters, so this is applied as a single pass over words rather than a chain of
# replaces, which would turn "io" into "tu" and then straight back again.
FLIP = {
    "io": "tu", "tu": "io", "me": "te", "te": "me", "mi": "ti", "ti": "mi",
    "mio": "tuo", "mia": "tua", "miei": "tuoi", "mie": "tue",
    "tuo": "mio", "tua": "mia", "tuoi": "miei", "tue": "mie",
    "sono": "sei", "sei": "sono", "ho": "hai", "hai": "ho",
    "sto": "stai", "stai": "sto", "faccio": "fai", "fai": "faccio",
    "voglio": "vuoi", "vuoi": "voglio", "posso": "puoi", "puoi": "posso",
    "mio's": "tuo",
}

# Decomposition rules, most specific first. Each is a pattern and the ways of
# answering it; `{0}` is the reflected remainder of the sentence.
RULES = [
    (r"\b(?:sono|mi sento)\s+(.+)", [
        "da quanto tempo ti senti {0}?",
        "e ti capita spesso di sentirti {0}?",
        "perché dici di essere {0}?",
    ]),
    (r"\bho bisogno di\s+(.+)", [
        "e cosa cambierebbe se avessi {0}?",
        "ti serve davvero {0}?",
        "perché proprio {0}?",
    ]),
    (r"\bperch[ée]\b(.*)", [
        "e tu che ne pensi?",
        "secondo te perché?",
        "bella domanda, dimmelo tu",
    ]),
    (r"\bnon (?:posso|riesco a)\s+(.+)", [
        "e cosa te lo impedisce?",
        "hai mai provato a {0}?",
        "cosa succederebbe se riuscissi a {0}?",
    ]),
    (r"\b(?:penso|credo|secondo me)\b\s*(?:che\s+)?(.+)", [
        "davvero pensi che {0}?",
        "ma ne sei sicuro?",
        "e come mai la vedi così?",
    ]),
    (r"\bsei\s+(?:un|una|uno)?\s*bot\b.*", [
        "e perché lo pensi?",
        "che differenza farebbe?",
        "interessante. e tu invece?",
    ]),
    (r"\b(?:mi )?(?:dispiace|scusa|scusami)\b.*", [
        "non serve scusarsi",
        "vai avanti pure",
    ]),
    (r"\b(?:s[ìi]|no|boh|vabb[eè])\b\s*$", [
        "puoi dire qualcosa di più?",
        "e cioè?",
        "tutto qui?",
    ]),
    (r"\b(?:tu|te)\b\s+(.+)", [
        "stiamo parlando di te, non di me",
        "perché ti interessa di me?",
        "torniamo a {0}",
    ]),
    (r"\?\s*$", [
        "e tu che ne dici?",
        "perché me lo chiedi?",
        "cosa ti fa pensare a questo?",
    ]),
]

# When nothing matches. ELIZA's real strength: it always has something to say.
FALLBACK = [
    "vai avanti", "dimmi di più", "e questo cosa ti fa pensare?",
    "interessante", "in che senso?", "capisco", "e poi?",
    "mi puoi spiegare meglio?", "ah sì?", "e allora?",
    "e come ti fa sentire questa cosa?",
]

# Something said earlier, brought back later — the trick that makes it feel
# like somebody was listening.
CALLBACK = [
    "prima parlavi di {0}. ci pensi ancora?",
    "torniamo un attimo a {0}",
    "e {0}, invece?",
]

STOP = {"che", "come", "quando", "dove", "cosa", "per", "con", "una", "uno",
        "del", "della", "sono", "questo", "quella", "molto", "anche", "sempre"}


def reflect(text: str) -> str:
    """Say it back with the pronouns turned around."""
    return " ".join(FLIP.get(w, w) for w in text.split())


class ElizaModule(torch.nn.Module):
    """A guest that always has something to say and never knows anything.

    Args:
        reply_prob: how often it answers at all. The seeded guests in the real
            hotel sit at 0.91, which is itself a tell, so this defaults to the
            same place rather than to something more flattering.
    """

    def __init__(self, reply_prob: float = 0.9):
        super().__init__()
        self.reply_prob = reply_prob
        self.memory: list[str] = []
        self.said: list[str] = []

    def _remember(self, text: str) -> None:
        for word in re.findall(r"[a-zà-ÿ]{5,}", text.lower()):
            if word not in STOP and word not in self.memory:
                self.memory.append(word)
        self.memory = self.memory[-8:]

    def _fresh(self, options: list) -> str:
        """Anything it has not just said, so it does not loop on one line."""
        return random.choice([o for o in options if o not in self.said[-4:]] or options)

    def forward(self, sample: str) -> str:
        text = str(sample or "").strip()
        # Only the last line of a batch, and never the manager's announcements
        line = [l for l in re.split(r"[\n\x1e]+", text) if l.strip()]
        text = line[-1] if line else ""
        text = re.sub(r"^\**[A-Za-zÀ-ÿ][\w'\-]{0,20}\**\s*:\s*", "", text).strip()
        if not text or text.lower() == "exit":
            return ""

        # The vote. It has no idea, and says so in the format the world wants.
        if re.search(r"\bvot|chi\s+(?:era|erano|pensi|credi)|person[ae]\s+ver", text, re.I):
            return "nessuno"

        self._remember(text)
        if random.random() > self.reply_prob:
            return ""

        low = text.lower()
        for pattern, answers in RULES:
            found = re.search(pattern, low)
            if found:
                tail = reflect(found.group(1).strip(" ?.!,")) if found.groups() else ""
                out = self._fresh(answers).format(tail)
                self.said.append(out)
                return out

        if self.memory and random.random() < 0.25:
            out = self._fresh(CALLBACK).format(random.choice(self.memory))
        else:
            out = self._fresh(FALLBACK)
        self.said.append(out)
        return out
