"""ELIZA, 1966, in Italian

Regular expressions and a dice roll. The first rule in RULES whose pattern
matches the last line wins, its answer is filled in with whatever the pattern
captured, and when nothing matches, which is most lines, it picks one of the
FALLBACK grunts instead. No model to load and nothing to install.

Two things are easy to get wrong once you start editing the rules. Captured
text goes through flip() before it is quoted back, so `mi sento come mio padre`
can come back as `ti capita spesso di sentirti come tuo padre?` with the
possessive pointing at the right person. Order matters too, since the list is
read top to bottom and stops at the first hit. Move the last rule, the one that
catches any question at all, up to the top and every question in the room comes
back as one of its three shrugs, whatever was actually asked.
"""

import re
import random
import torch

from utils import Conversation


RULES = [
    (r"(?:io )?sono (.*)",     ["da quanto sei {0}?", "perché dici di essere {0}?",
                                "e ti dà fastidio essere {0}?"]),
    (r"(?:mi serve|ho bisogno di) (.*)", ["e se lo avessi, cosa cambierebbe?"]),
    (r"non riesco (?:a |ad )?(.*)", ["cosa te lo impedisce?", "e se ci riuscissi?"]),
    (r"mi sento (.*)", ["ti capita spesso di sentirti {0}?", "da stamattina?"]),
    (r"(?:perch(?:é|e)|perche') (.*)", ["è davvero il motivo?", "e a parte quello?"]),
    (r".*\b(?:bot|ia|ai|robot|gpt|macchina)\b.*", ["me lo chiedono tutti",
                                                   "stavo per chiedertelo io",
                                                   "e secondo te?"]),
    (r".*\b(?:lavoro|ufficio|capo|studio)\b.*", ["che lavoro fai di preciso?",
                                                 "giornata lunga?"]),
    (r".*\b(?:stanc[oa]|sonno|sveglia)\b.*", ["uguale, ho dormito poco",
                                              "giornata lunga allora?"]),
    (r".*\b(?:roma|milano|napoli|torino|firenze|palermo)\b.*", ["ci sono stato una volta",
                                                                "com'è adesso lì?"]),
    (r"(?:ciao|ehi|hey|salve|buonasera|buongiorno)\b.*", ["ciao", "ehi", "oh ciao"]),
    (r".*\?", ["perché me lo chiedi?", "boh, tu che dici?", "non saprei"]),
]

# Applied to the captured group, never to the whole line.
FLIP = {"io": "tu", "mi": "ti", "me": "te", "mio": "tuo", "mia": "tua",
        "miei": "tuoi", "mie": "tue", "sono": "sei", "ho": "hai",
        "tu": "io", "ti": "mi", "te": "me", "tuo": "mio", "tua": "mia",
        "sei": "sono", "hai": "ho"}

# Where most lines end up.
FALLBACK = ["vai avanti", "mah", "in che senso?", "capito", "aspetta cosa",
            "eh", "boh"]

# For the first turn of a room, when there is no line to answer.
OPENERS = ["ciao", "buonasera", "ehi c'è nessuno", "ciao a tutti"]


def flip(text: str) -> str:
    return " ".join(FLIP.get(word.lower(), word) for word in text.split())


class Eliza(torch.nn.Module):

    def __init__(self, seed: int | None = None):
        super().__init__()
        self.rng = random.Random(seed)  # pass a seed to get the same room twice
        self.rules = [(re.compile(pattern, re.I), answers) for pattern, answers in RULES]
        self.conv = Conversation()

    def reply(self, text: str) -> str:
        """One line in, one line out. Nothing here reads the history."""
        for pattern, answers in self.rules:
            match = pattern.match(text.strip())
            if match:
                answer = self.rng.choice(answers)
                return answer.format(*[flip(group) for group in match.groups()])
        return self.rng.choice(FALLBACK)

    def forward(self, sample: str) -> str:
        # The whole history goes in and only the bottom line comes back out.
        # A rule that wanted context would read self.conv.history itself.
        self.conv.add(sample)

        heard = self.conv.last_message()   # None until somebody speaks
        reply = self.reply(heard.text) if heard else self.rng.choice(OPENERS)
        self.conv.remember(reply)
        return reply
