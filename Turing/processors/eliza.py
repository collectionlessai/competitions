"""An Italian adaptation of ELIZA, 1966.

The processor applies regular expressions to the latest event, selects an answer
from the first matching rule and fills it with any captured text. Most events
match no rule, in which case it chooses a short response from FALLBACK. It needs
no model or extra package.

Rule order matters because matching stops at the first hit. Captured text passes
through flip() before being reused. For example, `mi sento come mio padre`
becomes `ti capita spesso di sentirti come tuo padre?`, with the possessive
changed as well. The final rule matches any question, so moving it earlier would
hide the more specific question rules below it.

Conversation labels remembered local replies in its neutral transcript. ELIZA
does not assign chat-API roles to either local or remote messages.
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

# Apply these substitutions only to captured groups.
FLIP = {"io": "tu", "mi": "ti", "me": "te", "mio": "tuo", "mia": "tua",
        "miei": "tuoi", "mie": "tue", "sono": "sei", "ho": "hai",
        "tu": "io", "ti": "mi", "te": "me", "tuo": "mio", "tua": "mia",
        "sei": "sono", "hai": "ho"}

# Default replies for events that match no rule.
FALLBACK = ["vai avanti", "mah", "in che senso?", "capito", "aspetta cosa",
            "eh", "boh"]

# Greetings used before another guest has spoken.
OPENERS = ["ciao", "buonasera", "ehi c'è nessuno", "ciao a tutti"]


def flip(text: str) -> str:
    return " ".join(FLIP.get(word.lower(), word) for word in text.split())


class Eliza(torch.nn.Module):

    def __init__(self, seed: int | None = None):
        super().__init__()
        self.rng = random.Random(seed)  # A seed makes response choices repeatable.
        self.rules = [(re.compile(pattern, re.I), answers) for pattern, answers in RULES]
        self.conv = Conversation()

    def reply(self, text: str) -> str:
        """Produce one reply from an event without consulting older history."""
        for pattern, answers in self.rules:
            match = pattern.match(text.strip())
            if match:
                answer = self.rng.choice(answers)
                return answer.format(*[flip(group) for group in match.groups()])
        return self.rng.choice(FALLBACK)

    def forward(self, sample: str) -> str:
        # The current rules inspect only the latest remote event. A contextual
        # rule can read self.conv.history instead.
        self.conv.add(sample)

        heard = self.conv.last_message()   # None before another guest speaks.
        reply = self.reply(heard.text) if heard else self.rng.choice(OPENERS)
        self.conv.remember(reply)
        return reply


def main() -> None:
    """Run a small terminal conversation for testing the processor."""
    eliza = Eliza()
    print("Eliza is ready. Press Ctrl-D or Ctrl-C to stop.")
    try:
        while True:
            text = input("you> ").strip()
            if text:
                print(f"eliza> {eliza(f'**you:** {text}')}")
    except (EOFError, KeyboardInterrupt):
        print()


if __name__ == "__main__":
    main()
