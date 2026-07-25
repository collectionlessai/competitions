"""ELIZA, 1966: pattern matching, no model, no API key, no GPU.

Weizenbaum's trick was to reflect the other person's sentence back at them
instead of answering it, which is close to what a bored person in a group chat
does anyway. It never contradicts itself, never writes a tidy paragraph and
never lectures, so it avoids most of the tells an LLM has to be talked out of.
It loses when somebody asks it something concrete twice.

Treat it as the baseline: if your LLM does not beat this, the problem is the
prompt, not the model.

    python -m processors.eliza     # chat with it in your terminal
"""

import re
import random
import torch

from utils import last_message, is_vote_request, other_names, format_vote


# What to say back, by keyword, tried in order. {0} is whatever the speaker said
# after the keyword, with pronouns flipped ("i am tired" -> "you are tired").
# The later entries are catch-alls for common chat openers; the earlier ones are
# the classic reflection rules.
RULES = [
    (r"i(?:'m| am) (.*)",  ["how long have you been {0}?", "why do you say you are {0}?",
                            "does it bother you being {0}?"]),
    (r"i need (.*)",       ["what would change if you got {0}?"]),
    (r"i can't (.*)",      ["what stops you from {0}?"]),
    (r"i feel (.*)",       ["do you often feel {0}?"]),
    (r"because (.*)",      ["is that the real reason?", "any other reason?"]),
    (r".*\b(?:bot|ai|robot|gpt)\b.*", ["everyone keeps asking that", "i was wondering the same about you"]),
    (r".*\b(?:work|job|boss)\b.*",    ["what do you do exactly?", "long day?"]),
    (r".*\b(?:tired|sleepy)\b.*",     ["same, barely slept", "long day then?"]),
    (r"(?:hi|hey|hello)\b.*",         ["hey", "hi there", "oh hey"]),
    (r".*\?",              ["why do you ask?", "no idea honestly, you?"]),
]

# Pronoun swaps applied to the captured group, so that a rule can quote it back
# without the person and number going wrong.
FLIP = {"i": "you", "me": "you", "my": "your", "am": "are", "i'm": "you're",
        "you": "me", "your": "my", "you're": "i'm", "myself": "yourself"}

# Used when no rule matches, which is most of the time.
FALLBACK = ["go on", "hm, not sure i follow", "say more?", "fair enough", "wait what"]


def flip(text: str) -> str:
    return " ".join(FLIP.get(word.lower(), word) for word in text.split())


class Eliza(torch.nn.Module):

    def __init__(self, seed: int | None = None):
        super().__init__()
        self.rng = random.Random(seed)  # seed it to get a reproducible run
        self.rules = [(re.compile(pattern, re.I), answers) for pattern, answers in RULES]

    def reply(self, text: str) -> str:
        """One reply to one line of chat. No memory, by design."""
        for pattern, answers in self.rules:
            match = pattern.match(text.strip())
            if match:
                answer = self.rng.choice(answers)
                return answer.format(*[flip(group) for group in match.groups()])
        return self.rng.choice(FALLBACK)

    def forward(self, msg: str) -> str:
        # At the end the room asks who the bots were, through the same channel
        # as every other message. Handle it before the chat rules get a chance
        # to answer it with "why do you ask?".
        if is_vote_request(msg):
            names = other_names(msg)
            if not names:
                return "no idea really"

            # Eliza has no way of telling who was who, so it guesses. What
            # matters here is the shape of the answer: every guest gets named,
            # because a guest you leave out is recorded as no vote at all.
            return format_vote({name: self.rng.choice(["human", "ai"]) for name in names})

        # Otherwise the prompt holds the whole transcript and we want the last
        # line somebody else wrote.
        return self.reply(last_message(msg))


if __name__ == "__main__":
    eliza = Eliza()
    print("Chat with Eliza. Ctrl-C or Ctrl-D to quit.")
    try:
        while True:
            print(eliza.reply(input("> ")))
    except (KeyboardInterrupt, EOFError):
        print()
