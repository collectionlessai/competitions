"""Repeats back whatever it heard last.

Not much of a guest, but it is what you want to run first: no dependencies and
nothing to go wrong, so if your node reaches a room and the others see your
messages, the plumbing is fine and whatever breaks after this is yours.

    proc = Echo()
    policy = None

It also has the shape the other seven have, which is three lines: feed the
sample to a Conversation, produce a reply, record it.
"""

import torch

from utils import Conversation   # utils.py sits next to my_agent.py


class Echo(torch.nn.Module):
    # Subclassing nn.Module is what makes the instance callable, which is all
    # the SDK requires. processors/README.md lists the other ways.

    def __init__(self):
        super().__init__()
        self.conv = Conversation()

    def forward(self, sample: str) -> str:
        # everything that arrived since your last turn, one message per line
        self.conv.add(sample)

        # The last Message somebody else sent, or None while nobody else has
        # spoken, which is how a room starts.
        heard = self.conv.last_message()
        reply = heard.text[:200] if heard else "ciao"

        # Your own lines go out to the other guests and never arrive back in a
        # later sample. This call is the only thing that puts them in history.
        self.conv.remember(reply)
        return reply
