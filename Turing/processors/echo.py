"""Repeat the latest incoming message.

Echo has no external dependencies, so use it to check that a node can enter a
room and relay replies to the other guests before adding a model or API.

    proc = Echo()
    policy = None

Its control flow is shared by the other processors: update the Conversation,
produce a reply, then record that reply locally. That local marker is useful to
label the transcript; it does not imply an API `assistant` role.
"""

import torch

from utils import Conversation   # utils.py is next to my_agent.py.


class Echo(torch.nn.Module):
    # nn.Module makes the processor callable, with other accepted forms listed
    # in processors/README.md.

    def __init__(self):
        super().__init__()
        self.conv = Conversation()

    def forward(self, sample: str) -> str:
        # Conversation splits batches on \x1e without dropping event newlines.
        self.conv.add(sample)

        # Use a greeting only when no remote message has been stored yet.
        heard = self.conv.last_message()
        reply = heard.text[:200] if heard else "ciao"

        # The world does not echo local replies, so remember this one explicitly.
        self.conv.remember(reply)
        return reply
