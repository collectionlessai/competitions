"""The smallest processor that does anything: it repeats what it heard.

Useless in the game, but it is the right first thing to run. It has no
dependencies and no failure modes, so if you start it and watch your node reach
a room and get its messages relayed, the plumbing works, and anything that
breaks later is your model rather than your setup.

    proc = Echo()
    policy = None
"""

import torch

from utils import last_message   # utils.py sits next to my_agent.py


class Echo(torch.nn.Module):
    # torch.nn.Module is what makes the instance callable: the SDK calls the
    # processor, and nn.Module.__call__ forwards to forward(). See
    # processors/README.md for the alternatives.

    def forward(self, msg: str) -> str:
        # `msg` is the whole prompt: persona brief plus the entire transcript.
        # last_message() digs out the most recent line somebody else wrote.
        # The fallback covers the very first turn, when nobody has spoken yet
        # and returning "" would send an empty message into the room.
        return last_message(msg)[:200] or "hey"
