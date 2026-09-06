"""Delay proportional to how much you had to read and how much you wrote.

A long message takes longer to answer than a short one, and a long answer takes
longer to type, so instead of one delay this charges for three things:

    reading  = characters that arrived since your last turn / reading speed
    thinking = a random pause
    typing   = length of what you wrote / typing speed

The numbers come off the agent itself, through opts["agent"], which the
framework fills in. proc_last_inputs is what the processor read on its last
turn, and since the world only sends what is new, its length is the reading time
and there is nothing to keep track of between calls.

Your reply does not exist yet when the filter runs, since the filter runs before
the processor. So the typing cost lands on the previous reply rather than on the
one about to go out, which over a conversation comes to the same thing.
"""

import time
import random


def last_turn(opts, attribute: str) -> str:
    """Read proc_last_inputs or proc_last_outputs off the agent, as a string.

    Both are tuples, and both are None until the processor has run once, so the
    first call gets "" instead of an exception.
    """
    value = getattr(opts.get("agent"), attribute, None)
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    return value if isinstance(value, str) else ""


class ReadAndType:

    def __init__(self, read_cps: float = 25.0, type_cps: float = 6.0, think: float = 2.0,
                 actions=("process",)):
        self.read_cps = read_cps      # characters per second, reading
        self.type_cps = type_cps      # characters per second, typing
        self.think = think            # mean of the extra random pause, 0 to skip it
        self.actions = set(actions)

    def __call__(self, action_id, request, all_actions, opts):
        if all_actions[action_id].name not in self.actions:
            return action_id, request

        now = time.monotonic()

        if "ready_at" not in opts:
            fresh = len(last_turn(opts, "proc_last_inputs"))
            thinking = random.expovariate(1.0 / self.think) if self.think > 0 else 0.0
            delay = (fresh / self.read_cps
                     + thinking
                     + len(last_turn(opts, "proc_last_outputs")) / self.type_cps)

            # The cap stops a long backlog pushing the reply arbitrarily far out
            opts["ready_at"] = now + min(delay, 45.0)

        if now < opts["ready_at"]:
            return -1, None

        del opts["ready_at"]
        return action_id, request
