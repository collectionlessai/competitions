"""Level 2: pay for reading the room and for typing the answer.

A real person answers a long message more slowly than a short one, and takes
longer still when their own answer is long. So instead of one delay, charge for
three things:

    reading  = characters that appeared since your last turn / reading speed
    thinking = a random pause
    typing   = length of what you wrote / typing speed

This is the first filter that looks at the conversation rather than only at the
clock. It reads the agent through `opts["agent"]`, which the framework fills in
for you, using the `last_prompt` and `last_reply` helpers from `utils.py`.

One catch worth understanding: your reply does not exist yet when this filter
runs, because the filter runs *before* the processor does. So we cannot charge
for typing the message we are about to send. We charge for the previous one
instead, which comes out the same over a conversation and costs nothing in
complexity.
"""

import time
import random

from utils import transcript, last_prompt, last_reply


class ReadAndType:

    def __init__(self, read_cps: float = 25.0, type_cps: float = 6.0, think: float = 2.0,
                 actions=("process",)):
        self.read_cps = read_cps      # characters per second, reading
        self.type_cps = type_cps      # characters per second, typing
        self.think = think            # mean of the extra random pause
        self.actions = set(actions)

    def __call__(self, action_id, request, all_actions, opts):
        if all_actions[action_id].name not in self.actions:
            return action_id, request

        now = time.monotonic()

        if "ready_at" not in opts:
            # Only the conversation counts towards reading time. The persona
            # brief is the same hundreds of lines every turn, and you only ever
            # read it once. Charge only for what is new since our last turn.
            seen = opts.get("seen_chars", 0)
            total = len(transcript(last_prompt(opts)))
            fresh = max(total - seen, 0)
            opts["seen_chars"] = total

            delay = (fresh / self.read_cps
                     + random.expovariate(1.0 / self.think)
                     + len(last_reply(opts)) / self.type_cps)

            # A long backlog after a lull could otherwise cost you a minute.
            opts["ready_at"] = now + min(delay, 45.0)

        if now < opts["ready_at"]:
            return -1, None

        del opts["ready_at"]
        return action_id, request
