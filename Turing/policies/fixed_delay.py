"""Wait a few seconds before answering.

The framework calls a policy filter every time the agent is about to start an
action. Return the action unchanged and it runs; return (-1, None) and it is
dropped for now, the state machine tries whatever else is available in the same
state, and you get asked again on the next tick.

Nobody types a reply in 40 ms, so this holds it back for a few seconds. The
jitter is there because a constant delay is about as recognisable as none at
all: four replies exactly 6.0 seconds apart stand out in a chat log. The draw
here is uniform. A log-normal one would sit closer to how people actually react.
"""

import time
import random


class FixedDelay:

    def __init__(self, seconds: float = 6.0, jitter: float = 2.0, actions=("process",)):
        self.seconds = seconds
        self.jitter = jitter          # up to this much is added, uniformly
        self.actions = set(actions)   # which action names to slow down

    def __call__(self, action_id, request, all_actions, opts):
        # Only the named actions get paced. Slow down a protocol action and you
        # never reach a table
        if all_actions[action_id].name not in self.actions:
            return action_id, request

        now = time.monotonic()

        # opts is ours to write into, and it is the same dictionary on every call
        if "ready_at" not in opts:
            opts["ready_at"] = now + self.seconds + random.uniform(0, self.jitter)

        if now < opts["ready_at"]:
            return -1, None

        del opts["ready_at"]  # so the next message gets its own delay
        return action_id, request
