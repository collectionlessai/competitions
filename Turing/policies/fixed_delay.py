"""Hold selected actions for a short, variable delay.

The framework calls a policy filter every time the agent is about to start an
action. An unchanged selection runs immediately, whereas (-1, None) removes it
from the current candidate list. The state machine may then try another action
before asking about the delayed one again on the next tick.

The base delay prevents replies from appearing within a few milliseconds. A
uniform random jitter avoids a fixed interval such as four replies exactly 6.0
seconds apart, which would create another recognizable pattern. A log-normal
draw would approximate human reaction times more closely.
"""

import time
import random


class FixedDelay:

    def __init__(self, seconds: float = 6.0, jitter: float = 2.0, actions=("process",)):
        self.seconds = seconds
        self.jitter = jitter          # Maximum extra delay from a uniform draw.
        self.actions = set(actions)   # Action names to delay.

    def __call__(self, action_id, request, all_actions, opts):
        # Delay only configured actions because protocol actions move the agent
        # towards a room.
        if all_actions[action_id].name not in self.actions:
            return action_id, request

        now = time.monotonic()

        # opts persists across calls, so it can hold the current deadline.
        if "ready_at" not in opts:
            opts["ready_at"] = now + self.seconds + random.uniform(0, self.jitter)

        if now < opts["ready_at"]:
            return -1, None

        del opts["ready_at"]  # Draw a new delay for the next pending action.
        return action_id, request
