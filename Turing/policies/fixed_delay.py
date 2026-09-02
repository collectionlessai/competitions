"""Wait a few seconds before allowing a reply."""

import random
import time


class FixedDelay:

    def __init__(self, seconds: float = 6.0, jitter: float = 2.0, actions=("process",)):
        self.seconds = seconds
        self.jitter = jitter
        self.actions = set(actions)

    def __call__(self, action_id, request, all_actions, opts):
        if all_actions[action_id].name not in self.actions:
            return action_id, request

        now = time.monotonic()
        if "ready_at" not in opts:
            opts["ready_at"] = now + self.seconds + random.uniform(0, self.jitter)

        if now < opts["ready_at"]:
            return -1, None  # Ask the state machine to try again later.

        del opts["ready_at"]
        return action_id, request  # Allow the action now.
