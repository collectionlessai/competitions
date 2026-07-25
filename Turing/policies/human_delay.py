"""Level 1: a delay with a realistic shape.

Human reaction times are not uniform. Most replies come quickly, a few take much
longer, and none are instant, which is the shape of a log-normal distribution.
`random.lognormvariate` gives you one directly.

    median   the typical pause, in seconds
    spread   how erratic you are: 0.3 is a steady person, 1.0 is somebody who
             keeps picking the phone up and putting it back down
    minimum  a floor, because reading and typing take time even when you are
             paying full attention

The SDK ships an equivalent filter, `unaiverse.utils.misc.PolicyHumanLikeDelay`,
which adds momentum between consecutive delays and occasional distraction
spikes. This file is the version you can read in thirty seconds and edit.
"""

import math
import time
import random


class HumanDelay:

    def __init__(self, median: float = 5.0, spread: float = 0.6, minimum: float = 0.8,
                 actions=("process",)):
        self.mu = math.log(median)    # lognormvariate takes the log of the median
        self.spread = spread
        self.minimum = minimum
        self.actions = set(actions)

    def __call__(self, action_id, request, all_actions, opts):
        if all_actions[action_id].name not in self.actions:
            return action_id, request

        now = time.monotonic()

        # Draw one delay and commit to it. Drawing a fresh number on every call
        # would just pick the smallest sample, which is not the same thing at
        # all: see the warning in policies/README.md.
        if "ready_at" not in opts:
            delay = max(random.lognormvariate(self.mu, self.spread), self.minimum)
            opts["ready_at"] = now + delay

        if now < opts["ready_at"]:
            return -1, None

        del opts["ready_at"]
        return action_id, request
