"""Level 1: wait a few seconds before answering.

A policy filter is called whenever the agent is about to start a new action, and
decides whether that action runs now or not yet. Returning the action unchanged
lets it through; returning `(-1, None)` withdraws it, and the state machine
moves straight on to the other actions available in the same state.

This is that idea in its simplest form. Nobody types a reply in 40 ms, so hold
the reply back for a few seconds first.

The jitter matters more than it looks. A constant delay is as machine-like as no
delay: four replies exactly 6.0 seconds apart is a pattern anybody can spot.
`human_delay.py` takes this further with a distribution that actually resembles
human reaction times.
"""

import time
import random


class FixedDelay:

    def __init__(self, seconds: float = 6.0, jitter: float = 2.0, actions=("process",)):
        self.seconds = seconds
        self.jitter = jitter          # up to this much is added, uniformly
        self.actions = set(actions)   # which action names to slow down

    def __call__(self, action_id, request, all_actions, opts):
        # Only slow down the action that writes a message. Everything else, the
        # handshakes, moving between rooms, collecting incoming messages, must
        # run at full speed or you never reach a table.
        if all_actions[action_id].name not in self.actions:
            return action_id, request

        now = time.monotonic()

        # First time the agent wanted to speak since we last let it: set the
        # alarm. `opts` is our own dictionary and it survives across calls.
        if "ready_at" not in opts:
            opts["ready_at"] = now + self.seconds + random.uniform(0, self.jitter)

        if now < opts["ready_at"]:
            return -1, None

        del opts["ready_at"]  # clear it, so the next message is delayed too
        return action_id, request
