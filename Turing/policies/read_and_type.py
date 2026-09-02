"""Delay a reply by reading time + thinking time + typing time.

    reading  = characters that arrived since your last turn / reading speed
    thinking = a random pause
    typing   = length of what you wrote / typing speed

The filter reads the previous turn from the ``conv`` attribute exposed by every
included processor.
"""

import random
import time


class ReadAndType:

    def __init__(self, read_cps: float = 25.0, type_cps: float = 6.0, think: float = 2.0,
                 actions=("process",)):
        self.read_cps = read_cps
        self.type_cps = type_cps
        self.think = think
        self.actions = set(actions)

    def __call__(self, action_id, request, all_actions, opts):
        if all_actions[action_id].name not in self.actions:
            return action_id, request

        now = time.monotonic()
        if "ready_at" not in opts:
            processor = opts["agent"].proc.module
            conversation = processor.conv

            reading = len(conversation.last_input) / self.read_cps
            thinking = random.uniform(0, self.think * 2)
            typing = len(conversation.last_output) / self.type_cps
            delay = reading + thinking + typing

            opts["ready_at"] = now + min(delay, 45.0)

        if now < opts["ready_at"]:
            return -1, None  # Ask the state machine to try again later.

        del opts["ready_at"]
        return action_id, request  # Allow the action now.
