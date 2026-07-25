"""Level 2: do not answer everything.

The most robotic thing an agent does is reply to every single message. In a room
of four, if everyone answers everything, one message produces three, those three
produce nine, and the transcript turns into a wall of text no group of people
would ever generate.

Watch out for the obvious mistake. This filter is called about ten times a
second, so `if random.random() < 0.5: stay quiet` does not mean "answer half the
messages". It means "answer within the next fifth of a second", because the coin
is thrown again 0.1 s later and again after that until it comes up heads. A
probability only means something if you commit to the outcome and hold it, which
is what `quiet_until` is for.

Do not overdo it either. Votes about a guest who sent fewer than three messages
are discarded, so an agent that barely speaks scores nothing at all.
"""

import time
import random

from utils import is_voting


class SometimesSilent:

    def __init__(self, reply_chance: float = 0.6, quiet_for: float = 12.0,
                 delay: float = 4.0, actions=("process",)):
        self.reply_chance = reply_chance
        self.quiet_for = quiet_for      # how long a "no" lasts before rethinking
        self.delay = delay
        self.actions = set(actions)

    def __call__(self, action_id, request, all_actions, opts):
        if all_actions[action_id].name not in self.actions:
            return action_id, request

        # Casting the vote is also a `process` action, so without this line the
        # filter could refuse to vote for as long as it refuses to chat.
        if is_voting(opts):
            return action_id, request

        now = time.monotonic()

        # Still tuned out from an earlier "no".
        if now < opts.get("quiet_until", 0.0):
            return -1, None

        # A new chance to speak: decide once, then honour the decision.
        if "ready_at" not in opts:
            if random.random() > self.reply_chance:
                # Skip this one and ignore the room for a while, the way people
                # do when they read a message and do not feel like answering.
                opts["quiet_until"] = now + random.uniform(0.5, 1.5) * self.quiet_for
                return -1, None
            opts["ready_at"] = now + random.uniform(0.5, 1.5) * self.delay

        if now < opts["ready_at"]:
            return -1, None

        del opts["ready_at"]
        return action_id, request
