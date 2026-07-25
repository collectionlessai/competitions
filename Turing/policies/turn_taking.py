"""Level 3: yield the floor after you speak.

Silence on its own is not human either. What people actually do in a group chat
is take turns: you say something, then you let the others answer before you jump
back in. And occasionally you break that rule and send a second message straight
after the first, because you forgot a word or thought of something else.

Two timers do most of the work:

    refractory  just after speaking you are much less likely to speak again
    burst       except for a couple of seconds, when a quick follow-up is normal

The burst is the part worth keeping. A perfectly polite one-message-per-turn
agent is regular in a way people are not.
"""

import time
import random

from utils import is_voting


class TurnTaking:

    def __init__(self, reply_chance: float = 0.7, refractory: float = 15.0,
                 burst_chance: float = 0.2, burst_window: float = 4.0,
                 delay: float = 4.0, actions=("process",)):
        self.reply_chance = reply_chance
        self.refractory = refractory      # seconds spent yielding the floor
        self.burst_chance = burst_chance  # chance of a quick follow-up instead
        self.burst_window = burst_window  # how soon a follow-up still counts as one
        self.delay = delay
        self.actions = set(actions)

    def __call__(self, action_id, request, all_actions, opts):
        if all_actions[action_id].name not in self.actions:
            return action_id, request

        # Casting the vote is also a `process` action, and the refractory period
        # would otherwise apply to it right after your last message.
        if is_voting(opts):
            return action_id, request

        now = time.monotonic()
        since_i_spoke = now - opts.get("spoke_at", -1e9)

        if now < opts.get("quiet_until", 0.0):
            return -1, None

        if "ready_at" not in opts:
            if since_i_spoke < self.burst_window and random.random() < self.burst_chance:
                # "wait, i meant the other one": fast, and no second guessing.
                opts["ready_at"] = now + random.uniform(0.8, 2.0)
            else:
                chance = self.reply_chance
                if since_i_spoke < self.refractory:
                    chance *= 0.15  # I just talked, it is somebody else's turn
                if random.random() > chance:
                    opts["quiet_until"] = now + random.uniform(4.0, 12.0)
                    return -1, None
                opts["ready_at"] = now + random.uniform(0.5, 1.5) * self.delay

        if now < opts["ready_at"]:
            return -1, None

        del opts["ready_at"]
        opts["spoke_at"] = now   # starts the refractory period
        return action_id, request
