"""Level 4: be a different person every few minutes.

Every filter before this one is stationary: the same agent, from the first
second of the room to the last. People are not. They are on it for a while, then
something else takes their attention, then they come back and read the backlog.

So: a handful of moods, each with its own answering rate and pace, and a random
walk between them. Over a five minute room that produces bursts of activity
separated by gaps of silence, which is much harder to tell from a person than a
steady drip of well-timed messages.

This is the one worth tuning. Change the numbers in MOODS, add a mood of your
own, or drive the walk from the conversation instead of from a timer. The rest
of the file is bookkeeping.
"""

import time
import random

from utils import is_voting

#                  reply chance,  pace,  average duration (s)
MOODS = {
    "into it":     (0.85,         0.6,   70),
    "half here":   (0.45,         1.6,   90),
    "gone":        (0.00,         1.0,   35),
}


class Mood:

    def __init__(self, delay: float = 4.0, start: str = "into it", actions=("process",)):
        self.delay = delay
        self.start = start
        self.actions = set(actions)

    def _roll_mood(self, opts, now):
        """Pick the next mood and decide how long it lasts."""
        current = opts.get("mood", self.start)

        # Never draw the same mood twice in a row: a change should be a change.
        choices = [mood for mood in MOODS if mood != current]

        # Coming back from "gone" you are usually engaged again, not gone twice.
        weights = [3.0 if mood == "into it" and current == "gone" else 1.0
                   for mood in choices]

        mood = random.choices(choices, weights=weights)[0]
        opts["mood"] = mood
        opts["mood_until"] = now + random.uniform(0.5, 1.5) * MOODS[mood][2]

    def __call__(self, action_id, request, all_actions, opts):
        if all_actions[action_id].name not in self.actions:
            return action_id, request

        # Casting the vote is also a `process` action. Without this line, being
        # "gone" when the room ends would cost you the whole detection score.
        if is_voting(opts):
            return action_id, request

        now = time.monotonic()

        if now >= opts.get("mood_until", 0.0):
            self._roll_mood(opts, now)
        chance, pace, _ = MOODS[opts["mood"]]

        if now < opts.get("quiet_until", 0.0):
            return -1, None

        if "ready_at" not in opts:
            if random.random() > chance:
                opts["quiet_until"] = now + random.uniform(4.0, 15.0)
                return -1, None
            opts["ready_at"] = now + pace * random.uniform(0.5, 1.5) * self.delay

        if now < opts["ready_at"]:
            return -1, None

        del opts["ready_at"]
        return action_id, request
