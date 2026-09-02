"""Change reply probability and speed every few seconds."""

import random
import time

# mood: (probability of replying, delay multiplier)
MOODS = {
    "into it": (0.85, 0.6),
    "half here": (0.45, 1.6),
    "gone": (0.00, 1.0),
}


class Mood:

    def __init__(self, every: float = 60.0, delay: float = 4.0, start: str = "into it",
                 actions=("process",), max_hold: float | None = None):
        if start not in MOODS:
            raise ValueError(f"Unknown starting mood: {start}")

        self.every = every
        self.delay = delay
        self.start = start
        self.actions = set(actions)
        self.max_hold = max_hold

    def __call__(self, action_id, request, all_actions, opts):
        if all_actions[action_id].name not in self.actions:
            return action_id, request

        now = time.monotonic()

        if "mood" not in opts:
            opts["mood"] = self.start
            opts["mood_until"] = now + self.every
        elif now >= opts["mood_until"]:
            opts["mood"] = random.choice(tuple(MOODS))
            opts["mood_until"] = now + self.every

        if "ready_at" not in opts:
            chance, pace = MOODS[opts["mood"]]
            if random.random() < chance:
                wait = pace * self.delay
            else:
                wait = opts["mood_until"] - now  # Stay quiet until the next mood.
            if self.max_hold is not None:
                wait = min(wait, self.max_hold)
            opts["ready_at"] = now + wait

        if now < opts["ready_at"]:
            return -1, None  # Stay quiet for now.

        del opts["ready_at"]
        return action_id, request  # Allow the action now.
