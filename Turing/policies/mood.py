"""Vary the reply rate through a small set of attention states.

A fixed filter keeps the same timing for an entire room. This one draws a new
mood every `every` seconds, with a different reply probability and pace for each.
Across several minutes, those settings produce active stretches separated by
quieter ones.

The numbers in MOODS are guesses.
"""

import time
import random

#                  reply chance, pace
MOODS = {
    "into it":     (0.85,         0.6),
    "half here":   (0.45,         1.6),
    "gone":        (0.00,         1.0),
}


class Mood:

    def __init__(self, every: float = 60.0, delay: float = 4.0, start: str = "into it",
                 actions=("process",), max_hold: float | None = None):
        if start not in MOODS:
            raise ValueError(f"Unknown starting mood: {start}")
        self.every = every        # Seconds between mood changes.
        self.delay = delay        # Base pause scaled by the current mood.
        self.start = start
        self.actions = set(actions)
        # "gone" stays silent until the mood changes unless max_hold releases the
        # pending action first. A limit below 240 leaves time for this world's vote.
        self.max_hold = max_hold

    def _roll_mood(self, opts, now):
        current = opts.get("mood", self.start)

        # Do not repeat a mood. After "gone", favor a return to "into it".
        choices = [mood for mood in MOODS if mood != current]
        weights = [3.0 if mood == "into it" and current == "gone" else 1.0
                   for mood in choices]

        opts["mood"] = random.choices(choices, weights=weights)[0]
        opts["mood_until"] = now + self.every

    def __call__(self, action_id, request, all_actions, opts):
        if all_actions[action_id].name not in self.actions:
            return action_id, request

        now = time.monotonic()

        # held_since starts after the previous release and measures one continuous
        # stretch of silence.
        if self.max_hold is not None and now - opts.setdefault("held_since", now) >= self.max_hold:
            return self._release(opts, action_id, request)

        if "mood" not in opts:
            opts["mood"] = self.start
            opts["mood_until"] = now + self.every
        elif now >= opts.get("mood_until", 0.0):
            self._roll_mood(opts, now)
        chance, pace = MOODS[opts["mood"]]

        if now < opts.get("quiet_until", 0.0):
            return -1, None

        # Draw once per pending action. A failed draw holds the decision for 4 to
        # 15 seconds instead of trying again on the next tick.
        if "ready_at" not in opts:
            if random.random() > chance:
                opts["quiet_until"] = now + random.uniform(4.0, 15.0)
                return -1, None
            opts["ready_at"] = now + pace * random.uniform(0.5, 1.5) * self.delay

        if now < opts["ready_at"]:
            return -1, None

        return self._release(opts, action_id, request)

    @staticmethod
    def _release(opts, action_id, request):
        for key in ("ready_at", "quiet_until", "held_since"):
            opts.pop(key, None)
        return action_id, request
