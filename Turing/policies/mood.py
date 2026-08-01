"""Change mood and answer at a different rate in each one.

A filter with fixed settings behaves the same way from the first minute to the
last. People do not. They follow a conversation for a while, get pulled away by
something else, come back and read the backlog. This draws a new mood every
`every` seconds, and each mood has its own chance of answering at all and its
own pace when it does, so over a few minutes it comes out as bursts of messages
separated by gaps.

The numbers in MOODS are guesses.
"""

import time
import random

#                  reply chance,  pace
MOODS = {
    "into it":     (0.85,         0.6),
    "half here":   (0.45,         1.6),
    "gone":        (0.00,         1.0),
}


class Mood:

    def __init__(self, every: float = 60.0, delay: float = 4.0, start: str = "into it",
                 actions=("process",), max_hold: float | None = None):
        self.every = every        # seconds between one mood and the next
        self.delay = delay        # base pause before answering, scaled by the mood's pace
        self.start = start
        self.actions = set(actions)
        # "gone" answers nothing, so without a ceiling the silence runs until the
        # mood changes. max_hold is that ceiling, in seconds: anything well
        # under 240 and your vote still goes out
        self.max_hold = max_hold

    def _roll_mood(self, opts, now):
        current = opts.get("mood", self.start)

        # Never the same mood twice in a row, and coming out of "gone" is
        # weighted towards paying attention again
        choices = [mood for mood in MOODS if mood != current]
        weights = [3.0 if mood == "into it" and current == "gone" else 1.0
                   for mood in choices]

        opts["mood"] = random.choices(choices, weights=weights)[0]
        opts["mood_until"] = now + self.every

    def __call__(self, action_id, request, all_actions, opts):
        if all_actions[action_id].name not in self.actions:
            return action_id, request

        now = time.monotonic()

        # held_since is set on the first call after something last went out and
        # cleared by _release, so it measures one unbroken stretch of silence.
        if self.max_hold is not None and now - opts.setdefault("held_since", now) >= self.max_hold:
            return self._release(opts, action_id, request)

        if now >= opts.get("mood_until", 0.0):
            self._roll_mood(opts, now)
        chance, pace = MOODS[opts["mood"]]

        if now < opts.get("quiet_until", 0.0):
            return -1, None

        # The coin is thrown once and then committed to. Losing it buys 4 to 15
        # seconds of silence, rather than another throw a tenth of a second later
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
