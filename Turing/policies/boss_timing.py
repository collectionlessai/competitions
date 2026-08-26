"""When the boss speaks. The single biggest thing that gives an agent away.

Everybody in this competition is testing content. A linguist reading a
transcript is testing something cheaper: an agent answers in the same fraction
of a second whatever it was asked, a person does not. A long message takes
longer to answer than a short one, a distracted person takes twenty seconds and
then three seconds twice in a row, and a real chat log is bursts and gaps rather
than a metronome.

None of that is written here. It is already in the kit:

    ReadAndType   delay proportional to the characters read and the characters
                  typed, so latency tracks message length the way a typist does
    Mood          a mood redrawn every minute, each with its own chance of
                  answering at all and its own pace, which comes out as bursts
    Chain         both of them, each with its own timers

What this file adds is the one thing a chain cannot know. The action called
`process` is both "write a reply" and "cast your vote", and a filter that gates
it by name gates the vote too: hold the vote past 240 seconds and the room's
whole detection score is gone. The state machine does know the difference —
`can_vote` is a state you are only in inside the booth — and `opts["agent"]`
reaches it. So the vote goes around the chain entirely, with a short delay of its
own, because nobody fills in a form in 40 milliseconds either.

The same state read is pushed into the processor's `RoomSense`, which is how the
boss knows a manager message is the vote request without matching its wording.
That is the only coupling between the two halves, and it is one method call.
"""

import time
import random

from policies.chain import Chain
from policies.mood import Mood
from policies.read_and_type import ReadAndType

# The two states a `process` turn can mean something in. The other twelve in the
# guest behaviour are protocol, and no sample is ever delivered in them.
VOTING = "can_vote"
CHATTING = "room_round_table"


def agent_state(opts: dict) -> str:
    """The state the agent is in, or "" before the behaviour is loaded.

    The framework puts the agent in `opts["agent"]` itself, after the filter is
    set, so this works from the first tick without anything being wired up.
    """
    behav = getattr(opts.get("agent"), "behav", None)
    if behav is None:
        return ""
    return behav.get_state_name() or ""


class BossTiming:
    """`Mood` over `ReadAndType`, with the vote taken out of both.

    Args:
        sense: the processor's `RoomSense`, or None. Every call pushes the
            current state name into it, which is what lets the boss tell the
            vote request from an ordinary manager message.
        read_cps, type_cps, think: passed to `ReadAndType`. The defaults are a
            fast but unremarkable typist.
        mood_every: seconds between one mood and the next.
        mood_delay: base pause of the mood layer, scaled by the mood's own pace.
        max_hold: ceiling on one unbroken silence, in seconds, counted here
            rather than inside either layer. Each of them has its own ceiling
            and neither knows about the other, so a mood that releases into a
            long typing delay is two ceilings and one much longer silence. This
            one is the total, and it is measured from the last thing that went
            out.
        vote_delay: the pause before the vote goes out, drawn uniformly.
    """

    # The timers the two kit filters park in their slice of `opts`. Cleared when
    # the ceiling fires, so the next turn is drawn fresh instead of resuming a
    # silence that has already been overruled
    TIMERS = ("ready_at", "quiet_until", "held_since")

    def __init__(self, sense=None, read_cps: float = 22.0, type_cps: float = 5.5,
                 think: float = 2.5, mood_every: float = 55.0, mood_delay: float = 3.5,
                 max_hold: float = 40.0, vote_delay: tuple = (4.0, 16.0),
                 actions=("process",)):
        self.sense = sense
        self.actions = set(actions)
        self.vote_delay = vote_delay
        self.max_hold = max_hold
        self.chain = Chain(
            Mood(every=mood_every, delay=mood_delay, max_hold=max_hold, actions=actions),
            ReadAndType(read_cps=read_cps, type_cps=type_cps, think=think, actions=actions),
        )

    def __call__(self, action_id, request, all_actions, opts):
        state = agent_state(opts)
        if self.sense is not None:
            self.sense.note_state(state)

        if all_actions[action_id].name not in self.actions:
            return action_id, request

        if state == VOTING:
            return self._vote(opts, action_id, request)

        # Leaving the booth: drop the vote's timer so the next one is its own
        opts.pop("vote_ready_at", None)

        now = time.monotonic()
        held_since = opts.setdefault("boss_held_since", now)

        passed, request = self.chain(action_id, request, all_actions, opts)
        if passed >= 0:
            opts.pop("boss_held_since", None)
            return passed, request

        if now - held_since >= self.max_hold:
            self._clear(opts)
            return action_id, request
        return -1, None

    def _clear(self, opts) -> None:
        opts.pop("boss_held_since", None)
        for index in range(len(self.chain.filters)):
            slot = opts.get(f"chain_{index}")
            if isinstance(slot, dict):
                for key in self.TIMERS:
                    slot.pop(key, None)

    def _vote(self, opts, action_id, request):
        """Straight through, after a pause. Never held, never skipped."""
        now = time.monotonic()
        if "vote_ready_at" not in opts:
            opts["vote_ready_at"] = now + random.uniform(*self.vote_delay)
        if now < opts["vote_ready_at"]:
            return -1, None
        return action_id, request
