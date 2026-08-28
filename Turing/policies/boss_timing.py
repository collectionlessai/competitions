"""When the boss speaks. The single biggest thing that gives an agent away.

Everybody in this competition is testing content. A linguist reading a
transcript is testing something cheaper: an agent answers in the same fraction
of a second whatever it was asked, a person does not. A long message takes
longer to answer than a short one, a distracted person takes twenty seconds and
then three seconds twice in a row, and a real chat log is bursts and gaps rather
than a metronome.

The guest's behaviour splits one reply across two actions, and that is what
makes the timing honest:

    room_round_table --process--> msg_prepared --send_msg--> room_round_table

`process` runs the model. `send_msg` puts the answer on the wire. The filter is
called before each of them, so at `process` time the reply does not exist yet —
which is why the kit's `ReadAndType` charges typing on the *previous* turn's
answer. At `send_msg` time it does exist, and so does the model's own latency.
So the two halves are charged where they belong:

    process    reading what arrived, plus a pause, plus the occasional
               spell of having put the phone down
    send_msg   typing what was actually written, MINUS the time the model
               already spent generating it

That subtraction is the point. Adding a delay on top of generation time makes
the total depend on how fast the model happens to be that minute, which is not
a property any typist has. Taking generation out of the budget means the room
sees the same thing whether the model answered in 300ms or in four seconds: a
person typing at their own speed. When generation already overran the budget,
the message goes straight out — it is late enough.

The agent is reachable from `opts["agent"]`, and the processor through
`opts["agent"].proc.module`, which is where the generation time comes from.

`can_vote` goes around all of it. `process` there means "cast your vote", and
holding it past 240 seconds throws away the room's whole detection score.
"""

import time
import random

# The two states a `process` turn can mean something in. The other twelve in the
# guest behaviour are protocol, and no sample is ever delivered in them.
VOTING = "can_vote"
CHATTING = "room_round_table"

# Characters per second at a phone keyboard. Mobile text entry studies put the
# average adult near 36 wpm and practised thumbs past 60; at ~5 characters a
# word that is 3 to 5 cps. The default sits at the quick end, because a guest
# who takes 30 seconds over one line stops being able to hold a conversation.
TYPING_CPS = 4.5

# Reading. Silent reading runs 200-300 wpm, so ~20 cps, and nobody reads a chat
# message as carefully as a page.
READING_CPS = 22.0


def agent_state(opts: dict) -> str:
    """The state the agent is in, or "" before the behaviour is loaded."""
    behav = getattr(opts.get("agent"), "behav", None)
    if behav is None:
        return ""
    return behav.get_state_name() or ""


def last_output(opts: dict) -> str:
    """What the processor produced on its last turn, as a string."""
    value = getattr(opts.get("agent"), "proc_last_outputs", None)
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    return value if isinstance(value, str) else ""


def last_input(opts: dict) -> str:
    """What the processor read on its last turn. The world only sends what is new."""
    value = getattr(opts.get("agent"), "proc_last_inputs", None)
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    return value if isinstance(value, str) else ""


def generation_seconds(opts: dict) -> float:
    """How long the model took on the turn just finished.

    Through `opts["agent"].proc.module`, which is the processor object handed to
    the `Agent` at construction — our `Boss`, which times its own calls.
    """
    proc = getattr(opts.get("agent"), "proc", None)
    module = getattr(proc, "module", None)
    value = getattr(module, "last_call_seconds", 0.0)
    return value if isinstance(value, (int, float)) else 0.0


class BossTiming:
    """Reading on `process`, typing on `send_msg`, nothing in the way of the vote.

    Args:
        sense: the processor's `RoomSense`, or None. Every call pushes the
            current state name into it, which is what lets the boss tell the
            vote request from an ordinary manager message.
        typing_cps: characters a second at the keyboard. The one number to tune
            against real rooms.
        reading_cps: characters a second reading what arrived.
        think: mean of the extra pause before starting to answer, in seconds.
        distraction: chance that a turn also waits out a spell of somebody
            having put the phone down.
        max_hold: ceiling on any single hold, so nothing can eat the room.
        vote_delay: the pause before the vote goes out, drawn uniformly.
    """

    def __init__(self, sense=None, typing_cps: float = TYPING_CPS,
                 reading_cps: float = READING_CPS, think: float = 2.0,
                 distraction: float = 0.12, max_hold: float = 40.0,
                 vote_delay: tuple = (4.0, 16.0), jitter: float = 0.25):
        self.sense = sense
        self.typing_cps = typing_cps
        self.reading_cps = reading_cps
        self.think = think
        self.distraction = distraction
        self.max_hold = max_hold
        self.vote_delay = vote_delay
        self.jitter = jitter          # ± this fraction on every drawn delay

    def _wobble(self, seconds: float) -> float:
        return seconds * random.uniform(1.0 - self.jitter, 1.0 + self.jitter)

    def __call__(self, action_id, request, all_actions, opts):
        state = agent_state(opts)
        if self.sense is not None:
            self.sense.note_state(state)

        name = all_actions[action_id].name

        # The vote is a `process` turn too, and it is the one that must never be held
        if state == VOTING:
            if name == "process":
                return self._vote(opts, action_id, request)
            return action_id, request

        opts.pop("vote_ready_at", None)

        if name == "process":
            return self._read(opts, action_id, request)
        if name == "send_msg":
            return self._type(opts, action_id, request)
        return action_id, request

    # -- reading, before the model runs ------------------------------------

    def _read(self, opts, action_id, request):
        """Hold the turn for as long as reading what arrived would take.

        Deliberately not a second silence gate. Whether to speak at all is the
        director's decision, inside the processor, and it is already made with
        momentum so the log comes out in bursts and lulls. Stacking a filter
        that also declines turns on top of it triple-counts silence: measured
        with the kit's `Mood` in here, the median wait before the model even ran
        was 20.6s, and half of those waits ended in the director saying nothing
        anyway. What is left here is time, not choice.
        """
        now = time.monotonic()
        held_since = opts.setdefault("boss_held_since", now)

        if "read_until" not in opts:
            delay = len(last_input(opts)) / self.reading_cps
            delay += random.expovariate(1.0 / self.think) if self.think > 0 else 0.0

            # Somebody who put the phone down. Rare, and the only thing in here
            # that produces a gap longer than the message justifies
            if random.random() < self.distraction:
                delay += random.uniform(8.0, 25.0)

            opts["read_until"] = now + min(self._wobble(delay), self.max_hold)

        if now >= opts["read_until"] or now - held_since >= self.max_hold:
            self._clear(opts)
            opts["typed_from"] = time.monotonic()   # the typing clock starts here
            return action_id, request
        return -1, None

    # -- typing, once there is something to type ---------------------------

    def _type(self, opts, action_id, request):
        """Hold the finished reply until a person could have typed it.

        The budget is the length of what was actually written; what the model
        spent generating it comes out of that budget rather than being added to
        it, so the room never sees the model's speed.
        """
        now = time.monotonic()
        if "type_until" not in opts:
            reply = last_output(opts)
            budget = self._wobble(len(reply) / self.typing_cps)
            spent = now - opts.get("typed_from", now)          # generation + queueing
            spent = max(spent, generation_seconds(opts))
            opts["type_until"] = now + max(0.0, min(budget - spent, 45.0))

        if now < opts["type_until"]:
            return -1, None

        opts.pop("type_until", None)
        opts.pop("typed_from", None)
        return action_id, request

    # -- the vote ----------------------------------------------------------

    def _vote(self, opts, action_id, request):
        """Straight through, after a pause. Never held, never skipped."""
        now = time.monotonic()
        if "vote_ready_at" not in opts:
            opts["vote_ready_at"] = now + random.uniform(*self.vote_delay)
        if now < opts["vote_ready_at"]:
            return -1, None
        return action_id, request

    def _clear(self, opts) -> None:
        opts.pop("boss_held_since", None)
        opts.pop("read_until", None)
