"""When the boss speaks. The single biggest thing that gives an agent away.

Everybody in this competition is testing content. A linguist reading a
transcript is testing something cheaper: an agent answers in the same fraction
of a second whatever it was asked, a person does not. A long message takes
longer to answer than a short one, a distracted person takes twenty seconds and
then three seconds twice in a row, and a real chat log is bursts and gaps rather
than a metronome.

The guest's behaviour splits one reply across two actions:

    room_round_table --process--> msg_prepared --send_msg--> room_round_table

and the whole wait has to happen on the **first** of them. `get_msgs` is
registered from `room_round_table` and nowhere else, so an agent parked in
`msg_prepared` cannot receive: every message another guest sends while it waits
is refused outright, with `Requested action (get_msgs) not found` and `Cannot
enqueue a received interaction!` in its log. Holding `send_msg` to charge for
typing — which is the obvious place, because that is where the finished reply
finally exists — makes the agent deaf for exactly as long as it is pretending
to type. Tried in a live local room, it never heard a single guest message.

So the entire budget is spent before the model runs, while the agent is still
listening, and `send_msg` goes straight through:

    wait = reading what arrived
         + a pause
         + how long this reply will take to type   (estimated)
         - how long the model will take to write it (estimated)

The last two terms are estimates because neither is known yet, and both come
off the processor's own recent history through
`opts["agent"].proc.module` — the reference the framework hands the filter.
The subtraction is the point: adding a delay on top of generation makes the
total depend on how fast the model happens to be that minute, which is not a
property any typist has. Taking it out means the room sees the same thing
whether the model answered in 300ms or in four seconds.

`can_vote` goes around all of it. `process` there means "cast your vote", and
holding it past 240 seconds throws away the room's whole detection score.
"""

import time
import random

# The two states a `process` turn can mean something in. The other twelve in the
# guest behaviour are protocol, and no sample is ever delivered in them.
VOTING = "can_vote"
CHATTING = "room_round_table"

# Characters per second at a phone keyboard, used only when the processor does
# not say. The Aalto study of 37,000 people puts the mean at 36.2 WPM, which at
# five characters a word is 3.0 cps — the 4.5 that used to be here was 54 WPM,
# a number I had invented and well above the population mean. Each persona
# carries its own rate; the filter reads it off the processor.
TYPING_CPS = 3.0

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


def processor(opts: dict):
    """The processor object handed to the `Agent` at construction — our `Boss`.

    `opts["agent"].proc` is the SDK's `ModuleWrapper`; `.module` is what we
    passed in. This is how a filter reaches anything the processor knows.
    """
    return getattr(getattr(opts.get("agent"), "proc", None), "module", None)


def expected(opts: dict, name: str, fallback: float) -> float:
    """A running average off the processor, or a fallback before there is one."""
    value = getattr(processor(opts), name, None)
    return value if isinstance(value, (int, float)) and value > 0 else fallback


class BossTiming:
    """The whole wait before the model runs; nothing after it, nothing on the vote.

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
            return self._send(opts, action_id, request)
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

            # Typing, charged before the fact. The reply does not exist yet, so
            # the length comes from our own recent ones; the model's time comes
            # off the same way, and comes out of the budget rather than adding
            # to it. Floors keep a silent turn from waiting for nothing.
            cps = expected(opts, "typing_cps", self.typing_cps)
            # A follow-up the processor has already written — the rest of a
            # message sent too early, or a "*parola" correction — costs what
            # THAT costs to type, not what an average new message costs
            chars = expected(opts, "next_reply_chars", 0.0) \
                or expected(opts, "mean_reply_chars", 40.0)
            typing = chars / cps
            generating = expected(opts, "mean_call_seconds", 2.0)
            delay += max(0.0, typing - generating)

            # Somebody who put the phone down. Rare, and the only thing in here
            # that produces a gap longer than the message justifies
            if random.random() < self.distraction:
                delay += random.uniform(8.0, 25.0)

            # How long a wait is affordable depends on how fast the room moves.
            # The delay above is a reading time, and reading time alone is only
            # half of what a person is doing: the other half is noticing that
            # the conversation is running away and getting the message in before
            # it stops making sense. In a room turning over every three seconds,
            # a careful twenty-second read produces a reply to something seven
            # messages back — which is not what careful looks like, it is what
            # a queue looks like. So the ceiling comes down with the pace, and
            # a floor keeps a quiet room from being answered instantly.
            # Wobble last, or the ceiling eats it. Clipping a jittered delay
            # against a cap pins every busy-room turn to exactly the cap, and a
            # reply that lands 6.0 seconds after the last message every single
            # time is the metronome `Speaker.mechanical()` convicts other guests
            # for. The cap sets where the wait sits; the wobble is what stops it
            # from being a number.
            pace = expected(opts, "room_pace", 12.0)
            delay = min(delay, self.max_hold, max(2.5, 3.0 * pace))
            opts["read_until"] = now + self._wobble(delay)

        if now >= opts["read_until"] or now - held_since >= self.max_hold:
            self._clear(opts)
            opts["typed_from"] = time.monotonic()   # the typing clock starts here
            return action_id, request
        return -1, None

    # -- typing, once there is something to type ---------------------------

    def _send(self, opts, action_id, request):
        """Straight through. Never hold this one.

        Waiting here parks the agent in `msg_prepared`, where `get_msgs` is not
        registered, and every message another guest sends meanwhile is refused.
        The typing charge is already spent back in `_read`, while the agent was
        still able to listen.
        """
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
