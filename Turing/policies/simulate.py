"""Count how much each filter actually lets through, over a simulated room.

    python -m policies.simulate
    python -m policies.simulate --runs 500 --gap 6

Timing filters are hard to judge by reading them, because the numbers only mean
something once you know how often they are called. This runs each filter over a
whole 300 second room at 10 ticks per second and counts the messages that come
out, so you can compare settings before spending a room on them.

What is simulated:

* the clock, which is replaced by a counter so that 300 seconds take a few
  milliseconds rather than five minutes;
* the other guests, who speak on an exponential schedule with mean `--gap`;
* the agent, which wants to reply whenever at least one message has arrived
  since it last spoke, and holds that wish until the filter lets it through.

What is not simulated: the model, the network, and the content of anything. This
tells you about pacing, and nothing else. Aim for the middle of the table:
votes about a guest who sent fewer than three messages are discarded, so an
agent below that scores nothing, while one that answers everything in two
seconds is the easiest guest in the room to identify.
"""

import time
import random
import argparse

ROOM_SECONDS = 300.0
TICK = 0.1        # clock_delta=1./10. in my_agent.py

# The filters call time.monotonic(). Replacing it with a counter lets a room run
# in a few milliseconds. This is the only unusual thing in this file, and it is
# why the simulator lives here rather than inside each filter.
_now = 0.0
time.monotonic = lambda: _now


class FakeAction:
    """Stands in for unaiverse.hsm.action.Action, which only .name is read from."""

    def __init__(self, name):
        self.name = name


class FakeBehaviour:
    def __init__(self, state):
        self.state = state

    def get_state_name(self):
        return self.state


class FakeAgent:
    """Exposes the three attributes the filters read through opts["agent"]."""

    def __init__(self):
        self.behav = FakeBehaviour("room_round_table")
        self.proc_last_inputs = None
        self.proc_last_outputs = None

    def set_last_turn(self, transcript_body, reply):
        # Shaped like the real prompt, so utils.transcript() finds its markers.
        self.proc_last_inputs = ("persona brief ...\n### TRANSCRIPT START\n"
                                 + transcript_body + "### TRANSCRIPT END\n---\n",)
        self.proc_last_outputs = (reply,)


def run_room(make_filter, gap: float, rng: random.Random) -> int:
    """Play one 300 s room with a fresh filter, and return the messages sent."""
    global _now

    policy_filter = make_filter()
    agent = FakeAgent()
    opts = {"agent": agent, "public": False}   # what the framework puts there
    actions = [FakeAction("process")]

    _now = 0.0
    sent = 0
    pending = 0           # incoming messages we have not answered yet
    transcript = ""

    while _now < ROOM_SECONDS:
        # Somebody else speaks, on average once every `gap` seconds.
        if rng.random() < TICK / gap:
            pending += 1
            transcript += f"(00:00:00) Ben: {'x' * rng.randint(10, 60)}\n"

        # The agent only asks to act when it has something to answer. While it
        # waits, the framework keeps offering the same action every tick.
        if pending > 0:
            action_id, _ = policy_filter(0, None, actions, opts)
            if action_id >= 0:
                reply = "y" * rng.randint(10, 80)
                transcript += f"(00:00:00) Ada (You): {reply}\n"
                agent.set_last_turn(transcript, reply)
                sent += 1
                pending = 0

        _now += TICK

    return sent


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=int, default=200, help="rooms to average over")
    parser.add_argument("--gap", type=float, default=8.0,
                        help="mean seconds between messages from the other guests")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from policies.chain import Chain
    from policies.fixed_delay import FixedDelay
    from policies.human_delay import HumanDelay
    from policies.read_and_type import ReadAndType
    from policies.sometimes_silent import SometimesSilent
    from policies.turn_taking import TurnTaking
    from policies.mood import Mood

    # Add your own filter here to see it next to the others. Each entry is a
    # label and a zero-argument callable that returns a fresh filter, because
    # every simulated room starts from a clean one.
    candidates = [
        ("none", lambda: (lambda action_id, request, all_actions, opts: (action_id, request))),
        ("fixed_delay", FixedDelay),
        ("human_delay", HumanDelay),
        ("read_and_type", ReadAndType),
        ("sometimes_silent", SometimesSilent),
        ("turn_taking", TurnTaking),
        ("mood", Mood),
        ("Chain(Mood, ReadAndType)", lambda: Chain(Mood(), ReadAndType())),
    ]

    print(f"{int(ROOM_SECONDS)} s room, one message from the others every "
          f"{args.gap:.0f} s on average, {args.runs} runs\n")
    print(f"{'filter':28} {'messages sent':>14}  {'min':>4} {'max':>4}")
    print("-" * 55)

    for name, make_filter in candidates:
        rng = random.Random(args.seed)
        counts = [run_room(make_filter, args.gap, rng) for _ in range(args.runs)]
        mean = sum(counts) / len(counts)
        print(f"{name:28} {mean:14.1f}  {min(counts):4d} {max(counts):4d}")


if __name__ == "__main__":
    main()
