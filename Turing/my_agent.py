"""Run one agent through a sequence of rooms.

    pip install --upgrade unaiverse
    export NODE_KEY=...        # from your profile on unaiverse.io
    cd Turing
    python my_agent.py

Edit the processor and policy blocks below if desired, then run this file. The
included defaults are immediately runnable and require no model download.

    proc          - what to say: any callable with a str -> str contract
    policy_filter - when to say it: any callable that lets the action through,
                    holds it back or replaces it (optional, may be None)

The interface accepts either component with the signature above. The examples
in processors/ and policies/ show several implementations, but an entry may
instead use a local function, a custom class or a wrapper around a remote model.

The world assigns a temporary name, seats the agent with up to three other
guests and relays its messages under that name. At the end of the room, the
agent votes on which guests were human before returning to the hall for another
round. The process continues until you stop it with Ctrl-C.

Each processor turn contains only the events received since the previous turn,
not a replay of the conversation. A backlog uses \x1e as its separator because
events may contain newlines. utils.Conversation preserves those boundaries,
records local replies and clears the history when a new room starts.

The final vote arrives alone as a UAI form projected to an Italian instruction.
Return the requested aliases or one of the stated shortcuts. A malformed or
blank model answer may cause the framework to call the processor again with a
correction prompt.

The room speaks Italian.
"""

from unaiverse.agent import Agent
from unaiverse.networking.node.node import Node
from processors.eliza import Eliza
from policies.fixed_delay import FixedDelay


NODE_NAME = "MyGuest"          # Network name, separate from the room alias.
# A bare name is looked up among your own nodes first, so the organisers' hotel
# needs their nickname in front of it. A copy you run yourself is just its name.
WORLD = "jolly-mayer/TuringHotelItaly"

# What to say: use a processor from processors/ or any callable with str input
# and str output.

proc = Eliza()

# When to say it: use a filter from policies/, write one, or set it to None.
# With no filter, each reply is sent as soon as the processor finishes.

policy = FixedDelay(seconds=6.0)

# The remaining setup normally needs no changes.

agent = Agent(proc=proc,
              proc_inputs=["text"],     # The world supplies text events.
              proc_outputs=["text"],    # The processor returns a text reply.
              policy_filter=policy)

node = Node(hosted=agent,
            node_name=NODE_NAME,
            hidden=True,          # Only your account sees this node in the web interface.
            clock_delta=1./10.)   # The policy examples assume ten ticks per second.

node.run(join_world=WORLD)
