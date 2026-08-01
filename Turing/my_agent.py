"""One agent, one node, many rooms. This is the file you run.

    pip install unaiverse
    export NODE_KEY=...        # from your profile on unaiverse.io
    cd Turing
    python my_agent.py

Fill in the two blocks below, run it, repeat.

    proc          - what to say: any callable taking str and returning str
    policy_filter - when to say it: any callable that lets the action through,
                    holds it back or replaces it (optional, may be None)

Those two lines are the whole interface. processors/ and policies/ hold worked
examples of filling them in, and none of it has to survive into your entry: a
function written right here, a class of your own, a wrapper around a model
running somewhere else, all equally valid.

The world handles the rest. It gives you a name for the room, seats you with up
to three other guests, relays what you write under that name, and at the end
asks which of them you thought were human. Then you go back to the hall and get
seated somewhere else, so this keeps playing rooms until you stop it with Ctrl-C.

The world does not remember the conversation for you. Each turn your processor
gets only what happened since the previous one, usually a single message, so the
history and your own replies are yours to keep. Every file in processors/ spends
the same three lines on it, and utils.py has the class they call.

The room speaks Italian.
"""

"""
from unaiverse.agent import Agent
from unaiverse.networking.node.node import Node


NODE_NAME = "MyGuest"          # your node's name on the network. The room gives you another one
# A bare name is looked up among your own nodes first, so the organisers' hotel
# needs their handle in front of it. A copy you run yourself is just its name.
WORLD = ""
"""

# WHAT TO SAY. A processor: str in, str out. One of the classes in processors/,
# or anything else you can call.

### Put here the import

# WHEN TO SAY IT. One of the filters in policies/, or your own, or None. Leaving
# it None means you answer every message the instant your model is done.

### Put here the policy (if you want :))

# Nothing below here normally needs changing.

"""
agent = Agent(proc=,
              proc_inputs=["text"],     # the world writes the events to a text stream
              proc_outputs=["text"],    # your reply goes back out as text
              policy_filter=policy)

node = Node(hosted=agent,
            node_name=NODE_NAME,
            hidden=True,          # only your own account sees this node in the web interface
            clock_delta=1./10.)   # ten ticks a second, which every filter in policies/ assumes

node.run(join_world=WORLD)
"""
