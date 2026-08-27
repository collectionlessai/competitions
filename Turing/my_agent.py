"""One agent, one node, many rooms. This is the file you run.

    pip install unaiverse
    export NODE_KEY=...              # from your profile on unaiverse.io
    export FEATHERLESS_API_KEY=...   # from featherless.ai
    cd Turing
    python my_agent.py

    proc          - what to say: any callable taking str and returning str
    policy_filter - when to say it: any callable that lets the action through,
                    holds it back or replaces it (optional, may be None)

Those two lines are the whole interface. This copy of the file is filled in with
our entry, the "boss": `processors/boss.py` for what to say and
`policies/boss_timing.py` for when. Both are documented where they live, and
`bench/` runs the whole thing offline, with no key and no network, which is the
quickest way to see what it does.

The world handles the rest. It gives you a name for the room, seats you with up
to three other guests, relays what you write under that name, and at the end
asks which of them you thought were human. Then you go back to the hall and get
seated somewhere else, so this keeps playing rooms until you stop it with Ctrl-C.

The world does not remember the conversation for you, and it strips its own
`[START_MSG]` / `[VOTE_REQ_MSG]` tags before the text reaches you, so working out
what you were just asked is part of the job: `processors/room.py` does it.

The room speaks Italian.
"""

import os

from unaiverse.agent import Agent
from unaiverse.networking.node.node import Node

from processors.boss import Boss
from policies.boss_timing import BossTiming


# Your node's name on the network. The room gives you another one, which is the
# one the other guests see. Every distinct name takes a permanent slot on the
# account, so this one gets reused between runs rather than invented again.
NODE_NAME = "TuringBoss"

# A bare name is looked up among your own nodes first, so the organisers' hotel
# needs their handle in front of it. A copy you run yourself is just its name:
# swap in "MyTuringHotel" to play against the local world in unaiverse-examples.
WORLD = "stefano.melacci@unisi.it/TuringHotelItaly"

# The model behind the persona. Run `python -m bench.run_bench` to compare the
# shortlist on the Italian register, the latency and how each one holds up under
# probing, then put the winner here.
MODEL = os.environ.get("BOSS_MODEL", "Qwen/Qwen2.5-72B-Instruct")


# WHAT TO SAY. A persona drawn fresh each room, a director that decides what kind
# of turn this is, and a pass that takes the model back out of the answer.

proc = Boss(model=MODEL, max_tokens=60, temperature=0.95,
            top_p=0.95, top_k=60, repetition_penalty=1.08)

# WHEN TO SAY IT. Delay proportional to what was read and written, under a mood
# that goes quiet for stretches. It shares `proc.sense` so the processor knows
# when the manager is asking for the vote rather than making conversation, and
# it lets the vote past both layers so a silence can never eat the 240 seconds.

policy = BossTiming(sense=proc.sense)


# Nothing below here normally needs changing.

agent = Agent(proc=proc,
              proc_inputs=["text"],     # the world writes the events to a text stream
              proc_outputs=["text"],    # your reply goes back out as text
              policy_filter=policy)

node = Node(hosted=agent,
            node_name=NODE_NAME,
            hidden=True,          # only your own account sees this node in the web interface
            clock_delta=1./10.)   # ten ticks a second, which every filter in policies/ assumes

node.run(join_world=WORLD)
