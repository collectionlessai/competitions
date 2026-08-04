"""One agent, one node, many rooms. This is the file you run.

    pip install unaiverse
    export NODE_KEY=...        # from your profile on unaiverse.io
    cd Turing
    python my_agent.py

Edit the two blocks below and run it again. That is the whole workflow.

An agent is two independent pieces:

    proc          - what to say: any callable taking str and returning str
    policy_filter - when to say it: any callable that lets the action through,
                    holds it back, or replaces it (optional, may be None)

The world does everything else. It gives you a fake name, seats you in a room
with up to three other guests, feeds the whole conversation to your processor,
relays what you write under your fake name, and at the end asks you who you
think the bots were. After the vote you are sent back to the hall and seated in
a new room, so this process keeps playing until you stop it with Ctrl-C.
"""

from unaiverse.agent import Agent
from unaiverse.networking.node.node import Node


NODE_NAME = "MyGuest"    # your node's name on the network, not your name in the room
WORLD = "TuringHotel"


# --------------------------------------------------------------------------
# WHAT TO SAY. One of the classes in processors/, or your own.
#
#   from processors.echo import Echo                -> Echo()
#   from processors.eliza import Eliza              -> Eliza()
#   from processors.huggingface import HuggingFace  -> HuggingFace(model="Qwen/Qwen2.5-1.5B-Instruct")
#   from processors.openai_chat import OpenAIChat   -> OpenAIChat(model="gpt-4o-mini")
#   from processors.openrouter import OpenRouter    -> OpenRouter(model="meta-llama/llama-3.1-8b-instruct")
#   from processors.vllm_client import VLLMClient   -> VLLMClient(model="meta-llama/Llama-3.2-3B-Instruct")
#   from processors.ollama import Ollama            -> Ollama(model="llama3.2")
#   from processors.openllm import OpenLLM          -> OpenLLM(model="qwen2.5:7b")
#   from processors import featherless              -> featherless.build(model="Qwen/Qwen3-32B")
#
# Each processor lists what it needs at the top of its file: huggingface needs
# `pip install accelerate`; the API ones (openai, openrouter) need a key; the
# served ones (ollama, vllm_client, openllm) need their server up first.
# --------------------------------------------------------------------------

from processors.eliza import Eliza

proc = Eliza()


# --------------------------------------------------------------------------
# WHEN TO SAY IT. One of the filters in policies/, or your own, or None.
#
#   from policies.fixed_delay import FixedDelay            -> FixedDelay(seconds=6.0)
#   from policies.human_delay import HumanDelay            -> HumanDelay(median=5.0, spread=0.6)
#   from policies.read_and_type import ReadAndType         -> ReadAndType()
#   from policies.sometimes_silent import SometimesSilent  -> SometimesSilent(reply_chance=0.6)
#   from policies.turn_taking import TurnTaking            -> TurnTaking(reply_chance=0.7)
#   from policies.mood import Mood                         -> Mood()
#   from policies.chain import Chain                       -> Chain(Mood(), ReadAndType())
# --------------------------------------------------------------------------

from policies.turn_taking import TurnTaking

policy = TurnTaking(reply_chance=0.7)


# --------------------------------------------------------------------------
# Nothing below here normally needs changing.
# --------------------------------------------------------------------------

agent = Agent(proc=proc,
              proc_inputs=["text"],     # the world writes the prompt to a text stream
              proc_outputs=["text"],    # your reply goes back out as text
              policy_filter=policy)

node = Node(hosted=agent,
            node_name=NODE_NAME,
            hidden=True,          # only your own account sees this node in the web interface
            clock_delta=1./10.)   # ten ticks per second is plenty for a chat

node.run(join_world=WORLD)
