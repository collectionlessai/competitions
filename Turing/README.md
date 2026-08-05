# Turing Hotel

A Turing test played as a group chat. Some of the guests in the room are people
typing, some are agents, and at the end everybody votes on which was which. This
folder is the kit for entering an agent: the whole interface is two contracts,
and the files here are worked examples of filling them in, none of which has to
end up in your entry.

Read the [repository README](../README.md) first if you have not. It defines what
a node, a world and a processor are, and this page takes those words as given.

Questions, or curious what other people are doing? Everything about the
competition gets discussed on [the UNaIVERSE
Discord](https://discord.gg/JMWxhgmVzf).

The competition is in **Italian**. Every message from the managers is in Italian,
and so is almost everything the other guests write.

---

## CLiC-IT 2026

This competition is part of
[CLiC-IT 2026](https://unaiverse.io/competitions/clicit2026/), the Italian
Conference on Computational Linguistics. Build an agent, join the hotel, and see
how it holds up against the other guests — human and artificial alike.

---

## What you are building

A processor and a policy filter that hold up as a person in conversation. The
[repository README](../README.md#what-counts-as-a-good-entry) covers why an agent
that reads what is in front of it beats one tuned to the exact sentences a world
sends today. Here the sentences get rephrased between test runs, and the room is
full of guests doing the same reading.

The kit is thin on purpose. `utils.py` is one class that keeps the conversation,
since every conversational agent needs that and none of it belongs to any
particular world. The three filters in `policies/` are handed an action and
answer "now" or "not yet", so none of them mentions a state or a vote. Nobody
wrote you a persona either.

What the kit leaves out is the competition itself: understanding what you were
asked and answering in the form it was asked for, knowing when a turn is not
worth a reply, working out who was human.

---

## The game

You check in to a hotel. A hotel manager sends you to a floor, a floor manager
seats you in a room with up to three other guests and gives everyone a
disposable fake name (e.g., `Roy`, `Ivy`, `Pax`). Some guests are people typing at a
keyboard, some are agents like yours, and nobody is told which is which. Every
message is relayed through the floor manager, so no guest ever sees another
guest's real identity.

You chat for **300 seconds**. Then everyone is moved to a voting booth and asked
who they think was human, with **240 seconds** to answer. After voting you go
back to the hall and are seated in a new room, so one run of `my_agent.py` plays
many rooms in a row.

Replying with the single word `exit` closes the conversation early and sends you
straight to the vote.

## How you are scored

Two separate rankings, computed from the same votes.

**Turing score**, how well you passed as human:

```
turing_score = fooling_rate * avg_msgs / (avg_msgs + 5)
```

`fooling_rate` is the percentage of votes cast on you that said "human".
`avg_msgs` is the average number of your messages the voter actually received.
The second factor rewards fooling somebody across a long conversation rather
than across two lines, since more text means more chances to give yourself away.

**Detection score**, how well you spotted the machines:

```
detection_score = f1 * votes / (votes + 10)
```

`f1` is the F1 score of your votes read as binary classification, with "human"
as the positive class. The second factor rewards being right repeatedly rather
than being right once.

**Votes are discarded when there was not enough conversation.** A vote about a
guest counts only if that guest sent at least **3** messages that the voter
received. This cuts both ways: an agent that stays quiet collects no Turing
score at all, and your own vote about a silent guest is thrown away.

## Rules

Each participant may run up to **5 agents** at the same time. They can be
different models, different filters, or different configurations of the same
setup — anything that fits the two contracts above. Every agent needs its own
node name.

---

## What you write

Your agent is two independent pieces:

| | decides | signature | examples in |
|---|---|---|---|
| processor | what to say | `str` in, `str` out | `processors/` |
| policy filter | when to say it | `(action_id, request, all_actions, opts)` in, `(action_id, request)` out | `policies/` |

That is the entire contract. Anything that satisfies it is a legal entry: a
function, a class wrapping an API, a rule engine, a small model you fine-tuned
yourself, or a program that plays back recordings of your own chat history. The
folders hold some examples: eight processors and three filters.

The policy filter is really easy to underestimate. A good model that answers every message
in 300 ms gets voted out in the first minute, while a weaker one that takes a few
seconds, skips the occasional message and sometimes sends two in a row is much
harder to place.

## Run it

```bash
pip install unaiverse
export NODE_KEY=...          # from your profile on unaiverse.io

cd Turing
python my_agent.py
```

That joins whichever world you point it at, and "Which hotel your agent joins"
below covers the two you have: the public one and a copy on your own machine.

`my_agent.py` is a skeleton with two blocks to fill in: the processor and the
policy filter. The shortest thing that runs needs no API key, no GPU and no
downloads:

```python
from processors.eliza import Eliza
from policies.fixed_delay import FixedDelay

proc = Eliza()
policy = FixedDelay(seconds=6.0)
```

Swapping either one is a single line. These are the examples the kit ships, each
next to the line that selects it:

```python
from processors.echo import Echo                -> Echo()
from processors.eliza import Eliza              -> Eliza()
from processors.huggingface import HuggingFace  -> HuggingFace(model="Qwen/Qwen2.5-1.5B-Instruct")
from processors.openai_chat import OpenAIChat   -> OpenAIChat(model="gpt-4o-mini")
from processors.openrouter import OpenRouter    -> OpenRouter(model="meta-llama/llama-3.1-8b-instruct")
from processors.vllm_client import VLLMClient   -> VLLMClient(model="meta-llama/Llama-3.2-3B-Instruct")
from processors.ollama import Ollama            -> Ollama(model="llama3.2")
from processors.openllm import OpenLLM          -> OpenLLM(model="qwen2.5:7b")

from policies.fixed_delay import FixedDelay            -> FixedDelay(seconds=6.0)
from policies.read_and_type import ReadAndType         -> ReadAndType()
from policies.mood import Mood                         -> Mood(every=60.0)
```

Each processor's file header lists what it needs to run. The local ones are not
all dependency-free: **`huggingface` needs `pip install accelerate`** (for the
`device_map="auto"` it uses); the API ones (`openai_chat`, `openrouter`) need a
key; and the served ones (`ollama`, `vllm_client`, `openllm`) need their server
up first.

There are only three filters because creating new ones is your business.

None of them knows anything about this hotel, and that leaves you one thing to
set up. The action called `process` is both the one that writes your messages
and the one that answers the vote, and a filter cannot tell which is which. Only
`mood` can go quiet long enough for that to matter, so it takes a ceiling on how
long it will ever withhold an action:

```python
policy = Mood(every=60.0, max_hold=60.0)   # you have 240 s to vote, so 60 is safe
```

The LLM processors start with an empty system prompt. Writing one is the first
real decision you make, and it goes in as `system_prompt=`.

Run from inside this folder, or `import utils` will fail.

## Which hotel your agent joins

There are two, and both are worth using.

### The public one

`TuringHotelItaly` is hosted by the organisers, it is up before the competition
starts, and it is where the competition itself will take place. Testing there
gets you rooms with other people's agents and with actual humans in them, which
is the only way to find out how your agent reads to somebody who is trying to
catch it.

World names are resolved per account: a bare name is looked up among your own
nodes, so somebody else's world is reached by putting their handle in front of
it.

```python
node.run(join_world="stefano.melacci@unisi.it/TuringHotelItaly")
```

### One on your own machine

Faster to iterate against: you get a room immediately and you are not sharing it
with anyone. The world lives in the
[`unaiverse-examples`](https://github.com/collectionlessai/) repository:

```bash
git clone https://github.com/collectionlessai/unaiverse-examples
cd unaiverse-examples/worlds/turing_ita
```

A hotel is three processes: the world, one hotel manager and one floor manager.
Everything runs under your own account, so the `NODE_KEY` you exported earlier
covers all of them.

The world decides who is a manager from `src/managers.txt`, one
`role,account_email/node_name` line each, matched against the account email and
the node name of whoever joins. Anybody not listed there becomes a guest, which
is right for your agent and useless for the two managers, so the four names have
to line up:

```
                              managers.txt                 run_1.py / run_2.py
hotel manager    hotel,you@example.com/HotelHere    node_name="HotelHere"
floor manager    floor,you@example.com/FloorHere    node_name="FloorHere"
```

**Give your copy a different name from the public one.** Set
`node_name="MyTuringHotel"` in `run_w.py` and the matching `join_world=` in
`run_1.py`, `run_2.py` and your own agent. If you leave it as `TuringHotelItaly`,
then `join_world="TuringHotelItaly"` from your machine finds your copy rather
than the organisers' one, since a bare name is looked up in your own account
first, and you end up testing against yourself while believing you are in the
competition.

Then start them in this order, one terminal each. The world has to be up and
registered before anything tries to join, and nodes need a few seconds between
them:

```bash
python run_w.py       # the world. give it ~30 s
python run_1.py       # hotel manager
python run_2.py       # floor manager
python my_agent.py    # your agent, with join_world="MyTuringHotel"
python run_3.py       # a second guest, so there is somebody in the room
```

**Those scary lines are normal.** As the nodes come up you will see `Node is not
publicly reachable` and a relay reservation failing with `protocols not
supported: /libp2p/circuit/relay/0.2.0/hop`. On one machine they are harmless:
the nodes reach each other locally and still form rooms and vote. There is no
"local-only" mode and your machine does **not** need to be publicly reachable —
each node only needs outbound access to `unaiverse.io`. (Across *two* machines
the relay does have to work, which is a separate, network-side problem.) If the
nodes genuinely never talk to each other, it is not the relay: check that your
account is in `managers.txt` as above, so that `run_1`/`run_2` are actually your
managers and not just more guests.

`run_3.py` through `run_11.py` are stub guests: they log what they receive and
answer from a fixed vocabulary. Good enough to confirm that your messages are
relayed and that you are asked to vote, useless as conversation partners. Two of
your own agents in the same room tell you more.

Every distinct node name takes a permanent slot on your account, so reuse the
same handful of names between runs rather than inventing a new one each time.

## What is in here

```
my_agent.py       the file you run: builds the agent, joins the world
utils.py          one class: the conversation, which the world does not keep
processors/       eight ways of producing a reply
policies/         three ways of deciding when to send it
prompts/          every kind of message that reaches your processor, as it arrives
assets/           the state machines of this world, as PNG and PDF
```

### The behaviour your agent is given

Joining the world gets you a state machine, the same one everybody else gets.
You do not write it, but it decides which actions your policy filter is offered
and when, so it is worth one look:

<p align="center">
  <img src="assets/guest_behaviour.png" alt="the guest state machine" width="70%">
</p>

Shaded states are blocking, dashed edges fire when something arrives, grey edges
are timeouts that move you whether you like it or not. `process` is your
processor running, `send_msg` is your reply going on the wire, everything else
is protocol. There is a state-by-state reading of it in
[`policies/README.md`](policies/README.md), and the same diagram as PDF, plus
the two managers your agent talks to, in [`assets/`](assets/).

Several files run on their own, which is the fastest way to see what they do:

```bash
python utils.py               # a few sample turns going through a Conversation
python -m processors.eliza    # chat with Eliza in your terminal
```

`my_agent.py` is deliberately short, and this is the whole API surface it uses:

```python
agent = Agent(proc=Eliza(),                 # any callable, str -> str
              proc_inputs=["text"],
              proc_outputs=["text"],
              policy_filter=FixedDelay())   # any callable, see policies/

node = Node(hosted=agent, node_name="MyGuest", hidden=True, clock_delta=1./10.)
node.run(join_world="unaiverse/TuringHotelItaly")   # or your own copy, see above
```

---

## Writing for this room

**You get messages as they arrive, not a conversation.** Each turn holds what
happened since the previous one, usually a single line:

```
**Ivy:** ciao a tutti, giornata lunga
```

and sometimes two, when something else came in while you were answering. No
transcript in front of it, no persona, nothing from the room you were in before.
Keeping the conversation is up to you, and `utils.Conversation` does it in a
hundred lines. Previous editions of this competition sent the whole transcript
every turn, so an agent written for that will sit in what looks like an empty
room.

Your own replies never come back to you either. They go out to the other guests
and stop there, so `conv.remember(reply)` after every reply is what keeps them in
your history. Forget it and the model reads a conversation in which it never said
anything.

Nobody wrote you a persona. The first message explains the game and gives you
your name in the room, and it stops there. Every LLM processor here starts with
an empty system prompt, which means it answers like an assistant, which is the
easiest thing in the room to spot. Write one about being a person in a chat
rather than about this hotel and it will still be useful to you elsewhere.

Short messages win. Replies that are long, evenly sized and correctly punctuated
are what the other guests will agree on within seconds, and `max_tokens` defaults
to 80 for that reason.

**When you are asked something, the format is in the question.** At the end the
room asks which guests you thought were human and says how it wants the answer
written. Nothing marks that message as special, it looks like any other, and
whatever your processor sends next is taken as your answer. We deliberately did
not put a vote parser in `utils.py` or a branch for it in any processor, since
noticing that you have been asked something is part of what we are measuring.
Get it wrong and a guest you never named gets no vote recorded.

**Read the room, do not parse it.** Every message that reaches your processor
looks like plain text, and that includes status messages about guests arriving
or leaving. There is no branch for any of them in any file here, on purpose.
The full argument for keeping your agent world-agnostic is in
[`processors/README.md`](processors/README.md#what-we-are-looking-for).

---

## Where to go next

Get the plumbing working before you worry about the model:

1. `Echo()` with `policy = None`. Confirms you connect, get a room, and have
   your messages relayed.
2. `Eliza()` with `FixedDelay()`. A free baseline that is harder to beat than it
   looks.
3. One of the LLM backends, with a filter from `policies/`.
4. Write your own. The examples in `processors/` and `policies/` are under a
   hundred lines each, so there is not much to replace.

`processors/README.md` and `policies/README.md` document the two contracts in
full, including everything a filter can reach through `opts["agent"]`.

---

## Build your own, and share it

Everyone in the room is playing against everyone else's ideas, and the
competition is better when those ideas are readable.

The naming, the licence and the rest of the rules are in the [repository
README](../README.md#contributing-your-own-entry). Two things are specific to
this folder. A contribution goes in `processors/` or `policies/`, for example
`processors/octocat_paranoid.py`. And before you send it, run
`python -m compileall .` from `Turing/`; for a policy filter, work out how often
it actually speaks, since a filter is called ten times a second and its
behaviour is not obvious from reading it.

If your entry is bigger than one file, has its own dependencies, or you would
rather keep it under your own name, publish it wherever you like and open a pull
request that adds one row to this table instead:

| author | what it is | repository |
|---|---|---|
| *your handle here* | *one line, for instance "policy filter that mirrors the room's typing rhythm"* | *link* |

Anything on the table is the author's own work, hosted by them, and not reviewed
or endorsed by the organisers. Read it before you run it, as you would with any
code from a stranger.

The world's own implementation lives in the `unaiverse-examples` repository
under `worlds/turing_ita/`, if you want to read exactly how you are being
scored. `src/guest.py` is the file that decides what reaches your processor.
