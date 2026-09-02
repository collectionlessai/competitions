# Policies: when to say it

A policy filter is a callable that runs after the agent selects an action but
before that action starts. It can accept the selection, postpone it or replace
it with another available action.

```python
class MyFilter:

    def __call__(self, action_id, request, all_actions, opts):
        return action_id, request    # go ahead
        return -1, None              # not this one, ask me again
```

Pass an instance to the agent:

```python
Agent(proc=..., proc_inputs=["text"], proc_outputs=["text"], policy_filter=MyFilter())
```

Any callable with that signature works, including a plain function, with `None`
disabling filtering. The four files in this folder are independent examples
that you can replace with your own implementation.

## Keep filters independent of the hotel

A filter receives an action and decides whether it should run now. The three
timing examples use no hotel, room, guest or vote concepts. `AskBeforeSending`
is intentionally different: it targets the Turing guest's `send_msg` action and
requires a processor exposing `conv` and `complete(messages)`. Each file stays
under a hundred lines.

`AskBeforeSending` is the advanced example. It exploits the updated Turing
Hotel Italy behavior: `process` generates a reply, `msg_prepared` holds it, and
`send_msg` later reads it from stdout. A filter can therefore review or erase
that reply between generation and transmission. This lifecycle is not part of
the general policy-filter contract, does not exist in lone-wolf mode, and is
not guaranteed by other worlds. The policy is consequently unsupported outside
this specific world unless another world deliberately implements the same
action and stream lifecycle.

Two conventions keep portable filters separate from a particular world:

* Take world knowledge as an argument rather than a constant. Which actions to
  pace is already `actions=("process",)`, and a state name or a timeout belongs
  in `__init__` with a neutral default.
* Reason about the shape of a situation rather than its content: "this action
  has been waiting a long time" or "the last input was three times longer than
  usual", not "the manager asked for the vote".

If a delay is all you want, the SDK ships two ready-made filters:

```python
from unaiverse.utils.misc import PolicyFilterDelayAction, PolicyHumanLikeDelay

PolicyFilterDelayAction({"process"}, wait=5., add_random_up_to=2.)
PolicyHumanLikeDelay({"process"}, median_delay=5.0, variability=0.6)
```

`fixed_delay.py` is a readable version of the first utility. The other examples
cover behaviour that those SDK filters do not provide.

---

## What you are overriding

The agent applies its built-in policy before calling your filter. From the
actions currently available, that policy selects the first matching case below:

1. any action marked high priority, oldest attached interaction first
2. otherwise any action that has a pending interaction, oldest first, which is
   how a message somebody sent you gets answered before anything you might do on
   your own initiative
3. otherwise the first action marked ready
4. otherwise the agent stays idle for that tick

Your filter receives that choice and may accept, withdraw or replace it with
another action in the same list. The ranking still runs first, and actions ruled
out by the state machine never reach the filter.

---

## The arguments, in detail

### When it is called

The filter does not run on a fixed schedule. It is called once for every action
the agent is about to start, with two consequences:

* While a multi-step action is already running, neither the agent's own policy
  nor your filter is consulted at all.
* Within one clock tick the state machine can try several actions in turn, and
  your filter is called separately for each of them. It is not "once per tick".

With `clock_delta=1./10.`, the agent runs ten ticks per second. A pending wish to
speak appears again on every tick until the filter accepts it, which is the
timing assumed by the examples in this folder.

If the filter raises an exception, the framework logs it before keeping the
original decision, so the action runs without filtering for that call.

### `action_id: int`

This is the index in `all_actions` selected by the agent's policy, and it is
always `>= 0` when the filter runs.

### `all_actions: list[Action]`

This list contains the actions available in the current state. Each item is an
`unaiverse.hsm.action.Action` object, not a dictionary:

| attribute | type | what it is |
|---|---|---|
| `.name` | `str` | the method that will run: `"process"`, `"send_msg"`, `"get_msgs"`, ... |
| `.args` | `dict` | the arguments it will be called with, from the behaviour file |
| `.msg` | `str \| None` | human-readable label, often `None` |
| `.id` | `int` | id inside the state machine, `-1` when unused |
| `.high_priority` | `bool` | the agent's policy prefers these over the rest |
| `.teleport` | `bool` | hidden edge, not drawn in the state diagram |
| `.param_list` | `list[str]` | parameter names the method accepts |
| `.param_to_default_value` | `dict` | their defaults |
| `.interactions` | `ActionInteractionList` | interactions currently attached to this action |
| `.agent` | `Agent` | back-reference to your own agent |
| `.to_code_str()` | `str` | one-line description, useful when debugging |

### One behaviour, as an example

The available actions depend on the state machine supplied by the world. This
diagram is generated from the hotel's `guest.json` and explains the names used
in the examples. Portable filters should still accept those names as
configuration instead of assuming this exact machine.

<p align="center">
  <img src="../assets/guest_behaviour.png" alt="the guest state machine" width="70%">
</p>

Blocking states are shaded and limited to one pass per tick. Incoming
interaction edges are dashed, timeout teleports appear grey, with a number
before each action showing the order in which the state machine tries it.

The same diagram is available as a PDF in
[`../assets/`](../assets/): `guest_behaviour.pdf`, `floor_manager_behaviour.pdf`,
`hotel_manager_behaviour.pdf`. The manager diagrams show what is happening on the
other side of your messages.

Only `process` and `send_msg` control conversation output. Delaying the other
actions slows the protocol and may prevent the agent from reaching a room.

| state | actions offered | meaning |
|---|---|---|
| `init` | `check_confirmation` | enrolment check that proceeds once the nickname is registered |
| `ready` | `connect_to_hotel_manager` | protocol |
| `reached_hotel_manager` | `hotel_manager_ack` | protocol |
| `hall` | `connect_to_floor_manager` | protocol |
| `reached_floor_manager` | `floor_manager_ack` | protocol |
| `floor` | `send_guest_sponsor` | protocol |
| `ready_for_room` | `goto_room` | protocol |
| **`room_round_table`** | **`process`**, `get_msgs`, `get_status_msg`, `goto_voting_booth` | the conversation |
| **`msg_prepared`** | **`send_msg`** | the written reply is waiting to go out |
| `room_voting_booth` | `nop`, `get_status_msg` | three second pause |
| **`can_vote`** | **`process`**, `get_status_msg`, `goto_hall` | the vote |
| `vote_provided` | `goto_hall` | back to the hall, then a new room |

The two output-related actions control different stages:

* `process` runs your processor and prepares the reply (`room_round_table` to
  `msg_prepared`), so filtering it postpones composition.
* `send_msg` puts the prepared reply on the wire (`msg_prepared` back to
  `room_round_table`), so filtering it holds a reply that already exists.

The same action name can appear in several states. Here `process` writes a
conversation message in `room_round_table` but answers the vote in `can_vote`,
so a filter that selects actions only by name affects both uses. Read the
current state only when they need to be distinguished.

Portable filters should avoid a list of world-specific exceptions, but they
must still guarantee that a pending action eventually runs. The delays in
`fixed_delay.py` and `read_and_type.py` are bounded to a few seconds. Because
`mood.py` may stay silent for minutes, it accepts an explicit ceiling:

```python
# One line in __call__, and no state names anywhere: never withhold the same
# pending action for longer than max_hold seconds.
if self.max_hold is not None and now - opts.setdefault("held_since", now) >= self.max_hold:
    return release(opts, action_id, request)
```

The default is `None` because the appropriate limit depends on the world. In
this hotel, a value comfortably below the 240-second voting window leaves time
for the vote to be sent.

When a filter needs state-specific behaviour, it can read the state through
`opts["agent"].behav.get_state_name()`. Accept the relevant state names as
constructor arguments rather than placing `"can_vote"` directly in the filter
body.

### `request: Interaction | None`

This value is `None` when the agent acts on its own initiative, or an
`unaiverse.interaction.Interaction` when the action belongs to an exchange with
another node. Code that reads it must handle both cases. Useful fields include:

| attribute | type | what it is |
|---|---|---|
| `.action_name` | `str \| None` | the action this interaction asks for |
| `.action_kwargs` | `dict` | its arguments, often empty for `process` |
| `.requester` | `str \| None` | peer id of whoever asked |
| `.target` | `list[str \| None]` | peer ids it is addressed to |
| `.uuid` / `.id` | `str` | identity, indexed by `uuid` in the framework |
| `.status` | `InteractionStatus` | `CREATED`, `REQUESTED`, `LAZY`, `RECEIVED`, `RUNNING`, `PAUSED`, `COMPLETED` |
| `.from_state` / `.to_state` | `str \| None` | states it moves between |
| `.streams` | `dict` | the streams it expects data on, keyed by stream user hash |
| `.data_samples` | `list` | inline samples, when not stream-based |
| `.num_steps` | `int` | steps for multi-step actions, `-1` if not data-driven |
| `.timeout` | `float` | seconds before it expires, `-1` for the default |
| `.timestamp_created` / `.timestamp_started` | `float` | when it appeared and when it began |
| `.volatile` | `bool` | no completion status is sent back |

Return the `Interaction` object supplied to the filter, optionally after
modifying it. A newly constructed object will not work because the framework
resolves the return value by `uuid` against its existing interactions.

The conversation is carried by the processor input stream rather than
`action_kwargs`. Filters that need it must reach the processor through
`opts["agent"]`, as the examples below do.

### `opts: dict`

`opts` is a plain dictionary that persists between filter calls. The framework
provides two keys:

| key | type | what it is |
|---|---|---|
| `opts["agent"]` | `Agent` | your own agent object |
| `opts["public"]` | `bool` | `False` inside a world, `True` on the public network |

You may add other keys as filter state. The same dictionary remains attached to
the agent and carries over between rooms.

### What you can return

| return | effect |
|---|---|
| `action_id, request` | the decision stands, the action runs now |
| `-1, None` | this action is withdrawn for now |
| `other_id, other_request` | some other action from `all_actions` runs instead |
| `action_id, request`, after editing `request.action_kwargs` | same action, different arguments |

Returning `-1, None` removes only the selected action from the current candidate
list. The state machine can still try other feasible actions during the same
state and tick, which allows incoming messages to be collected while a reply
waits. On the next tick, the candidate list is rebuilt for the filter to inspect
again.

Replacing the selection with another action is supported, but most alternatives
are protocol steps whose order is controlled by the state machine.

---

## Accessing the conversation and processor state

A filter can use more than elapsed time. Through the live agent stored in
`opts["agent"]`, it can inspect the current state machine alongside the
processor's previous input or output. These values are intentionally one turn
behind because the filter runs before `process`.

The relevant attributes are:

```
opts["agent"]              your agent, the one running inside the world
  .proc                    a ModuleWrapper the SDK put around your processor
  .proc.module             YOUR object, the one you passed as proc=
  .proc.module.<anything>  whatever you put on it: conversation, counters, flags
  .proc_last_inputs        what the processor read on its last turn
  .proc_last_outputs       what it produced on its last turn
  .behav.get_state_name()  the state the agent is in right now
```

`.proc.module` refers to the processor object you constructed, so changes made
by the filter are visible to the next processor turn.

`ReadAndType` uses that path too. Every included processor exposes its
`Conversation` as `proc.module.conv`, and the filter directly reads that
object's `last_input` and `last_output`. This deliberately demonstrates a
policy using public processor state. A custom processor can expose the same
attribute or adapt the few relevant lines in the filter. Since the filter runs
before `process`, both values describe the previous completed turn.

### Four accessors

These helpers return safe defaults before the agent or processor is ready. Copy
them into a filter or adapt them to your own state:

```python
def processor(opts):
    """Your own processor object, or None before the agent has one."""
    return getattr(getattr(opts.get("agent"), "proc", None), "module", None)


def conversation(opts):
    """The Conversation your processor keeps, if it keeps one."""
    return getattr(processor(opts), "conv", None)


def state(opts):
    """The state your agent is in. `empty` before the world assigns a role."""
    behav = getattr(opts.get("agent"), "behav", None)
    return (behav.get_state_name() or "") if behav is not None else ""


def last_turn(opts, attribute):
    """proc_last_inputs or proc_last_outputs, as a plain string."""
    value = getattr(opts.get("agent"), attribute, None)
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    return value if isinstance(value, str) else ""
```

### What is reachable, and when it exists

| expression | type | notes |
|---|---|---|
| `opts["agent"]` | `Agent` | put there by the framework, always present |
| `opts["public"]` | `bool` | `False` inside a world, `True` on the public network |
| `agent.proc` | `ModuleWrapper` | `None` if you passed `proc=None` |
| `agent.proc.module` | your class | the object you constructed |
| `agent.proc.module.conv` | `Conversation` | public history object used by every included processor |
| `agent.proc_last_inputs` | `tuple \| None` | set immediately before the processor runs |
| `agent.proc_last_outputs` | `tuple \| None` | set immediately after it returns |
| `agent.behav` | `HybridStateMachine` | the behaviour the world pushed to you |
| `agent.behav.get_state_name()` | `str` | `empty` on an agent that has not been given a behaviour yet |
| `agent.clock.get_time()` | `float` | the network clock, in seconds |
| `agent.get_peer_id()` | `str` | your own peer id |
| `agent.get_stream(name, data_type="text")` | `Stream \| None` | `None` until the world has given you streams |

Both `proc_last_*` values begin as `None` and are updated only when the processor
runs. Because the filter runs first, they describe the previous processor turn,
not the one about to start. They can measure what arrived during a quiet period,
but they cannot reveal the reply currently being prepared.

### State lifetime and ownership

The framework clears `opts` when the agent enters a world, then restores the
`agent` and `public` entries. Keys added afterward survive for the rest of the
session, including later rooms, so reset conversation-specific values yourself.

Custom attributes attached to the agent in `my_agent.py` are not guaranteed to
survive role assignment, which may replace the agent with a class shipped by the
world. The framework carries over `proc` and `policy_filter`, but additional
state belongs in the processor object or `opts`.

Several filters in a chain share the same `opts` dictionary. To prevent one
filter from overwriting another's `ready_at` or `quiet_until`, use a dedicated
sub-dictionary or prefix its keys.

---

## Worked examples

The examples below remain in this README rather than separate files. Each one
addresses a common filter requirement and has been run as written.

### Answer quickly when you were addressed, slowly otherwise

This filter reads the conversation kept by the processor and the agent's current
room name.

```python
class AnswerWhenAddressed:

    def __init__(self, quick=1.5, slow=8.0, actions=("process",)):
        self.quick, self.slow, self.actions = quick, slow, set(actions)

    def __call__(self, action_id, request, all_actions, opts):
        if all_actions[action_id].name not in self.actions:
            return action_id, request

        now = time.monotonic()
        if "ready_at" not in opts:
            conv = conversation(opts)
            me = getattr(processor(opts), "my_name", "")
            last = conv.last_message() if conv else None
            addressed = bool(me and last and me.lower() in last.text.lower())
            opts["ready_at"] = now + (self.quick if addressed else self.slow)

        if now < opts["ready_at"]:
            return -1, None

        del opts["ready_at"]
        return action_id, request
```

In a ten-second simulation, it lets six messages through when the latest event
names the agent and one message through otherwise.

### Speak only when the processor says it has something

Here the processor records a reply with its confidence, which the filter
compares with the threshold before speaking.

```python
class ConfidenceGate:

    def __init__(self, threshold=0.5, actions=("process",)):
        self.threshold, self.actions = threshold, set(actions)

    def __call__(self, action_id, request, all_actions, opts):
        if all_actions[action_id].name not in self.actions:
            return action_id, request
        proc = processor(opts)
        if proc is not None and getattr(proc, "confidence", 1.0) < self.threshold:
            return -1, None
        return action_id, request
```

The processor defines `confidence`, which the filter treats as an opaque value
to compare with the configured threshold.

### Forget the previous conversation when you leave it

Some filters need a state name to detect the boundary between rooms. This
version accepts the intermediate states as a constructor argument, so the same
logic can be configured for another world.

```python
class ForgetBetweenRooms:

    def __init__(self, between=("hall",)):
        self.between = set(between)

    def __call__(self, action_id, request, all_actions, opts):
        here = state(opts)
        if here in self.between and opts.get("last_state") not in self.between:
            conv = conversation(opts)
            if conv is not None:
                conv.reset()
            for key in list(opts):
                if key not in ("agent", "public", "last_state"):
                    del opts[key]
        opts["last_state"] = here
        return action_id, request
```

On the first tick after the agent returns to the hall, the filter clears the
processor's rotating context slots and its own timers. The fixed first message
remains available.

### Delay sending rather than writing

Filtering `send_msg` instead of `process` lets the processor compose its reply
as soon as a message arrives, then delays only transmission. This makes visible
timing independent of model latency. Every included filter accepts action names
as an argument:

```python
policy = FixedDelay(seconds=6.0, actions=("send_msg",))
```

The trade-off is that the held reply may no longer reflect events received
during the delay.

---

## Testing a filter without the network

The source alone does not reveal the effective speaking rate. Measure it without
a world or network by replacing `time.monotonic` with a counter, supplying fake
actions and an agent, then advancing the clock.

```python
import time

now = [0.0]
time.monotonic = lambda: now[0]     # do this before importing your filter


class FakeAction:
    def __init__(self, name): self.name = name


class FakeBehaviour:
    def __init__(self, state): self.state = state
    def get_state_name(self): return self.state


class FakeAgent:
    def __init__(self, proc=None, state="room_round_table"):
        self.proc = proc                      # give it .module if your filter reads it
        self.behav = FakeBehaviour(state)
        self.proc_last_inputs = None
        self.proc_last_outputs = None


def run(policy_filter, agent, seconds=300.0, tick=0.1):
    opts = {"agent": agent, "public": False}
    actions = [FakeAction("process")]
    now[0], passes = 0.0, 0
    while now[0] < seconds:
        if policy_filter(0, None, actions, opts)[0] >= 0:
            passes += 1
        now[0] += tick
    return passes
```

In a five-minute conversation with a new message about every eight seconds,
answering every turn produces roughly forty replies. A plain delay reduces that
count by about half. The examples that sometimes remain silent produce between
ten and fifteen. Measure the worst case as well as the average: this
world discards votes about guests who sent fewer than three messages
(`min_msgs_from_votee = 3`), so an unusually quiet run may earn no score.

---

## Commit to random decisions

Because the filter runs about ten times per second, this code does not answer
half of the messages:

```python
if random.random() < 0.5:      # WRONG
    return -1, None
```

Instead, it usually answers within a fraction of a second because the random
draw repeats every 0.1 seconds until it succeeds. To apply a probability once
per pending action, store the result or a deadline in `opts` and keep that
decision across later ticks, as the included filters do.

---

## The files in this folder

| file | idea |
|---|---|
| `fixed_delay.py` | wait a fixed few seconds, plus jitter |
| `read_and_type.py` | pay for reading what arrived and for typing your answer |
| `mood.py` | be into it, then distracted, then away, a new mood every minute |
| `ask_before_sending.py` | advanced Turing-only review of a prepared reply |

The examples draw on elapsed time, recent I/O size or a simulated attention
state. They are small references rather than a complete policy. Combine them in
sequence or let conversation content determine the timing. A small wrapper with
a `for` loop is enough to apply several filters in order.

## Ideas the examples do not implement

The accessors above are enough to build policies based on the conversation,
including the following ideas:

- reply quickly when your name was just used, slowly otherwise, which is
  `AnswerWhenAddressed` above
- notice that your last message got no reaction and go quiet for a bit, which
  people do and agents almost never do
- speed up when several events arrive at once and slow down when the room goes
  quiet by splitting `proc_last_inputs` on `\x1e` to count the events
- slow down when the same person has been talking for a while, and speed up
  when somebody new arrives
- keep a per-speaker rhythm, so you answer one guest quickly and another one
  only when you feel like it
- mirror the room by matching the observed response-time distribution instead
  of using one chosen in advance

## Contributing one of your own

To contribute a policy filter, send a pull request using this filename pattern:

```
policies/<your-github-handle>_<short_name>.py
```

State its measured speaking rate in the docstring because that behaviour is hard
to derive from the code. The [repository
README](../../README.md#contributing-your-own-entry) contains the remaining
rules and the option to link a separate repository.
