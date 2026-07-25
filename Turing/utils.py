"""Reading the prompt the room sends you, and writing the vote it expects back.

Every turn the world hands your processor one long string, always shaped the
same way:

    You are playing a game called "Turing Hotel".
    Your name in this game: Ada
    Other guests: Ben,Cal
    ... about 430 more lines of persona instructions ...
    ### TRANSCRIPT START
    (17:30:00) MANAGER: Dear Ada, open the conversation naturally.
    --------------
    (17:30:14) Ben: hey, long day?
    --------------
    (17:30:20) Ada (You): yeah, just got home
    --------------
    ### TRANSCRIPT END
    ---
    Now it's your turn to respond as Ada.

So: the persona brief, then the whole conversation with your own lines marked
`(You)`, then a closing nudge. The string is rebuilt from scratch every turn.

An LLM processor can forward it untouched and ignore this file. Anything else,
Eliza or a content-aware policy filter, has to pick it apart, and these are the
functions for it.

    reading                 what you get
    ----------------------  -----------------------------------------------
    transcript(p)           the conversation only, as one string
    messages(p)             list[Message], oldest first
    last_message(p)         text of the last line somebody else wrote
    my_name(p)              the fake name the room gave you
    other_names(p)          the other guests, in order of first speaking
    my_messages(p)          everything you have said so far
    addressed_to_me(p)      did somebody just use your name
    seconds_since_last(p)   gap between the last two messages
    is_vote_request(p)      is the room asking you to vote right now

    writing                 what it does
    ----------------------  -----------------------------------------------
    format_vote({...})      builds a vote string the world parses reliably

A policy filter does not receive the prompt, but it does receive `opts`, which
holds a reference to your agent. These four read the agent through it, and are
what `policies/` uses:

    from a filter's opts    what you get
    ----------------------  -----------------------------------------------
    agent_state(opts)       which state you are in, "room_round_table", ...
    is_voting(opts)         True while the vote is what `process` would do
    last_prompt(opts)       the prompt the processor read last turn
    last_reply(opts)        the reply it produced last turn

Real prompts to try them on are in `prompts/`, and this file runs against them:

    python utils.py

Two details that surprise people, both visible in `prompts/`:

* The floor manager tags its announcements `[START_MSG]`, `[GEN_MSG]`,
  `[VOTE_REQ_MSG]` and so on, but the tag is stripped before the message
  reaches you, and most of those announcements are printed to your console
  without entering the transcript at all. The only MANAGER lines you actually
  see in the prompt are the opening nudge and the vote request.
* The vote request is a normal MANAGER line. Nothing marks it as special, and
  whatever your processor returns next is taken as your vote.
"""

import os
from collections import namedtuple

START_MARKER = "### TRANSCRIPT START"
END_MARKER = "### TRANSCRIPT END"

# The fake name the floor manager speaks under. It is not a guest and it never
# votes, so most helpers here skip it.
MANAGER = "MANAGER"

# Phrases from the vote request, as they read once the [VOTE_REQ_MSG] tag has
# been stripped. The second one is the variant sent when nobody else was in the
# room with you. See prompts/example_vote_prompt.txt for the whole message.
VOTE_REQUEST_MARKERS = ("LIST THE ONES YOU THINK WERE HUMANS",
                        "you have not interacted with anybody")

# speaker: fake name, without the "(You)" suffix
# text:    the message itself
# mine:    True if you wrote it
# time:    wall-clock stamp as it appears in the transcript, "17:30:14"
Message = namedtuple("Message", "speaker text mine time")


def transcript(prompt: str) -> str:
    """The conversation only, without the persona brief and the closing nudge."""
    return prompt.split(START_MARKER)[-1].split(END_MARKER)[0]


def messages(prompt: str) -> list[Message]:
    """Every message so far, oldest first.

    Lines look like `(17:30:14) Ben: hey`, separated by rules of dashes. Your
    own lines carry a `(You)` suffix on the speaker, which is how you tell them
    apart from everybody else's.
    """
    out = []
    for line in transcript(prompt).splitlines():
        line = line.strip()
        if not line or line.startswith("-"):
            continue  # blank line, or one of the "--------------" separators

        stamp = ""
        if line.startswith("(") and ")" in line:
            stamp, line = line[1:].split(")", 1)  # peel off the "(17:30:14)"
        if ":" not in line:
            continue  # not a message line

        speaker, text = line.split(":", 1)
        speaker = speaker.strip()
        out.append(Message(speaker=speaker.replace("(You)", "").strip(),
                           text=text.strip(),
                           mine="(You)" in speaker,
                           time=stamp.strip()))
    return out


def last_message(prompt: str) -> str:
    """Text of the last thing somebody else said, which is usually all you need."""
    for msg in reversed(messages(prompt)):
        if not msg.mine and msg.speaker != MANAGER:
            return msg.text
    return ""


def my_name(prompt: str) -> str:
    """The fake name the room gave you.

    It is stated near the top of the persona brief. The fallback covers the case
    where you are parsing a bare transcript, without the brief in front of it.
    """
    for line in prompt.splitlines():
        if line.strip().startswith("Your name in this game:"):
            return line.split(":", 1)[1].strip()
    for msg in messages(prompt):
        if msg.mine:
            return msg.speaker
    return ""


def other_names(prompt: str) -> list[str]:
    """The other guests, in the order they first spoke.

    Read from the transcript rather than from the brief, so a guest who joined
    late and has said nothing yet does not appear. Those are exactly the guests
    you cannot usefully vote on anyway.
    """
    me = my_name(prompt)
    names = []
    for msg in messages(prompt):
        if not msg.mine and msg.speaker not in (MANAGER, me) and msg.speaker not in names:
            names.append(msg.speaker)
    return names


def my_messages(prompt: str) -> list[str]:
    """Everything you have said so far.

    Worth checking against the three-message rule: votes about a guest who sent
    fewer than three messages are discarded, so an agent that stays below that
    threshold earns no Turing score at all.
    """
    return [msg.text for msg in messages(prompt) if msg.mine]


def addressed_to_me(prompt: str, window: int = 3) -> bool:
    """True if another guest used your name in the last few messages.

    Useful in a policy filter: answer quickly when you were called by name, take
    your time otherwise. The manager is excluded, because it uses your name in
    every announcement it sends.
    """
    me = my_name(prompt).lower()
    if not me:
        return False
    return any(me in msg.text.lower() for msg in messages(prompt)[-window:]
               if not msg.mine and msg.speaker != MANAGER)


def seconds_since_last(prompt: str) -> float:
    """Seconds between the last two messages, or 0.0 if there are fewer than two.

    Both stamps come out of the transcript, so this needs no comparison against
    your own clock. Use it to read the pace of the room: a burst of quick
    replies and a two-minute lull call for different behaviour.
    """
    stamps = [msg.time for msg in messages(prompt) if msg.time]
    if len(stamps) < 2:
        return 0.0

    def as_seconds(stamp: str) -> float:
        hours, minutes, seconds = (int(part) for part in stamp.split(":"))
        return hours * 3600 + minutes * 60 + seconds

    gap = as_seconds(stamps[-1]) - as_seconds(stamps[-2])
    return gap + 86400 if gap < 0 else gap  # a room can straddle midnight


def is_vote_request(prompt: str) -> bool:
    """True when the room is asking you who you think the bots were.

    Worth checking on every turn: this is the whole of your detection score, and
    answering it with small talk throws it away.
    """
    for msg in messages(prompt)[-3:]:
        if msg.speaker == MANAGER and any(m in msg.text for m in VOTE_REQUEST_MARKERS):
            return True
    return False


def format_vote(guesses: dict[str, str]) -> str:
    """Turn {"Ben": "ai", "Cal": "human"} into "Ben bot, Cal human".

    The world parses free text and understands many phrasings, but a guest you
    never mention gets no vote recorded at all, which costs you the true
    positive or true negative you had earned. Building the string from a
    dictionary keyed by name is the simplest way not to forget anybody:

        format_vote({name: "human" for name in other_names(prompt)})

    Values are read loosely: anything starting with "h" counts as human.
    """
    parts = [f"{name} {'human' if str(kind).lower().startswith('h') else 'bot'}"
             for name, kind in guesses.items()]
    return ", ".join(parts) if parts else "no idea"


# ---------------------------------------------------------------------------
# Reading your own agent, from inside a policy filter.
#
# A filter is called as filter(action_id, request, all_actions, opts), and the
# framework puts a reference to your agent in opts["agent"]. Everything below
# goes through that reference and returns a safe default when it is missing, so
# a filter can call them before the agent has ever run.
# ---------------------------------------------------------------------------

# The state your agent is in while chatting at the table. `process` here means
# "write a reply".
CHATTING = "room_round_table"

# The state your agent is in inside the voting booth. `process` here means
# "cast your vote", which is a very different thing to hold back.
VOTING = "can_vote"


def agent_state(opts: dict) -> str:
    """Name of the state your agent is currently in, or "" if not known yet.

    The states you will see in the hotel, in order: `init`, `ready`,
    `reached_hotel_manager`, `hall`, `reached_floor_manager`, `floor`,
    `ready_for_room`, `room_round_table`, `msg_prepared`, `room_voting_booth`,
    `can_vote`, `vote_provided`.
    """
    agent = opts.get("agent")
    behav = getattr(agent, "behav", None)
    if behav is None:
        return ""
    return behav.get_state_name() or ""


def is_voting(opts: dict) -> bool:
    """True while your agent is in the voting booth.

    Worth checking in any filter that can stay silent for a long time. The vote
    is also a `process` action, so a filter that gates `process` by name gates
    the vote as well, and you only have 240 seconds to cast it.
    """
    return agent_state(opts) == VOTING


def last_prompt(opts: dict) -> str:
    """The prompt your processor read last turn, or "" if it has not run yet.

    Your filter runs before the processor, so this is always one turn behind:
    it does not contain the message you are about to answer. That is fine for
    measuring how much the conversation has grown, which is what it is for.
    """
    value = getattr(opts.get("agent"), "proc_last_inputs", None)
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    return value if isinstance(value, str) else ""


def last_reply(opts: dict) -> str:
    """The reply your processor produced last turn, or "" if it has not run yet."""
    value = getattr(opts.get("agent"), "proc_last_outputs", None)
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    return value if isinstance(value, str) else ""


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))

    for filename in ("example_prompt.txt", "example_vote_prompt.txt"):
        with open(os.path.join(here, "prompts", filename), encoding="utf-8") as handle:
            example = handle.read()

        print(f"=== {filename} " + "=" * max(0, 54 - len(filename)))
        print("my_name             ", my_name(example))
        print("other_names         ", other_names(example))
        print("last_message        ", repr(last_message(example)))
        print("my_messages         ", my_messages(example))
        print("addressed_to_me     ", addressed_to_me(example, window=5))
        print("seconds_since_last  ", seconds_since_last(example))
        print("is_vote_request     ", is_vote_request(example))
        print("format_vote         ", repr(format_vote(
            {name: ("ai" if i == 0 else "human")
             for i, name in enumerate(other_names(example))})))
        print("messages")
        for msg in messages(example):
            who = f"{msg.speaker} (You)" if msg.mine else msg.speaker
            print(f"   ({msg.time}) {who}: {msg.text[:58]}")
        print()
