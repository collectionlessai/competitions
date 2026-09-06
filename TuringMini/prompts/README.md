# Prompts: what lands on your processor

Every kind of sample the room can hand to your `forward()`, one per file, taken
from the world's own source with the names filled in: you are `Roy`, the other
guests are `Ivy` and `Pax`. Read them once and you have seen your whole input
side. Nothing in here is something you write. A sample printed from inside your
processor should look like one of these, give or take the names and the seconds.

## What the input looks like

**One event per line.** Most samples are a single line. Two or three come
together when events piled up while you were busy answering, which is what
`04_batch.txt` shows, so splitting on newlines is safe. Length does not break
the rule, since the guest replaces the newlines inside an event with spaces
before pushing it to you: `01_start.txt` reads like a page of a document and is
one line.

**The manager speaks as `**MANAGER:**`, guests as `**Ivy:**`.** The fake names
are handed out per room and mean nothing between rooms. Some events carry no
name at all, `12_violation.txt` for one, so whatever parses these lines has to
keep a line that matches nothing rather than drop it. `utils.Conversation` keeps
it whole, with an empty speaker.

**The HTML is really there.** `<br/>` and `<strong>` reach you as text, since the
same message goes to human guests reading it in a browser. Strip it or leave it
in the prompt, as you prefer.

**The start message means a new room.** It is the first thing you get in every
room, and your history from the previous one is worth nothing from there on.

**The vote request arrives alone**, as the only line of its sample, and whatever
your processor returns next is recorded as your vote. Nothing in the text marks
it as special: it is a message from the manager, like the reminders are. It does
spell out the format it wants: bare names separated by commas or spaces,
"nessuno" if you think none of them was human, "tutti" if they all were. You get
**240 seconds**.

## The files

| file | arrives | world's tag |
|---|---|---|
| `01_start.txt` | first thing in a room: your name, who else is at the table, the rules | `[START_MSG]` |
| `02_start_alone.txt` | the same, when you are seated in a room by yourself | `[START_MSG_NOBODY]` |
| `03_chat.txt` | another guest said something | none |
| `04_batch.txt` | two events in one sample | mixed |
| `05_joined.txt` | somebody was seated at your table | `[JOINED_MSG]` |
| `06_left.txt` | somebody left the room | `[LEFT_MSG]` |
| `07_disconnected.txt` | somebody dropped off the network | `[DISCO_MSG]` |
| `08_reminder.txt` | every 65 s, with the seconds you have left | `[GEN_MSG]` |
| `09_vote_request.txt` | in the voting booth: name the guests you took for human | `[VOTE_REQ_MSG]` |
| `10_vote_request_alone.txt` | the same slot, with nobody to vote about: it asks for any message at all | `[VOTE_REQ_MSG]` |
| `11_reminder_vote.txt` | the vote is still missing and the clock is running | `[GEN_MSG]` |
| `12_violation.txt` | you are being disconnected | `[GEN_MSG]` |

The tags are how the guest sorts the events internally, and they come off before
the text reaches you. They are here so the files line up with
`worlds/turing_ita/src/guest.py` when you read it.

Two things never show up in a sample: the state machine moving you between hall,
room and voting booth, none of which is a message, and anything from a room you
have already left.

## Feeding them to a processor

From `Turing/`, which is where the imports work:

```python
from pathlib import Path
from processors.eliza import Eliza

proc = Eliza()
for path in sorted(Path("prompts").glob("*.txt")):
    print(path.name, "->", proc(path.read_text().strip()))
```

That answers "does it crash" and "does it notice it was asked to vote" in a
second, with no network. It is still not a room. The snippet never constructs a
policy filter, so the timing is missing, and the guests in these files never
answer back. For that you need the world running, which
[`../README.md`](../README.md#which-hotel-your-agent-joins) covers.

## Do not match on them

The wording changes. It changed between editions of this competition, and the
sentences get rephrased between test runs, so `if "ELENCA" in sample` is a branch
with an expiry date on it. What the files are good for is knowing the shape of
what arrives, and checking your processor against it before you spend a room
finding out.

They were rendered from `worlds/turing_ita/src/config.py` through the two
transformations in `src/guest.py`: the tag comes off the front, the newlines
become spaces. Run a copy of the world with an edited config and your rooms will
not match these files.
