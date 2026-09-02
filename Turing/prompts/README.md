# Prompts: what lands on your processor

Each file contains one kind of sample that the current `turing_ita` world can
pass to `forward()`. The examples use `Roy` for your room name, with `Ivy` and
`Pax` as the other guests, showing the processor view after the framework has
projected any UAI block to readable text.

## What the input looks like

A sample contains one or more events in arrival order, separated by ASCII Record
Separator (`\x1e`). Newlines remain part of the event itself, so `splitlines()`
would corrupt room guides, roster lists, multiline chat and UAI instructions.
Split only on `\x1e`.

Because the control character is invisible in an editor, `04_batch.txt` displays
it as `␞` (U+241E). Replace that symbol with `\x1e` when replaying the fixture.
`utils.Conversation` performs the split for real samples.

Every event begins with a sender. Manager events use `**MANAGER:**`, and guest
events use names such as `**Ivy:**`. Before broadcasting a multiline message,
the floor manager prevents its later lines from imitating another sender.

The start message is the first event of a new room and immediately triggers a
processor turn, which lets the agent open the conversation before another guest
speaks. Although its internal start tag has already been removed,
`utils.Conversation` recognises this visible greeting as the room boundary and
clears the completed room before storing it. No other prompt is interpreted.

The vote arrives alone as a UAI form. Although the wire carries a fenced `uai`
block, ordinary model processors see the Italian instruction shown in
`09_vote_request.txt`. The answer contains only the aliases judged human,
separated by commas, or the whole-room shortcut `tutti` or `nessuno`. The world
converts a valid answer to a canonical reply and retries blank or malformed
model output up to the framework limit. If the model remains silent, no empty
ballot is sent.

## The files

| file | arrives | world's internal tag |
|---|---|---|
| `01_start.txt` | first event in a populated room: your alias, roster and rules | `[START_MSG]` |
| `02_start_alone.txt` | the same when you are seated alone | `[START_MSG_NOBODY]` |
| `03_chat.txt` | one multiline guest message | none |
| `04_batch.txt` | two events in one sample, with a visible `␞` placeholder | mixed |
| `05_joined.txt` | somebody was seated at your table | `[JOINED_MSG]` |
| `06_left.txt` | somebody left the room | `[LEFT_MSG]` |
| `07_disconnected.txt` | somebody dropped off the network | `[DISCO_MSG]` |
| `08_reminder.txt` | periodic time and current-roster reminder | `[GEN_MSG]` |
| `09_vote_request.txt` | model view of the form asking who was human | `[VOTE_REQ_MSG]` |
| `10_vote_request_alone.txt` | the voting slot when there is nobody to judge | `[VOTE_REQ_MSG]` |
| `11_reminder_vote.txt` | the vote is still missing | `[GEN_MSG]` |
| `12_filter_mask.txt` | part of your message was masked before broadcast | `[GEN_MSG]` |
| `13_filter_severe.txt` | a severe-content warning | `[GEN_MSG]` |
| `14_filter_eject.txt` | queued when the severe-content limit is reached | `[GEN_MSG]` |

The world uses these tags for routing, then removes them from events selected
for the processor. The table includes each internal tag so its fixture can be
traced to `worlds/turing_ita/src/config.py` and `src/guest.py`. A
`[VIOLATION_MSG]` rejecting entry is printed locally but never pushed to the
processor. Unknown status tags are consumed without being printed. The
ejection notice is pushed immediately before the floor manager disconnects the
guest, so the process may end before `forward()` consumes that final payload.

## Feeding them to a processor

Run the following from `Turing/`, where the local imports resolve:

```python
from pathlib import Path
from processors.eliza import Eliza
from utils import DISPLAY_EVENT_SEPARATOR, EVENT_SEPARATOR

proc = Eliza()
for path in sorted(Path("prompts").glob("*.txt")):
    sample = path.read_text(encoding="utf-8").strip()
    sample = sample.replace(DISPLAY_EVENT_SEPARATOR, EVENT_SEPARATOR)
    print(path.name, "->", proc(sample))
```

This checks input shapes and processor errors without recreating a room, policy
timing or guest replies. The offline contract tests use the same fixtures:

```bash
python -m unittest discover -s tests
```

## Do not match on the wording

The manager wording may change between runs, but the transport contract stays
the same. Use these files as fixtures for event boundaries and input shapes,
not as a vocabulary for conditions such as `if "ELENCA" in sample`. Processor
state should remain useful when a sentence is rephrased.
