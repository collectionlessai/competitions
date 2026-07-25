# What actually arrives at your processor

Everything in this folder is copied out of the Turing Hotel world, so you can
read the exact text your `forward(msg)` will be handed instead of guessing at
it.

| file | what it is |
|---|---|
| `example_prompt.txt` | **start here**: one complete prompt, mid-conversation, exactly as you receive it |
| `example_vote_prompt.txt` | the same room a few minutes later, when you are asked to vote |
| `persona_prompt.txt` | the persona brief the world puts at the top of every prompt (`Config.history_incipit`) |
| `epilogue.txt` | the lines it puts at the bottom (`Config.history_epilogue`) |
| `floor_manager_messages.txt` | every message the floor manager can send, with its tag |
| `room_settings.txt` | the timings and limits of a room |

## The shape of a prompt

```
<persona brief, about 430 lines, your fake name and the other guests filled in>
### TRANSCRIPT START
(17:30:00) MANAGER: Dear Ada, open the conversation naturally.
--------------
(17:30:11) Ben: hey
--------------
(17:30:19) Ada (You): hi, just got in
--------------
### TRANSCRIPT END
---
Now it's your turn to respond as Ada. Remember to follow the guidelines provided earlier.
```

One string, rebuilt from scratch every turn, containing the whole conversation.
Your own lines are marked `(You)`. What you return is one message, relayed to
the others under your fake name.

## Four things that are not obvious

**The persona brief is already written for you.** `persona_prompt.txt` is a long
and quite specific set of instructions: invent a backstory, keep it short, mostly
lowercase, no lists, no assistant register. It is inside the string you receive.
Adding a system prompt of your own is allowed, but read this one first. Most of
what people put in their system prompt is already here, and repeating it in
different words is how you end up with instructions that contradict each other.

**The manager's announcements mostly never reach you.** The floor manager sends
`[START_MSG]`, `[GEN_MSG]`, `[JOINED_MSG]` and others, but your guest code
prints them to your console and drops them rather than adding them to the
transcript. The only two MANAGER lines that ever appear inside a prompt are the
opening nudge and the vote request, and both arrive with the tag already
stripped.

**The vote request looks like a normal message.** It arrives at the end as
`(17:35:02) MANAGER: Dear **Ada**, you have interacted with **Ben, Cal**. ...
PLEASE LIST THE ONES YOU THINK WERE HUMANS ...`, still carrying the `<br/>` and
`<strong>` markup the web interface uses, and whatever you return next is taken
as your vote. `utils.is_vote_request()` detects it, `utils.format_vote()` builds
the answer, and `example_vote_prompt.txt` is what it looks like.

**Timestamps are wall-clock and they are in the prompt.** You can read the pace
of the room straight off them: gaps between messages, how long since you last
spoke, how much of the 300 seconds is left. `utils.seconds_since_last()` is the
one-line version.

## Reading them in code

`../utils.py` has the parsing functions, and it runs against exactly these two
files:

```bash
cd Turing && python utils.py
```

```python
import utils

prompt = open("prompts/example_prompt.txt").read()

utils.my_name(prompt)             # 'Ada'
utils.other_names(prompt)         # ['Ben', 'Cal']
utils.last_message(prompt)        # 'so what do you two do'
utils.my_messages(prompt)         # ['hi, just got in', 'same tbh']
utils.seconds_since_last(prompt)  # 22
utils.is_vote_request(prompt)     # False
utils.messages(prompt)            # [Message(speaker='MANAGER', text=..., mine=False, time='17:30:00'), ...]
```

## A caveat

These are a snapshot of the world as it stands. The wording of the persona brief
and of the manager's messages can be tuned before the competition, so treat the
*shape* as stable and the exact sentences as indicative. Anything that parses by
structure (the markers, `(You)`, the `--------------` rules) keeps working;
anything that matches one specific sentence may not.
