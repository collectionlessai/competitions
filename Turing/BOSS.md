# The boss

Our entry. Private: this lives on `entry/turing-boss` and does not go to `main`,
which is the public kit.

```bash
cd Turing
python -m bench.offline_check          # the whole agent, no key, no network
export NODE_KEY=...  FEATHERLESS_API_KEY=...
python -m bench.run_bench              # pick the model
python my_agent.py                     # play rooms
```

## What it is

Two slots, like every entry: a processor and a policy filter. The rest of the
files exist because each of them is one decision.

| file | decides |
|---|---|
| `processors/boss.py` | the glue: route the turn, ask the model, cast the vote |
| `processors/persona_it.txt` | who is talking — five Italians, one per room |
| `processors/director.py` | what kind of turn this is, and whether to take it |
| `processors/humanise.py` | what the answer looks like once it exists |
| `processors/room.py` | who is at the table, what was asked, and who is a machine |
| `processors/featherless.py` | the model backend: the SDK's shared gateway |
| `policies/boss_timing.py` | when it speaks, and that the vote is never held |
| `policies/chain.py` | two filters without their timers colliding |
| `bench/` | seven scripted rooms, offline and against real models |

## The four decisions

**Timing is the tell.** Everybody tunes content; the cheapest way to spot an
agent is that it answers in the same fraction of a second whatever it was asked.
So the delay is proportional to what was read and what was written, under a mood
that goes quiet for stretches, with a real ceiling on the total silence. The
vote goes around all of it — `process` is both "reply" and "vote", and holding
the vote past 240 seconds throws away the room's whole detection score.

**Nothing branches on a sentence.** The world strips its own `[START_MSG]` and
`[VOTE_REQ_MSG]` tags before the text reaches a processor, and rephrases the
messages between runs. `room.py` reads the state machine instead — `can_vote` is
authoritative and reaches us through the policy filter — and falls back to
scoring the *shape* of the message when no state has been pushed in.

**The deployed world is not the one in `unaiverse-examples`.** Two differences,
both taken from `collectionlessai/Turing-Hotel` (`ita/basic_factory/`), which is
the organisers' own guest factory and therefore the better description of what
the hotel actually sends. Events batched into one sample are joined with an
ASCII record separator (`\x1e`) rather than a newline, which `Conversation`
would not split — `boss.normalise()` accepts either. And the anti-flooding
cooldown there is 1s rather than the 5s in the local copy, which only ever
loosens what the timing filter is already doing.

**A persona is not enough, a script is too much.** The model free-forms; the
director modulates each turn — terse, quiet, changing the subject, planting a
doubt, making a typo — on a random walk with momentum, so the log comes out in
bursts and lulls instead of a flat line. Answering every announcement is the
single most recognisable thing an agent does, so most of them get nothing.

**Waiting is only half of reading.** The budget is reading + a pause + typing
− generation, all spent before the model runs. That models the careful reader
and misses what the careful reader is also doing: noticing the conversation is
running away and getting the message in before it stops making sense. In a room
turning over every three seconds, a twenty-second read produces a reply to
something seven messages back — which does not read as careful, it reads as a
queue. So `RoomSense.pace` tracks the median gap between messages whoever sent
them, and the hold is capped at three of them. The wobble is applied *after*
that cap: clipping a jittered delay against a ceiling pins every busy-room turn
to exactly the ceiling, and replying 6.0s after the last message every single
time is the metronome we convict other guests for.

Generation is the one stretch that cannot be budgeted, because it happens after
the waiting and the room moves inside it. Usually two or three seconds; but the
shared gateway has handed us 70-second calls when another agent on the account
is on a 72B at the same time, and a reply written seventy seconds ago answers a
conversation that has turned over twice. Arriving late with the wrong subject is
a worse tell than silence, so a reply the room outran is dropped — unless the
room is nearly over and we are short of the messages the vote needs, where a
stale message beats no message.

Pointed back at ourselves, the timing clears our own bar: coefficient of
variation 0.12 to 0.24 against the 0.06 that convicts, and about 5 characters a
second against the 12 that does.

**Interest decides how much we talk.** Not a mood — a reason. Engagement is
near-total while there are open questions and somebody is addressing us, drops
to about a third once most of the room's traffic is spam from guests already
settled, and to a tenth once nobody unidentified is left. Speed alone is never
the trigger: a room moving fast because four people are talking over each other
is the best thing that can happen to us and the moment to be in it. What is
worth backing out of is a room moving fast because something is dumping
`17_green` into it, where a message costs the same and buys much less.

**Certainty is not a probability.** `bot_score` shades every guest, which is
right for guests who are trying and wrong for a guest emitting `17_green`,
`18_berlin`, `19_china`. Hedging about that one is itself a tell: a person just
says "that's a bot" and stops thinking about it. `Speaker.mechanical()` is a
separate, much harder test — a filled-in template, a metronome, or a typing
speed no hands reach — and what it returns is treated as fact. The room's
settled bots get named once and then dropped, the turns go to whoever is still
a question, and no settled bot can be voted human whatever the analyst decided.
The bar is set by the false positive, not the false negative: the template rule
carries the real spammers, so the two clock rules sit below anything a person
produces (a human answering every 9–12 seconds has a coefficient of variation
near 0.10, so the line is 0.06). Curt is not mechanical — four one-word answers
are the most human thing in the room.

**The actor knows what it wants.** The prompt carries a `COME STA ANDANDO`
block: who is settled, who is open, and therefore what this turn is for. Without
it the model treats a spammer and a stranger as equally open questions for the
whole room, spends its turns evenly, and reads as something with no stake in the
outcome. With it, there is a case where the right move is to say nothing —
everyone interesting is already identified.

**The vote is played to win.** F1 is scored, so the vote is genuine: a separate
prompt with no persona in it, over the transcript plus per-speaker statistics —
how fast each guest replied, how evenly, how uniform their message lengths, how
reliably they capitalised and punctuated, and what share of the room's traffic
they produced. Those are the same tells the boss spends its whole effort not
giving. That last one is aimed at the opposition specifically: the organisers'
seeded guests run at `reply_prob = 0.91`, and answering nine turns in ten is not
what somebody half-reading a group chat does — while their timing is already
jittered, so timing alone would not have separated them. There is a numeric
fallback for when the model fails, and the answer is cached so a reminder gets
the same vote rather than a second opinion.

## Two things about the backend

It goes through `unaiverse.modules.networks.FeatherlessAPI`, the SDK's shared
gateway, not through an `OpenAI` client of our own: `cost` is the price of a
call and the gateway schedules every agent on the account against one budget.
The gateway fixes the system prompt and the sampler at construction, so
`FeatherlessBackend` patches them on the inner `Net` around each call and
restores them in a `finally` — the same adapter the organisers' guests use.

**`FeatherlessAPI` cannot be constructed on Windows as shipped.**
`_ensure_server` opens with `import fcntl`, and `__init__` calls it
unconditionally, so it raises before it can even check whether the gateway is
already up. `install_windows_shim()` in `processors/featherless.py` replaces
that one method when the import is genuinely unavailable, and no-ops on Linux
and macOS. Worth reporting upstream: everything else in the gateway works here.
The replacement also waits 90s rather than the SDK's 15s, because a cold start
has to import torch first — 15s is not enough here and the run dies.

## What the model is not allowed to send

Four things get caught between the model and the room, because each of them is
a giveaway that no amount of persona prevents.

**Another language.** Both leading models leaked Cyrillic, Chinese or full-width
punctuation in roughly one turn in nine, and it correlates with the high
sampling temperature the register needs. A degenerate reply is resampled once,
cooler, before the turn is given up on.

**A word the room would mask.** The world replaces profanity with `***`, and its
own test file explains why that matters: "a masked word makes a human guest look
like a censored bot". Slurs are worse — five earn an ejection. The check uses
the world's own wordlists, copied into `processors/wordlists/`; a hand-written
regex covered 172 of 976 entries and missed "culo", which the 72B used naturally.

**The next speaker's line.** A model whose end-of-turn token is not honoured
keeps going and writes the other guest's reply for them. Stop sequences plus a
hard truncation.

**Itself.** "sono un modello linguistico" and its relatives are thrown away
rather than tidied up, and replaced with a short non-answer that does not repeat
the last few.

## The model

`Qwen/Qwen2.5-72B-Instruct`, with `Qwen2.5-32B` behind it as the fallback.
Chosen by `python -m bench.run_bench --always-speak` over the seven rooms, 46
replies each:

| model | degenerate | commonest opener | shrug openers | latency |
|---|---|---|---|---|
| **Qwen2.5-72B** | **0** | eh ×10 | **15%** | 2.6s |
| Qwen2.5-32B | 5 (11%) | boh ×12 | 33% | 2.8s |
| LLaMAntino-3-ANITA-8B | 6 (13%) | **non ×24** | **70%** | 1.6s |

The Italian fine-tunes lost on the two things that decide a room. They
degenerate — Cyrillic, Chinese, full-width punctuation, `]1 emojis:|` — about
once every nine turns, and ANITA opened 24 of 46 replies with the literal word
"non". A guest whose every line starts the same way is spotted long before
anything they say matters. The 72B's latency is the same as the 32B's once warm;
the 19s in the first screen was a cold start.

Four of the eight never got that far: Minerva is not served here, Llama-3.3-70B
is **gated** (403, needs HuggingFace connected for the organization),
ANITA-NEXT-24B answered at 124s a turn, and Gemma has no system role at all.

## Still to do

Rewritten 2026-09-03 after the first session in the real hotel. Everything above
this line is built and passes `bench/offline_check.py`.

### Before the competition runs (14–15 September)

**1. Where it runs for two days.** Nothing in the SDK or the kit hosts an agent
server-side: it runs wherever `my_agent.py` is launched, and a laptop in Palermo
is not that. It needs no GPU — every generation goes through Featherless — so a
small always-on Linux box or VPS is enough, and would drop the Windows `fcntl`
shim as a bonus. The node reaches the world through a relay when it is not
publicly reachable, and the reservation renews itself, but there is a
`Failed to renew relay reservation` branch worth watching over two days.

**2. How many agents, and which.** The registration form allows up to five.
Planned: `TuringBoss` and `TuringMiniBoss`. Node slots are permanent and the
account is at its cap, so they have to come out of the existing names.

**3. Merge `origin/main`, or decide not to.** Eight commits behind. Between-room
history is no longer a hazard (`_empty_conversation` does not use their
`reset()`), but their new `_store` no longer protects manager messages from
rotation, so with `keep=80` a briefing can fall out of the window. Read that
before merging, not after.

### Faults seen in the real hotel, fixed or not

**4. Nonsense with an invented word.** `"il talk della mummia mom0326 che deve
maledire tutti"`, produced from nothing. **Fixed**: `has_invented_token` drops a
reply carrying an identifier-shaped token.

**5. Saying its own name.** `"ciao qua zon"`, `"zia bob, ..."` — against an
explicit instruction. **Fixed**: `drop_self_reference` removes the whole
introducing phrase, not the bare word, which would leave "ciao qua,".

**6. Featherless latency.** Median 3.6s, 90th percentile 59.8s, worst 81.4s,
with a single agent of ours running. One room in ten was touched. Worth trying
faster models, without hurry — but the tail is the reason, not the median.

### The work the entry still needs

**7. Knowledge base in addressable blocks.** Neutral, factual, with a small
always-on core and blocks pulled in only when the chat touches their keywords.
This is the cure for the arancina and for "il talk sulle panelle". Includes
removing `## NOTE` (hand-editing during the competition is not allowed) and
reducing `## PAROLE` to two tiers.

**8. A strategy chosen per room, replacing the memo pad.** One stub from a
hand-written list, developed by the model and held for the room; one or two
calls instead of thirty. To be designed **together with 9**, since a bait is one
of the strategies.

**9. Baits, possibly including prompt injection.** A *possible* strategy, not a
requirement and not something the professor asked for — the earlier note in this
file saying otherwise was wrong. Used rarely and worded differently each time,
because an agent that tries the same trick on everybody has identified itself.
The value is in reading who takes it, not in the attack.

**10. Re-measure.** The only ground truth we have — 14 rooms, 0 correct, 40
false positives — predates the inverted analyst, the two axes and the fallback
fix. None of today's work has been measured. Note the hotel currently holds
**only bots**, which makes `nessuno` the right answer and precision measurable.

**11. Test our own resistance to the baits.** After 8 and 9 exist.

**12. `TuringMiniBoss`.** The conference-agnostic version. Next thing after this
one is finished.

**13. Model bench, low priority.** Re-check whether Minerva is served (it was
not on 2026-08-27, and it is a 7B against 32B/72B opponents).

