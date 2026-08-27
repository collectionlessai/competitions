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
- **Tune the register against real rooms.** The "middle" — some slang and
  imperfection, no crudeness — is a guess until people have played against it.
  `persona_it.txt` and the `Director` constants are the two dials.
- **Node name.** `NODE_NAME = "TuringBoss"` claims a permanent slot on the
  account. Reuse it between runs rather than inventing a new one.
