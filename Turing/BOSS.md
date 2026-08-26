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
| `processors/featherless.py` | the model backend, swappable for any other |
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

**A persona is not enough, a script is too much.** The model free-forms; the
director modulates each turn — terse, quiet, changing the subject, planting a
doubt, making a typo — on a random walk with momentum, so the log comes out in
bursts and lulls instead of a flat line. Answering every announcement is the
single most recognisable thing an agent does, so most of them get nothing.

**The vote is played to win.** F1 is scored, so the vote is genuine: a separate
prompt with no persona in it, over the transcript plus per-speaker statistics —
how fast each guest replied, how evenly, how uniform their message lengths, how
reliably they capitalised and punctuated. Those are the same tells the boss
spends its whole effort not giving. There is a numeric fallback for when the
model fails, and the answer is cached so a reminder gets the same vote rather
than a second opinion.

## Still to do

- **Pick the model.** `MODEL` in `my_agent.py` is a placeholder until
  `python -m bench.run_bench` has run against the shortlist. Confirm the ids
  with `--list-models` first: they are Hugging Face repo names and the Italian
  fine-tunes get re-uploaded under new tags.
- **Tune the register against real rooms.** The "middle" — some slang and
  imperfection, no crudeness — is a guess until people have played against it.
  `persona_it.txt` and the `Director` constants are the two dials.
- **Node name.** `NODE_NAME = "TuringBoss"` claims a permanent slot on the
  account. Reuse it between runs rather than inventing a new one.
