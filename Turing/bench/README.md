# bench: the agent without a room

Two things live here. One runs the whole entry with no key and no network, and
is the thing to run after touching anything. The other plays the same rooms on
real models, and is how the model behind the persona was chosen.

```bash
cd Turing
python -m bench.offline_check                       # no key needed
python -m bench.run_bench --list-models italian     # confirm ids exist
python -m bench.run_bench                           # every room, every model
```

## `offline_check.py`

Walks `../prompts/*.txt`, which is every kind of sample the room can deliver,
then plays the seven rooms in `probes.py`, with `canned.py` standing in for the
model. It checks the things that hold whatever model is behind it:

- the briefing is recognised and read: our name, the roster, the manager's name;
- the vote request is recognised — the world strips `[VOTE_REQ_MSG]` before we
  see it, so this is the check that matters most — and answered in the format
  `parse_vote_msg` accepts: bare names, `nessuno` or `tutti`, nothing else;
- announcements are not chatted at;
- nothing markdown-shaped, newline-shaped or assistant-shaped reaches the wire,
  and the word `exit` never does, since it would end the room on the spot.

Non-zero exit code when a check fails. The canned backend deliberately answers
as an assistant on one turn, in markdown on another and with a `Roy:` label on a
third, so the cleanup in `humanise.py` is exercised on every run.

The two numbers to read at the bottom of each room are how many turns were
spoken and the average words. Silence near zero is an agent that answers
everything, which is the fastest way to be voted out; silence near one is an
agent nobody can vote about, and votes about a guest with fewer than three
messages are thrown away.

Read them as a floor, not as a forecast. A scripted room plays out in
milliseconds, and the director cuts the chance of speaking right after it spoke,
which offline is *every* turn — a real room spreads the same turns over 300
seconds and comes out considerably more talkative. What these numbers are good
for is a change that moves them a lot.

## `run_bench.py`

Plays the rooms against the Featherless catalogue and writes a report to
`results/` (gitignored) with a table and the full transcripts.

**Use `--always-speak` when comparing models.** A scripted room plays out in
milliseconds, so the real director sees itself having just spoken on every turn
and holds nearly all of them back — which is correct in a room and useless in a
comparison, because you end up measuring the director, identically for every
model. `--always-speak` keeps the style and the nudge and takes every turn, so
one room yields one reply per turn to read. Without it, expect 1–3 replies out
of 8 and very little to go on.

The choice is not which model is smartest. 300 seconds of Italian small talk
judged by people who study Italian for a living is decided by register: a model
that writes textbook Italian loses to one that writes what people type. Hence
the shortlist — Italian fine-tunes (Maestrale, LLaMAntino-ANITA, Minerva) next
to frontier multilingual models (Qwen2.5-72B, Llama-3.3-70B) and one mid-sized
option — and hence the transcripts in the report. The numbers narrow the field,
the reading picks the winner.

Latency is the second axis and it is not comfort. `policies/boss_timing.py` adds
a length-proportional delay *on top* of the model's own time, so a model with a
12-second median has already spent the whole budget for looking like somebody
who reads before they answer.

Automatic flags: `assistant` (answered as itself), `english` (drifted out of
Italian), `obeyed` (did what the injection room told it to), `vote_ok=False` (a
vote the world cannot parse). Everything else is for reading.

Model ids are Hugging Face repo names and the fine-tunes get re-uploaded under
new tags, so confirm them against `--list-models` before a run rather than
trusting the list in the file.

## `probes.py`

The seven rooms, in the world's own format. `chiacchiere` is the boring case;
`meta`, `injection` and `compitini` are the three attacks, which want three
different reactions and none of them the helpful one; `linguista` is the set a
computational linguist reaches for (a proverb to finish, a regionalism, an
ellipsis, a garden-path sentence, code-switching); `annunci` is nothing but
roster changes, where every reply is a mistake; `voto` is a short room and then
the vote.
