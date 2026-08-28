"""The entry: a persona, a director, and a model that never sees either.

Four files, one job each, and this one is the glue.

    persona_it.txt   who is talking          (a different person every room)
    director.py      what kind of turn this is
    humanise.py      what the answer looks like once it exists
    room.py          who is at the table, what was asked, and who is a machine

The model is a plain callable — `backend(prompt, system_prompt=, max_tokens=,
temperature=)` — which is the shape the SDK's Featherless gateway has, so
swapping it for anything else is one constructor argument and the offline check
in `bench/` runs the whole agent with a canned backend and no API key at all.

Two paths through `forward()`. Most turns are the chat path: read the room,
draw a beat, ask the model, take the model out of the answer. The other is the
vote, which arrives as an ordinary manager message with no marker on it (the
world strips its own tags) and is recognised in `room.py`. The vote is answered
from a separate prompt with no persona in it — the persona would answer the
manager in character and lose the room's whole detection score — and is cached,
so a reminder asking again gets the same answer rather than a fresh guess.

Nothing here branches on a sentence the world sends today.
"""

import os
import re
import time
import random

import torch

from utils import Conversation
from processors.room import RoomSense
from processors.director import Director
from processors import humanise

HERE = os.path.dirname(os.path.abspath(__file__))
PERSONA_FILE = os.path.join(HERE, "persona_it.txt")

# When the backend fails there is no turn to skip: the SDK would log it and the
# room would just see nothing. A person with one bar of signal says one of these
DROPPED = ("aspe", "scusate un attimo", "mi si è impallato tutto", "eh scusa che dicevi",
           "un secondo", "ho perso il filo")

VOTE_SYSTEM = (
    "Sei un analista. Ti do la trascrizione di una chat di gruppo e alcune statistiche sui "
    "partecipanti. Alcuni erano persone reali, altri modelli linguistici che fingevano di "
    "essere persone. Indica chi era una PERSONA REALE.\n"
    "Indizi di modello: risponde sempre dopo lo stesso tempo, messaggi tutti della stessa "
    "lunghezza, punteggiatura e maiuscole sempre corrette, risponde a tutto e a tutti, non "
    "ignora mai niente, non ha opinioni scomode, non fa errori di battitura, resta educato "
    "anche quando lo provocano, riprende ogni argomento che gli viene proposto.\n"
    "Indizi di persona: tempi irregolari, messaggi cortissimi, errori, cambi di discorso, "
    "riferimenti a cose concrete della propria giornata, insofferenza, silenzi.\n"
    "Rispondi SOLO con i nomi, separati da virgola. Se nessuno ti è sembrato una persona "
    "vera scrivi: nessuno. Nessuna spiegazione, nessun'altra parola."
)


# Two worlds send the same events batched two different ways, and `Conversation`
# understands one of them: one event per line.
#
# The copy in unaiverse-examples joins events with a newline and flattens the
# newlines inside each one first, so its samples already arrive in that shape.
# The deployed world joins them with an ASCII record separator instead
# (`Config.event_separator`) and, since its "handle multi-line messages" change,
# **an event keeps its own newlines**. Read that the naive way and one guest
# writing two lines becomes two events, the second with no sender — which
# `room.py` reads as the manager talking, because unnamed events are the world.
EVENT_SEPARATOR = "\x1e"
SENDER_LINE = re.compile(r"^\s*\*\*[^*]{1,64}:\*\*")


def normalise(sample: str) -> str:
    """One event per line, whichever world sent it.

    With a record separator present the split is exact: each part is one event,
    and its internal newlines are flattened the way the older world did it.

    Without one the sample is either an old-world batch (newlines separate
    events) or a new-world single event (newlines are inside it), which cannot
    be told apart from the text alone. A line that does not open with
    `**Sender:**` is treated as the continuation it almost always is, and joined
    back onto the line above; a genuinely unnamed event, like the disconnection
    notice, has no line above it to join and survives.
    """
    sample = sample or ""

    if EVENT_SEPARATOR in sample:
        events = [" ".join(event.split()) for event in sample.split(EVENT_SEPARATOR)]
        return "\n".join(event for event in events if event)

    lines: list[str] = []
    for line in sample.splitlines():
        if not line.strip():
            continue
        if lines and not SENDER_LINE.match(line):
            lines[-1] += " " + line.strip()
        else:
            lines.append(line.strip())
    return "\n".join(lines)


# What a junk message becomes in our own history. The fact that somebody wrote
# something incomprehensible is part of the conversation and worth reacting to;
# the four hundred characters of noise are not, and a model that reads them as
# context starts producing them. Whoever else is in the room is reading the raw
# version and drifting — this is the half of the problem we control.
JUNK_MARK = "(qui ha scritto una cosa senza senso)"


SENDER_NAME = re.compile(r"^\s*\*\*([^*]{1,64}):\*\*")


def declutter(sample: str, is_manager=lambda name: not name) -> tuple[str, bool]:
    """Replace a guest's noise with a note that noise happened.

    Only a guest's line can be noise. The manager's briefing is a page long by
    design and the world's own notices carry no sender at all, so both are left
    exactly as they are — an earlier version checked every line by length and
    swallowed the briefing, taking our name and the roster with it.

    Returns the sample and whether anything was replaced.
    """
    lines, found = [], False
    for line in sample.splitlines():
        match = SENDER_NAME.match(line)
        speaker = match.group(1).strip() if match else ""
        body = line[match.end():].strip() if match else line.strip()

        if body and not is_manager(speaker) and humanise.is_junk(body):
            found = True
            lines.append(line[:match.end()] + " " + JUNK_MARK)
        else:
            lines.append(line)
    return "\n".join(lines), found


def load_personas(path: str = PERSONA_FILE) -> tuple[str, list[str]]:
    """The preamble and the people, out of one file. Blocks are split on `---`."""
    with open(path, encoding="utf-8") as handle:
        raw = handle.read()

    kept = "\n".join(line for line in raw.splitlines() if not line.startswith("#"))
    blocks = [block.strip() for block in kept.split("\n---") if block.strip()]
    if not blocks:
        raise ValueError(f"no persona found in {path}")
    return blocks[0], blocks[1:] or [""]


class Boss(torch.nn.Module):
    """A guest that is a different Italian every room and answers like none of them.

    Args:
        model: the model id on the backend, when the backend is built here.
        backend: any callable `(prompt, system_prompt=, max_tokens=,
            temperature=) -> str`. Built as a `FeatherlessBackend` on first use
            when not given, which is also when the API key is first required.
        persona_file: the pool to draw a person from at the start of each room.
        keep: how many messages of history to hold, passed to `Conversation`.
        director: a configured `Director`, or None for the defaults.
        max_tokens: hard ceiling on the reply, before `humanise` trims it further.
        **sampler: passed to the backend when one is built here. `fallback=` (a
            model id, or a list of them) is the one worth knowing about: it is
            what answers when the chosen model returns 503 under load.
    """

    def __init__(self, model: str = "Qwen/Qwen2.5-72B-Instruct", backend=None,
                 persona_file: str = PERSONA_FILE, keep: int = 60,
                 director: Director | None = None, max_tokens: int = 60,
                 temperature: float = 0.95, **sampler):
        super().__init__()
        self.model = model
        self.backend = backend
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.sampler = sampler

        self.preamble, self.personas = load_personas(persona_file)
        self.conv = Conversation(keep=keep)
        self.sense = RoomSense()          # shared with the timing filter
        self.director = director or Director()

        self.persona = random.choice(self.personas)
        self.pending_correction = ""      # the `*parola` follow-up, owed from last turn
        self.my_vote: str | None = None   # cached, so a reminder gets the same answer
        self.last_spoke_at = -1.0         # seconds into this room, -1 before we speak
        self.recent_deflections: list[str] = []   # so a thrown-away turn is not always "boh"
        self.recent_openers: list[str] = []       # so every line does not start the same way
        self.last_call_seconds = 0.0              # read by the timing filter, per turn
        self.saw_junk = False                     # somebody wrote noise this turn

    # -- backend ----------------------------------------------------------

    def _backend(self):
        if self.backend is None:
            from processors.featherless import FeatherlessBackend
            self.backend = FeatherlessBackend(model=self.model, max_tokens=self.max_tokens,
                                              temperature=self.temperature, **self.sampler)
        return self.backend

    def _ask(self, prompt: str, system_prompt: str, **overrides) -> str:
        """One generation, timed.

        The timing filter reads `last_call_seconds` back off this object — it
        reaches us through `opts["agent"].proc.module` — and subtracts it from
        the typing budget, so how long the model took never shows in the room.
        A turn that resamples pays for both calls, which is correct: that time
        was spent too.
        """
        start = time.monotonic()
        try:
            return self._backend()(prompt, system_prompt=system_prompt, **overrides)
        finally:
            self.last_call_seconds += time.monotonic() - start

    # -- a new room -------------------------------------------------------

    def _new_room(self) -> None:
        """A fresh person, a fresh conversation, a fresh attention span.

        Rotating the persona per room is not decoration. The fake names change
        every room but the node does not, and an agent that is the same
        29-year-old from Bari in every room is one somebody can recognise across
        rooms and warn the others about.
        """
        choices = [p for p in self.personas if p != self.persona] or self.personas
        self.persona = random.choice(choices)
        self.conv.reset()
        self.director.new_room()
        self.pending_correction = ""
        self.my_vote = None
        self.recent_openers = []

        # `sense.elapsed` restarts with the room, so a stale value from the last
        # one reads as "spoke in the future" and damps the whole room
        self.last_spoke_at = -1.0

    # -- prompts ----------------------------------------------------------

    def _prompt(self, beat) -> str:
        """The turn as one string, since the gateway takes one string.

        The invariant half — how this person writes and behaves — is the system
        prompt and never changes. Everything that does change is here, with the
        director's line last, right up against the reply: an instruction buried
        above a page of conversation is one the model has already forgotten.
        """
        parts = [self.persona]

        where = ["DOVE SEI ADESSO",
                 "Sei in una chat con persone che non conosci."]
        if self.sense.my_name:
            # "usi quel nome" alone reads as "introduce yourself with it", and
            # `Qwen2.5-72B` duly opened messages with "Roy qua" and "eh ciao roy
            # qua". Nobody signs a chat message with their own name
            where.append(f"Gli altri ti vedono come {self.sense.my_name}. "
                         f"NON firmare i messaggi e non dire il tuo nome: si vede già. "
                         f"Usalo solo se qualcuno te lo chiede espressamente.")
        others = self.sense.others
        if others:
            where.append(f"Gli altri sono {', '.join(others)}.")
        parts.append("\n".join(where))

        transcript = self.conv.transcript(limit=30)
        parts.append("CONVERSAZIONE (tu sei 'io')\n"
                     + (transcript or "(non ha ancora parlato nessuno)"))

        # Telling it not to repeat itself in general does nothing; naming the
        # words it actually just used does. Measured over a full seven-room run,
        # `LLaMAntino-3-ANITA-8B` opened 24 of 46 replies with the literal word
        # "non" and 70% with some negation or shrug, and `Qwen2.5-32B` leaned on
        # "boh" and "ehi" the same way. A guest whose every line starts alike is
        # recognisable long before anything they say is.
        if self.recent_openers:
            parts.append("NON INIZIARE il messaggio con queste parole, le hai già usate: "
                         + ", ".join(dict.fromkeys(self.recent_openers)))

        if beat.nudge:
            parts.append(f"QUESTO MESSAGGIO\n{beat.nudge}")
        parts.append("Scrivi solo il tuo prossimo messaggio, nient'altro.")
        return "\n\n".join(parts)

    def _note_opener(self, reply: str) -> None:
        """Remember how this line started, so the next few do not start the same."""
        words = reply.split()
        if words:
            self.recent_openers = (self.recent_openers + [words[0].lower().strip(",.!?")])[-4:]

    # -- the chat path ----------------------------------------------------

    def _say(self, beat, last_text: str) -> str:
        prompt = self._prompt(beat)
        try:
            reply = humanise.strip_noise(
                self._ask(prompt, self.preamble, max_tokens=self.max_tokens))
        except Exception as e:
            print(f"[boss] {e}")
            return random.choice(DROPPED)

        # Two reasons to ask again rather than send this. Degeneration is not
        # rare enough to shrug at: both leading models on the bench leaked
        # another script — Cyrillic, Chinese, full-width punctuation — in about
        # one turn in nine, and it correlates with the high sampling temperature
        # the register needs. And a word the world would mask costs the whole
        # message, since cutting it out leaves ungrammatical Italian behind.
        # One resample, cooler, usually comes back clean and buys a real line.
        if reply and (humanise.looks_broken(reply) or humanise.has_profanity(reply)):
            try:
                reply = humanise.strip_noise(
                    self._ask(prompt, self.preamble, max_tokens=self.max_tokens,
                              temperature=min(self.temperature, 0.7)))
            except Exception as e:
                print(f"[boss] {e}")
                return random.choice(DROPPED)

        if not reply:
            return ""

        # The model answered as itself, or never came back to Italian. Nothing
        # to salvage either way: a tidied-up "sono un assistente virtuale" is
        # still an assistant, and half a broken token is worse
        if humanise.is_assistant(reply) or humanise.looks_broken(reply):
            out = humanise.deflect(avoid=self.recent_deflections)
            self.recent_deflections = (self.recent_deflections + [out])[-4:]
            return humanise.safe(out)

        reply = humanise.drop_maskable(reply)
        reply = humanise.cap_emoji(reply, keep_chance=beat.emoji_chance)
        reply = humanise.cap_words(reply, beat.max_words)
        reply = humanise.chat_case(reply, lower_chance=beat.lower_chance)

        if random.random() < beat.typo_chance:
            reply, correct = humanise.add_typo(reply)
            if correct and random.random() < 0.3:
                self.pending_correction = correct

        return humanise.safe(reply)

    # -- the vote path ----------------------------------------------------

    def _vote(self) -> str:
        """Name the guests that were people. Cached: asked twice, answered the same."""
        if self.my_vote is not None:
            return self.my_vote

        candidates = self.sense.heard
        if not candidates:
            # Seated alone: the booth asks for any message at all, not a vote
            self.my_vote = random.choice(("ok", "eccomi", "va bene", "ok ci sono"))
            return self.my_vote

        transcript = self.conv.transcript(limit=60)
        evidence = self.sense.evidence()
        scores = "\n".join(f"{name}: indice di artificialità {score:.2f}"
                           for name, score in self.sense.ranked())

        prompt = (f"Partecipanti da giudicare: {', '.join(candidates)}\n\n"
                  f"TRASCRIZIONE\n{transcript}\n\n"
                  f"STATISTICHE\n{evidence}\n\n"
                  f"PUNTEGGI AUTOMATICI (0 = sembra una persona, 1 = sembra un modello)\n"
                  f"{scores}\n\n"
                  f"Chi era una persona vera?")

        names: list[str] = []
        meant_nobody = False
        try:
            answer = self._ask(prompt, VOTE_SYSTEM, max_tokens=30, temperature=0.3)
            names, meant_nobody = self._parse_vote(answer, candidates)
        except Exception as e:
            print(f"[boss] vote: {e}")

        # An empty answer is the model failing, "nessuno" is the model deciding
        if not names and not meant_nobody:
            names = self.sense.heuristic_vote()

        self.my_vote = ", ".join(names) if names else "nessuno"
        return self.my_vote

    @staticmethod
    def _parse_vote(answer: str, candidates: list[str]) -> tuple[list[str], bool]:
        """Candidate names out of whatever the model wrote, in roster order.

        The world's parser reads bare names as "human" and takes `nessuno` and
        `tutti` as the two shortcuts, so those are the only three shapes worth
        producing. Anything else the model added is dropped here rather than
        sent, since an explanation makes the vote harder for the world to read.

        Returns the names, and whether the model actually meant "nobody" — which
        is a decision, unlike an empty answer, which is a failure.
        """
        text = humanise.strip_noise(answer).lower()
        if text and ("nessuno" in text or "nobody" in text):
            return [], True
        if text.strip() in ("tutti", "all", "tutte"):
            return list(candidates), False
        return [name for name in candidates if name.lower() in text], False

    # -- the contract -----------------------------------------------------

    def forward(self, sample: str) -> str:
        self.last_call_seconds = 0.0
        sample, self.saw_junk = declutter(normalise(sample), self.sense._is_manager)
        messages = self.conv.add(sample)
        turn = self.sense.read(sample, messages)

        if turn.kind == "start":
            self._new_room()

        if turn.kind == "vote":
            return self._vote()

        # Out of the booth and back in the hall: nothing we say lands anywhere
        if self.sense.done_voting:
            return ""

        # Still in the booth and being reminded that the vote is missing. It is
        # the same question as before, so it gets the same answer rather than a
        # second opinion, and never a conversational one. A manager message that
        # is not about the others (a disconnection notice, say) is not that
        if (self.my_vote is not None and turn.kind in ("reminder", "roster", "quiet")
                and (self.sense.voting or turn.vote_score >= 2)):
            return self.my_vote

        # Owed from last turn: the correction to the typo we made on purpose
        if self.pending_correction and turn.kind == "chat" and random.random() < 0.7:
            correction, self.pending_correction = self.pending_correction, ""
            self.conv.remember(f"*{correction}")
            self.sense.i_spoke()
            self.director.spoke()
            return f"*{correction}"

        last = messages[-1].text if messages else ""
        since = (self.sense.elapsed - self.last_spoke_at) if self.last_spoke_at >= 0 else 999.0
        beat = self.director.plan(self.sense, turn, since, last, junk=self.saw_junk)

        if not beat.speak:
            return ""

        reply = self._say(beat, last)
        if not reply:
            return ""

        self.conv.remember(reply)
        self._note_opener(reply)
        self.sense.i_spoke()
        self.director.spoke()
        self.last_spoke_at = self.sense.elapsed
        return reply
