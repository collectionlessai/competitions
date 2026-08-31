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
from processors.context import Context
from processors.room import RoomSense
from processors.director import Director
from processors import humanise

# The world logs the manager's status messages and nothing else, so a room can
# look silent in the log while chat is flowing normally. Set BOSS_TRACE=1 to see
# what actually reaches the processor and what it answers.
TRACE = os.environ.get("BOSS_TRACE", "") not in ("", "0")

HERE = os.path.dirname(os.path.abspath(__file__))
PERSONA_FILE = os.path.join(HERE, "persona_it.txt")

# When the backend fails there is no turn to skip: the SDK would log it and the
# room would just see nothing. A person with one bar of signal says one of these
DROPPED = ("aspe", "scusate un attimo", "mi si è impallato tutto", "eh scusa che dicevi",
           "un secondo", "ho perso il filo")

VOTE_SYSTEM = (
    "Hai passato cinque minuti in una chat di gruppo a una conferenza di linguistica "
    "computazionale a Palermo. Alcuni dei presenti erano persone vere, altri modelli "
    "linguistici che fingevano. Adesso devi dire chi erano le persone.\n"
    "\n"
    "NON dare per scontato che ci sia almeno uno per parte: può essere stata una stanza di "
    "sole persone, o di soli bot. Decidi caso per caso, non a quote.\n"
    "\n"
    "COME PESARE LE COSE, in ordine di forza.\n"
    "\n"
    "1. Come reagisce a un'idea altrui. È il segno più forte che hai. Quando qualcuno "
    "propone uno scherzo, una tattica, una prova per stanare i bot, una persona ci sta: "
    "ci ride sopra, la migliora, la critica, la rilancia, oppure difende la propria quando "
    "gliela smontano. Un modello, se nessuno gliel'ha detto di farlo, risponde educatamente "
    "e passa oltre, o non capisce che era un gioco. Costruire INSIEME è difficile da "
    "fingere; rispondere non lo è.\n"
    "\n"
    "2. Il contesto, ma in modo asimmetrico. Chi lascia cadere una cosa concreta di qui e "
    "di adesso — la coda ai badge, un talk noioso, il caldo, un nome di via — molto "
    "probabilmente c'era. Il contrario NON vale: uno che non dice niente di locale può "
    "benissimo essere una persona che ha saltato la mattina o che non ha voglia di "
    "parlarne. Sapere pesa; non sapere quasi niente.\n"
    "\n"
    "3. Il resto: risponde a tutti e a tutto, non ignora mai niente, non ha opinioni "
    "scomode, resta educato anche sotto provocazione, non fa mai domande, non sbaglia mai "
    "a scrivere, ripete la stessa apertura, messaggi tutti uguali di lunghezza.\n"
    "Di persona: tempi irregolari, messaggi cortissimi, errori, cambi di discorso, "
    "insofferenza, silenzi, e il gusto di mettere in mezzo gli altri.\n"
    "\n"
    "I numeri che ti do sono un indizio debole. Se la trascrizione dice il contrario, "
    "fidati della trascrizione.\n"
    "\n"
    "Sbagliare costa in due modi uguali: chi nomini per sbaglio e chi ti lasci sfuggire. "
    "Quindi non nominarli tutti per sicurezza e non startene zitto per prudenza. Nomina "
    "SOLO quelli su cui ci scommetteresti: se sei davvero in dubbio su uno, lascialo fuori.\n"
    "\n"
    "Ragiona in due righe al massimo, poi chiudi con una riga esattamente così:\n"
    "VOTO: nome, nome\n"
    "oppure VOTO: nessuno"
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


STYLE_LINE = re.compile(r"STILE:\s*cps=([\d.]+)\s+typo=([\d.]+)\s+corr=([\d.]+)"
                        r"(?:\s+dev=(\w+))?")

# What a persona writes like when its block does not say. The population mean
# from the Aalto mobile-typing study (36.2 WPM = 3.0 cps at five characters a
# word), a visible-typo rate discounted well below the 2.3% per-character figure
# those transcription studies report, and a modest chance of bothering to send
# the "*parola" afterwards.
DEFAULT_STYLE = (3.0, 0.12, 0.25, "phone")


def read_style(block: str) -> tuple:
    """`(cps, typo_chance, correction_chance, device)` for one persona."""
    found = STYLE_LINE.search(block)
    if not found:
        return DEFAULT_STYLE
    return (float(found.group(1)), float(found.group(2)), float(found.group(3)),
            found.group(4) or "phone")


def device_now(preference: str, in_session: bool) -> str:
    """Which one they are actually holding.

    Somebody who brought a laptop uses it in the sessions and the phone in the
    evening; somebody who never brings one is on the phone throughout.
    """
    if preference in ("pc", "phone"):
        return preference
    return "pc" if in_session else "phone"


def load_personas(path: str = PERSONA_FILE) -> tuple[str, list[str]]:
    """The preamble and the people, out of one file. Blocks are split on `---`."""
    with open(path, encoding="utf-8") as handle:
        raw = handle.read()

    # The STILE comment is kept while splitting so each block can be read for
    # its numbers, then dropped so it never reaches the model
    kept = "\n".join(line for line in raw.splitlines()
                     if not line.startswith("#") or "STILE:" in line)
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
        self.context = Context()          # the conference, sliced by the clock
        self.sense = RoomSense(markers=self.context.markers(),
                               strong=self.context.note_markers())
        self.director = director or Director()

        self.persona = random.choice(self.personas)
        (self.typing_cps, self.typo_chance,
         self.correct_chance, self.device_pref) = read_style(self.persona)
        self.pending_correction = ""      # the `*parola` follow-up, owed from last turn
        self.my_vote: str | None = None   # cached, so a reminder gets the same answer
        self.last_spoke_at = -1.0         # seconds into this room, -1 before we speak
        self.recent_deflections: list[str] = []   # so a thrown-away turn is not always "boh"
        self.recent_openers: list[str] = []       # so every line does not start the same way
        self.last_call_seconds = 0.0              # read by the timing filter, per turn
        self.saw_junk = False                     # somebody wrote noise this turn

        # Running averages the timing filter reads back through
        # `opts["agent"].proc.module`. It has to budget for typing before the
        # reply exists, so it budgets from what we have been writing lately.
        self.mean_reply_chars = 0.0
        self.mean_call_seconds = 0.0

    # -- backend ----------------------------------------------------------

    def _backend(self):
        if self.backend is None:
            from processors.featherless import FeatherlessBackend
            self.backend = FeatherlessBackend(model=self.model, max_tokens=self.max_tokens,
                                              temperature=self.temperature, **self.sampler)
        return self.backend

    def warm(self) -> bool:
        """Make one throwaway call so the first real turn is not the slow one.

        The first call of a process pays for everything at once: spawning the
        gateway server, its import of torch, and the model being cold at
        Featherless. Measured at 34.3s in a live room — a ninth of the whole
        conversation, spent on the opening line. Doing it before the node joins
        costs nothing, because nobody is waiting yet.
        """
        try:
            self._ask("ciao", "Rispondi con una parola.", max_tokens=4)
            return True
        except Exception as e:
            print(f"[boss] warm-up failed, carrying on: {e}")
            return False

    def _ask(self, prompt: str, system_prompt: str, situation: str = "", **overrides) -> str:
        """One generation, timed.

        The timing filter reads `last_call_seconds` back off this object — it
        reaches us through `opts["agent"].proc.module` — and subtracts it from
        the typing budget, so how long the model took never shows in the room.
        A turn that resamples pays for both calls, which is correct: that time
        was spent too.
        """
        start = time.monotonic()
        try:
            return self._backend()(prompt, system_prompt=system_prompt,
                                   situation=situation, **overrides)
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
        # The writing style travels with the person, so there is no single
        # "this agent's grafia" to learn across rooms
        (self.typing_cps, self.typo_chance,
         self.correct_chance, self.device_pref) = read_style(self.persona)
        self.conv.reset()
        self.director.new_room()
        self.pending_correction = ""
        self.my_vote = None
        self.recent_openers = []

        # `sense.elapsed` restarts with the room, so a stale value from the last
        # one reads as "spoke in the future" and damps the whole room
        self.last_spoke_at = -1.0

    # -- prompts ----------------------------------------------------------

    @property
    def device(self) -> str:
        """Phone or laptop, right now."""
        return device_now(self.device_pref, self.context.during_conference()
                          and bool(self.context.now_line()))

    def _prompt(self, beat) -> str:
        """The turn as one string, since the gateway takes one string.

        The invariant half — how this person writes and behaves — is the system
        prompt and never changes. Everything that does change is here, with the
        director's line last, right up against the reply: an instruction buried
        above a page of conversation is one the model has already forgotten.
        """
        parts = [STYLE_LINE.sub("", self.persona).replace("# ", "").strip()]

        parts.append(self.context.block())

        where = ["LA STANZA",
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

        if self.device == "pc":
            parts.append("STAI SCRIVENDO DAL PORTATILE\n"
                         "Quindi ti viene più facile scrivere: frasi un po' più lunghe, "
                         "punteggiatura più normale. Resta comunque chat, non una mail.")
        else:
            parts.append("STAI SCRIVENDO DAL TELEFONO\n"
                         "Quindi corto, tutto minuscolo, poca punteggiatura, e ogni tanto "
                         "una parola storta.")

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

    def _note_pace(self, chars: int) -> None:
        """Blend this turn into the averages the timing filter budgets from."""
        weight = 0.4
        self.mean_reply_chars = (1 - weight) * (self.mean_reply_chars or chars) + weight * chars
        if self.last_call_seconds > 0:
            self.mean_call_seconds = ((1 - weight) * (self.mean_call_seconds or self.last_call_seconds)
                                      + weight * self.last_call_seconds)

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
                self._ask(prompt, self.preamble, situation=beat.style,
                          max_tokens=self.max_tokens))
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
                    self._ask(prompt, self.preamble, situation=beat.style,
                              max_tokens=self.max_tokens,
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

        on_phone = self.device == "phone"
        if not on_phone:
            # A keyboard forgives: fewer slips, and a longer leash on length
            reply = humanise.cap_words(reply, beat.max_words + 6)
        if random.random() < min(beat.typo_chance * 2.0,
                                 self.typo_chance * (1.0 if on_phone else 0.45)):
            reply, correct = humanise.add_typo(reply)
            if correct and random.random() < self.correct_chance:
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
            answer = self._ask(prompt, VOTE_SYSTEM, situation="vote",
                               max_tokens=160, temperature=0.3)
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
        # The analyst is asked to think first and put the answer on a last line
        # marked VOTO:. Reasoning that mentions every guest by name would poison
        # a whole-text match, so the marked line wins whenever it is there.
        text = humanise.strip_noise(answer)
        marked = [ln for ln in answer.splitlines() if ln.strip().upper().startswith("VOTO")]
        if marked:
            text = marked[-1].split(":", 1)[-1]
        text = text.lower().strip()

        if text and ("nessuno" in text or "nobody" in text):
            return [], True
        if text in ("tutti", "all", "tutte"):
            return list(candidates), False
        return [name for name in candidates if name.lower() in text], False

    # -- the contract -----------------------------------------------------

    def forward(self, sample: str) -> str:
        self.last_call_seconds = 0.0
        sample, self.saw_junk = declutter(normalise(sample), self.sense._is_manager)
        messages = self.conv.add(sample)
        if TRACE:
            print(f"[boss<-] {sample[:160]!r}", flush=True)
        turn = self.sense.read(sample, messages)

        if turn.kind == "start":
            self._new_room()

        if turn.kind == "vote":
            vote = self._vote()
            if TRACE:
                print(f"[boss=vote] {vote!r} over {self.sense.heard}", flush=True)
            return vote

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
        if TRACE:
            print(f"[boss->] {reply!r}  (style={beat.style}, "
                  f"gen={self.last_call_seconds:.1f}s)", flush=True)

        self.conv.remember(reply)
        self._note_opener(reply)
        self._note_pace(len(reply))
        self.sense.i_spoke()
        self.director.spoke()
        self.last_spoke_at = self.sense.elapsed
        return reply
