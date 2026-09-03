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
import json
import re
import time
import random

import torch

from utils import Conversation
from processors.context import Context
from processors.room import RoomSense
from processors.director import Director
from processors.pad import Pad
from processors.tactics import Playbook
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

# The world throws away a vote about anyone it did not see us receive this many
# messages from (`Config.min_msgs_from_votee` in the hotel's own config), so it
# is also the point below which naming somebody is all cost and no reward.
MIN_MSGS_TO_JUDGE = 3


def only_names(text: str, names) -> bool:
    """Whether an answer is nothing but the roster handed back.

    This failure does not look like a failure downstream: the vote prompt gets a
    section headed COME SI SONO RELAZIONATI containing a list of the very people
    it was supposed to say something about.
    """
    words = [w for w in re.findall(r"[A-Za-zÀ-ÿ]+", text or "") if len(w) > 1]
    if not words:
        return True
    known = {n.lower() for n in names}
    return sum(1 for w in words if w.lower() not in known) <= 1

# A directory to write the room journal into, or "" for off. Debug instrument.
JOURNAL = os.environ.get("BOSS_JOURNAL", "")

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


# The judge's pre-pass, and the part that decides whether any of this works.
#
# The first version asked who behaved oddly. Measured over fourteen rooms of
# known bots, that produced 40 names called human and none correct, because a
# well-built agent behaves oddly in no way at all — being attentive, coherent,
# engaged and on-topic is the *cheapest* thing a language model does, and the
# old prompt reported all four as findings. Worse, it fed them to `VOTE_SYSTEM`
# whose strongest stated criterion is "costruire INSIEME è difficile da
# fingere", so "raccoglieva quello che dicevano gli altri, faceva domande" read
# as exactly the proof the vote had been told to weigh most heavily. The two
# prompts were calibrated against each other and this one was wrong: it could
# not tell collaboration from compliance, so it called compliance collaboration.
#
# So the question is inverted: not "who slipped up" — nobody good slips up — but
# what each guest showed that bears on the difference. The ranking matters more
# than the inversion. **Friction is first**, because it is what a model does
# worst: being bored and saying so, holding an uncomfortable opinion, refusing a
# request, being wrong and not minding. Compliance is cheap; cost is not.
#
# Place-grounding is third and deliberately not decisive, which is a correction
# of a first attempt that made it the whole test. It fails in both directions:
# a model given the programme has it too — this entry is that model — and a
# person who arrived late, is only half reading, or is playing dumb on purpose
# has none. It confirms; it does not decide.
#
# The prompt still receives the conference context, because without it the
# analyst could not see that "Conferenza? Non ne so nulla" was worth noting at
# all. But that observation is filed as *ambiguous*, not as proof.
INTERACTION_SYSTEM = (
    "Sei a una conferenza e hai passato cinque minuti in una chat con altri "
    "partecipanti. Ti do la trascrizione e quello che sai della giornata vera.\n"
    "Per ciascun partecipante scrivi UNA riga: che cosa ha fatto o detto che "
    "aiuta a capire se è una persona o un modello linguistico.\n"
    "\n"
    "PESA COSÌ, dal segnale più forte al più debole.\n"
    "1. Attrito che costa qualcosa. È il segnale migliore che hai, perché è "
    "quello che a un modello riesce peggio: annoiarsi e dirlo, rifiutare di "
    "stare al gioco, tenere un'opinione scomoda anche quando gli altri non sono "
    "d'accordo, ignorare una richiesta seccante, sbagliare e fregarsene, "
    "cambiare discorso perché quello di prima non interessava.\n"
    "2. Come reagisce a uno scherzo o a una provocazione: rilancia e ci gioca, "
    "oppure risponde educatamente e passa oltre.\n"
    "3. Un dettaglio concreto di oggi e di questo posto che combacia con la "
    "giornata vera. ATTENZIONE: da solo vale poco, perché un modello preparato "
    "bene può avere lo stesso il programma della conferenza. Conta come "
    "conferma insieme ai punti 1 e 2, non da solo.\n"
    "\n"
    "NON sono prove che si tratti di una persona, e vanno segnalate come tali "
    "quando le vedi: essere gentile, disponibile, coerente, partecipe, "
    "interessato, fare domande di cortesia, raccogliere quello che dicono gli "
    "altri, restare sempre in argomento, non ignorare mai nessuno. Sono le cose "
    "che a una macchina riescono MEGLIO che a una persona.\n"
    "\n"
    "Due casi ambigui da riportare come ambigui, senza deciderli:\n"
    "- chi dice di non sapere niente della conferenza o non capire di che si "
    "parla: può essere un modello che non sa dove si trova, ma può benissimo "
    "essere una persona distratta, arrivata tardi, o che fa il finto tonto "
    "apposta per far sbagliare gli altri;\n"
    "- chi parla a lungo di una vita che non c'entra niente con l'essere qui: "
    "stessa cosa, può essere fuori contesto o può semplicemente avere altro per "
    "la testa.\n"
    "\n"
    "\n"
    "FORMATO, una riga per persona ed esattamente così:\n"
    "Nome: [PERSONA|MACCHINA|NIENTE] - cosa hai visto, CITANDO le sue parole\n"
    "\n"
    "L'etichetta dice in che direzione punta l'elemento che hai trovato, non chi "
    "è: la decisione la prende un altro. [NIENTE] va usata spesso, ed è la "
    "risposta giusta per chi ha parlato poco o non ha fatto niente di rivelatore.\n"
    "La citazione è OBBLIGATORIA quando scrivi [PERSONA] o [MACCHINA]: riporta fra "
    "virgolette le parole precise da cui lo deduci. Se non riesci a citare niente "
    "di specifico, allora non hai un elemento e la riga è [NIENTE]. Serve a non "
    "riempire le righe con impressioni generiche tipo 'sembra coinvolto', che "
    "vanno bene per chiunque e non decidono niente.\n"
    "NON ricopiare l'elenco dei partecipanti."
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
                        r"(?:\s+early=([\d.]+))?"
                        r"(?:\s+dev=([\w/]+))?")

# What a persona writes like when its block does not say. The population mean
# from the Aalto mobile-typing study (36.2 WPM = 3.0 cps at five characters a
# word), a visible-typo rate discounted well below the 2.3% per-character figure
# those transcription studies report, and a modest chance of bothering to send
# the "*parola" afterwards.
DEFAULT_STYLE = (3.0, 0.12, 0.25, 0.0, "phone")


def read_style(block: str) -> tuple:
    """`(cps, typo_chance, correction_chance, early_send, device)` for one persona."""
    found = STYLE_LINE.search(block)
    if not found:
        return DEFAULT_STYLE
    return (float(found.group(1)), float(found.group(2)), float(found.group(3)),
            float(found.group(4) or 0.0), found.group(5) or "phone")


def device_now(preference: str, in_session: bool) -> str:
    """Which one they are actually holding.

    `pc` or `phone` means always that one. `a/b` means a during sessions and b
    outside them — and which way round is the persona's business, not a rule:
    one person keeps the laptop open in talks and writes from bed at night, the
    next takes notes on paper and only opens it back at the hotel.
    """
    if "/" in preference:
        during, _, outside = preference.partition("/")
        return during if in_session else outside
    return preference if preference in ("pc", "phone") else "phone"


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
        # No `strong` tier any more: it came from `## NOTE`, and hand-editing
        # the context while the competition runs is not allowed.
        self.sense = RoomSense(markers=self.context.markers())
        self.pad = Pad()                  # what we are holding in mind
        self.director = director or Director()

        self.persona = random.choice(self.personas)
        (self.typing_cps, self.typo_chance, self.correct_chance,
         self.early_chance, self.device_pref) = read_style(self.persona)
        self.pending_correction = ""      # the `*parola` follow-up, owed from last turn
        self.pending_tail = ""            # the rest of a message sent too early
        self.next_reply_chars = 0.0       # what the timing filter should budget next
        self.my_vote: str | None = None   # cached, so a reminder gets the same answer
        self.last_spoke_at = -1.0         # seconds into this room, -1 before we speak
        self.recent_deflections: list[str] = []   # so a thrown-away turn is not always "boh"
        self.play = Playbook()          # which game this room is
        self.node_name = os.environ.get("BOSS_NODE", "boss")
        self.room_count = 0
        self.recent_openers: list[str] = []       # so every line does not start the same way
        self.recent_mine: list[str] = []          # our own last few, to not repeat a subject
        self.entry_line = ""                      # a hello, written before the room started
        self.entry_at = 0.0                       # and when it is due
        self.last_greeted = -99.0                 # so hellos are not traded all night
        self.entries = humanise.load_lines(os.path.join(HERE, "openers_it.txt"), "ENTRATA")
        self.hellos = humanise.load_lines(os.path.join(HERE, "openers_it.txt"), "SALUTO")
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
        answer = ""
        try:
            answer = self._backend()(prompt, system_prompt=system_prompt,
                                     situation=situation, **overrides)
            return answer
        finally:
            self.last_call_seconds += time.monotonic() - start
            # Both halves of every call, verbatim. The journal recorded what the
            # agent decided and not what it was asked, which is the half you
            # need when a decision looks wrong: a vote that names a bot is not
            # explicable from the vote alone.
            self._journal("call", situation=situation, system=system_prompt,
                          prompt=prompt, answer=answer,
                          seconds=round(time.monotonic() - start, 1))

    # -- a new room -------------------------------------------------------

    def _empty_conversation(self) -> None:
        """Leave nothing of the last room behind, whichever kit we are sitting on.

        `Conversation.reset()` used to clear the history outright. Upstream
        changed the contract (`changed Conversation class contract`, 2026-09-02):
        the first stored message became a pinned anchor that `reset()` keeps on
        purpose, and the automatic new-room detection was replaced by a list of
        phrases — "nuova conversazione", "clear context" — that a hotel manager
        never says. Under that version, calling `reset()` between rooms would
        carry the *previous* room's opening line into the next one, under a new
        name, which is the one piece of history that must not survive.

        Neither policy is wrong: the file says so itself, that the retention
        policy is the starter kit's choice and not the world's contract. But it
        is the kit's choice, and ours is a different one, so this stops
        borrowing theirs. Clearing the list directly does the same thing on both
        versions and will keep doing it on the next one.
        """
        self.conv.history.clear()
        self.conv.last_input = ""
        self.conv.last_output = ""
        if hasattr(self.conv, "speakers"):      # gone upstream, present here
            self.conv.speakers = []

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
        (self.typing_cps, self.typo_chance, self.correct_chance,
         self.early_chance, self.device_pref) = read_style(self.persona)
        self._empty_conversation()
        self.play.new_room(0.0)
        self.pad.clear()
        self.director.new_room()
        self.pending_correction = ""
        self.pending_tail = ""
        self.next_reply_chars = 0.0
        self.my_vote = None
        self.room_count += 1
        self._journal("room", me=self.sense.my_name,
                      roster=list(self.sense.heard), persona=self.persona[:120])
        self.recent_openers = []
        self.recent_mine = []

        # Say hello without asking anybody's permission. The room is at its
        # emptiest in the first seconds, which is exactly when a model-backed
        # guest is slowest: cold gateway, a long briefing to read, and a typing
        # charge computed for an average-length sentence. Measured, that put the
        # first word about eighteen seconds after the door opened, and a person
        # who walks in and says nothing for eighteen seconds has already
        # answered the question the room is there to ask. So the greeting is
        # picked here, off a list, and owes the model nothing.
        #
        # The spread matters more than the delay. Everyone arriving at 2.0s is
        # its own signature, so this runs from "before I have even read the
        # briefing" to "I was doing something else for a bit".
        self.last_greeted = -99.0
        self.entry_line = random.choice(self.entries) if self.entries else ""
        self.entry_at = random.choice([random.uniform(0.8, 4.0),
                                       random.uniform(4.0, 14.0),
                                       random.uniform(14.0, 35.0)])

        # `sense.elapsed` restarts with the room, so a stale value from the last
        # one reads as "spoke in the future" and damps the whole room
        self.last_spoke_at = -1.0

    # -- prompts ----------------------------------------------------------

    def _journal(self, kind: str, **fields) -> None:
        """Append one event to this room's journal, when journalling is on.

        The trace prints are for watching a run go past; this is for reading it
        afterwards. Everything the agent decided and could not otherwise be
        asked about — what it made of each guest, what it was holding in mind,
        what the analyst said before the vote — exists only inside one process
        for five minutes unless it is written down here.

        Set `BOSS_JOURNAL` to a directory to turn it on. Off by default: it is
        a debugging instrument, not part of the entry.
        """
        if not JOURNAL:
            return
        fields["kind"] = kind
        fields["t"] = round(self.sense.elapsed, 1)
        fields["who"] = self.node_name
        fields["room"] = self.room_count
        try:
            os.makedirs(JOURNAL, exist_ok=True)
            path = os.path.join(JOURNAL, f"{self.node_name}.jsonl")
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(fields, ensure_ascii=False) + "\n")
        except OSError:
            pass

    @property
    def opening(self) -> bool:
        """True until we have said anything in this room, for the timing filter."""
        return self.director.said == 0

    @property
    def room_pace(self) -> float:
        """Seconds between messages, for the timing filter to budget against."""
        return self.sense.pace

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

        # Only the context the room has actually opened. `said` is the recent
        # conversation, and a block whose trigger does not appear in it stays
        # out of the prompt entirely.
        recent = self.conv.transcript(limit=12)
        parts.append(self.context.block(said=recent))

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

        # What this room is for, and how it is being played. The tactic gives
        # the turn a purpose; without one the model writes correct Italian that
        # goes nowhere, which is what the real hotel showed it doing — a
        # question about a talk asked straight after two guests had ignored the
        # same question, because the turn had to be filled and nothing said
        # what for.
        switched = self.play.reconsider(
            self.sense.elapsed, self.sense.still_open(), self.sense.obvious_bots(),
            beat.style in ("dueallavolta", "allea", "sonda"))
        if switched and TRACE:
            print(f"[boss~tattica] passo a {switched}", flush=True)

        tactic = self.play.block()
        if tactic:
            parts.append(tactic)

        state = self.pad.game_state(self.sense.obvious_bots(), self.sense.still_open())
        if state:
            parts.append("COME STA ANDANDO\n" + state)

        pad = self.pad.block()
        if pad:
            parts.append(pad)

        if beat.nudge:
            parts.append(f"QUESTO MESSAGGIO\n{beat.nudge}")
        parts.append("Scrivi solo il tuo prossimo messaggio, nient'altro.")
        return "\n\n".join(parts)

    # Something proposed rather than merely said. Kept loose on purpose: it is
    # matched against our OWN line, which the persona wrote, so it has to catch
    # "ho un'idea", "proviamo", "facciamo una cosa" without listing them all.
    PROPOSAL = re.compile(r"\b(?:ho un(?:')?idea|propong|proviamo|facciamo|"
                          r"secondo me dovremmo|senti(?:te)? qua|"
                          r"vi va se|scommetto)", re.IGNORECASE)
    # Something asserted about ourselves that we must not contradict later
    ABOUT_ME = re.compile(r"\b(?:io )?(?:sono|faccio|lavoro|vengo|abito|studio|"
                          r"stamattina ho|ieri ho|oggi ho|ho appena)\b", re.IGNORECASE)

    def _note_pad(self, reply: str, beat) -> None:
        """Write down what this line commits us to."""
        if self.PROPOSAL.search(reply):
            self.pad.add("plan", reply, ttl=100.0)
        elif self.ABOUT_ME.search(reply) and len(reply.split()) >= 4:
            self.pad.add("claim", reply, ttl=240.0)

        # A verdict we voiced about somebody is one we should hold to
        if beat.style in ("smaschera", "accusa") and self.sense.heard:
            worst = self.sense.ranked()[-1][0]
            self.pad.add("read", "secondo te è un bot", about=worst, ttl=180.0)
        elif beat.style == "allea" and self.sense.heard:
            best = self.sense.ranked()[0][0]
            self.pad.add("read", "ti sembra una persona vera", about=best, ttl=180.0)

    # The profiler used to live here: a second model call on quiet turns that
    # wrote one line about each guest. Removed on evidence, not taste.
    #
    # It cost 20.7% of all generation time over fourteen measured rooms — 30
    # calls, 7.3 minutes — and 48% of what it produced was truncated mid-word by
    # a 90-character cap. Formally it only ran on turns we were sitting out, so
    # it never stole a turn of speech; but a cost-4 model holds a concurrency
    # slot while it runs, and it is part of why generations of 62 and 74 seconds
    # were measured on a shared budget.
    #
    # The fatal part was the output. "Socievole, suggerisce cibo, risponde in
    # tempi regolari" is a compliment a language model earns for free, and the
    # vote read those lines as evidence of humanity. There is no wording of
    # "describe what sort of person this seems" that stops rewarding
    # conversational competence, which is the cheapest thing an LLM produces.
    # What replaces it is the analyst asking the opposite question, once, at
    # vote time — see `INTERACTION_SYSTEM`.

    def _journal_pad(self) -> None:
        live = [{"kind": n.kind, "text": n.text, "about": n.about}
                for n in self.pad.notes if n.live]
        if live:
            self._journal("pad", notes=live)

    def _note_pace(self, chars: int) -> None:
        """Blend this turn into the averages the timing filter budgets from."""
        weight = 0.4
        self.mean_reply_chars = (1 - weight) * (self.mean_reply_chars or chars) + weight * chars
        if self.last_call_seconds > 0:
            self.mean_call_seconds = ((1 - weight) * (self.mean_call_seconds or self.last_call_seconds)
                                      + weight * self.last_call_seconds)

    def _note_opener(self, reply: str) -> None:
        """Remember how this line started, so the next few do not start the same.

        Shapes, not words. Measured live, ten of thirteen messages opened by
        addressing somebody by name — `ben,` `cal,` `zon,` — and the guard never
        fired once, because as literal strings those are three different
        openers. They are one habit. A guest whose every line begins "name," is
        as findable as one whose every line begins "non", which is why a model
        was rejected in the bench for exactly this at half the rate.
        """
        words = reply.split()
        if not words:
            return
        first = words[0].lower().strip(",.!?:")
        if first in {n.lower() for n in self.sense.heard} or first == "raga":
            first = "@nome"
        self.recent_openers = (self.recent_openers + [first])[-4:]

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

        # Ration the vocative. Addressing somebody by name is normal; opening
        # every message that way is a signature, and the model will not stop on
        # its own because each instance in isolation is the natural thing to
        # write. A person reaches for a name to disambiguate or to switch who
        # they are talking to — call it one message in three or four.
        #
        # Pressure, not a quota. A hard "never twice running" strips exactly
        # every third message, which is its own detectable rhythm; the chance
        # rises with how much the habit has been indulged lately and leaves the
        # rate ragged.
        used = self.recent_openers[-3:].count("@nome")
        if used and random.random() < (0.55 if used == 1 else 0.9):
            reply = humanise.drop_vocative(reply, self.sense.heard)

        # Already said, and recently. Silence beats saying it a third time —
        # and it costs nothing, because the alternative is a second model call
        # for a turn nobody is waiting on.
        # Somebody asking us directly is not us going on about it, so staying
        # on a subject we raised is only a fault when nobody asked
        asked_us = "?" in last_text and (self.sense.my_name or "").lower() in last_text.lower()
        if not asked_us and humanise.repeats_self(reply, self.recent_mine[-2:]):
            self._journal("dropped", text=reply, why="ripetizione")
            if TRACE:
                print(f"[boss--] ripetizione {reply!r}", flush=True)
            return ""

        # A word nobody typed. Not a style problem — the message has stopped
        # meaning anything, so it goes rather than gets tidied, the same way a
        # broken-character reply does.
        if humanise.has_invented_token(reply):
            self._journal("dropped", text=reply, why="parola inventata")
            if TRACE:
                print(f"[boss--] parola inventata {reply!r}", flush=True)
            return ""

        # Our own name, out of our own mouth. Both the announcement
        # ("ciao qua zon") and the weld onto somebody else's ("zia bob, ...").
        reply = humanise.drop_self_reference(reply, self.sense.my_name, self.sense.heard)

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

        # Occasionally the thumb goes before the sentence does — but only for a
        # persona whose STILE says so, and only on a phone. A tic everybody has
        # is a tic somebody can learn.
        early = self.early_chance if on_phone else 0.0
        if early and not self.pending_correction and random.random() < early:
            reply, tail = humanise.send_too_early(reply)
            if tail:
                self.pending_tail = tail
                # The rest follows at the pace of typing the rest, not at the
                # pace of writing a whole new message
                self.next_reply_chars = float(len(tail))

        if self.pending_correction:
            # Reread, notice, and type "*parola": a short beat, not a full turn
            self.next_reply_chars = float(len(self.pending_correction) + 1)

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

        # Two axes, written out in words. A single number could not say the
        # difference between "not enough to tell" and "the evidence disagrees
        # with itself", and the vote treated both as a middling suspicion.
        rows = []
        for name, (human, bot) in self.sense.readings().items():
            if human < 0.2 and bot < 0.2:
                verdict = "quasi nessun elemento in nessuna delle due direzioni"
            elif human >= 0.4 and bot >= 0.4:
                verdict = "elementi in CONTRASTO, decidi tu guardando la trascrizione"
            elif bot > human:
                verdict = "qualche elemento da macchina"
            elif human > bot:
                verdict = "qualche elemento da persona"
            else:
                verdict = "niente di netto"
            rows.append(f"{name}: persona {human:.2f} / macchina {bot:.2f} — {verdict}")
        scores = "\n".join(rows)

        # Work out what happened between people BEFORE judging it. Two cheap
        # calls inside a 240-second window we were using five seconds of, and
        # the first one is deliberately not allowed to name anybody: separating
        # "what occurred" from "who was a machine" keeps the second question
        # honest instead of letting it rationalise a hunch.
        # Four analyst calls in six came back as nothing but the roster echoed
        # straight back — `Ada, Rye, Mae, Zac`, with no analysis under it. The
        # prompt leads with "Partecipanti da analizzare: ..." and the model
        # finishes that sentence instead of starting its own. A second ask costs
        # one call inside a window we are using seconds of, and an answer that is
        # only names is worse than none: it feeds the vote a section heading with
        # nothing underneath it, which reads as "nothing was found".
        interactions = ""
        try:
            # The day itself, which is what the whole question now turns on. The
            # analyst could not previously tell that "Conferenza? Non ne so
            # nulla" was remarkable, because nothing told it everybody in the
            # room was supposed to be at one.
            # The statistics too. The analyst was judging the words with no idea
            # how fast or how evenly they arrived, while the vote got the
            # numbers and no idea what they meant.
            ask = (f"{self.context.block(said=transcript)}\n\n"
                   f"Partecipanti da analizzare: {', '.join(candidates)}\n\n"
                   f"COME HANNO SCRITTO\n{evidence}\n\n"
                   f"{transcript}")
            interactions = self._ask(ask, INTERACTION_SYSTEM, situation="vote",
                                     max_tokens=260, temperature=0.3)
            if only_names(interactions, candidates):
                interactions = self._ask(ask, INTERACTION_SYSTEM, situation="vote",
                                         max_tokens=200, temperature=0.6)
            if only_names(interactions, candidates):
                print("[boss] l'analista ha reso solo l'elenco, lo ignoro")
                interactions = ""
            self._journal("analyst", text=interactions)
            if TRACE:
                print(f"[boss~interactions] {interactions[:300]}", flush=True)
        except Exception as e:
            print(f"[boss] interactions: {e}")

        # The heading matters. "COME SI SONO RELAZIONATI" invited the vote to
        # read sociability as evidence. "CHI ERA DAVVERO QUI" was worse in the
        # other direction: it made the place the whole test, which a prepared
        # model passes and a distracted person fails. This one names the section
        # for what it holds and leaves the weighing to `VOTE_SYSTEM`.
        prompt = (f"{self.context.block(said=transcript)}\n\n"
                  f"Partecipanti da giudicare: {', '.join(candidates)}\n\n"
                  f"TRASCRIZIONE\n{transcript}\n\n"
                  + (f"COSA HA MOSTRATO CIASCUNO\n{interactions}\n\n"
                     if interactions else "")
                  + f"STATISTICHE\n{evidence}\n\n"
                  f"PUNTEGGI AUTOMATICI (due misure separate, 0 = nessun elemento, "
                  f"1 = elementi forti. Sono un indizio debole: se la trascrizione "
                  f"dice il contrario, fidati della trascrizione)\n"
                  f"{scores}\n\n"
                  "Chi era una persona vera?")

        names: list[str] = []
        meant_nobody = False
        try:
            answer = self._ask(prompt, VOTE_SYSTEM, situation="vote",
                               max_tokens=160, temperature=0.3)
            names, meant_nobody = self._parse_vote(answer, candidates)
        except Exception as e:
            print(f"[boss] vote: {e}")

        # Whatever the analyst concluded, a guest who spent the room emitting
        # `17_green` is not getting a human vote out of us. This is the one
        # place the hard evidence overrules the model rather than informing it,
        # and it only ever removes a name — it never adds one.
        # And nobody gets called a person on the strength of one message. Live,
        # a spammer joined a room near the end, said two things, and the
        # analyst named it the only human present — while the same guest was
        # emitting `50_cat` and `52_green` in the room next door, where four
        # messages would have settled it outright. The world discards any vote
        # about a guest we received fewer than three messages from
        # (`Config.min_msgs_from_votee`), so this costs nothing it could have
        # won, and asymmetry is the point: silence about a stranger is free,
        # calling a stranger human is a precision error.
        # An empty answer is the model failing, "nessuno" is the model deciding
        if not names and not meant_nobody:
            names = self.sense.heuristic_vote()

        # Both guards apply to whatever produced the list, model or fallback.
        # They used to sit above, where the fallback ran after them and slipped
        # past both: live in the competition hotel that voted `Oli, Bob` human on
        # two messages and one, when the numbers said nothing at all about either.
        thin = [n for n in names
                if self.sense.speakers[n].count < MIN_MSGS_TO_JUDGE]
        if thin:
            names = [n for n in names if n not in thin]
            print(f"[boss] vote: {', '.join(thin)} visti troppo poco per dirli persone")

        refused = [n for n in names if self.sense.settled(n)]
        if refused:
            names = [n for n in names if n not in refused]
            print(f"[boss] vote: ignorati {', '.join(refused)}, palesemente bot")

        self.my_vote = ", ".join(names) if names else "nessuno"
        self._journal("vote", vote=self.my_vote, candidates=candidates,
                      scores={n: round(v, 2) for n, v in self.sense.ranked()},
                      settled=self.sense.obvious_bots(),
                      counts={n: self.sense.speakers[n].count for n in candidates},
                      evidence=evidence, raw=answer if "answer" in dir() else "")
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
        for line in turn.lines:
            if line.speaker and not line.mine:
                self._journal("heard", speaker=line.speaker, text=line.text[:400])

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

        # A hello, if one is due and nothing has been said yet. Before the
        # director, before the model, before anything that can decline it.
        # ...unless we already opened our mouth. The scheduled hello exists to
        # cover the silence before the model can produce anything; when the
        # model got there first the silence never happened, and firing anyway
        # introduces us a second time. Seen live: "ciao gente, qui zia" from the
        # model, then "ma quanti siete" from here, then a reflex "ehilà" — three
        # greetings in a row, which is the opposite of the problem this solves.
        if self.director.said:
            self.entry_line = ""

        if self.entry_line and turn.kind != "vote":
            if self.sense.elapsed >= self.entry_at:
                line, self.entry_line = self.entry_line, ""
                self.conv.remember(line)
                self.sense.i_spoke()
                self.director.spoke()
                self.last_spoke_at = self.sense.elapsed
                if TRACE:
                    print(f"[boss->] {line!r}  (entrata, {self.sense.elapsed:.1f}s)", flush=True)
                return line

        # Somebody said hello. This one is answered by reflex and not by the
        # model: the point of a greeting is that it comes back straight away,
        # and anything that has to be generated first cannot.
        greeted_by = [m.speaker for m in turn.lines
                      if m.speaker and not m.mine
                      and not self.sense._is_manager(m.speaker)
                      and not self.sense.settled(m.speaker)
                      and humanise.is_greeting(m.text)]
        # Once per room, and only before we have said anything else. The reflex
        # hello is here to cover the silence at the door; once we are already
        # talking it covers nothing and just adds a second greeting. Live it
        # produced "ciao!" and then "eccomi" inside one room, with nothing in
        # between, because the guard was a 25-second cooldown rather than a
        # once-only.
        if (turn.kind == "chat" and self.hellos and greeted_by
                and self.director.said == 0
                and self.sense.elapsed - self.last_greeted > 25.0):
            self.last_greeted = self.sense.elapsed
            line = random.choice([h for h in self.hellos if h not in self.recent_mine]
                                 or self.hellos)
            self.entry_line = ""          # that was the hello
            self.conv.remember(line)
            self.recent_mine = (self.recent_mine + [line])[-3:]
            self.sense.i_spoke()
            self.director.spoke()
            self.last_spoke_at = self.sense.elapsed
            if TRACE:
                print(f"[boss->] {line!r}  (saluto)", flush=True)
            return line

        # Owed from last turn: the rest of a message the thumb cut short. This
        # one is not optional — a fragment nobody ever finishes is the bug this
        # was modelled on, not the behaviour
        if self.pending_tail:
            tail, self.pending_tail = self.pending_tail, ""
            self.conv.remember(tail)
            self.sense.i_spoke()
            self.director.spoke()
            self.last_spoke_at = self.sense.elapsed
            self.next_reply_chars = 0.0
            if TRACE:
                print(f"[boss->] {tail!r}  (il resto)", flush=True)
            return tail

        # Owed from last turn: the correction to the typo we made on purpose
        if self.pending_correction and turn.kind == "chat" and random.random() < 0.7:
            correction, self.pending_correction = self.pending_correction, ""
            self.conv.remember(f"*{correction}")
            self.sense.i_spoke()
            self.director.spoke()
            return f"*{correction}"

        last = messages[-1].text if messages else ""
        since = (self.sense.elapsed - self.last_spoke_at) if self.last_spoke_at >= 0 else 999.0
        self.director.drive = self.play.drive
        beat = self.director.plan(self.sense, turn, since, last, junk=self.saw_junk)

        if not beat.speak:
            return ""

        reply = self._say(beat, last)
        if not reply:
            return ""

        # Overtaken while we were thinking. The whole waiting budget is spent
        # before the model runs, so generation is the one stretch where the room
        # moves and we cannot see it — normally two or three seconds, but the
        # shared gateway has handed us 70-second calls when another agent on the
        # account is on a 72B at the same time. A reply written seventy seconds
        # ago answers a conversation that has moved on twice over, and arriving
        # late with the wrong subject is a worse tell than saying nothing. The
        # floor still wins: if the room is nearly over and we are short of the
        # messages the vote needs, a stale message beats no message.
        # Only against a room that has been seen to move. A cold gateway made
        # the very first call of a session take 44 seconds and this threw the
        # opening line away — measured against a pace of 12s that was the
        # fallback, not a measurement, in a room where nothing had been said
        # yet. Nothing can overtake an opener.
        stale_after = max(10.0, 3.0 * self.sense.pace)
        short_on_messages = (self.sense.elapsed > 170.0
                             and self.director.said < self.director.min_msgs)
        if (self.sense.pace_known and self.last_call_seconds > stale_after
                and not short_on_messages):
            self.pending_tail = ""
            self.pending_correction = ""
            self.next_reply_chars = 0.0
            self._journal("dropped", text=reply, why="vecchia",
                          gen=round(self.last_call_seconds, 1), pace=round(self.sense.pace, 1))
            if TRACE:
                print(f"[boss--] scartato {reply!r} "
                      f"(gen={self.last_call_seconds:.1f}s, ritmo={self.sense.pace:.1f}s)",
                      flush=True)
            return ""
        if TRACE:
            print(f"[boss->] {reply!r}  (style={beat.style}, "
                  f"gen={self.last_call_seconds:.1f}s)", flush=True)

        self._journal("said", text=reply, style=beat.style, gen=round(self.last_call_seconds, 1),
                      settled=self.sense.obvious_bots(), open=self.sense.still_open(),
                      pace=round(self.sense.pace, 1))
        self.conv.remember(reply)
        self._note_opener(reply)
        self.recent_mine = (self.recent_mine + [reply])[-3:]
        self._note_pace(len(reply))
        self._note_pad(reply, beat)
        self._journal_pad()
        self.sense.i_spoke()
        self.director.spoke()
        self.last_spoke_at = self.sense.elapsed
        return reply
