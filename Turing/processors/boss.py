"""The entry: a persona, a director, and a model that never sees either.

Four files, one job each, and this one is the glue.

    persona_it.txt   who is talking          (a different person every room)
    director.py      what kind of turn this is
    humanise.py      what the answer looks like once it exists
    room.py          who is at the table, what was asked, and who is a machine

The model is a plain backend with a `complete(messages)` method, so swapping
Featherless for anything else is one constructor argument, and the offline check
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
        backend: anything with `complete(messages, **overrides) -> str`. Built as
            a `Featherless` on first use when not given, which is also when the
            API key is first required.
        persona_file: the pool to draw a person from at the start of each room.
        keep: how many messages of history to hold, passed to `Conversation`.
        director: a configured `Director`, or None for the defaults.
        max_tokens: hard ceiling on the reply, before `humanise` trims it further.
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
        self.last_spoke_at = 0.0
        self.turns_since_spoke = 99

    # -- backend ----------------------------------------------------------

    def _backend(self):
        if self.backend is None:
            from processors.featherless import Featherless
            self.backend = Featherless(model=self.model, max_tokens=self.max_tokens,
                                       temperature=self.temperature, **self.sampler)
        return self.backend

    def _ask(self, messages: list[dict], **overrides) -> str:
        return self._backend().complete(messages, **overrides)

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
        self.turns_since_spoke = 99

    # -- prompts ----------------------------------------------------------

    def _system(self, beat) -> str:
        parts = [self.preamble, self.persona]

        where = ["DOVE SEI ADESSO",
                 "Sei in una chat con persone che non conosci."]
        if self.sense.my_name:
            where.append(f"In questa chat ti chiamano {self.sense.my_name}, "
                         f"e usi quel nome se te lo chiedono.")
        others = self.sense.others
        if others:
            where.append(f"Gli altri sono {', '.join(others)}.")
        parts.append("\n".join(where))

        if beat.nudge:
            parts.append(f"QUESTO MESSAGGIO\n{beat.nudge}")
        return "\n\n".join(parts)

    # -- the chat path ----------------------------------------------------

    def _say(self, beat, last_text: str) -> str:
        messages = self.conv.as_messages(system=self._system(beat),
                                         nudge="(tocca a te, dì qualcosa di tuo)")
        try:
            reply = self._ask(messages, max_tokens=self.max_tokens)
        except Exception as e:
            print(f"[boss] {e}")
            return random.choice(DROPPED)

        reply = humanise.strip_noise(reply)
        if not reply:
            return ""

        # The model answered as itself. Nothing to salvage: a tidied-up
        # "sono un assistente virtuale" is still an assistant
        if humanise.is_assistant(reply):
            reply = humanise.deflect()
            return humanise.safe(reply)

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
            answer = self._ask([{"role": "system", "content": VOTE_SYSTEM},
                                {"role": "user", "content": prompt}],
                               max_tokens=30, temperature=0.3)
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
        since = self.sense.elapsed - self.last_spoke_at if self.last_spoke_at else 999.0
        beat = self.director.plan(self.sense, turn, since, last)

        if not beat.speak:
            self.turns_since_spoke += 1
            return ""

        reply = self._say(beat, last)
        if not reply:
            return ""

        self.conv.remember(reply)
        self.sense.i_spoke()
        self.director.spoke()
        self.last_spoke_at = self.sense.elapsed
        self.turns_since_spoke = 0
        return reply
