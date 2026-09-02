"""Keep a local conversation across processor turns.

Each turn your processor gets one string with whatever arrived since the
previous turn. The guest role selects which world events to forward, removes
their internal routing tags, then separates a backlog with ASCII Record
Separator (``\x1e``). Newlines cannot mark boundaries because one delivered
event may itself span several lines:

    **Ada:** ciao a tutti,
    giornata lunga

The world does not echo your replies. Add incoming samples with add(), then save
each outgoing reply with remember():

    conv = Conversation(keep=80)

    def forward(self, sample):
        conv.add(sample)
        reply = my_model(conv.as_messages(system=persona))
        conv.remember(reply)
        return reply

This conversation manager is an optional starter-kit implementation, not a
competition rule. Competitors may change, replace or omit it; a processor only
has to honour its public ``str -> str`` contract. The demo inputs under
``prompts/`` may also be used to build a manager that deliberately exploits
their wording or structure, accepting that those details may change between
runs.
"""

import re
from collections import namedtuple


EVENT_SEPARATOR = "\x1e"
DISPLAY_EVENT_SEPARATOR = "␞"
SPEAKER = r"^\*\*(.+?):\*\*\s?(.*)$"
RESET_PHRASES = (
    "nuova conversazione",
    "cancella contesto",
    "inizia una nuova chat",
    "new conversation",
    "clear context",
    "start a new chat",
)


def reset_on_phrase(event: str) -> bool:
    """Match the small Italian and English reset vocabulary."""
    event = event.casefold()
    return any(phrase in event for phrase in RESET_PHRASES)


DEFAULT_RESET_RULES = (reset_on_phrase,)

# speaker: sender name, or "" for an event with no name
# text:    event body with surrounding whitespace removed
# mine:    True when remember() stored the reply
Message = namedtuple("Message", "speaker text mine")


class Conversation:
    """Keep one fixed first message and a rotating tail of `keep - 1` messages.

    This retention policy is the starter kit's choice. It is not prescribed by
    the world, and competitors are free to implement conversation state in a
    different way, including by exploiting the demo inputs under ``prompts/``.

    Args:
        keep: Total number of messages to retain, or 0 for no limit. The first
            message is fixed; later messages rotate through the other slots.
        speaker_pattern: Pattern with speaker and text groups, applied to each
            event. Unmatched events keep their full text with an empty speaker.
            Supply another pattern for worlds with a different format.
        me: Label used for local replies in the transcript.
        reset_rules: Callables that receive one raw event and return True when
            the rotating tail should be cleared. By default, common Italian and
            English requests to start a new conversation or clear context match.

    A reset rule can be a simple heuristic, a regular-expression wrapper or a
    neural classifier. Conversation does not know which world produced the
    event. ``last_input`` and ``last_output`` expose the most recent completed
    processor turn to policies or other cooperating components.
    """

    def __init__(self, keep: int = 80, speaker_pattern: str = SPEAKER, me: str = "Io",
                 reset_rules=DEFAULT_RESET_RULES):
        self.keep = keep
        self.pattern = re.compile(speaker_pattern, re.S)
        self.me = me
        self.reset_rules = tuple(reset_rules)
        self.history: list[Message] = []
        self.last_input = ""
        self.last_output = ""

    def reset(self) -> None:
        """Clear the rotating slots while preserving the first message."""
        self.history[:] = self.history[:1]
        self.last_input = ""
        self.last_output = ""

    def add(self, sample: str) -> list[Message]:
        """Store one processor input and return its delivered events.

        Only ``EVENT_SEPARATOR`` divides events, so internal newlines remain in
        the text. Routing tags have already been removed by the guest role. If
        the speaker pattern does not match, the whole event is stored with an
        empty speaker. Empty messages are omitted from the history.
        """
        new = []
        for event in sample.split(EVENT_SEPARATOR):
            event = event.strip()
            if not event:
                continue

            # Clear the rotating tail first; the trigger is stored below.
            if any(rule(event) for rule in self.reset_rules):
                self.reset()

            match = self.pattern.match(event)
            if match:
                speaker, text = match.group(1).strip(), match.group(2).strip()
            else:
                speaker, text = "", event   # Preserve the full unmatched event.
            message = Message(speaker=speaker, text=text, mine=False)
            new.append(message)
            self._store(message)
        self.last_input = sample
        return new

    def remember(self, text: str) -> None:
        """Store a local reply that the world will not echo through add()."""
        self.last_output = text.strip()
        self._store(Message(speaker="", text=self.last_output, mine=True))

    def discard_last_output(self) -> None:
        """Forget the latest local reply when a policy decides not to send it."""
        if self.history and self.history[-1].mine:
            self.history.pop()
        self.last_output = ""

    def _store(self, message: Message) -> None:
        if not message.text:
            return

        # The first stored message is the fixed context anchor.
        if not self.history:
            self.history.append(message)
            return

        # With one slot, only the fixed first message fits.
        if self.keep == 1:
            return

        # Rotate the oldest tail message when all k slots are occupied.
        if self.keep and len(self.history) >= self.keep:
            self.history.pop(1)
        self.history.append(message)

    def last_message(self, mine: bool = False) -> Message | None:
        """Return the latest remote message, or a local one when mine=True."""
        for message in reversed(self.history):
            if message.mine == mine:
                return message
        return None

    def transcript(self, limit: int | None = None) -> str:
        """Render up to `limit` recent messages as `Speaker: text` entries.

        Local replies use `me`, and events without a sender use `?`. A limit
        of ``None`` or 0 includes the entire history. A positive limit still
        includes the fixed first message.
        """
        if not limit or len(self.history) <= limit:
            messages = self.history
        elif limit == 1:
            messages = self.history[:1]
        else:
            messages = self.history[:1] + self.history[-(limit - 1):]
        return "\n".join(f"{self.me if m.mine else (m.speaker or '?')}: {m.text}"
                         for m in messages)

    def as_messages(self, system: str = "", nudge: str = "") -> list[dict]:
        """Render the transcript as one neutral user message.

        ``mine`` selects only the local speaker label. It does not imply an API
        ``assistant`` role. Competitors who want role-based turns can implement
        that mapping in their own conversation manager.
        """
        out: list[dict] = []
        if system:
            out.append({"role": "system", "content": system})

        content = self.transcript()
        if nudge:
            content = f"{content}\n{nudge}" if content else nudge
        out.append({"role": "user", "content": content})
        return out
