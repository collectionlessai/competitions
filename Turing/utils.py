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
"""

import re
from collections import namedtuple


EVENT_SEPARATOR = "\x1e"
DISPLAY_EVENT_SEPARATOR = "␞"
SPEAKER = r"^\*\*(.+?):\*\*\s?(.*)$"

# speaker: sender name, or "" for an event with no name
# text:    event body with surrounding whitespace removed
# mine:    True when remember() stored the reply
Message = namedtuple("Message", "speaker text mine")


class Conversation:
    """Store messages in arrival order, optionally capped by `keep`.

    Args:
        keep: Maximum number of messages to retain, or 0 for no limit. Once the
            cap is reached, new messages replace the oldest ones. The limit
            counts messages rather than tokens.
        speaker_pattern: Pattern with speaker and text groups, applied to each
            event. Unmatched events keep their full text with an empty speaker.
            Supply another pattern for worlds with a different format.
        me: Label for local replies in transcript(), replaced by the
            `assistant` role inside as_messages().

    The world removes internal tags such as ``[START_MSG]`` before the processor
    sees a sample. This class deliberately does not infer them by matching prompt
    wording or world-specific structures. A competitor could implement such a
    heuristic and call reset(), but the starter kit does not exploit prompts.
    """

    def __init__(self, keep: int = 80, speaker_pattern: str = SPEAKER, me: str = "io"):
        self.keep = keep
        self.pattern = re.compile(speaker_pattern, re.S)
        self.me = me
        self.reset()

    def reset(self) -> None:
        """Clear the conversation history and known speakers."""
        self.history: list[Message] = []
        self.speakers: list[str] = []   # in the order they first spoke

    def add(self, sample: str) -> list[Message]:
        """Store one processor input and return its delivered events.

        Only ``EVENT_SEPARATOR`` divides events, so internal newlines remain in
        the text. Routing tags have already been removed by the guest role. If
        the speaker pattern does not match, the whole event is stored with an
        empty speaker. A named event with no body still registers that speaker,
        although empty messages are omitted from the history.
        """
        new = []
        for event in sample.split(EVENT_SEPARATOR):
            event = event.strip()
            if not event:
                continue
            match = self.pattern.match(event)
            if match:
                speaker, text = match.group(1).strip(), match.group(2).strip()
            else:
                speaker, text = "", event   # Preserve the full unmatched event.
            message = Message(speaker=speaker, text=text, mine=False)
            new.append(message)
            if message.speaker and message.speaker not in self.speakers:
                self.speakers.append(message.speaker)
            self._store(message)
        return new

    def remember(self, text: str) -> None:
        """Store a local reply that the world will not echo through add()."""
        self._store(Message(speaker="", text=text.strip(), mine=True))

    def _store(self, message: Message) -> None:
        if not message.text:
            return
        self.history.append(message)
        if self.keep and len(self.history) > self.keep:
            del self.history[:-self.keep]

    def last_message(self, mine: bool = False) -> Message | None:
        """Return the latest remote message, or a local one when mine=True."""
        for message in reversed(self.history):
            if message.mine == mine:
                return message
        return None

    def transcript(self, limit: int | None = None) -> str:
        """Render up to `limit` recent messages as `Speaker: text` entries.

        Local replies use `me`, and events without a sender use `?`. A limit
        of ``None`` or 0 includes the entire history.
        """
        messages = self.history[-limit:] if limit else self.history
        return "\n".join(f"{self.me if m.mine else (m.speaker or '?')}: {m.text}"
                         for m in messages)

    def as_messages(self, system: str = "", nudge: str = "") -> list[dict]:
        """Render history as chat messages, using `assistant` for local replies.

        Remote events use the `user` role but retain the speaker name in their
        text, so the model can distinguish several guests inside one role.
        Consecutive turns with the same role are merged. If a local reply is
        last, the list ends with `nudge`, or with `(tocca a te)` when no nudge
        is supplied.
        """
        out: list[dict] = []
        if system:
            out.append({"role": "system", "content": system})

        for message in self.history:
            role = "assistant" if message.mine else "user"
            text = message.text if message.mine else f"{message.speaker or '?'}: {message.text}"
            if out and out[-1]["role"] == role and role != "system":
                out[-1]["content"] += "\n" + text
            else:
                out.append({"role": role, "content": text})

        if not out or out[-1]["role"] != "user":
            out.append({"role": "user", "content": nudge or "(tocca a te)"})
        return out
