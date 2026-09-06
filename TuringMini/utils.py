"""Keeps the conversation.

Each turn your processor gets one string with whatever arrived since the
previous turn, one message per line:

    **Ada:** ciao a tutti, giornata lunga

Your own replies are not in there and never will be, so the two calls that keep
the history whole are add() on the way in and remember() on the way out:

    conv = Conversation(keep=80)

    def forward(self, sample):
        conv.add(sample)
        reply = my_model(conv.as_messages(system=persona))
        conv.remember(reply)
        return reply
"""

import re
from collections import namedtuple


SPEAKER = r"^\*\*(.+?):\*\*\s?(.*)$"

# speaker: who sent it, "" when the line carries no name
# text:    what they said, whitespace stripped and nothing else touched
# mine:    True for the lines you added with remember()
Message = namedtuple("Message", "speaker text mine")


class Conversation:
    """Messages, oldest first, capped at the last `keep`.

    Args:
        keep: how many messages to hold on to, 0 for all of them. Old ones fall
            off as new ones arrive, so an agent left running for hours keeps a
            bounded history. The cap counts messages, not tokens.
        speaker_pattern: two groups, speaker and text, matched against each line
            of a sample. A line that does not match is kept whole, with no
            speaker, so a world with its own format needs its own pattern here.
        me: how your own lines are labelled in transcript(). as_messages() does
            not use it, since there your lines are `assistant` turns.
    """

    def __init__(self, keep: int = 80, speaker_pattern: str = SPEAKER, me: str = "io"):
        self.keep = keep
        self.pattern = re.compile(speaker_pattern, re.S)
        self.me = me
        self.reset()

    def reset(self) -> None:
        """Drop everything, for when you decide the conversation you were in is over."""
        self.history: list[Message] = []
        self.speakers: list[str] = []   # in the order they first spoke

    def add(self, sample: str) -> list[Message]:
        """Store one processor input and return the messages it contained.

        One message per line. A line the pattern does not match becomes a
        message with no speaker, and a line carrying a name with nothing after
        it registers the speaker without going into the history, since empty
        messages are dropped on the way in.
        """
        new = []
        for line in sample.splitlines():
            line = line.strip()
            if not line:
                continue
            match = self.pattern.match(line)
            if match:
                speaker, text = match.group(1).strip(), match.group(2).strip()
            else:
                speaker, text = "", line   # keep unrecognised lines whole
            new.append(Message(speaker=speaker, text=text, mine=False))

        for message in new:
            if message.speaker and message.speaker not in self.speakers:
                self.speakers.append(message.speaker)
            self._store(message)
        return new

    def remember(self, text: str) -> None:
        """Store a line you sent, since it never arrives back through add()."""
        self._store(Message(speaker="", text=text.strip(), mine=True))

    def _store(self, message: Message) -> None:
        if not message.text:
            return
        self.history.append(message)
        if self.keep and len(self.history) > self.keep:
            del self.history[:-self.keep]

    def last_message(self, mine: bool = False) -> Message | None:
        """The last message somebody else sent, or your own with mine=True, or None."""
        for message in reversed(self.history):
            if message.mine == mine:
                return message
        return None

    def transcript(self, limit: int | None = None) -> str:
        """The last `limit` messages as `Speaker: text` lines, or all of them.

        Your own are labelled with `me`, anything that arrived without a name
        with `?`.
        """
        messages = self.history[-limit:] if limit else self.history
        return "\n".join(f"{self.me if m.mine else (m.speaker or '?')}: {m.text}"
                         for m in messages)

    def as_messages(self, system: str = "", nudge: str = "") -> list[dict]:
        """The history as OpenAI-style chat messages.

        Your lines are `assistant` turns, everybody else's are `user` turns with
        the speaker's name in front, which is how the model can tell three people
        apart inside one role. Runs of the same role collapse into one message,
        and the list always ends on a `user` turn: when your own line is the
        most recent thing in the history, `nudge` goes on the end, or
        `(tocca a te)` if you did not pass one.
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


