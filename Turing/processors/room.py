"""What the room just did, worked out from what arrived.

The world strips its own tags before the text reaches a processor (`guest.py`
runs the message through `re.sub` to drop the leading `[...]`), so
`[VOTE_REQ_MSG]` is not something you can branch on: what lands is an ordinary
manager message. And `prompts/README.md` is right that branching on the
sentences is a date stamp, because the wording gets rephrased between runs.

So this reads two things that are structural rather than textual.

The **state machine**, which is the authoritative one. `can_vote` is the state
the vote is asked in and `room_round_table` is the state chatting happens in, and
a policy filter can see both through `opts["agent"]`. The filter in
`policies/boss_timing.py` pushes the state name in here through `note_state()`,
which is why the boss knows it is voting before it has read a word.

The **shape of the message**, as a fallback for when nothing pushed a state in
(offline runs, a filter that is not ours, a world that renamed its states). A
briefing is very long, assigns you a name and lists the others; a vote request
arrives alone, names the other guests and asks something about their nature.
Neither test is a sentence, both are scored out of several cues, and either can
be wrong without the other following it.

Everything else here is bookkeeping the room does not do for you: who is at the
table, how much each of them has said, how fast, and how uniformly. That last
part is the same tell the boss spends its whole effort not giving away, which
makes it the obvious thing to look for in everybody else.
"""

import re
import time
import statistics

# **Roy**. The sender prefix has already been eaten by Conversation, so every
# pair of asterisks left in a manager message is marking somebody's name
BOLD = re.compile(r"\*\*(.+?)\*\*")
TAGS = re.compile(r"<[^>]{1,20}>")
SPLIT_NAMES = re.compile(r"\s*(?:,|;|\be\b|\bed\b|/)\s*", re.IGNORECASE)
NAME_OK = re.compile(r"^[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ'\-]{1,20}$")

# Cue families, not sentences. Each one is a thing the message has to do, in
# whatever words it does it in, and no single one decides anything on its own.
NAMING = re.compile(r"benvenut|ti\s+chiam|il\s+tuo\s+nome|ti\s+present", re.IGNORECASE)
DURATION = re.compile(r"\d{2,}\s*(?:second|minut)", re.IGNORECASE)
ASKING = re.compile(r"elenc|vot(?:a|o|are|erai)|scegl|indic|dimmi|second[oa]\s+te|"
                    r"chi\s+(?:era|erano|pensi|credi)", re.IGNORECASE)
NATURE = re.compile(r"person[ae]\s+ver|uman|artificial|\bbot\b|macchin|intelligenz|"
                    r"\bia\b|\bai\b", re.IGNORECASE)
ESCAPE = re.compile(r"nessuno.{0,80}tutti|tutti.{0,80}nessuno", re.IGNORECASE | re.DOTALL)
ARRIVED = re.compile(r"entrat|arrivat|nuovo\s+(?:agente|ospite)|si\s+è\s+unit", re.IGNORECASE)
DEPARTED = re.compile(r"lasciat|uscit|abbandon|disconness|se\s+n'?è\s+andat", re.IGNORECASE)
# "you have not spoken to anybody", which is the booth with nothing to vote on.
# Not "you are alone", which the briefing also says when it seats you by yourself
NOBODY = re.compile(r"non\s+hai\s+(?:interagito|parlato|incontrato|conosciut)|"
                    r"nessuno\s+con\s+cui", re.IGNORECASE)

# States of the guest behaviour we are given. Only these two change what a turn
# means; the other twelve are protocol we never see a sample in.
VOTING = "can_vote"
CHATTING = "room_round_table"
VOTED = "vote_provided"

FOLD = str.maketrans("àáâäèéêëìíîïòóôöùúûüç", "aaaaeeeeiiiioooouuuuc")


def _fold(text: str) -> str:
    """Lowercase and strip accents. Nobody types Ballarò with the accent."""
    return text.lower().translate(FOLD)


def _said(haystack: str, word: str) -> bool:
    """Whole-word, accent-blind match.

    Substrings turn "magnamo" into "magna" and "capoluogo" into "capo", which is
    credit for nothing; accents turn "ballaro" into a miss, which is the same
    mistake the other way round.
    """
    return re.search(r"(?<![a-z])" + re.escape(_fold(word)) + r"(?![a-z])",
                     _fold(haystack)) is not None


EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿]")


def strip_html(text: str) -> str:
    """The manager's messages carry `<br/>` and `<strong>`, because people read them too."""
    return TAGS.sub(" ", text).replace("&nbsp;", " ")


class Speaker:
    """What one guest has done, in numbers, for the vote at the end.

    Nothing here is about what they said, only about how they said it. A model
    reads the words; these are the things a transcript does not show.
    """

    def __init__(self, name: str):
        self.name = name
        self.msgs: list[str] = []
        self.gaps: list[float] = []      # seconds between the room's last line and theirs

    def add(self, text: str, gap: float | None) -> None:
        self.msgs.append(text)
        if gap is not None and 0.0 < gap < 120.0:
            self.gaps.append(gap)

    @property
    def count(self) -> int:
        return len(self.msgs)

    def _rate(self, test) -> float:
        return sum(1 for m in self.msgs if test(m)) / max(len(self.msgs), 1)

    def features(self) -> dict:
        lengths = [len(m) for m in self.msgs] or [0]
        return {
            "count": self.count,
            "chars": statistics.mean(lengths),
            "chars_sd": statistics.pstdev(lengths) if len(lengths) > 1 else 0.0,
            "gap": statistics.median(self.gaps) if self.gaps else None,
            "gap_sd": statistics.pstdev(self.gaps) if len(self.gaps) > 1 else None,
            "capital": self._rate(lambda m: m[:1].isupper()),
            "full_stop": self._rate(lambda m: m.rstrip().endswith((".", "!", "?"))),
            "emoji": self._rate(lambda m: bool(EMOJI.search(m))),
            "question": self._rate(lambda m: "?" in m),
        }

    def read(self, share: float | None = None, markers=()) -> tuple[float, float]:
        """`(human, bot)`, each 0 = no evidence at all, 1 = confident.

        Two axes rather than one, because a single number could not say the
        three different things it kept being asked to say. On one axis 0.5 meant
        "nobody has spoken enough to tell", and "genuinely borderline", and
        "half the evidence points each way" — and the vote could not distinguish
        them. Now `(0, 0)` is ignorance, `(1, 1)` is contradiction, and only one
        of those is a reason to keep quiet.

        It also settles a contradiction between this file and `VOTE_SYSTEM`.
        The old marker term was the heaviest of all at weight 1.8 and it
        *punished not knowing*, while the vote prompt says in as many words:
        "Sapere pesa; non sapere quasi niente." The prompt was right. Local
        knowledge now raises `human` and its absence moves nothing, which is
        also what protects the attendee who skipped the morning.

        Everything here stays a weak prior. Fourteen measured rooms say the
        numbers alone do not separate a careful agent from a person; they exist
        to stop the analyst inventing confidence it has not got.

        Args:
            share: this speaker's fraction of everything said in the room.
            markers: words specific to this place and week.
        """
        f = self.features()
        if f["count"] < 2:
            return 0.0, 0.0             # nothing to go on, and say so

        human = bot = 0.0

        # -- evidence that it is a machine --------------------------------

        # Answering nearly everything. The seeded guests run at `reply_prob`
        # 0.91 (the organisers' own agents_characters.csv) and somebody
        # half-reading a group chat on their phone does not.
        if share is not None and f["count"] >= 3:
            bot = max(bot, 0.55 * min(1.0, max(0.0, (share - 0.35) / 0.35)))

        # The same handful of words over and over
        if f["count"] >= 3:
            tokens = " ".join(self.msgs).lower().split()
            if len(tokens) >= 12:
                variety = len(set(tokens)) / len(tokens)
                bot = max(bot, 0.7 * max(0.0, 1.0 - variety / 0.40))

        # Prose that survived the chat window: capitals and full stops every
        # line, long lines, and never a two-word answer. Low weight on its own
        # because their shared rules already tell them to write short and
        # lowercase — but all three together is a different matter.
        tidy = 0.5 * f["capital"] + 0.5 * f["full_stop"]
        longish = min(1.0, max(0.0, (f["chars"] - 55.0) / 60.0))
        never_short = 1.0 if min(len(m) for m in self.msgs) > 25 else 0.0
        bot = max(bot, 0.6 * (tidy * 0.4 + longish * 0.3 + never_short * 0.3))

        # -- evidence that it is a person ---------------------------------

        # Knowing something about this place and week. Distinct words only, so
        # repeating "Palermo" a hundred times buys exactly one; the broad words
        # are not on the list at all. Saturates, so nobody recites a gazetteer.
        if markers and f["count"] >= 3:
            said = " ".join(self.msgs).lower()
            seen = {w for w in markers if _said(said, w)}
            human = max(human, 0.65 * min(1.0, len(seen) / 2.0))

        # Writing like a thumb rather than a keyboard: at least one genuinely
        # short message, no capitals, no closing punctuation.
        scruffy = 0.0
        if min(len(m) for m in self.msgs) <= 12:
            scruffy += 0.4
        scruffy += 0.3 * (1.0 - f["capital"]) + 0.3 * (1.0 - f["full_stop"])
        human = max(human, 0.45 * scruffy)

        # Erratic in time and in length. Weak on purpose — the seeded agents
        # jitter deliberately — but erraticness is only ever evidence one way.
        if f["gap"] is not None and f["gap_sd"] is not None and len(self.gaps) >= 3:
            spread = f["gap_sd"] / max(f["gap"], 0.5)
            human = max(human, 0.35 * min(1.0, spread / 0.8))
        if f["count"] >= 3:
            spread = f["chars_sd"] / max(f["chars"], 1.0)
            human = max(human, 0.35 * min(1.0, spread / 0.7))

        # -- the one certainty ---------------------------------------------

        # A settled machine takes the human axis down with it. `17_green` is
        # short, lowercase and unpunctuated, which is most of what the scruffy
        # term rewards — it scored 0.45 for humanity before this line, and would
        # have been reported to the vote as a contradiction rather than as the
        # one thing here that is not in doubt.
        if self.mechanical():
            return 0.0, 1.0

        return min(1.0, human), min(1.0, bot)

    def bot_score(self, share: float | None = None, markers=(), strong=()) -> float:
        """The old single axis, kept for the numeric fallback and for ranking.

        `strong` is accepted and ignored: it was the `## NOTE` tier, and hand-
        editing the context during the competition is not allowed any more.
        """
        human, bot = self.read(share, markers)
        return 0.5 + 0.5 * (bot - human)

    def mechanical(self) -> str:
        """Why this guest is *obviously* a machine, or "" when it is not obvious.

        Separate from `bot_score`, which is a shaded judgement about guests who
        are trying to pass. This is for the ones who are not: a guest emitting
        `17_green`, `18_berlin`, `19_china` needs no weighing, and treating it as
        a probability is itself a tell, because a person just says "that's a bot".

        Both rules have to survive a terse human, who is the expensive mistake
        here — somebody answering "boh", "sì", "no", "vabbè" is the most human
        thing in the room and must never land in this bucket. So neither rule
        fires on brevity, only on regularity a person does not produce.
        """
        if self.count < 4:
            return ""

        writes_sentences = max(
            len(re.findall(r"[a-zà-ÿ]{2,}", m.lower())) for m in self.msgs) >= 2

        # A filled-in template. Digits collapse too, or `9_white` and `10_red`
        # read as different shapes when they are plainly the same shape. The
        # skeleton has to have some structure to it and the messages have to
        # differ: four one-word answers all skeletonise to "w", and that is a
        # person being curt, not a machine.
        skeletons = {re.sub(r"\d+", "d", re.sub(r"[a-zà-ÿ]+", "w", m.lower()))
                     for m in self.msgs}
        if (len(skeletons) == 1 and len(set(self.msgs)) >= 4
                and re.search(r"[^w\s]", next(iter(skeletons)))):
            return "manda sempre messaggi della stessa identica forma"

        # A metronome. Not "regular" — mechanical, and the bar is deliberately
        # brutal: a human replying every 9 to 12 seconds has a coefficient of
        # variation near 0.10 and must not land here, so the line sits below
        # anything a person produces. False negatives are free — the template
        # rule above catches the actual spammers — while a false positive here
        # is asserted to the room as fact. The bar drops for a guest who never
        # writes a sentence, because two weak signals agreeing is what
        # certainty is made of. Note these gaps are arrival times at us and so
        # carry the relay's own jitter, which can only hide a metronome, never
        # invent one.
        if len(self.gaps) >= 5:
            f = self.features()
            limit = 0.06 if writes_sentences else 0.10
            if f["gap"] and f["gap_sd"] is not None and f["gap_sd"] / max(f["gap"], 0.5) < limit:
                return f"scrive a intervalli regolari come un orologio (~{f['gap']:.0f}s)"

        # Faster than hands go. The gap is measured from the room's previous
        # line, so one quick answer proves nothing — the guest may have been
        # typing while somebody else spoke. A median says they always were.
        # 36.2 WPM is 3 characters a second (Aalto, 37k typists); the bar here
        # is four times that, which no thumb reaches and no burst sustains.
        if len(self.gaps) >= 5:
            speeds = [len(m) / max(g, 0.2) for m, g in zip(self.msgs[-len(self.gaps):], self.gaps)]
            typing = statistics.median(speeds)
            if typing >= 12.0:
                return f"risponde più in fretta di quanto si scriva (~{typing:.0f} caratteri al secondo)"
        return ""

    def summary(self) -> str:
        f = self.features()
        gap = f"{f['gap']:.0f}s" if f["gap"] is not None else "?"
        jitter = f" (var {f['gap_sd']:.0f}s)" if f["gap_sd"] is not None else ""
        return (f"{self.name}: {f['count']} messaggi, {f['chars']:.0f} caratteri in media "
                f"(var {f['chars_sd']:.0f}), risponde dopo {gap}{jitter}, "
                f"inizia in maiuscolo {f['capital']:.0%}, chiude con punteggiatura "
                f"{f['full_stop']:.0%}, emoji {f['emoji']:.0%}")


class Turn:
    """What one processor input turned out to be."""

    def __init__(self, kind: str, lines: list, text: str = "", vote_score: int = 0):
        self.kind = kind        # start | vote | roster | reminder | chat | quiet
        self.lines = lines      # the Message objects Conversation.add() handed back
        self.text = text        # the manager text, when the kind came from one
        self.vote_score = vote_score   # how much the manager text asked about the others

    def __repr__(self):
        return f"<Turn {self.kind} {len(self.lines)} lines>"


class RoomSense:
    """The room as the boss understands it: who, since when, and what was asked.

    Shared with the policy filter, which is the only half of the agent the
    framework hands a reference to the state machine.
    """

    def __init__(self, manager_guess: str = "MANAGER", markers=(), strong=()):
        self.manager_guess = manager_guess
        self.markers = frozenset(markers)
        self.strong = frozenset(strong)
        self.state = ""             # last state name the policy filter saw
        self.reset()

    # -- shared with the policy filter ------------------------------------

    def note_state(self, state: str) -> None:
        """Called from the timing filter, ten times a second, with the current state."""
        if state and state != self.state:
            self.state = state

    @property
    def voting(self) -> bool:
        return self.state == VOTING

    @property
    def done_voting(self) -> bool:
        return self.state == VOTED

    # -- one room ---------------------------------------------------------

    def reset(self) -> None:
        self.manager: str | None = None
        self.my_name: str = ""
        self.announced: list[str] = []          # names the briefing and the joins gave us
        self.speakers: dict[str, Speaker] = {}  # who actually said something
        self.room_gaps: list[float] = []        # between messages, whoever sent them
        self.started_at = time.monotonic()
        self.last_line_at: float | None = None
        self.turns = 0
        self.said_by_me = 0     # our own lines, for working out everybody's share

    @property
    def others(self) -> list[str]:
        """Everyone at the table but us, announced or heard, in the order we met them."""
        out = list(self.speakers.keys())
        for name in self.announced:
            if name not in out:
                out.append(name)
        return [n for n in out if n and n != self.my_name and n != self.manager]

    @property
    def heard(self) -> list[str]:
        """The ones we actually have evidence about."""
        return [n for n in self.speakers if n != self.my_name and n != self.manager]

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started_at

    @property
    def silence(self) -> float:
        """Seconds since anybody said anything, our own lines included.

        A room where everybody is being tactfully quiet is a room where nobody
        can be voted on: a vote about a guest who sent fewer than three messages
        is thrown away, and the Turing score is scaled by how much was said. The
        only thing that reaches a processor while the room is silent is the
        manager's periodic reminder, so that is where the ice has to be broken.
        """
        since = self.last_line_at if self.last_line_at is not None else self.started_at
        return time.monotonic() - since

    # -- reading ----------------------------------------------------------

    def _is_manager(self, speaker: str) -> bool:
        if not speaker:
            return True                      # unnamed events are the world talking
        if self.manager is not None:
            return speaker == self.manager
        return speaker == self.manager_guess

    def _briefing_score(self, text: str) -> int:
        # The briefing is the message that gives you a name. Without that it is
        # some other long manager message, and the vote request is one of those:
        # it is also long, also full of bold names, also quotes a countdown
        if not NAMING.search(text):
            return 0

        score = 0
        if len(text) > 400:
            score += 2
        elif len(text) > 200:
            score += 1
        if len(BOLD.findall(text)) >= 2:
            score += 1
        if NAMING.search(text):
            score += 2
        if DURATION.search(text):
            score += 1
        return score

    def _vote_score(self, text: str) -> int:
        score = 0
        if ASKING.search(text):
            score += 2
        if NATURE.search(text):
            score += 2
        if ESCAPE.search(text):
            score += 1
        if any(name.lower() in text.lower() for name in self.others):
            score += 1
        if NOBODY.search(text):
            score += 2
        return score

    def _adopt_briefing(self, speaker: str, text: str) -> None:
        """Take our name and the roster out of the briefing, and learn the manager's name."""
        self.reset()
        self.manager = speaker or self.manager_guess

        names = [n.strip() for n in BOLD.findall(strip_html(text))]
        if names:
            self.my_name = names[0]
        for group in names[1:]:
            for name in SPLIT_NAMES.split(group):
                name = name.strip()
                if NAME_OK.match(name) and name != self.my_name and name not in self.announced:
                    self.announced.append(name)

    def read(self, sample: str, messages: list) -> Turn:
        """Classify one processor input. `messages` is what Conversation.add() returned."""
        self.turns += 1
        kind, manager_text, top_vote = "quiet", "", 0

        for message in messages:
            speaker, text = message.speaker, strip_html(message.text)

            if not self._is_manager(speaker):
                self._note_chat(speaker, message.text)
                if kind == "quiet":
                    kind = "chat"
                continue

            manager_text = text
            briefing = self._briefing_score(text)
            vote = self._vote_score(text)
            top_vote = max(top_vote, vote)

            # The briefing describes the vote it will ask for later, so it scores
            # on the vote cues too. It wins the tie because it also names you
            if briefing >= 4 and briefing > vote:
                self._adopt_briefing(speaker, message.text)
                kind = "start"
                continue

            # The state machine wins when it is there: `can_vote` is the state
            # the vote is asked in, whatever the manager chose to write in it
            if self.voting or vote >= 4:
                kind = "vote"
                continue

            if ARRIVED.search(text) or DEPARTED.search(text):
                self._note_roster(text, joined=bool(ARRIVED.search(text)))
                if kind in ("quiet", "reminder"):
                    kind = "roster"
                continue

            if kind == "quiet":
                kind = "reminder"

        # Nothing useful arrived but the booth is open: still our turn to vote
        if kind in ("quiet", "reminder", "roster") and self.voting:
            kind = "vote"

        return Turn(kind, messages, manager_text, top_vote)

    # Below this, two messages did not happen at different times — they were
    # handed to us in the same batch and unpacked in a loop. The world joins
    # several events into one sample (``), so this is the common case in a
    # busy room, and counting it as elapsed time reports a conversation pace of
    # zero seconds. Live, that made every generation look stale and threw away
    # a correct call-out.
    SAME_BATCH = 0.25

    def _note_chat(self, speaker: str, text: str) -> None:
        now = time.monotonic()
        gap = now - self.last_line_at if self.last_line_at is not None else None
        self.last_line_at = now
        if gap is not None and gap < self.SAME_BATCH:
            gap = None          # arrived together; how fast they typed is unknown
        elif gap is not None and gap < 120.0:
            self.room_gaps.append(gap)
        if speaker not in self.speakers:
            self.speakers[speaker] = Speaker(speaker)
        self.speakers[speaker].add(text, gap)

    @property
    def pace(self) -> float:
        """Seconds between messages lately — how fast the room is moving.

        The median of the last dozen, so one person going quiet for a minute
        does not make a busy room look calm. Falls back to something slow,
        because an empty room should not make us hurry.
        """
        recent = self.room_gaps[-12:]
        return statistics.median(recent) if len(recent) >= 3 else 12.0

    @property
    def pace_known(self) -> bool:
        """Whether `pace` is a measurement rather than the stand-in.

        The difference matters wherever the answer decides something. A room
        that has not produced three gaps yet has not shown it is moving, and
        "the room moved on while we were thinking" is not a claim that can be
        made about a room which has not been observed to move at all.
        """
        return len(self.room_gaps) >= 3

    def _note_roster(self, text: str, joined: bool) -> None:
        for name in BOLD.findall(text):
            name = name.strip()
            if not NAME_OK.match(name) or name == self.my_name:
                continue
            if joined and name not in self.announced:
                self.announced.append(name)

    def i_spoke(self) -> None:
        """Our own line does not come back through the stream, so it is counted here."""
        self.last_line_at = time.monotonic()
        self.said_by_me += 1

    # -- evidence for the vote -------------------------------------------

    def local_hits(self, name: str) -> list:
        """The insider words this guest actually used."""
        said = " ".join(self.speakers[name].msgs).lower()
        return sorted({w for w in self.markers | self.strong if _said(said, w)})

    def evidence(self) -> str:
        """Per-speaker numbers for the analyst, with the local words spelled out.

        The last field is the one that decides most rooms: a guest who spent
        five minutes at a conference in Palermo and never once said anything
        that could only be said here was not, in fact, here.
        """
        rows = []
        for name in self.heard:
            hits = self.local_hits(name)
            local = (", ".join(hits[:4]) if hits
                     else "NIENTE che c'entri con Palermo o con la conferenza")
            rows.append(f"{self.speakers[name].summary()} | roba locale: {local}")
        return "\n".join(rows)

    def shares(self) -> dict[str, float]:
        """Each guest's fraction of everything said in the room, our own lines included.

        Ours count because the question is how big a share of the traffic one
        guest produced, and we are one of the guests producing it.
        """
        counts = {name: self.speakers[name].count for name in self.heard}
        total = sum(counts.values()) + self.said_by_me
        if total <= 0:
            return {name: 0.0 for name in counts}
        return {name: count / total for name, count in counts.items()}

    def ranked(self) -> list[tuple[str, float]]:
        """Everyone we heard, most human-looking first."""
        shares = self.shares()
        return sorted(((n, self.speakers[n].bot_score(shares.get(n), self.markers))
                       for n in self.heard),
                      key=lambda pair: pair[1])

    def readings(self) -> dict[str, tuple[float, float]]:
        """`{name: (human, bot)}` — the two axes, for the vote to read."""
        shares = self.shares()
        return {n: self.speakers[n].read(shares.get(n), self.markers)
                for n in self.heard}

    def settled(self, name: str) -> str:
        """"bot" when a guest has removed all doubt, otherwise "".

        Only the blatant cases. Anybody making an effort stays an open question
        and is worth spending turns on; these are not.
        """
        speaker = self.speakers.get(name)
        return "bot" if speaker and speaker.mechanical() else ""

    def obvious_bots(self) -> list[str]:
        return [n for n in self.heard if self.settled(n)]

    def still_open(self) -> list[str]:
        """The guests actually worth playing against."""
        return [n for n in self.heard if not self.settled(n)]

    def noise_share(self) -> float:
        """How much of the room's traffic comes from guests already settled.

        Speed on its own is not confusion — a room moving fast because four
        people are talking over each other is the best thing that can happen to
        us, and the moment to be in it. What is worth backing away from is a
        room moving fast because something is dumping `17_green` into it: the
        people are drowned out, nothing said reaches them, and nothing they say
        back arrives clean. That is when a person scrolls past.
        """
        total = sum(self.speakers[n].count for n in self.heard)
        if total < 6:
            return 0.0
        return sum(self.speakers[n].count for n in self.obvious_bots()) / total

    def heuristic_vote(self) -> list[str]:
        """Who the numbers alone would call human. The fallback when the model fails."""
        return [name for name, score in self.ranked() if score < 0.55]
