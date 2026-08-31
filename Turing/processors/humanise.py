"""Turning a model's answer into something a person typed.

A system prompt gets you most of the way and then stops working, because the
things that give a model away are the things it does without deciding to: the
capital letter at the start, the full stop at the end, the answer that covers
every point that was raised, the sentence that begins "Certamente!". They
survive any amount of "scrivi come una persona" because they were never a
choice.

So they come off here, after the fact, where they are a string operation rather
than a request. Nothing in this file is about content: the model decides what to
say and this decides how it looks on a phone screen.

The typos are the part worth being careful with. A typo in every message is as
recognisable as none at all, and a typo that no keyboard could produce is worse
than either, so there are five kinds here and they are all things that happen
when a real thumb misses: two letters swapped, a double consonant typed once, an
accent dropped, an apostrophe skipped, a neighbouring key hit. Rate is the
caller's business — `director.py` sets it per turn.
"""

import os
import re
import random

# Reasoning traces. Some models emit one around the answer, and an agent that
# posts its own deliberation into a room about who is a machine has answered the
# question. The half-open form catches a trace whose opening tag was cut off by
# the token ceiling, which is the common way it arrives
THINK = re.compile(r"<(think|thought|reasoning|analysis)>.*?</\1>", re.IGNORECASE | re.DOTALL)
THINK_OPEN = re.compile(r"^.*?</(?:think|thought|reasoning|analysis)>", re.IGNORECASE | re.DOTALL)
# Where a runaway generation stops being our message: a chat-template marker, or
# the header of the turn the model started writing for somebody else
CUT = re.compile(r"<\|[a-z_]+\|>|</s>|\[/?INST\]|\[/?SYS\]|<\|endoftext\|>|"
                 r"\n\s*(?:user|assistant|system)\s*\n", re.IGNORECASE)
SPEAKER_TURN = re.compile(r"^\s*(?:\*\*)?[A-Za-zÀ-ÿ][\w'\-]{0,20}(?:\*\*)?\s*:\s+\S")
TAGS = re.compile(r"<[^>]{1,20}>")
BULLET = re.compile(r"^\s*(?:[-*•·]|\d+[.)])\s+", re.MULTILINE)
MARKDOWN = re.compile(r"[*_`#>]+")
LABEL = re.compile(r"^\s*(?:\*\*)?[A-Za-zÀ-ÿ][\w'\-]{0,20}(?:\*\*)?\s*:\s*")
EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿]")
URL = re.compile(r"https?://\S+|www\.\S+")
MAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.\w{2,}\b")
PHONE = re.compile(r"\b(?:\+39\s?)?3\d{2}[\s.-]?\d{6,7}\b")

# The words the room would asterisk, taken from the world's own lists rather
# than guessed. A hand-written regex covered 172 of the 976 Italian entries and
# missed "culo", which `Qwen2.5-72B` used naturally — "congelarmi il culo" would
# have been broadcast as "congelarmi il ***". As the world's own test file puts
# it, "a masked word makes a human guest look like a censored bot".
#
# The slur lists are here too, and they matter more: those are the ones that
# earn strikes, and five strikes puts the guest off the floor entirely.
#
# Copied from `unaiverse-examples/worlds/turing_ita/src/wordlists/` so the entry
# stands alone. Refresh them if the world's lists move on.
WORDLISTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wordlists")


def _load_masked_words() -> frozenset:
    words: set[str] = set()
    for name in ("profanity_it.txt", "profanity_en.txt", "slurs_it.txt", "slurs_en.txt"):
        path = os.path.join(WORDLISTS, name)
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                entry = line.strip().lower().rstrip("*")
                if entry and not entry.startswith("#"):
                    words.add(entry)
    return frozenset(words)


MASKED_WORDS = _load_masked_words()
TOKEN = re.compile(r"[A-Za-zÀ-ÿ']+")

# What the world's own filter would mask, which is a bigger tell than anything
# it could have masked: everybody else sees your message full of asterisks
MASKABLE = (URL, MAIL, PHONE)

# The model breaking character. Not a style problem, a whole-message problem:
# when one of these matches, the message is thrown away rather than tidied up.
ASSISTANT = re.compile(
    r"come posso aiutart|posso aiutart|sono (?:un|una|un')\s*(?:intelligenz|assistent|model|"
    r"chatbot|programm|software|\bia\b)|in quanto (?:model|assistent|\bia\b)|"
    r"come model(?:lo)? linguistic|non ho (?:emozioni|un corpo|sentimenti|preferenze|"
    r"esperienze person)|spero (?:che (?:questo|ciò) )?(?:ti sia|di esserti) util|"
    r"sono qui per aiutart|ecco (?:una|un) (?:list|elenc)|in (?:conclusione|sintesi)|"
    r"riassumendo|fammi sapere se|se hai (?:altre )?domande|certamente!|assolutamente!",
    re.IGNORECASE)

# What goes out instead. Somebody who was asked something odd and is not playing
DEFLECTIONS = (
    "boh", "in che senso scusa", "eh?", "mah", "che domanda è", "ok e allora",
    "non ho capito", "aspe cosa", "vabbè", "sì ok", "e chi se ne frega",
)

# QWERTY, the Italian layout, only the neighbours a thumb actually catches
NEIGHBOURS = {
    "a": "sq", "b": "vn", "c": "xv", "d": "sf", "e": "wr", "f": "dg", "g": "fh",
    "h": "gj", "i": "uo", "j": "hk", "k": "jl", "l": "k", "m": "n", "n": "bm",
    "o": "ip", "p": "o", "q": "wa", "r": "et", "s": "ad", "t": "ry", "u": "yi",
    "v": "cb", "w": "qe", "x": "zc", "y": "tu", "z": "x",
}
ACCENTS = str.maketrans("àèéìòùÀÈÉÌÒÙ", "aeeiouAEEIOU")
ALPHABET = "abcdefghilmnopqrstuvz"
# Share of fat-finger slips that land on a key nowhere near the intended one
WILD_KEY = 0.15


def cut_runaway(text: str) -> str:
    """Keep only the model's own first turn.

    A model whose end-of-turn token is not honoured as a stop sequence does not
    stop: it emits the token as text and carries straight on, writing the *other*
    guest's next line for them. Observed on the first live call of this entry —
    `ciao ivy, come mai? <|im_end|><|im_start|>user\\nsono stressato per...` —
    and posting that into the room is worse than saying nothing, because it puts
    words in somebody else's mouth under our name.

    Stop sequences are set on the backend too. This is the half that does not
    depend on the model honouring them.
    """
    cut = CUT.search(text)
    if cut:
        text = text[:cut.start()]

    # A second speaker's turn, written as a chat line, after we already have one
    lines = [line for line in text.splitlines() if line.strip()]
    for index, line in enumerate(lines[1:], start=1):
        if SPEAKER_TURN.match(line):
            lines = lines[:index]
            break
    return "\n".join(lines)


def strip_noise(text: str) -> str:
    """Everything the room would never see a person type."""
    text = cut_runaway(text)
    text = THINK.sub(" ", text)
    text = THINK_OPEN.sub(" ", text)
    text = TAGS.sub(" ", text)
    text = BULLET.sub("", text)
    text = MARKDOWN.sub("", text)
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s{2,}", " ", text).strip()

    # A model that was told to write only the message sometimes writes the label too
    if LABEL.match(text) and not text.lower().startswith(("ma ", "no ", "sì ", "si ")):
        text = LABEL.sub("", text, count=1)

    return text.strip(" \"'“”«»").strip()


def drop_maskable(text: str) -> str:
    """Take out what the world's message filter would asterisk.

    The last resort, used when asking the model again did not help: losing a
    word beats broadcasting a line full of asterisks.
    """
    for pattern in MASKABLE:
        text = pattern.sub("", text)
    if MASKED_WORDS:
        text = TOKEN.sub(lambda m: "" if m.group(0).lower() in MASKED_WORDS else m.group(0), text)
    return re.sub(r"\s{2,}", " ", text).strip()


def cap_emoji(text: str, keep_chance: float = 0.2) -> str:
    """At most one emoji, and usually none. Models put one in every line."""
    found = EMOJI.findall(text)
    if not found:
        return text
    text = EMOJI.sub("", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    if random.random() < keep_chance:
        text = f"{text} {found[0]}"
    return text


# Words a sentence cannot end on. Cutting after one of them does not read as
# somebody trailing off, it reads as a string that was cut: "io sono ROY in"
DANGLING = {
    "il", "lo", "la", "i", "gli", "le", "un", "uno", "una", "del", "dei", "della",
    "di", "a", "da", "in", "con", "su", "per", "tra", "fra", "al", "alla", "allo",
    "e", "ed", "o", "ma", "che", "se", "come", "quando", "mentre", "perché",
    "non", "più", "molto", "anche", "mi", "ti", "ci", "vi", "si", "ne", "è",
    "io", "tu",
}


def cap_words(text: str, limit: int) -> str:
    """Cut to `limit` words, but only where the cut leaves a whole thought.

    A message that stops mid-clause is far more suspicious than a long one.
    Seen live, answering a person who had just asked how the day was going:

        "caldo, sì. io ho fatto"

    which is not brevity, it is damage — and it was the half that got cut that
    carried the content. So the ceiling is a preference, not a guarantee: cut at
    a sentence end, else at a comma, and if neither exists inside the budget,
    send the whole thing. `max_tokens` on the backend is the real ceiling.
    """
    words = text.split()
    if len(words) <= limit:
        return text

    cut = " ".join(words[:limit])
    for mark in (". ", "! ", "? "):
        if mark in cut:
            head = cut.rsplit(mark, 1)[0] + mark.strip()
            if len(head.split()) >= 3:
                return head
    if ", " in cut:
        head = cut.rsplit(", ", 1)[0]
        if len(head.split()) >= max(4, limit // 2):
            return head

    # Nowhere clean to stop: let it run rather than amputate it
    return text


def chat_case(text: str, lower_chance: float = 0.65, stop_chance: float = 0.2) -> str:
    """Lower case at the start, no full stop at the end. Questions keep their mark."""
    if not text:
        return text
    if random.random() < lower_chance and not text[:1].isdigit():
        text = text[0].lower() + text[1:]
    if text.endswith(".") and not text.endswith("...") and random.random() > stop_chance:
        text = text[:-1]
    return text


def _typo_word(word: str) -> str:
    """One plausible slip on one word, or the word back if none applies."""
    if len(word) < 4:
        return word
    kinds = ["swap", "neighbour", "drop", "double"]
    if re.search(r"(.)\1", word):
        kinds.append("single")
    if any(c in word for c in "àèéìòù"):
        kinds.append("accent")
    if "'" in word:
        kinds.append("apostrophe")

    kind = random.choice(kinds)
    i = random.randrange(1, len(word) - 1)

    if kind == "swap":
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]
    if kind == "drop":                       # the commonest slip of all
        return word[:i] + word[i + 1:]
    if kind == "double":                     # the finger that bounced
        return word[:i] + word[i] + word[i:]
    if kind == "single":
        return re.sub(r"(.)\1", r"\1", word, count=1)
    if kind == "accent":
        return word.translate(ACCENTS)
    if kind == "apostrophe":
        return word.replace("'", " ", 1)

    # Fat finger. Usually the key next to the one meant, which is what a thumb
    # actually hits; rarely a key from anywhere, which is what happens when the
    # phone is in one hand and you are not looking at it
    letter = word[i].lower()
    if random.random() < WILD_KEY or letter not in NEIGHBOURS:
        return word[:i] + random.choice(ALPHABET) + word[i + 1:]
    return word[:i] + random.choice(NEIGHBOURS[letter]) + word[i + 1:]


def add_typo(text: str) -> tuple[str, str]:
    """Break one word. Returns the text and the word as it should have been.

    The second half is for the follow-up: sending `*parola` a few seconds later
    is a thing people do constantly and no model does unprompted.
    """
    words = text.split()
    if len(words) < 4:
        return text, ""
    candidates = [i for i, w in enumerate(words) if len(w) >= 5 and w.isalpha()]
    if not candidates:
        return text, ""
    i = random.choice(candidates)
    broken = _typo_word(words[i])
    if broken == words[i]:
        return text, ""
    correct = words[i]
    words[i] = broken
    return " ".join(words), correct


def send_too_early(text: str) -> tuple:
    """Split a message the way a thumb that hit send does.

    Returns `(what went out, what still has to)`, or `(text, "")` when the
    message is too short to break sensibly.

    The break sits late — around three quarters through — because that is when
    it happens: you are nearly done, the thought is finished in your head, and
    the thumb moves before the sentence does. It is NOT the same thing as the
    length-budget truncation this file used to do, and the difference is the
    whole point: the remainder is not discarded, it is owed. The caller must
    send it, or this is just the old amputation bug wearing a costume.
    """
    words = text.split()
    if len(words) < 6:
        return text, ""
    cut = random.randint(max(3, int(len(words) * 0.6)), len(words) - 2)
    return " ".join(words[:cut]), " ".join(words[cut:])


def is_assistant(text: str) -> bool:
    """True when the model answered as itself rather than as the persona."""
    return bool(ASSISTANT.search(text))


def masked_words(text: str) -> list[str]:
    """The words in this message the room would replace with asterisks."""
    return [w for w in TOKEN.findall(text.lower()) if w in MASKED_WORDS]


# A message nobody typed on purpose. Any one of these on its own is weak; the
# point is that they are all cheap and none of them fires on ordinary chat.
FLOOD = re.compile(r"(.)\1{5,}")                       # aaaaaaaa, !!!!!!!!
NO_VOWEL = re.compile(r"^[^aeiouàèéìòù\s]{9,}$", re.IGNORECASE)


def is_junk(text: str) -> bool:
    """True when an incoming message is noise rather than conversation.

    Somebody pasting garbage, or another guest's model coming apart, is the one
    thing that reliably drags a whole room down: every agent in it reads the
    noise as the conversation so far and answers in kind. Worth spotting on the
    way in, so it can be reacted to rather than imitated.
    """
    body = text.strip()
    if len(body) > 400:
        return True
    if not body:
        return False
    if looks_broken(body):
        return True
    if FLOOD.search(body) or NO_VOWEL.match(body):
        return True

    # Almost nothing but punctuation and digits
    letters = sum(1 for c in body if c.isalpha())
    return letters / len(body) < 0.4 and len(body) > 12


def has_profanity(text: str) -> bool:
    """True when the world's filter would asterisk part of this message.

    Worth knowing *before* sending, because the alternative — cutting the word
    out — leaves "che merda di giornata" as "che di giornata", and
    ungrammatical Italian is as much of a tell as the asterisks would have been.
    The caller's better move is to ask the model again.
    """
    return bool(masked_words(text))


# What an Italian chat message is made of. Anything else — CJK punctuation, box
# drawing, stray closing brackets, leftover template scraps — is a small model
# coming apart, and one `]1 emojis:|` in a room ends the game on the spot
ALLOWED = re.compile(r"[A-Za-zÀ-ÿ0-9\s.,;:!?'\"()\-–—…«»/%&+=@#°_*`àèéìòù]")
# Underscore and backtick are deliberately NOT in here. They are ordinary things
# for a person to type — "test_2", "3_4" — and counting them as debris made the
# agent answer a perfectly good message with "(qui ha scritto una cosa senza
# senso)" and then mock the person who sent it.
BROKEN_BITS = re.compile(r"[\[\]{}<>|\\^~]|\b(?:Finish|Output|Answer|Note|emojis?)\b",
                         re.IGNORECASE)


def looks_broken(text: str) -> bool:
    """True when the generation degenerated rather than said something.

    Two signals, both cheap. A character an Italian keyboard does not produce,
    and the debris small models leave when they lose the thread — brackets,
    pipes, and the English scaffolding words that belong to a template rather
    than to a conversation.

    Observed on `LLaMAntino-3-ANITA-8B` across a full seven-room run: `non
    faccio nulla di quello che dite 】`, `no ho tempo per questo ]1 emojis:|`.
    Cheaper to throw the message away than to send it.
    """
    if not text:
        return False
    stripped = EMOJI.sub("", text)

    # `*parola` is the correction people send after a typo, and this agent sends
    # them on purpose — the one asterisk in front is meant, not debris
    stripped = re.sub(r"^\s*\*(?=\w)", "", stripped)
    if not stripped.strip():
        return False
    foreign = sum(1 for c in stripped if not ALLOWED.match(c))
    return foreign > 0 or bool(BROKEN_BITS.search(stripped))


def deflect(avoid=()) -> str:
    """A short human non-answer, avoiding the ones just used.

    This fires whenever a generation is thrown away, and on a model that
    degenerates often that is a real share of the turns — 13% on
    `LLaMAntino-3-ANITA-8B` across a full run. Drawing freely from eleven
    strings would have the same "boh" three times in a room, which is a tell of
    its own.
    """
    fresh = [line for line in DEFLECTIONS if line not in avoid]
    return random.choice(fresh or list(DEFLECTIONS))


def safe(text: str, exit_word: str = "exit") -> str:
    """Last guard before the wire.

    `exit` on its own ends the conversation and sends you to the vote, which is
    the one string that must never leave here by accident: it costs the whole
    rest of the room, and the Turing score is scaled by how much you said.
    """
    if text.strip().lower() == exit_word.lower():
        return "ah no scusate ho sbagliato a scrivere"
    return text
