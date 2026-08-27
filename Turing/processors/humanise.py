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
    """Take out what the world's message filter would asterisk."""
    for pattern in MASKABLE:
        text = pattern.sub("", text)
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


def cap_words(text: str, limit: int) -> str:
    """Cut to `limit` words, at the last punctuation before it when there is one."""
    words = text.split()
    if len(words) <= limit:
        return text
    cut = " ".join(words[:limit])
    for mark in (". ", "! ", "? ", ", "):
        if mark in cut:
            head = cut.rsplit(mark, 1)[0]
            if len(head.split()) >= max(3, limit // 2):
                return head.rstrip(" ,")
    return cut.rstrip(" ,")


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
    kinds = ["swap", "neighbour"]
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
    if kind == "single":
        return re.sub(r"(.)\1", r"\1", word, count=1)
    if kind == "accent":
        return word.translate(ACCENTS)
    if kind == "apostrophe":
        return word.replace("'", " ", 1)

    letter = word[i].lower()
    if letter in NEIGHBOURS:
        return word[:i] + random.choice(NEIGHBOURS[letter]) + word[i + 1:]
    return word


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


def is_assistant(text: str) -> bool:
    """True when the model answered as itself rather than as the persona."""
    return bool(ASSISTANT.search(text))


def deflect() -> str:
    return random.choice(DEFLECTIONS)


def safe(text: str, exit_word: str = "exit") -> str:
    """Last guard before the wire.

    `exit` on its own ends the conversation and sends you to the vote, which is
    the one string that must never leave here by accident: it costs the whole
    rest of the room, and the Turing score is scaled by how much you said.
    """
    if text.strip().lower() == exit_word.lower():
        return "ah no scusate ho sbagliato a scrivere"
    return text
