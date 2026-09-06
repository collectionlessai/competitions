"""Which language the room is speaking, worked out from what it says.

TuringBoss assumed Italian, because the competition was in Italian and somebody
told it so. That is exactly the kind of in-group fact this agent is not allowed
to carry, and it is load-bearing: the check for "a guest wrote a whole message
in another language" is meaningless until you know which language the *room*
is in.

Function words, not vocabulary. Content words travel between languages and the
loanwords a technical room uses travel most of all — a chat about "meeting",
"paper" and "coffee break" is still Italian. Function words do not travel: no
Italian sentence of any length avoids `di`, `che`, `non`, `per`, and no English
one avoids `the`, `and`, `is`, `to`.

No model call. This runs on every message that arrives, costs nothing, and is
deterministic, which matters because everything downstream trusts its answer.
"""
import re

# Small, deliberately: only words that are frequent in their language AND rare
# in the others here. "la" is Italian, Spanish and French, so it is in none of
# them; "come" is Italian and English, so it is in neither.
FUNZIONALI = {
    "italiano": frozenset((
        "di", "che", "non", "per", "con", "una", "sono", "anche", "ma", "come",
        "questo", "quello", "perché", "però", "molto", "sempre", "ancora",
        "quando", "cosa", "adesso", "gli", "delle", "degli", "nella", "sulla",
        "boh", "vabbè", "cioè", "magari", "grazie", "scusa", "ciao",
    )),
    "inglese": frozenset((
        "the", "and", "is", "are", "was", "you", "your", "this", "that", "with",
        "for", "from", "have", "has", "what", "when", "where", "would", "could",
        "just", "about", "there", "they", "them", "been", "than", "because",
        "hey", "yeah", "thanks", "sorry", "guys",
    )),
    "spagnolo": frozenset((
        "que", "los", "las", "una", "por", "para", "con", "pero", "muy", "está",
        "estoy", "también", "porque", "cuando", "cómo", "hola", "gracias",
        "ahora", "aquí", "todo", "algo", "nada", "bueno",
    )),
    "francese": frozenset((
        "les", "des", "une", "est", "sont", "pour", "avec", "mais", "vous",
        "nous", "être", "très", "aussi", "parce", "quand", "comment", "bonjour",
        "merci", "alors", "bien", "tout", "rien", "voilà",
    )),
    "tedesco": frozenset((
        "der", "die", "das", "und", "ist", "sind", "nicht", "mit", "auch",
        "aber", "sehr", "immer", "wenn", "was", "wie", "hallo", "danke",
        "schon", "noch", "hier", "alles", "nichts",
    )),
}

# Italian and Spanish share a lot of shape; "una" and "con" are in both lists on
# purpose, because dropping every shared word would leave Spanish with nothing.
# The margin below is what keeps a near-tie from being called.
MARGINE = 1.5
MINIMO = 3          # hits before any claim is made at all


def parole(testo: str) -> list:
    return re.findall(r"[a-zà-ÿäöüßñçéèêëíóúâîô']+", (testo or "").lower())


def punteggi(testi) -> dict:
    """How many function words of each language turned up, across `testi`."""
    conteggi = {nome: 0 for nome in FUNZIONALI}
    for testo in testi:
        for parola in parole(testo):
            for nome, insieme in FUNZIONALI.items():
                if parola in insieme:
                    conteggi[nome] += 1
    return conteggi


def della_stanza(testi) -> str:
    """The room's language, or "" when it is too early or too close to call."""
    conteggi = punteggi(testi)
    ordinati = sorted(conteggi.items(), key=lambda kv: -kv[1])
    primo, secondo = ordinati[0], ordinati[1]
    if primo[1] < MINIMO:
        return ""
    if secondo[1] and primo[1] < secondo[1] * MARGINE:
        return ""                       # two languages, or one that looks like two
    return primo[0]


def estranea(testo: str, lingua: str, minimo: int = 5) -> bool:
    """Whether this one message is in a language that is not the room's.

    Same shape as the old `foreign_language`, with the room's answer in place of
    the hard-coded Italian. Short messages say nothing — "ok", "no problem" and
    a bare loanword are what everybody types in every language.
    """
    if not lingua or lingua not in FUNZIONALI:
        return False
    trovate = parole(testo)
    if len(trovate) < minimo:
        return False
    if any(p in FUNZIONALI[lingua] for p in trovate):
        return False
    altre = sum(1 for p in trovate
                for nome, insieme in FUNZIONALI.items()
                if nome != lingua and p in insieme)
    return altre >= 2
