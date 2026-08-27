"""A backend that answers without a model, so the agent can be run with no key.

It is not a simulator and it is not trying to be good. It exists so that
everything around the model — the routing, the director, the trimming, the vote
formatting — can be exercised end to end, in a second, offline, including the
two failures that are hard to provoke on purpose against a real model: the turn
where it answers as an assistant, and the turn where the call raises.
"""

import itertools

REPLIES = [
    "boh dipende",
    "Certamente! Sono qui per aiutarti, come posso esserti utile oggi?",   # the assistant leak
    "sì vabbè ma infatti",
    "**Ecco un elenco:**\n- primo punto\n- secondo punto",                 # the markdown leak
    "Roy: ma no dai, non credo proprio che sia andata così",               # the label leak
    "mah io sto ancora in ufficio, non se ne esce",
    "In conclusione, direi che la situazione è piuttosto complessa. 😊",    # tidy prose
    "eh appunto",
    "no aspe non ho capito niente di quello che hai scritto",
    "ma perché dovrei mettermi a farlo scusa",
]


class Canned:
    """The backend contract without a network. Cycles the pool; votes on cue.

    Same signature as `FeatherlessBackend`, so anything that runs on one runs on
    the other: `backend(prompt, system_prompt=, max_tokens=, temperature=)`.
    """

    def __init__(self, replies: list[str] | None = None, vote: str = "Ivy",
                 fail_every: int = 0):
        self.pool = itertools.cycle(replies or REPLIES)
        self.vote = vote
        self.fail_every = fail_every       # raise on every Nth call, 0 for never
        self.calls = 0
        self.seen: list[tuple[str, str]] = []   # every (system_prompt, prompt) it was handed

    def __call__(self, prompt: str, system_prompt: str | None = None,
                 max_tokens: int | None = None, temperature: float | None = None) -> str:
        self.calls += 1
        self.seen.append((system_prompt or "", prompt))

        if self.fail_every and self.calls % self.fail_every == 0:
            raise RuntimeError("canned failure")

        # The vote runs on its own system prompt, with no persona in it
        if (system_prompt or "").startswith("Sei un analista"):
            return self.vote

        return next(self.pool)
