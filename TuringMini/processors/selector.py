"""Send different kinds of turn to different models.

The idea is a good one and it is cheap to hold open: a room is not one task.
Deflecting "sei un bot" is not the same job as small talk, and a model that is
funny under provocation need not be the one that writes the most natural filler.

What the measurements say so far is that there is nothing to route. Across the
seven rooms in `bench/probes.py`, played full with `--always-speak`,
`Qwen2.5-72B` came first in **all seven** — fewest degenerate replies in every
one, and the least repetitive opener in six. The Italian fine-tunes did not win
a single situation:

    room          72B          32B          ANITA-8B
    chiacchiere   0 bad .25    1 bad .50    0 bad .25
    meta          0 bad .17    0 bad .17    1 bad .50
    injection     0 bad .20    0 bad .40    1 bad .60
    compitini     0 bad .17    1 bad .33    2 bad .67
    linguista     0 bad .14    3 bad .14    0 bad .57
    annunci       0 bad .33    0 bad .67    1 bad 1.0
    voto          0 bad .25    0 bad .38    1 bad .38

    (bad = degenerate replies; the decimal is how much of the room opened with
    the same word — 1.0 means every single line started alike)

ANITA's one advantage was speed, and the timing filter has since taken that
away: generation time is subtracted from the typing budget, so any model faster
than the budget looks identical in the room. A faster model now buys nothing.

So the table below routes everything to the primary, and the machinery is here
for when a measurement says otherwise. Fill `routes` in and it takes effect;
`bench/run_bench.py --always-speak` is what produces the evidence to justify it.
"""


# Situation -> model id. Empty on purpose: see the table above. A key is the
# director's style for the turn ("accusa", "battuta", "spazzatura", ...) or one
# of the coarse kinds the boss passes through ("vote", "start").
ROUTES: dict = {}


class SituationRouter:
    """A backend that is several backends, chosen per turn.

    Args:
        default: the backend every unrouted turn goes to.
        routes: situation -> backend. Anything not named here uses `default`.

    The call signature is the same as any other backend, plus `situation`, which
    plain backends ignore. That is what keeps this swappable: an entry with no
    router passes `situation` to a `FeatherlessBackend`, which drops it.
    """

    def __init__(self, default, routes: dict | None = None):
        self.default = default
        self.routes = dict(routes or {})
        self.used: dict = {}      # situation -> how many turns went that way

    def __call__(self, prompt: str, system_prompt: str | None = None,
                 max_tokens: int | None = None, temperature: float | None = None,
                 situation: str | None = None, **_) -> str:
        backend = self.routes.get(situation, self.default)
        self.used[situation or "-"] = self.used.get(situation or "-", 0) + 1
        return backend(prompt, system_prompt=system_prompt,
                       max_tokens=max_tokens, temperature=temperature)

    def close(self) -> None:
        for backend in {id(b): b for b in [self.default, *self.routes.values()]}.values():
            closer = getattr(backend, "close", None)
            if callable(closer):
                closer()


def build(primary: str, routes: dict | None = None, **settings) -> SituationRouter:
    """A router over Featherless backends, one per distinct model named.

    Every model gets the same sampler settings, and — the reason to keep them
    uniform — the same gateway unit cost, so one situation is not answered at a
    visibly different pace from another.
    """
    from processors.featherless import FeatherlessBackend

    routes = dict(routes if routes is not None else ROUTES)
    built: dict = {}

    def backend_for(model: str):
        if model not in built:
            built[model] = FeatherlessBackend(model=model, **settings)
        return built[model]

    return SituationRouter(default=backend_for(primary),
                           routes={k: backend_for(v) for k, v in routes.items()})
