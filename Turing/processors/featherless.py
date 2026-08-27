"""Featherless, through the gateway that ships with the SDK.

    pip install unaiverse
    export FEATHERLESS_API_KEY=rc_...

`unaiverse.modules.networks.FeatherlessAPI` is already a processor: it owns a
persistent registration socket onto a shared gateway server, which it spawns on
first use, and that server is what actually talks to Featherless. Going through
it rather than through an `OpenAI` client of our own is what keeps every agent
on this account inside one concurrency budget — `cost` is the price of a call,
and the gateway schedules against it — instead of each process opening its own
connection and discovering the ceiling the hard way.

The catch is that the gateway fixes the system prompt and the sampler **at
construction**, and takes one prompt string per call. The boss needs neither:
the persona changes every room, the director's line changes every turn, and the
vote runs at a different temperature and token ceiling from the chat. So
`FeatherlessBackend` patches `system_prompt` and `sampler` on the inner `Net`
around each call and restores them in a `finally` — the same adapter the
organisers' own guests use (`ita/basic_factory/processors.py` in
collectionlessai/Turing-Hotel), which is worth matching rather than inventing a
second way of doing it.

The contract it exposes is the one `boss.py` calls and `bench/canned.py` fakes:

    backend(prompt, system_prompt=..., max_tokens=..., temperature=...) -> str

One instance per agent. Call `close()` (or let the process die) to release the
registration; when the last one goes, the server shuts itself down.
"""

import re
import time
import socket
import subprocess
import sys

from unaiverse.modules.networks import FeatherlessAPI
from unaiverse.modules.utils import APIGatewayServer

# What the gateway charges per call, by model size. The accepted values are
# 1, 2 and 4, and the number has to match the model actually asked for
COSTS = ((re.compile(r"\b(?:65|70|72|8x22|120|180|235|405)b\b", re.IGNORECASE), 4),
         (re.compile(r"\b(?:deepseek|kimi)", re.IGNORECASE), 4),
         (re.compile(r"\b(?:22|24|27|30|32|34|35)b\b", re.IGNORECASE), 2))

# Not every model on the catalogue has its end-of-turn token wired up as a stop,
# and one that does not runs straight past it and writes the next speaker's line
# for them. These are the markers that end our turn, whichever family the model
# comes from. `humanise.cut_runaway` catches what gets through anyway
STOP = ["<|im_end|>", "<|im_start|>", "<|eot_id|>", "</s>", "<|endoftext|>", "[INST]"]

# Not every chat template has a system turn. Gemma's does not, and the API says
# so with a 400 rather than by ignoring it, which fails the whole call
NO_SYSTEM = re.compile(r"system role not supported|system.{0,24}not supported", re.IGNORECASE)
BAD_REQUEST = re.compile(r"bad_request|invalid_request|rejected as invalid", re.IGNORECASE)


def _ensure_gateway(cls, timeout: float = 30.0) -> None:
    """Bring the shared gateway server up, without `fcntl`.

    Same contract as the SDK's `FeatherlessAPI._ensure_server`, minus the lock.
    That lock guards exactly one thing — two processes racing to spawn the
    server — and losing that race is already harmless, because the loser fails
    to bind the port and dies. Everything else here is the same: fast-path the
    check, spawn detached, wait for the port.
    """
    if cls._server_is_up():
        return

    subprocess.Popen([sys.executable, "-c",
                      "from unaiverse.modules.utils import serve_api_gateway; serve_api_gateway()"],
                     close_fds=True)

    deadline = time.time() + timeout
    while time.time() < deadline:
        if cls._server_is_up():
            return
        time.sleep(0.1)
    raise RuntimeError(f"the Featherless gateway did not come up on "
                       f"{APIGatewayServer.HOST}:{APIGatewayServer.PORT} within {timeout:.0f}s")


def install_windows_shim() -> bool:
    """Make `FeatherlessAPI` constructible on Windows. No-op everywhere else.

    `FeatherlessAPI._ensure_server` begins with `import fcntl`, a Unix-only
    module, and `__init__` calls it unconditionally — so on Windows the class
    raises `ModuleNotFoundError` before it can even check whether the server it
    wants is already running. Starting the server first does not help for the
    same reason: the import is the first statement in the method.

    So the method is replaced, and only when the import genuinely is not
    available. This is an SDK portability bug rather than something about this
    entry, and it is worth reporting upstream — the whole gateway works on
    Windows once past this one line.

    Returns True when the shim was installed.
    """
    try:
        import fcntl  # noqa: F401
        return False
    except ImportError:
        FeatherlessAPI._ensure_server = classmethod(_ensure_gateway)
        return True


WINDOWS_SHIM = install_windows_shim()


def cost_for(model: str) -> int:
    """The gateway's unit price for a model id, guessed from the size in its name.

    A guess, because the id is the only thing we have: pass `cost=` explicitly
    when it is wrong. Too low and the gateway over-schedules the model; the
    table it comes from is in the SDK's own Featherless notes.
    """
    for pattern, cost in COSTS:
        if pattern.search(model):
            return cost
    return 1


class FeatherlessBackend:
    """A `FeatherlessAPI` handle with a per-call system prompt and sampler.

    Args:
        model: the model id, which on Featherless is its Hugging Face repo name.
        cost: the gateway's unit price, or None to guess it from the model id.
        max_tokens, temperature, top_p, top_k, repetition_penalty: the defaults
            for every call, each overridable per call where the caller passes one.
        **kwargs: forwarded to `FeatherlessAPI` (`min_p`, `sampler`, ...).
    """

    def __init__(self, model: str, cost: int | None = None, max_tokens: int = 60,
                 temperature: float = 0.95, top_p: float = 0.95, top_k: int = 60,
                 repetition_penalty: float = 1.08, stop: list | None = None, **kwargs):
        self.model = model
        self.no_system = False       # set on the first "system role not supported"

        sampler = dict(kwargs.pop("sampler", None) or {})
        sampler.setdefault("stop", STOP if stop is None else stop)

        self.api = FeatherlessAPI(model=model, cost=cost if cost is not None else cost_for(model),
                                  max_tokens=max_tokens, temperature=temperature,
                                  top_p=top_p, top_k=top_k,
                                  repetition_penalty=repetition_penalty,
                                  sampler=sampler, **kwargs)

    def __call__(self, prompt: str, system_prompt: str | None = None,
                 max_tokens: int | None = None, temperature: float | None = None) -> str:
        """One generation, with the system prompt folded in where it has to be.

        Some chat templates have no system turn — Gemma's is the one on the
        shortlist — and the API rejects the whole call with a 400 rather than
        ignoring the role. The first time that happens the prompt is rebuilt
        with the system text on the front and the decision is remembered, so a
        model like that costs one failed call per process rather than one per
        turn, and no model that does support the role loses it.
        """
        if self.no_system and system_prompt:
            prompt, system_prompt = f"{system_prompt}\n\n{prompt}", ""

        try:
            return self._generate(prompt, system_prompt, max_tokens, temperature)
        except Exception as e:
            merged = bool(system_prompt) and (NO_SYSTEM.search(str(e)) or BAD_REQUEST.search(str(e)))
            if not merged:
                raise
            self.no_system = True
            return self._generate(f"{system_prompt}\n\n{prompt}", "", max_tokens, temperature)

    def _generate(self, prompt: str, system_prompt: str | None,
                  max_tokens: int | None, temperature: float | None) -> str:
        net = self.api.module        # the inner Net: the sockets, the system prompt, the sampler
        assert net is not None

        saved_prompt = net.system_prompt
        saved_sampler = dict(net.sampler)
        try:
            if system_prompt is not None:
                net.system_prompt = system_prompt
            if max_tokens is not None and max_tokens > 0:
                net.sampler["max_tokens"] = int(max_tokens)
            if temperature is not None and temperature >= 0.:
                net.sampler["temperature"] = float(temperature)

            out = self.api(prompt)   # ModuleWrapper.forward returns a tuple
        finally:
            net.system_prompt = saved_prompt
            net.sampler.clear()
            net.sampler.update(saved_sampler)

        if isinstance(out, tuple):
            out = out[0] if out else ""
        return out if isinstance(out, str) else ""

    def close(self) -> None:
        """Release the gateway sockets. The server stops when the last caller goes."""
        self.api.close()
