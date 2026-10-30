"""OpenRouter, as a backend for the boss (the SAINET day runs on OpenRouter, not Featherless).

    pip install openai
    export OPENROUTER_API_KEY=sk-or-...     # never written in any file
    export BOSS_BACKEND=openrouter          # see my_agent.py

Same contract as `FeatherlessBackend`, which is what `boss.py` calls and
`bench/canned.py` fakes:

    backend(prompt, system_prompt=..., max_tokens=..., temperature=...) -> str

and it fails the same way: an exception, which the boss turns into a stall line,
after the fallback models have been tried in order. Per-call system prompt,
ceiling and temperature, because the persona changes every room and the vote
runs colder than the chat.

The requests go straight through an OpenAI client pointed at OpenRouter. Not
through `unaiverse.modules.networks.OpenRouterAPI`: its `forward` (unaiverse-src
add8622) builds the request in `_kwargs` and then sends `**kwargs`, so no model
and no messages ever leave, and it is missing from older unaiverse releases.
`processors/openrouter.py` is the kit's simple processor, a different contract.
"""

import os

# OpenAI-native sampler keys; the rest (top_k, repetition_penalty, min_p) ride
# in extra_body, where OpenRouter reads them and the SDK does not reject them
NATIVE = {"max_tokens", "temperature", "top_p", "stop", "frequency_penalty", "presence_penalty", "seed"}


class OpenRouterBackend:
    """A chat-completions client on OpenRouter with a per-call system prompt and sampler.

    Args:
        model: the OpenRouter slug, e.g. "google/gemma-4-31b-it".
        fallback: slug, or list of them, tried in order when this one fails.
        max_tokens, temperature, top_p, top_k, repetition_penalty: the defaults
            for every call, max_tokens and temperature overridable per call.
        timeout: seconds for one request; a room lasts 300, so keep it short.
        api_key: None reads OPENROUTER_API_KEY.
    """

    def __init__(self, model: str, fallback: str | list | None = None, max_tokens: int = 60,
                 temperature: float = 0.95, top_p: float = 0.95, top_k: int = 60,
                 repetition_penalty: float = 1.08, stop: list | None = None,
                 timeout: float = 20.0, api_key: str | None = None, client=None, **extra):
        from openai import OpenAI

        self.models = [model] + ([fallback] if isinstance(fallback, str) else list(fallback or []))
        self.sampler = {"max_tokens": max_tokens, "temperature": temperature, "top_p": top_p,
                        "top_k": top_k, "repetition_penalty": repetition_penalty, **extra}
        if stop:
            self.sampler["stop"] = stop
        key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        if client is None and not key:
            raise RuntimeError("OPENROUTER_API_KEY is not set")
        self.client = client or OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key,
                                       timeout=timeout, max_retries=0,
                                       default_headers={"HTTP-Referer": "https://unaiverse.io",
                                                        "X-Title": "UNaIVERSE"})

    def __call__(self, prompt: str, system_prompt: str | None = None,
                 max_tokens: int | None = None, temperature: float | None = None, **_) -> str:
        sampler = dict(self.sampler)
        if max_tokens is not None and max_tokens > 0:
            sampler["max_tokens"] = int(max_tokens)
        if temperature is not None and temperature >= 0.:
            sampler["temperature"] = float(temperature)
        native = {k: v for k, v in sampler.items() if k in NATIVE}
        extra = {k: v for k, v in sampler.items() if k not in NATIVE}
        messages = ([{"role": "system", "content": system_prompt}] if system_prompt else []) + \
                   [{"role": "user", "content": prompt}]

        error = None
        for model in self.models:
            try:
                answer = self.client.chat.completions.create(model=model, messages=messages,
                                                             extra_body=extra or None, **native)
                text = (answer.choices[0].message.content or "").strip() if answer.choices else ""
                if text:
                    return text
                error = RuntimeError(f"{model}: empty answer")
            except Exception as e:
                error = e
        raise error if error is not None else RuntimeError("no model to ask")

    def close(self) -> None:
        try:
            self.client.close()
        except Exception:
            pass
