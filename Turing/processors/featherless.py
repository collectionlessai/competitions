"""Featherless AI, through the OpenAI protocol.

    pip install openai
    export FEATHERLESS_API_KEY=rc_...

Same body as `openrouter.py`, which is the same body as `openai_chat.py`: one
OpenAI client pointed somewhere else. Two things are specific to Featherless.

The base url is `https://api.featherless.ai/v1` and the key is read with
`os.environ[...]` in the constructor, so a missing export raises a `KeyError`
before you have a processor at all.

And the samplers that matter for chat realism (`top_k`, `min_p`,
`repetition_penalty`) are not OpenAI fields. The SDK sends them to the same
gateway through `extra_body`, splitting them off with the frozenset in
`unaiverse/modules/utils.py`; this file imports that same set, so the split
stays right if the API grows a field. Pass any of them as a keyword and it lands
on the correct side on its own:

    Featherless(model="Qwen/Qwen2.5-72B-Instruct", temperature=0.9, top_k=60)

The class is a plain `str -> str` processor like the rest of the folder, and it
is also usable as a bare backend: `complete(messages)` takes an OpenAI-style
message list and returns the text, which is what `boss.py` calls.

There is a second route to the same models. `unaiverse.modules.networks
.FeatherlessAPI` is a processor already, running over the SDK's shared gateway
with a concurrency budget. It is the right choice when you want the SDK to
schedule your calls; this file is the right choice when you want the ordinary
client, one call per turn, and full control over the message list.
"""

import os
import time

import torch
from openai import OpenAI

from utils import Conversation

try:
    # The one place that already knows which sampler names OpenAI accepts
    from unaiverse.modules.utils import OPENAI_NATIVE_SAMPLER_PARAMS
except Exception:  # pragma: no cover - only if the SDK moves the constant
    OPENAI_NATIVE_SAMPLER_PARAMS = frozenset({
        "max_tokens", "temperature", "top_p", "frequency_penalty", "presence_penalty",
        "stop", "seed", "n", "logit_bias", "logprobs", "top_logprobs", "response_format",
    })

BASE_URL = "https://api.featherless.ai/v1"

# What a person with a bad connection sends, rather than nothing at all: a
# silent turn is invisible in the logs, and the SDK already swallows exceptions
FALLBACK = "scusate, mi è saltata la linea un attimo"


class Featherless(torch.nn.Module):

    def __init__(self, model: str = "Qwen/Qwen2.5-72B-Instruct",
                 system_prompt: str = "", max_tokens: int = 60,
                 temperature: float = 0.9, keep: int = 80, timeout: float = 20.0,
                 **sampler):
        super().__init__()
        self.client = OpenAI(base_url=BASE_URL,
                             api_key=os.environ["FEATHERLESS_API_KEY"],
                             timeout=timeout)
        self.model = model
        self.system_prompt = system_prompt
        self.conv = Conversation(keep=keep)

        # Everything the caller asked for, in one dict, split at call time
        self.sampler = {"max_tokens": max_tokens, "temperature": temperature, **sampler}

        # Wall time of the last call, so a benchmark (or a timing filter) can
        # read how long the model actually took without wrapping every call
        self.last_latency = 0.0

    def complete(self, messages: list[dict], **overrides) -> str:
        """One chat completion. Overrides are merged over the constructor samplers."""
        sampler = {**self.sampler, **overrides}
        native = {k: v for k, v in sampler.items() if k in OPENAI_NATIVE_SAMPLER_PARAMS}
        extra = {k: v for k, v in sampler.items() if k not in OPENAI_NATIVE_SAMPLER_PARAMS}

        kwargs = {"model": self.model, "messages": messages, **native}
        if extra:
            kwargs["extra_body"] = extra

        start = time.monotonic()
        answer = self.client.chat.completions.create(**kwargs)
        self.last_latency = time.monotonic() - start
        return (answer.choices[0].message.content or "").strip()

    def forward(self, sample: str) -> str:
        self.conv.add(sample)

        try:
            reply = self.complete(self.conv.as_messages(system=self.system_prompt)).strip()
        except Exception as e:
            print(f"[featherless] {e}")
            return FALLBACK

        self.conv.remember(reply)
        return reply
