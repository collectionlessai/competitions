"""OpenRouter

    pip install openai
    export OPENROUTER_API_KEY=sk-or-...

Same body as `openai_chat.py`, which is where it is explained. Two things are
different here. The key is read with `os.environ[...]` inside the constructor,
so forgetting the export raises a `KeyError` before you have a processor at all.
And the model id carries the vendor in front of it: the default is
`meta-llama/llama-3.1-8b-instruct`.
"""

import os

import torch
from openai import OpenAI

from utils import Conversation


class OpenRouter(torch.nn.Module):

    def __init__(self, model: str = "meta-llama/llama-3.1-8b-instruct",
                 system_prompt: str = "", max_tokens: int = 80,
                 temperature: float = 1.0, keep: int = 80):
        super().__init__()
        self.client = OpenAI(base_url="https://openrouter.ai/api/v1",
                             api_key=os.environ["OPENROUTER_API_KEY"],
                             timeout=20.0)
        self.model = model
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.conv = Conversation(keep=keep)

    def complete(self, messages: list[dict]) -> str:
        answer = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return (answer.choices[0].message.content or "").strip()

    def forward(self, sample: str) -> str:
        self.conv.add(sample)

        try:
            reply = self.complete(self.conv.as_messages(system=self.system_prompt)).strip()
        except Exception as e:
            print(f"[openrouter] {e}")
            return "aspetta un secondo"

        self.conv.remember(reply)
        return reply
