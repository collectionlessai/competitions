"""OpenRouter

    pip install openai
    export OPENROUTER_API_KEY=sk-or-...

This processor follows `openai_chat.py` with two service-specific changes. The
constructor reads its key through `os.environ[...]`, so a missing export raises
`KeyError` before the processor is created. Model identifiers include the
vendor prefix, as in the default `meta-llama/llama-3.1-8b-instruct`.

It sends an optional `system` message followed by one neutral `user` message
containing the labelled transcript.
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
        messages = self.conv.as_messages(system=self.system_prompt)

        try:
            reply = self.complete(messages).strip()
        except Exception as e:
            print(f"[openrouter] {e}")
            reply = "aspetta un secondo"

        self.conv.remember(reply)
        return reply
