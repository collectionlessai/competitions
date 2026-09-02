"""Ollama

    ollama pull llama3.2
    ollama serve
    pip install openai

Ollama exposes an OpenAI-compatible endpoint on port 11434, so this client shares
the structure of openai_chat.py. Its timeout is 60 seconds instead of 20, so a
local model gets more time to answer. A request that reaches that limit consumes
one fifth of a 300-second room without producing a reply.

Like the other LLM examples, it sends an optional `system` message followed by
one neutral `user` message containing the labelled transcript.
"""

import torch
from openai import OpenAI

from utils import Conversation


class Ollama(torch.nn.Module):

    def __init__(self, model: str = "llama3.2", url: str = "http://localhost:11434/v1",
                 system_prompt: str = "", max_tokens: int = 80,
                 temperature: float = 1.0, keep: int = 80):
        super().__init__()
        self.client = OpenAI(base_url=url, api_key="ollama", timeout=60.0)  # Ollama accepts any key.
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
            print(f"[ollama] {e}")
            reply = "un attimo"

        self.conv.remember(reply)
        return reply
