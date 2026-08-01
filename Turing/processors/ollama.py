"""Ollama

    ollama pull llama3.2
    ollama serve
    pip install openai

Ollama exposes an OpenAI-compatible endpoint on port 11434, so this is the same
body as openai_chat.py with one number changed: a 60 second timeout, against 20
there. A local model gets three times as long to answer, and one request that
never comes back is a fifth of a 300 second room spent waiting for a line you
will not send.
"""

import torch
from openai import OpenAI

from utils import Conversation


class Ollama(torch.nn.Module):

    def __init__(self, model: str = "llama3.2", url: str = "http://localhost:11434/v1",
                 system_prompt: str = "", max_tokens: int = 80,
                 temperature: float = 1.0, keep: int = 80):
        super().__init__()
        self.client = OpenAI(base_url=url, api_key="ollama", timeout=60.0)  # any string does
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
            print(f"[ollama] {e}")
            return "un attimo"

        self.conv.remember(reply)
        return reply
