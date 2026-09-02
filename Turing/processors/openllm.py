"""OpenLLM

    pip install openllm openai
    openllm serve qwen2.5:7b

BentoML's server listens on port 3000 and implements the OpenAI API. This client
therefore follows `openai_chat.py`, but uses the 60-second timeout from
`ollama.py`.
"""

import torch
from openai import OpenAI

from utils import Conversation


class OpenLLM(torch.nn.Module):

    def __init__(self, model: str = "qwen2.5:7b", url: str = "http://localhost:3000/v1",
                 system_prompt: str = "", max_tokens: int = 80,
                 temperature: float = 1.0, keep: int = 80):
        super().__init__()
        self.client = OpenAI(base_url=url, api_key="EMPTY", timeout=60.0)  # The server ignores this key.
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
            print(f"[openllm] {e}")
            reply = "un secondo"

        self.conv.remember(reply)
        return reply
