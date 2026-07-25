"""Ollama: the shortest path to a model running on your own laptop.

    ollama pull llama3.2
    ollama serve
    pip install openai

Ollama exposes an OpenAI-compatible endpoint on port 11434, so this is the same
body as `openai_chat.py`, which is where the shared logic is explained. The
timeout is longer because a local model on CPU can take several seconds for the
first tokens.
"""

import torch
from openai import OpenAI


class Ollama(torch.nn.Module):

    def __init__(self, model: str = "llama3.2", url: str = "http://localhost:11434/v1",
                 system_prompt: str = "", max_tokens: int = 80, temperature: float = 1.0):
        super().__init__()
        self.client = OpenAI(base_url=url, api_key="ollama", timeout=60.0)  # the key is ignored
        self.model = model
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.temperature = temperature

    def forward(self, msg: str) -> str:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": msg})

        try:
            answer = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            return answer.choices[0].message.content.strip()
        except Exception as e:
            print(f"[ollama] {e}")
            return "hang on"
