"""OpenRouter: hundreds of models behind one OpenAI-compatible API.

    pip install openai
    export OPENROUTER_API_KEY=sk-or-...

Useful when you want to compare models without touching any code: only the
`model` string changes, for instance "mistralai/mistral-7b-instruct",
"google/gemma-2-9b-it", "meta-llama/llama-3.1-8b-instruct".

Same body as `openai_chat.py`, which is where the shared logic is explained.
"""

import os
import torch
from openai import OpenAI


class OpenRouter(torch.nn.Module):

    def __init__(self, model: str = "meta-llama/llama-3.1-8b-instruct",
                 system_prompt: str = "", max_tokens: int = 80, temperature: float = 1.0):
        super().__init__()
        self.client = OpenAI(base_url="https://openrouter.ai/api/v1",
                             api_key=os.environ["OPENROUTER_API_KEY"],
                             timeout=20.0)
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
            print(f"[openrouter] {e}")
            return "hm, one sec"
