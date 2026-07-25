"""A model you serve yourself with vLLM.

Start the server first; it speaks the OpenAI API:

    pip install vllm openai
    vllm serve meta-llama/Llama-3.2-3B-Instruct --port 8000

Then point this processor at it. No API key, no rate limit and no cost per
token: the only limit is your own GPU. Same body as `openai_chat.py`, which is
where the shared logic is explained.
"""

import torch
from openai import OpenAI


class VLLMClient(torch.nn.Module):

    def __init__(self, model: str = "meta-llama/Llama-3.2-3B-Instruct",
                 url: str = "http://localhost:8000/v1", system_prompt: str = "",
                 max_tokens: int = 80, temperature: float = 1.0):
        super().__init__()
        self.client = OpenAI(base_url=url, api_key="EMPTY", timeout=20.0)  # the key is ignored
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
            print(f"[vllm] {e}")
            return "wait, brb"
