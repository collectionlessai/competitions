"""OpenLLM (BentoML): serve an open model with one command.

    pip install openllm openai
    openllm serve qwen2.5:7b

It listens on port 3000 and speaks the OpenAI API, so this is the same body as
`openai_chat.py`, which is where the shared logic is explained. Run
`openllm model list` to see what you can serve.
"""

import torch
from openai import OpenAI


class OpenLLM(torch.nn.Module):

    def __init__(self, model: str = "qwen2.5:7b", url: str = "http://localhost:3000/v1",
                 system_prompt: str = "", max_tokens: int = 80, temperature: float = 1.0):
        super().__init__()
        self.client = OpenAI(base_url=url, api_key="EMPTY", timeout=60.0)  # the key is ignored
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
            print(f"[openllm] {e}")
            return "one moment"
