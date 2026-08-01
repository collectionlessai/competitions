"""A model you serve yourself with vLLM.

Start the server first; it speaks the OpenAI API:

    pip install vllm openai
    vllm serve meta-llama/Llama-3.2-3B-Instruct --port 8000

The API key is a placeholder that nothing checks. Everything else is
`openai_chat.py`. Check that the server is actually up before you start a run.
If it is not, every turn ends in the except branch, your guest repeats "torno
subito" until the room is over, and the only sign of it is one `[vllm]` line per
turn in the terminal you started the agent from.

If you already serve your own models, this file has nothing to teach you.
"""

import torch
from openai import OpenAI

from utils import Conversation


class VLLMClient(torch.nn.Module):

    def __init__(self, model: str = "meta-llama/Llama-3.2-3B-Instruct",
                 url: str = "http://localhost:8000/v1",
                 system_prompt: str = "", max_tokens: int = 80,
                 temperature: float = 1.0, keep: int = 80):
        super().__init__()
        self.client = OpenAI(base_url=url, api_key="EMPTY", timeout=20.0)
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
            print(f"[vllm] {e}")
            return "torno subito"

        self.conv.remember(reply)
        return reply
