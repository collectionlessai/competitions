"""A model you serve yourself with vLLM.

Start the OpenAI-compatible server before the agent:

    pip install vllm openai
    vllm serve meta-llama/Llama-3.2-3B-Instruct --port 8000

The API key is an unchecked placeholder. The rest of the client follows
`openai_chat.py`. Check that the server is available before starting a run
because an unavailable endpoint sends every turn to the exception branch,
returning "torno subito" and recording one `[vllm]` error in the terminal.

It sends an optional `system` message followed by one neutral `user` message
containing the labelled transcript.
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
        messages = self.conv.as_messages(system=self.system_prompt)

        try:
            reply = self.complete(messages).strip()
        except Exception as e:
            print(f"[vllm] {e}")
            reply = "torno subito"

        self.conv.remember(reply)
        return reply
