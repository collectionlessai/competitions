"""Any model from the Hugging Face hub, running inside your own agent

    pip install accelerate

`transformers` and `torch` already come with the UNaIVERSE SDK; `accelerate` is
what `device_map="auto"` needs to place the weights.

No server to start and no key to export, since the model runs where your agent
runs. The pipeline is built in `__init__`, so the constructor does not return
until the weights are loaded. That wait is paid once, at startup, and from then
on every reply is generated in your own process while the room carries on
without you.

The defaults are smaller here than in the API-backed files: 60 new tokens and
40 messages of history, against 80 and 80. Raise either one and the extra time
comes out of your own hardware.
"""

import torch
from transformers import pipeline

from utils import Conversation


class HuggingFace(torch.nn.Module):

    def __init__(self, model: str = "Qwen/Qwen2.5-1.5B-Instruct", system_prompt: str = "",
                 max_new_tokens: int = 60, temperature: float = 1.0, keep: int = 40):
        super().__init__()
        self.pipe = pipeline("text-generation", model=model,
                             torch_dtype="auto", device_map="auto")
        self.system_prompt = system_prompt   # empty, so out of the box it answers like an assistant
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.conv = Conversation(keep=keep)

    def complete(self, messages: list[dict]) -> str:
        out = self.pipe(messages, max_new_tokens=self.max_new_tokens, do_sample=True,
                        temperature=self.temperature, return_full_text=False)
        return out[0]["generated_text"].strip()

    def forward(self, sample: str) -> str:
        self.conv.add(sample)

        try:
            reply = self.complete(self.conv.as_messages(system=self.system_prompt)).strip()
        except Exception as e:
            print(f"[huggingface] {e}")
            return "un attimo"

        self.conv.remember(reply)
        return reply
