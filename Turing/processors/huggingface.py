"""Run a model from the Hugging Face Hub inside the agent.

    pip install accelerate

The UNaIVERSE SDK already installs `transformers` and `torch`. The automatic
weight placement requested by `device_map="auto"` also requires `accelerate`.

The model runs in the agent process, so it needs neither an API key nor a
separate server. `__init__` constructs the pipeline and waits for the weights to
load before returning. Later replies are generated locally while the room
continues to receive other events.

This processor defaults to 60 new tokens with 40 messages of history, compared
with 80 for both settings in the API-backed examples. Larger values increase
the work performed by your hardware.
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
        self.system_prompt = system_prompt   # Empty by default, with no supplied persona.
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
            reply = "un attimo"

        self.conv.remember(reply)
        return reply
