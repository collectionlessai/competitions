"""OpenAI, through the official client

    pip install openai
    export OPENAI_API_KEY=sk-...

Four other files in this folder are this body with a different `base_url`, and
they all point back here rather than repeat what follows.

`as_messages` renders the history with your own lines as `assistant` turns and
everybody else's as `user` turns with the speaker's name in front of the text.
That is how one model keeps three guests apart inside a single role.

The client carries a 20 second timeout. A request that hangs therefore costs you
20 seconds of a 300 second room, and for all of them your guest says nothing.

`forward` catches its own exceptions rather than letting them out. If it did
not, the SDK would catch them, log them and skip the turn, and the only trace
of a dead API key would be one line in a log you are not watching. A short
Italian sentence about the connection is louder, and it is roughly what
somebody with a bad line would type anyway.
"""

import torch
from openai import OpenAI

from utils import Conversation


class OpenAIChat(torch.nn.Module):

    def __init__(self, model: str = "gpt-4o-mini", system_prompt: str = "",
                 max_tokens: int = 80, temperature: float = 1.0, keep: int = 80):
        super().__init__()
        self.client = OpenAI(timeout=20.0)  # reads OPENAI_API_KEY from the environment
        self.model = model
        # Empty, and it stays empty. Nobody wrote you a persona.
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.conv = Conversation(keep=keep)   # the last `keep` messages, older ones fall off

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
            print(f"[openai] {e}")
            return "scusate, mi è saltata la connessione"

        self.conv.remember(reply)
        return reply
