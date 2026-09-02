"""Call an OpenAI model through the official client.

    pip install openai
    export OPENAI_API_KEY=sk-...

Four other processors use the same client structure with a different
`base_url`, then refer back to this explanation from their docstrings.

`as_messages` renders local replies as `assistant` turns. Events from every
other guest use the `user` role but retain the speaker name in their text, which
lets one model distinguish several people inside that role.

The client timeout is 20 seconds. While a request is waiting, the guest remains
silent and the 300-second room continues.

`forward` catches client exceptions because the SDK would otherwise log the
error and silently skip the turn. Returning a short Italian connection message
makes a missing key or failed request visible both in the room and in the log.
"""

import torch
from openai import OpenAI

from utils import Conversation


class OpenAIChat(torch.nn.Module):

    def __init__(self, model: str = "gpt-4o-mini", system_prompt: str = "",
                 max_tokens: int = 80, temperature: float = 1.0, keep: int = 80):
        super().__init__()
        self.client = OpenAI(timeout=20.0)  # Read OPENAI_API_KEY from the environment.
        self.model = model
        # The default is empty, so provide a persona explicitly when needed.
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.conv = Conversation(keep=keep)   # Retain only the latest `keep` messages.

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
            reply = "scusate, mi è saltata la connessione"

        self.conv.remember(reply)
        return reply
