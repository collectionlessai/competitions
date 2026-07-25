"""OpenAI, through the official SDK.

    pip install openai
    export OPENAI_API_KEY=sk-...

This is the reference implementation for the five API-backed processors in this
folder. The other four (openrouter, vllm_client, ollama, openllm) are the same
code pointed at a different `base_url`, so the comments here are not repeated
there.

Three things are worth understanding once:

* `msg` is already a complete prompt. The world rebuilds it every turn with the
  persona brief and the entire transcript inside it. Forward it as one user
  message and keep no history of your own, or the model reads every line twice
  and starts repeating itself.
* `system_prompt` is optional and empty by default. The persona brief inside
  the prompt already tells the model to write short, lowercase, non-assistant
  text. Read `../prompts/persona_prompt.txt` before adding instructions of your
  own, so that you do not contradict it.
* `max_tokens=80` is the most useful knob in this file. Long, evenly sized,
  well-punctuated replies are the clearest machine tell in the room.
"""

import torch
from openai import OpenAI


class OpenAIChat(torch.nn.Module):

    def __init__(self, model: str = "gpt-4o-mini", system_prompt: str = "",
                 max_tokens: int = 80, temperature: float = 1.0):
        super().__init__()
        self.client = OpenAI(timeout=20.0)  # reads OPENAI_API_KEY from the environment
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
            # Never let this escape. The SDK catches processor exceptions, logs
            # them and drops the turn, so a flaky API would leave you silent
            # without any obvious sign. Returning a short line keeps you in the
            # conversation and reads like somebody whose connection hiccuped.
            print(f"[openai] {e}")
            return "sorry, my connection dropped for a sec"
