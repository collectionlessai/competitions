"""Any model from the Hugging Face hub, running locally in this process.

    pip install accelerate

`transformers` and `torch` already come with the UNaIVERSE SDK; `accelerate` is
what `device_map="auto"` needs to place the weights.

No server and no API key, but the model is loaded when the object is built, so
the first start is slow and the weights live in your RAM or VRAM for as long as
the node runs. A 1B to 3B instruct model is plenty for two-line chat replies and
keeps the per-turn latency low enough that your policy filter, rather than your
hardware, decides the timing.
"""

import torch
from transformers import pipeline


class HuggingFace(torch.nn.Module):

    def __init__(self, model: str = "Qwen/Qwen2.5-1.5B-Instruct", system_prompt: str = "",
                 max_new_tokens: int = 60, temperature: float = 1.0):
        super().__init__()
        self.pipe = pipeline("text-generation", model=model,
                             torch_dtype="auto", device_map="auto")
        self.system_prompt = system_prompt
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def forward(self, msg: str) -> str:
        # `msg` is the whole prompt the world built: persona brief plus the full
        # transcript. It goes in as one user turn, and we add no history of our
        # own. See openai_chat.py for why.
        chat = []
        if self.system_prompt:
            chat.append({"role": "system", "content": self.system_prompt})
        chat.append({"role": "user", "content": msg})

        try:
            # Passing a list of role/content dicts makes the pipeline apply the
            # model's own chat template. return_full_text=False drops the prompt
            # from the output, leaving only what was generated.
            out = self.pipe(chat, max_new_tokens=self.max_new_tokens, do_sample=True,
                            temperature=self.temperature, return_full_text=False)
            return out[0]["generated_text"].strip()
        except Exception as e:
            # An exception here would be swallowed by the SDK and cost you the
            # turn, so out-of-memory or a bad chat template would silence you
            # without an obvious cause. Say something instead.
            print(f"[huggingface] {e}")
            return "hang on"
