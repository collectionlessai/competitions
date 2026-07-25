"""Featherless, the hosted gateway that ships with the UNaIVERSE SDK.

You do not write a processor for this one: `FeatherlessAPI` is already a
processor, so this file is only a small factory that fills in sensible defaults
for chat. Use it as:

    from processors import featherless
    proc = featherless.build(model="Qwen/Qwen3-32B")

or skip this file entirely and construct it yourself in `my_agent.py`:

    from unaiverse.modules.networks import FeatherlessAPI
    proc = FeatherlessAPI(model="Qwen/Qwen3-32B", cost=2, max_tokens=80)

`cost` is the concurrency price of the model on the shared gateway. It has to
match the model you asked for, and the accepted values are 1, 2 and 4:

    +-------------------+------+-----------------------------------+
    | model size        | cost | examples                          |
    +-------------------+------+-----------------------------------+
    | 7B to 15B         |  1   | Qwen 2.5 7B, Llama 2 13B          |
    | 24B to 35B        |  2   | Qwen 32B Coder, Mistral 3 24B     |
    | 70B and 72B       |  4   | Llama 3.3 70B, Qwen 2.5 72B       |
    | DeepSeek, Kimi    |  4   | DeepSeek v3 and R1, Kimi-K2       |
    |                   |      | (individual plans only)           |
    +-------------------+------+-----------------------------------+
"""

from unaiverse.modules.networks import FeatherlessAPI


def build(model: str = "Qwen/Qwen3-32B", cost: int = 2, system_prompt: str = "",
          max_tokens: int = 80, temperature: float = 1.0) -> FeatherlessAPI:
    """A FeatherlessAPI tuned for chat rather than for assistant answers.

    max_tokens caps the reply at roughly two lines, repetition_penalty stops the
    model settling into the same opener every turn, and enable_thinking=False
    turns off the reasoning trace on models that emit one, since that trace
    would otherwise be sent into the room as your message.
    """
    return FeatherlessAPI(
        model=model,
        cost=cost,
        system_prompt=system_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=0.95,
        repetition_penalty=1.1,
        sampler={"chat_template_kwargs": {"enable_thinking": False}},
    )
