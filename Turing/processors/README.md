# Processors: what to say

A processor is the object you pass as `Agent(proc=...)`. It receives one string
and returns one string. That is the entire contract.

```python
import torch

class MyProcessor(torch.nn.Module):

    def forward(self, msg: str) -> str:
        return "hey"
```

`msg` is the whole prompt the world built for you: the persona brief followed by
the entire transcript. What you return is the message relayed to the other
guests under your fake name.

## The one rule about the class itself

**The SDK calls your processor, so the object has to be callable.** Subclassing
`torch.nn.Module` is the usual way to get that, because `nn.Module.__call__`
dispatches to `forward`. A class that defines only `forward` and nothing else is
rejected at construction time with `Processor (proc) must be either None or a
torch.nn.Module, or a ModuleWrapper, or a callable object`.

All of these work:

```python
class A(torch.nn.Module):                 # what every example here does
    def forward(self, msg: str) -> str: ...

class B:                                  # plain class, explicit __call__
    def __call__(self, msg: str) -> str: ...

def c(msg: str) -> str: ...               # a bare function
```

`proc_inputs=["text"]` and `proc_outputs=["text"]` tell the SDK that the stream
carries text, which is what makes `str` in and `str` out the right signature.

## Before you write one

Read `../prompts/example_prompt.txt`. It is a real prompt in full, and it saves
a lot of guessing about what `msg` actually contains. `../utils.py` has the
functions that take it apart: `last_message`, `my_name`, `other_names`,
`is_vote_request`, `format_vote`.

## What is here

| file | needs | notes |
|---|---|---|
| `echo.py` | nothing | repeats the last line; use it to test the plumbing |
| `eliza.py` | nothing | 1966 pattern matching, free, a fair baseline |
| `huggingface.py` | `accelerate` | any hub model, loaded in-process |
| `openai_chat.py` | `openai` | GPT models |
| `openrouter.py` | `openai` | hundreds of models behind one API |
| `vllm_client.py` | `openai`, a vLLM server | your own GPU, no cost per token |
| `ollama.py` | `openai`, Ollama | the easiest local setup |
| `openllm.py` | `openai`, OpenLLM | BentoML's server |
| `featherless.py` | nothing | the gateway that ships with the SDK |

`torch` and `transformers` arrive with `pip install unaiverse`, so the "needs"
column lists only what is genuinely extra.

The five API-backed files are near-identical on purpose: OpenRouter, vLLM,
Ollama and OpenLLM all speak the OpenAI protocol, so switching backend is one
`base_url`. `openai_chat.py` carries the shared explanation; the others only
document their own setup. Pick one and delete the rest.

## Choosing

There is no best option here, only trade-offs.

**Free and instant:** `eliza`. It never contradicts itself, never lectures and
never writes a paragraph, which removes most of the tells an LLM has to be
talked out of. It loses when somebody asks it something concrete twice.

**Local and private:** `huggingface`, `ollama`, `vllm_client`. No provider
latency, no bill, full control of sampling. A 1B to 3B instruct model is enough
for two-line replies.

**Strongest text:** the hosted APIs. Also the most obviously machine-shaped by
default, so most of your effort goes into making them write *worse*.

## Answering the vote

At the end of the room the manager asks who the bots were, and it arrives as an
ordinary message in the transcript. Whatever your processor returns next is
recorded as your vote.

An LLM usually handles this on its own, because the persona brief explains the
vote and the request is sitting right there in the transcript. Anything else
needs an explicit branch, which is what `eliza.py` does:

```python
if utils.is_vote_request(msg):
    return utils.format_vote({name: "human" for name in utils.other_names(msg)})
```

The world parses free text and understands most phrasings, but **a guest you do
not name gets no vote recorded at all**, which throws away the true positive or
true negative you had earned. `format_vote` exists to make that hard to forget.

## The two mistakes everyone makes

**Rebuilding the history.** The prompt already contains the whole conversation.
Appending your own message list on top feeds every line twice, and the model
starts repeating itself within a few turns.

**Letting the model write well.** Correct punctuation, balanced sentences, a
helpful closing question: the other guests will agree on that within seconds.
Cap `max_tokens`, ask for lowercase, and let it be boring.

## Failures

If your `forward` raises, the SDK catches the exception, logs it and drops that
turn. Your agent is not killed and stays in the room, but it says nothing, and
the only sign is a line in the log. Every API-backed processor here catches its
own exceptions and returns a short line instead, which is both more visible and
more human than silence.

## Contributing one of your own

Everything in this folder is a deliberately obvious approach, and an entry that
owes it nothing is more interesting than a variation on it. If you write one,
you are invited to send it back as a pull request:

```
processors/<your-github-handle>_<short_name>.py
```

One self-contained file, a docstring at the top saying what it does and what it
needs, no keys and no weights committed. The full note, including the option of
linking your own repository instead, is in [`../README.md`](../README.md) under
"Build your own, and share it".
