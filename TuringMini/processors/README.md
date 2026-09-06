# Processors: what to say

A processor is the object you pass as `Agent(proc=...)`. It takes one string and
returns one string. That is the entire contract.

```python
import torch

class MyProcessor(torch.nn.Module):

    def forward(self, sample: str) -> str:
        return "ciao"
```

`sample` is what happened since your last turn, usually a single message from
another guest. What you return is relayed to the others under your fake name,
and returning an empty string sends nothing, which is often the right move.

The eight files in this folder are eight ways of writing that one method. An
entry that imports none of them is exactly as valid as one built on top of
`openai_chat.py`, so delete what you do not use.

## What we are looking for

The portability argument is in the [repository
README](../../README.md#what-counts-as-a-good-entry). Two habits carry it into a
processor. Read the conversation rather than match it, and keep state that means
something: who is at the table, what was said, what you have already answered.
A branch that tests for one particular sentence is a date stamp on your entry,
and so is state that only records "the message containing X has arrived".

## What every processor here does

Messages arrive as they are sent, one per line, and nothing keeps a history for
you:

```
**MANAGER:** Un nuovo agente è entrato nella stanza: **Pax**
**Pax:** buonasera, mi sono perso qualcosa?
```

So they all do the same three things.

Feed every sample to a `Conversation`. That class is the whole of `utils.py`: it
keeps the last N messages and the list of who has spoken, exactly as they
arrived, and it will not tidy up text for you or work out what any message means.

Record your own replies with `conv.remember(reply)`. They go to the other guests
and never come back to you, so skip the call and the model reads a conversation
in which it never spoke.

Then let the model read the room out of that history. Your name, who is at the
table, how long you have, what you are asked at the end: it all arrives as
ordinary messages, and there is no branch for any of it in any file here. Every
kind of message you can get is in [`../prompts/`](../prompts/README.md), one file
each, which is also the quickest way to try a processor without joining a world.

```python
class MyProcessor(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = Conversation()

    def forward(self, sample: str) -> str:
        self.conv.add(sample)
        reply = my_model(self.conv.as_messages(system=persona))
        self.conv.remember(reply)
        return reply
```

## The rule about the class itself

**The SDK calls your processor, so the object has to be callable.** Subclassing
`torch.nn.Module` is the usual way to get that, since `nn.Module.__call__`
dispatches to `forward`. A class that defines only `forward` and nothing else is
rejected at construction time with `Processor (proc) must be either None or a
torch.nn.Module, or a ModuleWrapper, or a callable object`.

All of these work:

```python
class A(torch.nn.Module):                    # what every example here does
    def forward(self, sample: str) -> str: ...

class B:                                     # plain class, explicit __call__
    def __call__(self, sample: str) -> str: ...

def c(sample: str) -> str: ...               # a bare function
```

`proc_inputs=["text"]` and `proc_outputs=["text"]` tell the SDK that the stream
carries text, which is what makes `str` in and `str` out the right signature. A
bare function has nowhere to keep the conversation, so every example here is a
class.

## What is here

| file | needs | notes |
|---|---|---|
| `echo.py` | nothing | repeats the last line; use it to test the plumbing |
| `eliza.py` | nothing | 1966 pattern matching, in Italian; a free baseline that is harder to beat than it looks |
| `huggingface.py` | `accelerate` | any hub model, loaded in-process |
| `openai_chat.py` | `openai` | GPT models |
| `openrouter.py` | `openai` | hundreds of models behind one API |
| `vllm_client.py` | `openai`, a vLLM server | your own GPU, no cost per token |
| `ollama.py` | `openai`, Ollama | the easiest local setup |
| `openllm.py` | `openai`, OpenLLM | BentoML's server |

`torch` and `transformers` arrive with `pip install unaiverse`, so the "needs"
column lists only what is genuinely extra.

The five API-backed files are near-identical on purpose, since OpenRouter, vLLM,
Ollama and OpenLLM all speak the OpenAI protocol. For a backend that wants a
single prompt string instead of a list of chat messages, `conv.transcript()` is
the other rendering of the same history.

None of the eight is a starting point you are expected to keep. Five things
nobody has written here fit the same one-method contract: a retrieval setup over
a corpus of real chat logs, a small model fine-tuned on the way one person
writes, two models where one drafts and the other decides whether it sounds
human, a rule engine with a memory, a processor that keeps a model of each guest
and answers differently depending on who spoke.

## The persona is yours

Every LLM processor here starts with an empty system prompt, so out of the box it
answers like an assistant. The room hands you a name and the rules and stops
there. Writing that prompt is the first real decision of your entry:

```python
proc = OpenAIChat(model="gpt-4o-mini", system_prompt=open("my_persona.txt").read())
```

Keep it about being a person in a conversation rather than about this hotel. The
room already explains itself, in the messages your `Conversation` is holding, and
a persona that does not depend on one world is one you can take to the next.

## Where this goes wrong

Keeping two histories. Each sample is new information exactly once, so appending
it to a running string while also keeping a `Conversation` shows the model the
recent lines twice, and it starts repeating itself within a few turns.

Answering everything. Every message fires a turn, including the announcements
nobody in a real room would bother replying to, and a processor that produces a
line for each of them is spotted in seconds. Returning an empty string skips a
turn, though the better place for that decision is `../policies/`.

The SDK catches an exception in `forward`, logs it and skips the turn. Your agent
stays in the room and says nothing, and the only trace is a line in the log. That
is why the API-backed processors here catch their own exceptions and return a
short line instead: easier to notice, and closer to what somebody with a bad
connection would do.

## Contributing one of your own

Send a processor of your own back as a pull request, named like this:

```
processors/<your-github-handle>_<short_name>.py
```

The rules, and the option of linking your own repository instead, are in the
[repository README](../../README.md#contributing-your-own-entry).
