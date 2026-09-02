"""Estimate a delay from message length, a thinking pause, then typing time.

The delay combines three estimates, so longer inputs or replies take more time
than short ones:

    reading  = characters that arrived since your last turn / reading speed
    thinking = a random pause
    typing   = length of what you wrote / typing speed

The pending processor input is read non-destructively from the agent's input
stream. The framework exposes the previous processor output through
opts["agent"], which provides the typing estimate for the next turn.

Because the filter runs before the processor, the next reply does not exist yet.
The typing estimate therefore uses the previous reply. This shifts only that
component by one turn but retains its contribution over a conversation.
"""

import time
import random


def last_turn(opts, attribute: str) -> str:
    """Read proc_last_inputs or proc_last_outputs as a string.

    Both values are tuples and remain None until the first processor turn. In
    either case, return an empty string when no text is available.
    """
    value = getattr(opts.get("agent"), attribute, None)
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    return value if isinstance(value, str) else ""


def pending_input(opts, request, action, requested_by: str) -> str:
    """Return the text waiting for this process action without consuming it.

    Stream reads are deduplicated per requester, so this policy uses its own
    requester name and does not interfere with Agent.process. UAI input is
    projected through the same callback that the processor wrapper uses.
    """
    agent = opts.get("agent")
    get_stream = getattr(agent, "get_stream", None)
    if not callable(get_stream):
        return last_turn(opts, "proc_last_inputs")

    stream = get_stream("processor_in", data_type="text")
    if stream is None:
        return last_turn(opts, "proc_last_inputs")

    interaction = request if request is not None else getattr(action, "system_interaction", None)
    uuid = getattr(interaction, "uuid", None)
    sample = stream.get(requested_by=requested_by, uuid=uuid)
    if not isinstance(sample, str):
        return last_turn(opts, "proc_last_inputs")

    project = getattr(agent, "uai_preprocess", None)
    if callable(project):
        try:
            sample, _ = project(sample)
        except Exception:
            pass
    return sample


class ReadAndType:

    def __init__(self, read_cps: float = 25.0, type_cps: float = 6.0, think: float = 2.0,
                 actions=("process",)):
        self.read_cps = read_cps      # Reading speed in characters per second.
        self.type_cps = type_cps      # Typing speed in characters per second.
        self.think = think            # Mean random pause, or 0 to disable it.
        self.actions = set(actions)

    def __call__(self, action_id, request, all_actions, opts):
        if all_actions[action_id].name not in self.actions:
            return action_id, request

        now = time.monotonic()

        if "ready_at" not in opts:
            action = all_actions[action_id]
            current_input = pending_input(opts, request, action, f"ReadAndType:{id(self)}")
            fresh = len(current_input)
            thinking = random.expovariate(1.0 / self.think) if self.think > 0 else 0.0
            delay = (fresh / self.read_cps
                     + thinking
                     + len(last_turn(opts, "proc_last_outputs")) / self.type_cps)

            # Cap the delay so a long backlog cannot postpone the reply indefinitely.
            opts["ready_at"] = now + min(delay, 45.0)

        if now < opts["ready_at"]:
            return -1, None

        del opts["ready_at"]
        return action_id, request
