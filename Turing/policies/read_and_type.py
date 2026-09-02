"""Estimate a delay from message length, a thinking pause, then typing time.

The delay combines three estimates, so longer inputs or replies take more time
than short ones:

    reading  = characters that arrived since your last turn / reading speed
    thinking = a random pause
    typing   = length of what you wrote / typing speed

The filter runs before the processor, so both measurements describe the
previous completed turn. It first checks whether the processor saved
``last_input`` or ``last_output`` attributes itself, then uses the equivalent
framework hooks. This keeps the example small and lets a custom processor
expose its own state.
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


def processor_turn(opts, attribute: str, fallback: str) -> str:
    """Read a string saved by the processor, then try the framework hook."""
    processor = getattr(getattr(opts.get("agent"), "proc", None), "module", None)
    value = getattr(processor, attribute, None)
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    if isinstance(value, str):
        return value
    return last_turn(opts, fallback)


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
            previous_input = processor_turn(opts, "last_input", "proc_last_inputs")
            previous_output = processor_turn(opts, "last_output", "proc_last_outputs")
            thinking = random.expovariate(1.0 / self.think) if self.think > 0 else 0.0
            delay = (len(previous_input) / self.read_cps
                     + thinking
                     + len(previous_output) / self.type_cps)

            # Cap the delay so a long backlog cannot postpone the reply indefinitely.
            opts["ready_at"] = now + min(delay, 45.0)

        if now < opts["ready_at"]:
            return -1, None

        del opts["ready_at"]
        return action_id, request
