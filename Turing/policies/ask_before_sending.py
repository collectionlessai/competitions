"""Advanced policy for the updated Turing Hotel Italy world only.

It deliberately exploits that world's ``process -> msg_prepared -> send_msg``
sequence. At ``send_msg``, ``forward`` has already run and its reply is waiting
in the guest's stdout stream. The policy asks the same model for a second,
private decision:

- ``silenzio`` discards the prepared reply;
- any other answer keeps it and adds a typing delay.

The processor must expose ``conv`` and ``complete(messages)``, as the included
LLM processors do. This policy is not portable to other worlds or lone-wolf
agents, whose state machines do not provide this Turing-specific send stage.
"""

import time


class AskBeforeSending:

    def __init__(self, type_cps: float = 6.0, silence: str = "silenzio"):
        self.type_cps = type_cps
        self.silence = silence.casefold()

    def __call__(self, action_id, request, all_actions, opts):
        # This action exists in the updated Turing Hotel Italy guest only.
        if all_actions[action_id].name != "send_msg":
            return action_id, request

        processor = opts["agent"].proc.module
        conversation = processor.conv
        reply = conversation.last_output
        if not reply:
            return action_id, request

        if "ready_at" not in opts:
            prompt = (
                "Decidi se inviare davvero l'ultima risposta nella conversazione.\n\n"
                f"{conversation.transcript()}\n\n"
                "Rispondi solo SILENZIO per non inviarla. "
                "Qualsiasi altra risposta significa: inviala."
            )
            decision = processor.complete([{"role": "user", "content": prompt}])

            if decision.strip().casefold() == self.silence:
                conversation.discard_last_output()
                opts["agent"].stdout.clear_all_data()
                return action_id, request  # Complete send_msg without a message.

            typing_time = len(reply) / self.type_cps
            opts["ready_at"] = time.monotonic() + typing_time

        if time.monotonic() < opts["ready_at"]:
            return -1, None  # The prepared reply is still being typed.

        del opts["ready_at"]
        return action_id, request
