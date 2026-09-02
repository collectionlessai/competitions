import unittest
from types import SimpleNamespace
from unittest.mock import patch

from policies.mood import Mood
from policies.read_and_type import ReadAndType


class FakeStream:

    def __init__(self, sample):
        self.sample = sample
        self.calls = []

    def get(self, **kwargs):
        self.calls.append(kwargs)
        return self.sample


class FakeAgent:

    def __init__(self, sample):
        self.stream = FakeStream(sample)
        self.proc_last_inputs = ("stale" * 1000,)
        self.proc_last_outputs = ("123456",)

    def get_stream(self, name, data_type=None):
        return self.stream if (name, data_type) == ("processor_in", "text") else None

    @staticmethod
    def uai_preprocess(text):
        return text.replace("WIRE", "view"), None


class PolicyTests(unittest.TestCase):

    def test_mood_honours_configured_start(self):
        actions = [SimpleNamespace(name="process")]
        opts = {}

        with patch("policies.mood.time.monotonic", return_value=100.0), \
                patch("policies.mood.random.random", return_value=0.0), \
                patch("policies.mood.random.uniform", return_value=1.0), \
                patch("policies.mood.random.choices") as choices:
            Mood(start="into it")(0, None, actions, opts)

        self.assertEqual(opts["mood"], "into it")
        self.assertEqual(opts["mood_until"], 160.0)
        choices.assert_not_called()

    def test_mood_rejects_unknown_start(self):
        with self.assertRaises(ValueError):
            Mood(start="unknown")

    def test_read_and_type_uses_pending_projected_input(self):
        agent = FakeAgent("WIRE")
        action = SimpleNamespace(name="process", system_interaction=SimpleNamespace(uuid="system-uuid"))
        opts = {"agent": agent}

        with patch("policies.read_and_type.time.monotonic", return_value=10.0):
            result = ReadAndType(read_cps=2.0, type_cps=2.0, think=0.0)(0, None, [action], opts)

        self.assertEqual(result, (-1, None))
        self.assertEqual(opts["ready_at"], 15.0)  # len("view") / 2 + len("123456") / 2
        self.assertEqual(agent.stream.calls[0]["uuid"], "system-uuid")

    def test_read_and_type_uses_received_interaction_uuid(self):
        agent = FakeAgent("ciao")
        action = SimpleNamespace(name="process", system_interaction=SimpleNamespace(uuid="system-uuid"))
        request = SimpleNamespace(uuid="vote-uuid")

        with patch("policies.read_and_type.time.monotonic", return_value=10.0):
            ReadAndType(read_cps=5.0, type_cps=6.0, think=0.0)(
                0, request, [action], {"agent": agent}
            )

        self.assertEqual(agent.stream.calls[0]["uuid"], "vote-uuid")


if __name__ == "__main__":
    unittest.main()
