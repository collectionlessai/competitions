import runpy
import unittest
from pathlib import Path
from unittest.mock import patch

from policies.fixed_delay import FixedDelay
from processors.eliza import Eliza


class FakeAgent:
    calls = []

    def __init__(self, **kwargs):
        self.options = kwargs
        self.calls.append(kwargs)


class FakeNode:
    calls = []
    runs = []

    def __init__(self, **kwargs):
        self.options = kwargs
        self.calls.append(kwargs)

    def run(self, **kwargs):
        self.runs.append(kwargs)


class EntrypointTests(unittest.TestCase):

    def setUp(self):
        FakeAgent.calls.clear()
        FakeNode.calls.clear()
        FakeNode.runs.clear()

    def test_starter_constructs_and_joins_public_world(self):
        entrypoint = Path(__file__).parents[1] / "my_agent.py"

        with patch("unaiverse.agent.Agent", FakeAgent), \
                patch("unaiverse.networking.node.node.Node", FakeNode):
            runpy.run_path(str(entrypoint), run_name="__main__")

        self.assertEqual(len(FakeAgent.calls), 1)
        agent_options = FakeAgent.calls[0]
        self.assertIsInstance(agent_options["proc"], Eliza)
        self.assertIsInstance(agent_options["policy_filter"], FixedDelay)
        self.assertEqual(agent_options["proc_inputs"], ["text"])
        self.assertEqual(agent_options["proc_outputs"], ["text"])

        self.assertIsInstance(FakeNode.calls[0]["hosted"], FakeAgent)
        self.assertEqual(FakeNode.runs, [{"join_world": "jolly-mayer/TuringHotelItaly"}])


if __name__ == "__main__":
    unittest.main()
