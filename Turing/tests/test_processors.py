import sys
import unittest
from types import ModuleType
from unittest.mock import patch

import torch

from utils import Conversation


if "openai" not in sys.modules:
    openai = ModuleType("openai")
    openai.OpenAI = object
    sys.modules["openai"] = openai

from processors.huggingface import HuggingFace
from processors.ollama import Ollama
from processors.openai_chat import OpenAIChat
from processors.openllm import OpenLLM
from processors.openrouter import OpenRouter
from processors.vllm_client import VLLMClient
from processors.eliza import main as eliza_main


FALLBACKS = {
    HuggingFace: "un attimo",
    Ollama: "un attimo",
    OpenAIChat: "scusate, mi è saltata la connessione",
    OpenLLM: "un secondo",
    OpenRouter: "aspetta un secondo",
    VLLMClient: "torno subito",
}


class ProcessorTests(unittest.TestCase):

    def test_failure_fallbacks_are_remembered(self):
        for processor_class, expected in FALLBACKS.items():
            with self.subTest(processor=processor_class.__name__):
                class Broken(processor_class):

                    def __init__(self):
                        torch.nn.Module.__init__(self)
                        self.conv = Conversation(room_start_pattern=None)
                        self.system_prompt = ""

                    @staticmethod
                    def complete(messages):
                        raise RuntimeError("offline")

                processor = Broken()
                with patch("builtins.print"):
                    self.assertEqual(processor("**Pax:** ciao"), expected)
                self.assertEqual(processor.conv.last_message(mine=True).text, expected)

    def test_eliza_module_entrypoint_handles_end_of_input(self):
        with patch("builtins.input", side_effect=EOFError), patch("builtins.print") as output:
            eliza_main()

        output.assert_any_call("Eliza is ready. Press Ctrl-D or Ctrl-C to stop.")


if __name__ == "__main__":
    unittest.main()
