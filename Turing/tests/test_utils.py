import unittest
from pathlib import Path

from utils import Conversation, DISPLAY_EVENT_SEPARATOR, EVENT_SEPARATOR, SPEAKER


PROMPTS = Path(__file__).parents[1] / "prompts"


class ConversationTests(unittest.TestCase):

    def test_record_separator_splits_a_batch(self):
        conv = Conversation(room_start_pattern=None)
        events = conv.add(f"**MANAGER:** entra Pax{EVENT_SEPARATOR}**Pax:** ciao")

        self.assertEqual([(m.speaker, m.text) for m in events],
                         [("MANAGER", "entra Pax"), ("Pax", "ciao")])

    def test_existing_positional_me_argument_stays_compatible(self):
        conv = Conversation(80, SPEAKER, "me", room_start_pattern=None)
        conv.remember("ciao")

        self.assertEqual(conv.transcript(), "me: ciao")

    def test_newlines_stay_inside_one_event(self):
        conv = Conversation(room_start_pattern=None)
        events = conv.add("**Ivy:** prima riga\nseconda riga")

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].speaker, "Ivy")
        self.assertEqual(events[0].text, "prima riga\nseconda riga")

    def test_room_guide_resets_the_previous_history(self):
        conv = Conversation()
        conv.add("**Ivy:** messaggio della vecchia stanza")
        conv.remember("mia vecchia risposta")
        conv.add("**MANAGER:** Benvenuto/a, ti chiami **Roy** e per ora sei solo/a.")

        self.assertEqual(len(conv.history), 1)
        self.assertEqual(conv.history[0].speaker, "MANAGER")
        self.assertNotIn("vecchia", conv.transcript())

    def test_room_guide_resets_earlier_event_in_the_same_batch(self):
        conv = Conversation()
        conv.add(
            "**Ivy:** ultimo messaggio della vecchia stanza"
            f"{EVENT_SEPARATOR}"
            "**MANAGER:** Benvenuto/a, ti chiami **Roy** e per ora sei solo/a."
            f"{EVENT_SEPARATOR}"
            "**Pax:** ciao dalla nuova stanza"
        )

        self.assertEqual([message.speaker for message in conv.history], ["MANAGER", "Pax"])
        self.assertNotIn("vecchia", conv.transcript())

    def test_current_multiline_prompt_is_one_manager_event(self):
        sample = (PROMPTS / "01_start.txt").read_text(encoding="utf-8").strip()
        events = Conversation().add(sample)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].speaker, "MANAGER")
        self.assertIn("- **Ivy**\n- **Pax**", events[0].text)

    def test_display_separator_fixture_replays_as_a_real_batch(self):
        sample = (PROMPTS / "04_batch.txt").read_text(encoding="utf-8").strip()
        sample = sample.replace(DISPLAY_EVENT_SEPARATOR, EVENT_SEPARATOR)
        events = Conversation(room_start_pattern=None).add(sample)

        self.assertEqual([m.speaker for m in events], ["MANAGER", "Pax"])


if __name__ == "__main__":
    unittest.main()
