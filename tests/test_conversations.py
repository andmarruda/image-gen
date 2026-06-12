import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.conversations import get_manifest, normalize_id, prepare, save_revision


class FakeImage:
    def save(self, path, format=None):
        Path(path).write_bytes(b"fake-png")


class ConversationMemoryTests(unittest.TestCase):
    def test_rejects_unsafe_conversation_id(self):
        with self.assertRaises(ValueError):
            normalize_id("../outside")

    def test_saves_and_reuses_latest_image(self):
        with tempfile.TemporaryDirectory() as directory:
            environment = {
                "CONVERSATION_DIR": directory,
                "CONVERSATION_MEMORY_ENABLED": "true",
                "CONVERSATION_MAX_REVISIONS": "2",
            }
            with patch.dict(os.environ, environment, clear=True):
                image = FakeImage()
                save_revision("study-session", image, "create a red square")

                with patch("src.conversations.load_latest", return_value=image):
                    prepared, conversation_id = prepare(
                        {"conversation_id": "study-session", "prompt": "make it blue"}
                    )

                self.assertEqual(conversation_id, "study-session")
                self.assertEqual(len(prepared["images"]), 1)
                self.assertEqual(len(get_manifest("study-session")["revisions"]), 1)

    def test_limits_revision_history(self):
        with tempfile.TemporaryDirectory() as directory:
            environment = {
                "CONVERSATION_DIR": directory,
                "CONVERSATION_MEMORY_ENABLED": "true",
                "CONVERSATION_MAX_REVISIONS": "2",
            }
            with patch.dict(os.environ, environment, clear=True):
                image = FakeImage()
                for index in range(3):
                    save_revision("limited", image, f"revision {index}")

                self.assertEqual(len(get_manifest("limited")["revisions"]), 2)

    def test_rejects_missing_requested_revision(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {"CONVERSATION_DIR": directory}, clear=True):
                with patch("src.conversations.load_revision", return_value=None):
                    with self.assertRaisesRegex(ValueError, "revision_id was not found"):
                        prepare(
                            {
                                "conversation_id": "study-session",
                                "revision_id": "old-version",
                                "prompt": "try another direction",
                            }
                        )


if __name__ == "__main__":
    unittest.main()
