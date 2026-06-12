from __future__ import annotations

import unittest

from kmmad_minimal_openai_server import final_user_text, normalize_backend


class MinimalOpenAIServerTests(unittest.TestCase):
    def test_auto_backend_selects_translategemma(self) -> None:
        self.assertEqual(normalize_backend("auto", "google/translategemma-27b-it"), "translategemma")
        self.assertEqual(normalize_backend("auto", "Qwen/Qwen2.5-7B-Instruct"), "causal_lm")

    def test_final_user_text_handles_openai_string_messages(self) -> None:
        messages = [
            {"role": "system", "content": "Translate to Korean."},
            {"role": "user", "content": "A red square."},
        ]
        self.assertEqual(final_user_text(messages), "A red square.")

    def test_final_user_text_handles_multimodal_text_parts(self) -> None:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "First sentence."},
                    {"type": "image_url", "image_url": {"url": "ignored"}},
                    {"type": "text", "text": "Second sentence."},
                ],
            }
        ]
        self.assertEqual(final_user_text(messages), "First sentence.\nSecond sentence.")


if __name__ == "__main__":
    unittest.main()
