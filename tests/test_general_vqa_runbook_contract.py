from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNBOOK = ROOT / "docs" / "kmmad_translation_smoke_runbook.md"


class GeneralVqaRunbookContractTest(unittest.TestCase):
    def test_general_vqa_section_separates_mmad_gate(self) -> None:
        text = RUNBOOK.read_text(encoding="utf-8")
        self.assertIn("## General VQA multi-benchmark smoke", text)
        section = text.split("## General VQA multi-benchmark smoke", 1)[1]
        self.assertIn("MMAD full-download and H200 gate above applies to the original MMAD smoke only", section)
        self.assertIn("completion gate for this General VQA first pass", section)
        self.assertIn("**not**", section)
        self.assertIn("Mock output is sanity-only", section)
        self.assertIn("actual local GPU LLM translation samples", section)
        self.assertIn("actual `openai-oauth` endpoint translation samples", section)

    def test_openai_oauth_and_api_key_boundaries_are_documented(self) -> None:
        section = RUNBOOK.read_text(encoding="utf-8").split("## General VQA multi-benchmark smoke", 1)[1]
        self.assertIn("ghcr.io/pjh6029/snupi-personal-codex:20260414", section)
        self.assertIn("CODEX_HOME=/root/work/.codex", section)
        self.assertIn("127.0.0.1", section)
        self.assertIn("no paid/live OpenAI API call", section)
        self.assertIn('endpoint_provider = "openai_oauth"', section)
        self.assertIn('auth_mode = "bearer_env"', section)


if __name__ == "__main__":
    unittest.main()
