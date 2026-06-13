from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from kmmad_full_report import package_stats  # noqa: E402


class FullReportTest(unittest.TestCase):
    def make_package(self, row: dict[str, object]) -> Path:
        root = Path(tempfile.mkdtemp())
        package_dir = root / "package"
        image_path = package_dir / "test" / "images" / "mme_realworld" / "sample.png"
        image_path.parent.mkdir(parents=True)
        image_path.write_bytes(b"fake")
        metadata = package_dir / "test" / "metadata.jsonl"
        metadata.parent.mkdir(parents=True, exist_ok=True)
        metadata.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
        (package_dir / "hf_package_manifest.json").write_text(
            json.dumps({"splits": {"test": {"num_rows": 1}}, "media_packaging": {"copied_files": 1, "missing_media_refs": 0}}),
            encoding="utf-8",
        )
        (package_dir / "hf_package_validation.json").write_text(json.dumps({"status": "passed", "errors": []}), encoding="utf-8")
        self.addCleanup(lambda: __import__("shutil").rmtree(root, ignore_errors=True))
        return package_dir

    def base_row(self) -> dict[str, object]:
        return {
            "record_id": "mme_realworld/test/assistant-title",
            "source_record_id": "mme_realworld/test/assistant-title",
            "question_original": "When will the Administrative Assistant screening test be held?",
            "question_ko": "행정 보조원 선발 시험은 언제 열리나요?",
            "file_name": "images/mme_realworld/sample.png",
            "media_files": ["test/images/mme_realworld/sample.png"],
            "options_original_json": json.dumps(["(A) 1", "(B) 2"], ensure_ascii=False),
            "options_ko_json": json.dumps(["(A) 1", "(B) 2"], ensure_ascii=False),
        }

    def test_package_stats_does_not_flag_assistant_job_title(self) -> None:
        stats = package_stats(self.make_package(self.base_row()))

        self.assertEqual(stats["review_candidate_count"], 0)

    def test_package_stats_flags_actual_assistant_artifact(self) -> None:
        row = self.base_row()
        row["question_ko"] = "Assistant: 이 응답은 번역이 아닙니다."

        stats = package_stats(self.make_package(row))

        self.assertEqual(stats["review_candidate_count"], 1)
        self.assertEqual(stats["review_candidates_sample"][0]["reasons"], ["assistant_artifact_text"])

    def test_package_stats_flags_option_cardinality_mismatch(self) -> None:
        row = self.base_row()
        row["options_ko_json"] = json.dumps(["(A) 1"], ensure_ascii=False)

        stats = package_stats(self.make_package(row))

        self.assertEqual(stats["review_candidate_count"], 1)
        self.assertEqual(stats["review_candidates_sample"][0]["reasons"], ["options_cardinality_mismatch"])

    def test_package_stats_flags_missing_or_unparseable_translated_options(self) -> None:
        for translated_options in ("", "{not json"):
            with self.subTest(translated_options=translated_options):
                row = self.base_row()
                row["options_ko_json"] = translated_options

                stats = package_stats(self.make_package(row))

                self.assertEqual(stats["review_candidate_count"], 1)
                self.assertEqual(stats["review_candidates_sample"][0]["reasons"], ["options_missing_or_unparseable"])


if __name__ == "__main__":
    unittest.main()
