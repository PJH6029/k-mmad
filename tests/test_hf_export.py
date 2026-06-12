from __future__ import annotations

import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from kmmad_hf_export import export_translations, main as hf_export_main, validate_export  # noqa: E402
from kmmad_translate_smoke import main as translate_main  # noqa: E402


class HfExportTest(unittest.TestCase):
    def test_general_vqa_translation_bundle_exports_to_loadable_hf_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            translations = root / "translations"
            rc = translate_main([
                "--dataset",
                str(ROOT / "tests" / "fixtures" / "general_vqa"),
                "--benchmarks",
                "blink,mmmu_pro",
                "--output-dir",
                str(translations),
                "--sample-size",
                "1",
                "--mock",
                "--concurrency",
                "2",
            ])
            self.assertEqual(rc, 0)

            export_dir = root / "hf_export"
            manifest = export_translations(
                translation_outputs=[translations],
                output_dir=export_dir,
                dataset_name="k-general-vqa-ko",
                original_hf_dataset="fixture/general-vqa",
                original_hf_config="adapter-fixture",
                original_hf_revision="test-revision",
                default_split="test",
                fallback_benchmark_id="general_vqa",
                translation_run_id="fixture-run",
                translation_qc_status="passed",
                translation_qc_report="qc/report.json",
                hub_repo_id="pjh6029/k-general-vqa-ko",
            )

            self.assertEqual(manifest["validation"]["status"], "passed")
            self.assertFalse(manifest["export"]["hub_upload_performed"])
            self.assertIn("push_to_hub", manifest["export"]["push_to_hub_command"])
            validation = validate_export(export_dir)
            self.assertEqual(validation["status"], "passed")
            self.assertEqual(validation["splits"], {"val": 1, "validation": 1})

            datasets = importlib.import_module("datasets")
            loaded = datasets.load_from_disk(str(export_dir / "dataset"))
            self.assertEqual(set(loaded), {"val", "validation"})
            row = loaded["val"][0]
            self.assertEqual(row["benchmark_id"], "blink")
            self.assertEqual(row["translation_qc_status"], "passed")
            self.assertIn("한국어 번역 초안", row["question_ko"])
            self.assertIn("blink-001", row["source_record_json"])
            self.assertIn("한국어 번역 초안", json.loads(row["options_ko_json"])["A"])
            self.assertEqual(row["translation_artifact_path"], "input-00-translations/blink/translation_smoke.jsonl")

    def test_mmad_translation_bundle_exports_with_default_split_and_fallback_benchmark(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            translations = root / "mmad-translations"
            rc = translate_main([
                "--dataset",
                str(ROOT / "tests" / "fixtures" / "mmad_sample"),
                "--output-dir",
                str(translations),
                "--sample-size",
                "1",
                "--mock",
            ])
            self.assertEqual(rc, 0)

            export_dir = root / "hf-export"
            self.assertEqual(
                hf_export_main([
                    "--translation-output",
                    str(translations),
                    "--output-dir",
                    str(export_dir),
                    "--dataset-name",
                    "k-mmad-ko-fixture",
                    "--original-hf-dataset",
                    "jiang-cc/MMAD",
                    "--default-split",
                    "test",
                    "--translation-qc-status",
                    "passed",
                ]),
                0,
            )
            self.assertEqual(hf_export_main(["--validate-only", str(export_dir)]), 0)

            manifest = json.loads((export_dir / "hf_export_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["splits"]["test"]["num_rows"], 1)
            record = json.loads((export_dir / "data" / "test.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(record["benchmark_id"], "mmad")
            self.assertEqual(record["original_hf_dataset"], "jiang-cc/MMAD")
            self.assertIn("한국어 번역 초안", record["question_ko"])
            self.assertIn("mmad", record["record_id"])

    def test_export_refuses_non_empty_output_directory_without_deleting_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "existing"
            output_dir.mkdir()
            sentinel = output_dir / "keep.txt"
            sentinel.write_text("do not delete", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                export_translations(
                    translation_outputs=[],
                    output_dir=output_dir,
                    dataset_name="unused",
                    original_hf_dataset="unused",
                    original_hf_config=None,
                    original_hf_revision=None,
                    default_split="test",
                    fallback_benchmark_id="mmad",
                    translation_run_id=None,
                    translation_qc_status="unchecked",
                    translation_qc_report=None,
                    hub_repo_id=None,
                    skip_translation_validation=True,
                )

            self.assertEqual(sentinel.read_text(encoding="utf-8"), "do not delete")


    def test_export_redacts_secrets_and_validation_scans_data_shards(self) -> None:
        secret = "sk-" + "testsecretvalue123456789"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            translations = root / "translations"
            translations.mkdir()
            (translations / "translation_smoke.jsonl").write_text(
                json.dumps(
                    {
                        "source": {
                            "id": "secret-row",
                            "image_path": f"images/{secret}.png",
                            "question": "Original question",
                            "answer": "A",
                            "api_key": secret,
                        },
                        "translated": {
                            "question": f"한국어 질문 {secret}",
                            "options": [f"A. 한국어 선택 {secret}"],
                        },
                        "translation_scope": ["question", "options"],
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            export_dir = root / "hf-export"
            export_translations(
                translation_outputs=[translations],
                output_dir=export_dir,
                dataset_name="k-secret-test",
                original_hf_dataset="fixture/secret",
                original_hf_config=None,
                original_hf_revision=None,
                default_split="test",
                fallback_benchmark_id="mmad",
                translation_run_id=None,
                translation_qc_status="passed",
                translation_qc_report="qc.json",
                hub_repo_id=None,
                skip_translation_validation=True,
            )

            data_text = (export_dir / "data" / "test.jsonl").read_text(encoding="utf-8")
            self.assertNotIn(secret, data_text)
            self.assertIn("sk-<redacted>", data_text)
            datasets = importlib.import_module("datasets")
            loaded = datasets.load_from_disk(str(export_dir / "dataset"))
            self.assertNotIn(secret, json.dumps(loaded["test"][0], ensure_ascii=False))
            self.assertEqual(validate_export(export_dir)["status"], "passed")

            row = json.loads(data_text.splitlines()[0])
            row["question_ko"] = secret
            (export_dir / "data" / "test.jsonl").write_text(
                json.dumps(row, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            validation = validate_export(export_dir)
            self.assertEqual(validation["status"], "failed")
            self.assertTrue(any("data/test.jsonl" in error for error in validation["errors"]))

            orphan = dict(row)
            orphan["record_id"] = "mmad/test/orphan"
            orphan["question_ko"] = secret
            (export_dir / "data" / "orphan.jsonl").write_text(
                json.dumps(orphan, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            (export_dir / "data" / "test.jsonl").write_text(
                json.dumps({**row, "question_ko": "한국어 정상"}, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            validation = validate_export(export_dir)
            self.assertEqual(validation["status"], "failed")
            self.assertTrue(any("unexpected split JSONL shard" in error for error in validation["errors"]))
            self.assertTrue(any("data/orphan.jsonl" in error for error in validation["errors"]))

    def test_mmad_multi_turn_records_get_unique_stable_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            translations = root / "translations"
            translations.mkdir()
            rows = [
                {
                    "source_id": "shared-top-level",
                    "source": {
                        "mmad_image_key": "images/shared.png",
                        "conversation_index": idx,
                        "Question": f"Question {idx}",
                        "Options": ["A. yes", "B. no"],
                        "answer": "A",
                    },
                    "translated": {
                        "Question": f"한국어 질문 {idx}",
                        "Options": ["A. 예", "B. 아니요"],
                    },
                    "translation_scope": ["Question", "Options"],
                }
                for idx in range(2)
            ]
            (translations / "translation_smoke.jsonl").write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )

            export_dir = root / "hf-export"
            export_translations(
                translation_outputs=[translations],
                output_dir=export_dir,
                dataset_name="k-mmad-turn-test",
                original_hf_dataset="jiang-cc/MMAD",
                original_hf_config=None,
                original_hf_revision=None,
                default_split="test",
                fallback_benchmark_id="mmad",
                translation_run_id=None,
                translation_qc_status="passed",
                translation_qc_report="qc.json",
                hub_repo_id=None,
                skip_translation_validation=True,
            )

            records = [
                json.loads(line)
                for line in (export_dir / "data" / "test.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            record_ids = [record["record_id"] for record in records]
            self.assertEqual(len(record_ids), len(set(record_ids)))
            self.assertTrue(any("shared-top-level#turn-0" in record_id for record_id in record_ids))
            self.assertTrue(any("shared-top-level#turn-1" in record_id for record_id in record_ids))
            self.assertEqual(validate_export(export_dir)["status"], "passed")


    def test_directory_without_summary_or_direct_bundle_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stale = root / "stale"
            stale.mkdir()
            (stale / "translation_smoke.jsonl").write_text(
                json.dumps({"translated": {"question": "한국어"}, "source": {"id": "stale"}}, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(FileNotFoundError, "requires general_vqa_translation_summary"):
                export_translations(
                    translation_outputs=[root],
                    output_dir=root / "hf-export",
                    dataset_name="reject-stale",
                    original_hf_dataset="fixture/reject",
                    original_hf_config=None,
                    original_hf_revision=None,
                    default_split="test",
                    fallback_benchmark_id="mmad",
                    translation_run_id=None,
                    translation_qc_status="unchecked",
                    translation_qc_report=None,
                    hub_repo_id=None,
                    skip_translation_validation=True,
                )


    def test_missing_id_fallback_is_order_independent_and_content_addressed(self) -> None:
        rows = [
            {
                "source": {"question": "First no id", "answer": "A"},
                "translated": {"question": "한국어 첫 번째"},
                "translation_scope": ["question"],
            },
            {
                "source": {"question": "Second no id", "answer": "B"},
                "translated": {"question": "한국어 두 번째"},
                "translation_scope": ["question"],
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ids_by_question: list[dict[str, str]] = []
            for name, ordered_rows in (("forward", rows), ("reverse", list(reversed(rows)))):
                translations = root / name
                translations.mkdir()
                (translations / "translation_smoke.jsonl").write_text(
                    "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in ordered_rows),
                    encoding="utf-8",
                )
                export_dir = root / f"hf-{name}"
                export_translations(
                    translation_outputs=[translations],
                    output_dir=export_dir,
                    dataset_name="k-order-test",
                    original_hf_dataset="fixture/order",
                    original_hf_config=None,
                    original_hf_revision=None,
                    default_split="test",
                    fallback_benchmark_id="mmad",
                    translation_run_id=None,
                    translation_qc_status="passed",
                    translation_qc_report="qc.json",
                    hub_repo_id=None,
                    skip_translation_validation=True,
                )
                exported = [
                    json.loads(line)
                    for line in (export_dir / "data" / "test.jsonl").read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                ids_by_question.append({json.loads(row["source_record_json"])["question"]: row["record_id"] for row in exported})

            self.assertEqual(ids_by_question[0], ids_by_question[1])
            self.assertTrue(all("row-" in record_id for record_id in ids_by_question[0].values()))

    def test_same_named_input_roots_get_disambiguated_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundles = []
            for parent, row_id in (("a", "row-a"), ("b", "row-b")):
                bundle = root / parent / "bundle"
                bundle.mkdir(parents=True)
                (bundle / "translation_smoke.jsonl").write_text(
                    json.dumps(
                        {
                            "source": {"id": row_id, "question": f"Question {row_id}", "answer": "A"},
                            "translated": {"question": f"한국어 {row_id}"},
                            "translation_scope": ["question"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                bundles.append(bundle)

            export_dir = root / "hf-export"
            manifest = export_translations(
                translation_outputs=bundles,
                output_dir=export_dir,
                dataset_name="k-root-label-test",
                original_hf_dataset="fixture/roots",
                original_hf_config=None,
                original_hf_revision=None,
                default_split="test",
                fallback_benchmark_id="mmad",
                translation_run_id=None,
                translation_qc_status="passed",
                translation_qc_report="qc.json",
                hub_repo_id=None,
                skip_translation_validation=True,
            )

            self.assertEqual(manifest["translation"]["input_roots"], ["input-00-bundle", "input-01-bundle"])
            records = [
                json.loads(line)
                for line in (export_dir / "data" / "test.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(
                sorted(record["translation_artifact_path"] for record in records),
                ["input-00-bundle/translation_smoke.jsonl", "input-01-bundle/translation_smoke.jsonl"],
            )

    def test_validate_export_rejects_missing_required_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            (output_dir / "README.md").write_text("# incomplete", encoding="utf-8")
            result = validate_export(output_dir)
            self.assertEqual(result["status"], "failed")
            self.assertTrue(any("manifest missing" in error for error in result["errors"]))
            self.assertTrue(any("HF dataset directory missing" in error for error in result["errors"]))


if __name__ == "__main__":
    unittest.main()
