from __future__ import annotations

import importlib
import base64
import contextlib
import functools
import http.server
import json
import sys
import tempfile
import threading
import unittest
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from kmmad_hf_export import (  # noqa: E402
    export_translations,
    main as hf_export_main,
    validate_export,
    validate_self_contained_package,
)
from kmmad_translate_smoke import main as translate_main  # noqa: E402


PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
)


class HfExportTest(unittest.TestCase):
    def fetch_from_static_dir(self, directory: Path, relative_url: str) -> int:
        class QuietHandler(http.server.SimpleHTTPRequestHandler):
            def log_message(self, format: str, *args: object) -> None:  # noqa: A002 - stdlib signature.
                return

        handler = functools.partial(QuietHandler, directory=str(directory))
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            url = f"http://127.0.0.1:{server.server_port}/{relative_url}"
            with contextlib.closing(urllib.request.urlopen(url, timeout=5)) as response:
                response.read()
                return int(response.status)
        finally:
            server.shutdown()
            thread.join(timeout=5)
            server.server_close()

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

    def test_self_contained_imagefolder_package_copies_media_and_omits_process_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            media_root = root / "source-media"
            (media_root / "images").mkdir(parents=True)
            (media_root / "images" / "sample.png").write_bytes(PNG_1X1)
            translations = root / "translations"
            translations.mkdir()
            (translations / "translation_smoke.jsonl").write_text(
                json.dumps(
                    {
                        "source": {
                            "id": "clean-row",
                            "image_path": "images/sample.png",
                            "question": "Which defect is visible?",
                            "options": {"A": "scratch", "B": "dent"},
                            "answer": "A",
                        },
                        "translated": {
                            "question": "어떤 결함이 보이나요?",
                            "options": {"A": "긁힘", "B": "찌그러짐"},
                        },
                        "translation_scope": ["question", "options"],
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            export_dir = root / "clean-package"
            visualizer_dir = root / "visualizer"
            manifest = export_translations(
                translation_outputs=[translations],
                output_dir=export_dir,
                dataset_name="k-clean-package",
                original_hf_dataset="fixture/clean",
                original_hf_config=None,
                original_hf_revision=None,
                default_split="test",
                fallback_benchmark_id="mmad",
                translation_run_id="process-run-id-should-not-appear",
                translation_qc_status="passed",
                translation_qc_report="process/qc.json",
                hub_repo_id="pjh6029/k-clean-package",
                skip_translation_validation=True,
                self_contained_package=True,
                media_roots=[media_root],
                visualizer_dir=visualizer_dir,
            )

            self.assertEqual(manifest["validation"]["status"], "passed")
            self.assertTrue((export_dir / "README.md").exists())
            metadata_path = export_dir / "test" / "metadata.jsonl"
            self.assertTrue(metadata_path.exists())
            row = json.loads(metadata_path.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(row["question_ko"], "어떤 결함이 보이나요?")
            self.assertNotIn("translation_artifact_path", row)
            self.assertNotIn("translation_run_id", row)
            self.assertNotIn("source_record_json", row)
            self.assertTrue((export_dir / "test" / row["file_name"]).exists())
            package_text = "\n".join(
                path.read_text(encoding="utf-8")
                for path in [export_dir / "README.md", export_dir / "hf_package_manifest.json", metadata_path]
            )
            self.assertNotIn("run_records", package_text)
            self.assertNotIn("/mnt/ddn/", package_text)
            self.assertNotIn("process-run-id-should-not-appear", package_text)
            self.assertEqual(validate_self_contained_package(export_dir)["status"], "passed")

            datasets = importlib.import_module("datasets")
            loaded = datasets.load_dataset("imagefolder", data_dir=str(export_dir))
            self.assertEqual(len(loaded["test"]), 1)
            self.assertIn("question_ko", loaded["test"].column_names)

            viewer_data = json.loads((visualizer_dir / "viewer_data.json").read_text(encoding="utf-8"))
            self.assertEqual(viewer_data["records"][0]["question_original"], "Which defect is visible?")
            media_url = viewer_data["records"][0]["media"][0]
            self.assertIn("media/test/images/mmad/", media_url)
            self.assertTrue((visualizer_dir / media_url).exists())
            self.assertEqual(self.fetch_from_static_dir(visualizer_dir, media_url), 200)
            self.assertIn("side-by-side", (visualizer_dir / "index.html").read_text(encoding="utf-8"))

    def test_self_contained_package_rejects_missing_media_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            translations = root / "translations"
            translations.mkdir()
            (translations / "translation_smoke.jsonl").write_text(
                json.dumps(
                    {
                        "source": {
                            "id": "missing-media",
                            "image_path": "images/missing.png",
                            "question": "Question",
                            "answer": "A",
                        },
                        "translated": {"question": "질문"},
                        "translation_scope": ["question"],
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "missing media references"):
                export_translations(
                    translation_outputs=[translations],
                    output_dir=root / "clean-package",
                    dataset_name="k-missing-media",
                    original_hf_dataset="fixture/missing",
                    original_hf_config=None,
                    original_hf_revision=None,
                    default_split="test",
                    fallback_benchmark_id="mmad",
                    translation_run_id=None,
                    translation_qc_status="passed",
                    translation_qc_report=None,
                    hub_repo_id=None,
                    skip_translation_validation=True,
                    self_contained_package=True,
                    media_roots=[root / "media"],
                )

    def test_self_contained_package_rejects_absolute_media_outside_media_roots(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            outside = root / "outside.png"
            outside.write_bytes(PNG_1X1)
            allowed_root = root / "allowed"
            allowed_root.mkdir()
            translations = root / "translations"
            translations.mkdir()
            (translations / "translation_smoke.jsonl").write_text(
                json.dumps(
                    {
                        "source": {
                            "id": "absolute-outside",
                            "image_path": str(outside),
                            "question": "Question",
                            "answer": "A",
                        },
                        "translated": {"question": "질문"},
                        "translation_scope": ["question"],
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "missing media references"):
                export_translations(
                    translation_outputs=[translations],
                    output_dir=root / "clean-package",
                    dataset_name="k-absolute-outside",
                    original_hf_dataset="fixture/absolute",
                    original_hf_config=None,
                    original_hf_revision=None,
                    default_split="test",
                    fallback_benchmark_id="mmad",
                    translation_run_id=None,
                    translation_qc_status="passed",
                    translation_qc_report=None,
                    hub_repo_id=None,
                    skip_translation_validation=True,
                    self_contained_package=True,
                    media_roots=[allowed_root],
                )
            self.assertFalse(list((root / "clean-package").rglob("outside.png")))

    def test_self_contained_package_rejects_visualizer_inside_package_tree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            media_root = root / "source-media"
            (media_root / "images").mkdir(parents=True)
            (media_root / "images" / "sample.png").write_bytes(PNG_1X1)
            translations = root / "translations"
            translations.mkdir()
            (translations / "translation_smoke.jsonl").write_text(
                json.dumps(
                    {
                        "source": {"id": "row", "image_path": "images/sample.png", "question": "Question"},
                        "translated": {"question": "질문"},
                        "translation_scope": ["question"],
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            export_dir = root / "clean-package"

            with self.assertRaisesRegex(ValueError, "visualizer_dir must be separate"):
                export_translations(
                    translation_outputs=[translations],
                    output_dir=export_dir,
                    dataset_name="k-bad-visualizer",
                    original_hf_dataset="fixture/visualizer",
                    original_hf_config=None,
                    original_hf_revision=None,
                    default_split="test",
                    fallback_benchmark_id="mmad",
                    translation_run_id=None,
                    translation_qc_status="passed",
                    translation_qc_report=None,
                    hub_repo_id=None,
                    skip_translation_validation=True,
                    self_contained_package=True,
                    media_roots=[media_root],
                    visualizer_dir=export_dir / "viewer",
                )

    def test_self_contained_validation_rejects_unexpected_clean_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            media_root = root / "source-media"
            (media_root / "images").mkdir(parents=True)
            (media_root / "images" / "sample.png").write_bytes(PNG_1X1)
            translations = root / "translations"
            translations.mkdir()
            (translations / "translation_smoke.jsonl").write_text(
                json.dumps(
                    {
                        "source": {"id": "row", "image_path": "images/sample.png", "question": "Question"},
                        "translated": {"question": "질문"},
                        "translation_scope": ["question"],
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            export_dir = root / "clean-package"
            export_translations(
                translation_outputs=[translations],
                output_dir=export_dir,
                dataset_name="k-extra-column",
                original_hf_dataset="fixture/extra",
                original_hf_config=None,
                original_hf_revision=None,
                default_split="test",
                fallback_benchmark_id="mmad",
                translation_run_id=None,
                translation_qc_status="passed",
                translation_qc_report=None,
                hub_repo_id=None,
                skip_translation_validation=True,
                self_contained_package=True,
                media_roots=[media_root],
            )
            metadata = export_dir / "test" / "metadata.jsonl"
            row = json.loads(metadata.read_text(encoding="utf-8").splitlines()[0])
            row["provider"] = "should be rejected"
            metadata.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
            validation = validate_self_contained_package(export_dir)
            self.assertEqual(validation["status"], "failed")
            self.assertTrue(any("unexpected clean package column" in error for error in validation["errors"]))

    def test_self_contained_validation_rejects_empty_splits_and_top_level_process_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            package = Path(tmp)
            (package / "README.md").write_text("# malformed package", encoding="utf-8")
            (package / "hf_package_manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "dataset_name": "malformed",
                        "layout": "huggingface_imagefolder_self_contained",
                        "splits": {},
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            (package / "run_records").mkdir()

            validation = validate_self_contained_package(package)
            self.assertEqual(validation["status"], "failed")
            self.assertTrue(any("manifest splits is empty" in error for error in validation["errors"]))
            self.assertTrue(any("unexpected top-level package artifact" in error for error in validation["errors"]))

    def test_self_contained_package_resolves_media_inside_named_zip_archives(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            media_root = root / "media"
            media_root.mkdir()
            with zipfile.ZipFile(media_root / "DS-MVTec.zip", "w") as zf:
                zf.writestr("DS-MVTec/leather/image/poke/012.png", PNG_1X1)
            translations = root / "translations"
            translations.mkdir()
            (translations / "translation_smoke.jsonl").write_text(
                json.dumps(
                    {
                        "source": {
                            "id": "zip-row",
                            "query_image": "DS-MVTec/leather/image/poke/012.png",
                            "question": "What defect is visible?",
                            "answer": "A",
                        },
                        "translated": {"question": "어떤 결함이 보이나요?"},
                        "translation_scope": ["question"],
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            export_dir = root / "zip-package"
            manifest = export_translations(
                translation_outputs=[translations],
                output_dir=export_dir,
                dataset_name="k-zip-media",
                original_hf_dataset="fixture/zip",
                original_hf_config=None,
                original_hf_revision=None,
                default_split="test",
                fallback_benchmark_id="mmad",
                translation_run_id=None,
                translation_qc_status="passed",
                translation_qc_report=None,
                hub_repo_id=None,
                skip_translation_validation=True,
                self_contained_package=True,
                media_roots=[media_root],
            )

            self.assertEqual(manifest["validation"]["status"], "passed")
            row = json.loads((export_dir / "test" / "metadata.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertTrue((export_dir / "test" / row["file_name"]).exists())

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
