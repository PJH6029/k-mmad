from __future__ import annotations

import json
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from kmmad_benchmarks import BENCHMARKS, load_benchmark_sample, normalize_benchmark_id, parse_benchmark_ids  # noqa: E402
from kmmad_translate_smoke import (  # noqa: E402
    build_request_headers,
    build_benchmark_translation_row,
    configured_concurrency,
    resolve_benchmark_root,
    translate_rows_ordered,
    validate_output,
)

FIXTURE_ROOT = ROOT / "tests" / "fixtures" / "general_vqa"


class DelayedOpenAIHandler(BaseHTTPRequestHandler):
    seen: list[str] = []

    def log_message(self, format: str, *args: Any) -> None:
        return

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length).decode("utf-8"))
        text = body["messages"][-1]["content"]
        idx = int(text.rsplit(" ", 1)[-1])
        # Higher indexes return first, so completion order differs from input order.
        time.sleep(0.01 * (5 - idx))
        self.__class__.seen.append(text)
        response = {
            "choices": [
                {
                    "message": {"content": f"한국어 지연 응답 {idx}"},
                    "finish_reason": "stop",
                    "index": 0,
                }
            ]
        }
        payload = json.dumps(response, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


class GeneralVqaAdapterTest(unittest.TestCase):
    def test_registry_contains_exact_first_pass_benchmarks(self) -> None:
        self.assertEqual(set(BENCHMARKS), {"mme_realworld", "blink", "mmmu_pro", "mega_bench"})
        self.assertEqual(parse_benchmark_ids("MME-RealWorld,blink,mmmu-pro,mega-bench"), ["mme_realworld", "blink", "mmmu_pro", "mega_bench"])
        with self.assertRaises(KeyError):
            normalize_benchmark_id("mmstar")

    def test_all_fixture_adapters_emit_common_schema(self) -> None:
        for benchmark_id in BENCHMARKS:
            with self.subTest(benchmark_id=benchmark_id):
                record_path, records = load_benchmark_sample(FIXTURE_ROOT / benchmark_id, benchmark_id, sample_size=2, seed=1)
                self.assertIsNotNone(record_path)
                self.assertEqual(len(records), 1)
                row = records[0]
                self.assertEqual(row["benchmark_id"], benchmark_id)
                self.assertTrue(row["source_id"])
                self.assertIn("text_fields", row)
                self.assertTrue(row["text_fields"])
                self.assertIn("preserve_fields", row)
                self.assertIn("media", row)
                self.assertIn("translation_scope", row)
                self.assertNotIn("answer", row["text_fields"])
                self.assertTrue(any(key.lower() in {"answer", "target"} for key in row["preserve_fields"]))
                if benchmark_id == "mega_bench":
                    self.assertIn("output_format", row["preserve_fields"])
                    self.assertIn("rubric", row["skip_fields"])
                if benchmark_id == "mmmu_pro":
                    self.assertIn("ocr_text", row["skip_fields"])

    def test_loader_fails_closed_without_benchmark_named_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "unrelated.json").write_text(
                json.dumps([{"question": "Wrong file", "answer": "A"}]),
                encoding="utf-8",
            )

            record_path, records = load_benchmark_sample(root, "blink", sample_size=1, seed=1)

        self.assertIsNone(record_path)
        self.assertEqual(records, [])

    def test_missing_benchmark_directory_does_not_scan_other_benchmark_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            other = root / "mme_realworld"
            other.mkdir()
            (other / "annotations.json").write_text(
                json.dumps([{"question": "Wrong benchmark", "answer": "A"}]),
                encoding="utf-8",
            )

            benchmark_root = resolve_benchmark_root(root, "blink")

        self.assertIsNone(benchmark_root)

    def test_parallel_translation_preserves_input_order(self) -> None:
        rows = [
            {
                "benchmark_id": "blink",
                "benchmark_name": "BLINK",
                "source_id": f"row-{idx}",
                "text_fields": {"question": f"question {idx}"},
                "preserve_fields": {"answer": "A"},
                "skip_fields": {},
                "media": [],
                "translation_scope": ["question"],
            }
            for idx in range(5)
        ]
        config: dict[str, Any] = {"inference": {"concurrency": 4}}
        translated = translate_rows_ordered(rows, config, mock=True, concurrency=4, benchmark_mode=True)
        self.assertEqual([item[0]["source_id"] for item in translated], [f"row-{idx}" for idx in range(5)])
        self.assertEqual(configured_concurrency({"smoke": {"concurrency": 3}, "inference": {}}, None), 3)

    def test_parallel_endpoint_responses_merge_in_input_order(self) -> None:
        DelayedOpenAIHandler.seen = []
        server = ThreadingHTTPServer(("127.0.0.1", 0), DelayedOpenAIHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(thread.join, 2)
        self.addCleanup(server.shutdown)
        rows = [
            {
                "benchmark_id": "blink",
                "benchmark_name": "BLINK",
                "source_id": f"row-{idx}",
                "text_fields": {"question": f"question {idx}"},
                "preserve_fields": {"answer": "A"},
                "skip_fields": {},
                "media": [],
                "translation_scope": ["question"],
            }
            for idx in range(5)
        ]
        config: dict[str, Any] = {
            "inference": {
                "endpoint_provider": "contract",
                "auth_mode": "none",
                "openai_compatible_base_url": f"http://127.0.0.1:{server.server_port}/v1",
                "model": "delayed-test",
                "timeout_seconds": 5,
                "max_tokens": 32,
                "temperature": 0.0,
            }
        }

        translated = translate_rows_ordered(rows, config, mock=False, concurrency=5, benchmark_mode=True)

        self.assertEqual([item[0]["source_id"] for item in translated], [f"row-{idx}" for idx in range(5)])
        self.assertEqual(
            [item[0]["translated"]["text_fields"]["question"] for item in translated],
            [f"한국어 지연 응답 {idx}" for idx in range(5)],
        )
        self.assertNotEqual(DelayedOpenAIHandler.seen, [f"question {idx}" for idx in range(5)])

    def test_leading_option_labels_are_structurally_preserved(self) -> None:
        row = {
            "benchmark_id": "mme_realworld",
            "benchmark_name": "MME-RealWorld",
            "source_id": "label-row",
            "text_fields": {"options": ["A. School zone", "B) No parking"]},
            "preserve_fields": {"answer": "A"},
            "skip_fields": {},
            "media": [],
            "translation_scope": ["options"],
        }

        translated, _, _ = build_benchmark_translation_row(row, {"inference": {}}, mock=True)

        self.assertEqual(
            translated["translated"]["text_fields"]["options"],
            ["A. 한국어 번역 초안: A. School zone", "B) 한국어 번역 초안: B) No parking"],
        )

    def test_directory_validation_requires_full_artifact_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            benchmark_dir = root / "blink"
            benchmark_dir.mkdir()
            (benchmark_dir / "translation_smoke.jsonl").write_text(
                json.dumps(
                    {
                        "benchmark_id": "blink",
                        "source_id": "row-1",
                        "source": {"id": "row-1"},
                        "translated": {"text_fields": {"question": "한국어 질문"}},
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            (benchmark_dir / "untranslated_fields.json").write_text("{}", encoding="utf-8")

            result = validate_output(root)

        self.assertEqual(result["status"], "failed")
        self.assertTrue(any("general_vqa_translation_summary.json" in error for error in result["errors"]))
        self.assertTrue(any("inspection_examples.json" in error for error in result["errors"]))
        self.assertTrue(any("translation_smoke_summary.json" in error for error in result["errors"]))

    def test_directory_validation_requires_summary_benchmark_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "general_vqa_translation_summary.json").write_text(
                json.dumps({"benchmarks": ["blink", "mega_bench"]}),
                encoding="utf-8",
            )
            benchmark_dir = root / "blink"
            benchmark_dir.mkdir()
            row = {
                "benchmark_id": "blink",
                "source_id": "row-1",
                "source": {"id": "row-1"},
                "translated": {"text_fields": {"question": "한국어 질문"}},
            }
            (benchmark_dir / "translation_smoke.jsonl").write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
            (benchmark_dir / "untranslated_fields.json").write_text("{}", encoding="utf-8")
            (benchmark_dir / "inspection_examples.json").write_text("[]", encoding="utf-8")
            (benchmark_dir / "translation_validation.json").write_text("{}", encoding="utf-8")
            (benchmark_dir / "translation_smoke_summary.json").write_text(json.dumps({"benchmark_id": "blink"}), encoding="utf-8")

            result = validate_output(root)

        self.assertEqual(result["status"], "failed")
        self.assertTrue(any("benchmark output missing: mega_bench" in error for error in result["errors"]))

    def test_validation_allows_numeric_symbolic_options_without_hangul(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "translation_smoke.jsonl"
            report = Path(tmp) / "untranslated_fields.json"
            row = {
                "benchmark_id": "mmmu_pro",
                "source_id": "numeric-options",
                "source": {"id": "numeric-options"},
                "translated": {
                    "text_fields": {
                        "question": "한국어 질문",
                        "options": "['A: 5.76%; B: 6.30%', '17940 N', '29.193 * 10^6 m^3']",
                    }
                },
            }
            output.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
            report.write_text("{}", encoding="utf-8")

            result = validate_output(output, report)

        self.assertEqual(result["status"], "passed")

    def test_validation_still_rejects_untranslated_english_words(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "translation_smoke.jsonl"
            report = Path(tmp) / "untranslated_fields.json"
            row = {
                "benchmark_id": "blink",
                "source_id": "english-options",
                "source": {"id": "english-options"},
                "translated": {
                    "text_fields": {
                        "question": "한국어 질문",
                        "options": ["A. School zone", "B. No parking"],
                    }
                },
            }
            output.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
            report.write_text("{}", encoding="utf-8")

            result = validate_output(output, report)

        self.assertEqual(result["status"], "failed")
        self.assertTrue(any("has no Korean text" in error for error in result["errors"]))

    def test_nonlinguistic_validation_policy_is_benchmark_and_field_scoped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "translation_smoke.jsonl"
            report = Path(tmp) / "untranslated_fields.json"
            row = {
                "benchmark_id": "blink",
                "source_id": "numeric-question",
                "source": {"id": "numeric-question"},
                "translated": {
                    "text_fields": {
                        "question": "17940 N",
                        "options": "['17940 N', '38750 N']",
                    }
                },
            }
            output.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
            report.write_text("{}", encoding="utf-8")

            result = validate_output(output, report)

        self.assertEqual(result["status"], "failed")
        self.assertTrue(any("text_fields.question" in error for error in result["errors"]))
        self.assertFalse(any("text_fields.options" in error for error in result["errors"]))


class OpenAiOauthAliasContractTest(unittest.TestCase):
    def test_endpoint_provider_openai_oauth_defaults_to_no_authorization(self) -> None:
        config = {"inference": {"endpoint_provider": "openai_oauth", "openai_compatible_base_url": "http://127.0.0.1:10531/v1", "model": "test"}}
        self.assertNotIn("Authorization", build_request_headers(config))

    def test_auth_mode_openai_oauth_alias_is_no_authorization(self) -> None:
        config = {"inference": {"endpoint_provider": "contract", "auth_mode": "openai_oauth", "openai_compatible_base_url": "http://127.0.0.1:10531/v1", "model": "test"}}
        self.assertNotIn("Authorization", build_request_headers(config))


if __name__ == "__main__":
    unittest.main()


def test_numeric_only_options_are_valid_nonlinguistic_translations(tmp_path: Path) -> None:
    output = tmp_path / "translation_smoke.jsonl"
    output.write_text(
        json.dumps(
            {
                "benchmark_id": "blink",
                "source_id": "counting-row",
                "source": {"text_fields": {"question": "How many?", "options": ["0", "1"]}},
                "translated": {
                    "text_fields": {
                        "question": "몇 개인가요?",
                        "options": ["0", "1"],
                    }
                },
                "translation_scope": ["question", "options"],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    result = validate_output(output)
    assert result["status"] == "passed"


def test_visual_labels_and_unit_options_are_valid_nonlinguistic_translations(tmp_path: Path) -> None:
    output = tmp_path / "translation_smoke.jsonl"
    rows = [
        {
            "benchmark_id": "blink",
            "source_id": "box-row",
            "source": {"text_fields": {"question": "Which box?", "options": ["Box A", "Box B"]}},
            "translated": {
                "text_fields": {
                    "question": "어느 상자인가요?",
                    "options": ["Box A", "Box B"],
                }
            },
            "translation_scope": ["question", "options"],
        },
        {
            "benchmark_id": "mme_realworld",
            "source_id": "unit-row",
            "source": {"text_fields": {"question": "What size?", "options": ["(A) NA", "(B) 20.00 sq ft"]}},
            "translated": {
                "text_fields": {
                    "question": "크기가 얼마인가요?",
                    "options": ["(A) NA", "(B) 20.00 sq ft"],
                }
            },
            "translation_scope": ["question", "options"],
        },
    ]
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    result = validate_output(output)
    assert result["status"] == "passed"


def test_table_chart_literal_options_allow_dates_currency_and_ranges(tmp_path: Path) -> None:
    output = tmp_path / "translation_smoke.jsonl"
    rows = [
        {
            "benchmark_id": "mme_realworld",
            "source_id": "date-row",
            "source": {"text_fields": {"question": "What date?", "options": ["30-Apr-22", "30-Jun-22"]}},
            "translated": {
                "text_fields": {
                    "question": "날짜는 무엇인가요?",
                    "options": ["(A) 30-Apr-22", "(B) 30-Jun-22"],
                }
            },
            "translation_scope": ["question", "options"],
        },
        {
            "benchmark_id": "mme_realworld",
            "source_id": "currency-row",
            "source": {"text_fields": {"question": "What currency?", "options": ["RMB", "USD", "EURO"]}},
            "translated": {
                "text_fields": {
                    "question": "통화는 무엇인가요?",
                    "options": ["(A) RMB", "(B) USD", "(C) EURO"],
                }
            },
            "translation_scope": ["question", "options"],
        },
        {
            "benchmark_id": "mme_realworld",
            "source_id": "range-row",
            "source": {"text_fields": {"question": "What range?", "options": ["40.00 to 60.00", "$7.5 bn to $10.0 bn"]}},
            "translated": {
                "text_fields": {
                    "question": "범위는 무엇인가요?",
                    "options": ["(A) 40.00 to 60.00", "(B) $7.5 bn to $10.0 bn"],
                }
            },
            "translation_scope": ["question", "options"],
        },
        {
            "benchmark_id": "mme_realworld",
            "source_id": "compact-unit-row",
            "source": {"text_fields": {"question": "What value?", "options": ["1200K", "+806.7k", "R$45 Mi", "4800.00PAX", "20..%"]}},
            "translated": {
                "text_fields": {
                    "question": "값은 무엇인가요?",
                    "options": ["(A) 1200K", "(B) +806.7k", "(C) R$45 Mi", "(D) 4800.00PAX", "(E) 20..%"],
                }
            },
            "translation_scope": ["question", "options"],
        },
        {
            "benchmark_id": "mmad",
            "source_id": "volume-unit-row",
            "source": {"question": "What volume?", "options": "A: 350 ml\nB: 6.5% vol"},
            "translated": {
                "text_fields": {
                    "question": "부피는 얼마인가요?",
                    "options": "A: 350 ml\nB: 6.5% vol",
                }
            },
            "translation_scope": ["question", "options"],
        },
        {
            "benchmark_id": "mmad",
            "source_id": "connector-code-row",
            "source": {"question": "What connector?", "options": "A: USB\nB: RJ11\nC: HDMI\nD: RJ45"},
            "translated": {
                "text_fields": {
                    "question": "커넥터는 무엇인가요?",
                    "options": "A: USB\nB: RJ11\nC: HDMI\nD: RJ45",
                }
            },
            "translation_scope": ["question", "options"],
        },
        {
            "benchmark_id": "mme_realworld",
            "source_id": "proper-name-row",
            "source": {"text_fields": {"question": "Which client?", "options": ["Prophet LLC", "Orange Inc", "Paseo", "PvL1", "a1"]}},
            "translated": {
                "text_fields": {
                    "question": "어느 클라이언트인가요?",
                    "options": ["(A) Prophet LLC", "(B) Orange Inc", "(C) Paseo", "(D) PvL1", "(E) a1"],
                }
            },
            "translation_scope": ["question", "options"],
        },
        {
            "benchmark_id": "mme_realworld",
            "source_id": "fiscal-period-row",
            "source": {"text_fields": {"question": "Which year?", "options": ["FY 2032", "YTD 31-03-2018", "TTM 31-mar-19"]}},
            "translated": {
                "text_fields": {
                    "question": "어느 연도인가요?",
                    "options": ["(A) FY 2032", "(B) YTD 31-03-2018", "(C) TTM 31-mar-19"],
                }
            },
            "translation_scope": ["question", "options"],
        },
        {
            "benchmark_id": "mme_realworld",
            "source_id": "reasoning/diagram_and_table/table/0398",
            "source": {"text_fields": {"question": "Which client?", "options": ["xyz", "lmn", "dd", "byz co."]}},
            "translated": {
                "text_fields": {
                    "question": "어느 고객인가요?",
                    "options": ["(A) xyz", "(B) lmn", "(C) dd", "(D) byz co."],
                }
            },
            "translation_scope": ["question", "options"],
        },
        {
            "benchmark_id": "mme_realworld",
            "source_id": "company-code-row",
            "source": {"text_fields": {"question": "Which client?", "options": ["ABC Co. 1", "XYZ Co. 2"]}},
            "translated": {
                "text_fields": {
                    "question": "어느 고객인가요?",
                    "options": ["(A) ABC Co. 1", "(B) XYZ Co. 2"],
                }
            },
            "translation_scope": ["question", "options"],
        },
        {
            "benchmark_id": "mmad",
            "source_id": "quoted-label-row",
            "source": {"question": "What text?", "options": 'A: "Soup Daren Signature"\nB: "Soup Darren"'},
            "translated": {
                "text_fields": {
                    "question": "어떤 글자인가요?",
                    "options": 'A: "Soup Daren Signature"\nB: "Soup Darren"',
                }
            },
            "translation_scope": ["question", "options"],
        },
        {
            "benchmark_id": "mme_realworld",
            "source_id": "blank-placeholder-row",
            "source": {"text_fields": {"question": "Yes/no?", "options": ["Yes", "No", "", ""]}},
            "translated": {
                "text_fields": {
                    "question": "예/아니오 질문인가요?",
                    "options": ["(A) 예", "(B) 아니요", "(C) ", "(D) "],
                }
            },
            "translation_scope": ["question", "options"],
        },
    ]
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")

    result = validate_output(output)

    assert result["status"] == "passed"


def test_table_chart_literal_policy_still_rejects_plain_english_options(tmp_path: Path) -> None:
    output = tmp_path / "translation_smoke.jsonl"
    row = {
        "benchmark_id": "mme_realworld",
        "source_id": "plain-english-option",
        "source": {"text_fields": {"question": "What currency?", "options": ["money"]}},
        "translated": {
            "text_fields": {
                "question": "통화는 무엇인가요?",
                "options": ["(A) money"],
            }
        },
        "translation_scope": ["question", "options"],
    }
    output.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

    result = validate_output(output)

    assert result["status"] == "failed"
    assert any("text_fields.options" in error for error in result["errors"])


def test_short_color_words_are_not_treated_as_source_literals(tmp_path: Path) -> None:
    output = tmp_path / "translation_smoke.jsonl"
    row = {
        "benchmark_id": "mme_realworld",
        "source_id": "perception/remote_sensing/color/0001",
        "source": {"text_fields": {"question": "Which color?", "options": ["red", "blue"]}},
        "translated": {
            "text_fields": {
                "question": "어떤 색인가요?",
                "options": ["(A) red", "(B) blue"],
            }
        },
        "translation_scope": ["question", "options"],
    }
    output.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

    result = validate_output(output)

    assert result["status"] == "failed"
    assert any("text_fields.options" in error for error in result["errors"])


def test_option_cardinality_mismatch_is_invalid(tmp_path: Path) -> None:
    output = tmp_path / "translation_smoke.jsonl"
    row = {
        "benchmark_id": "mme_realworld",
        "source_id": "perception/ocr_cc/book_map_poster/0463",
        "source": {
            "text_fields": {
                "question": "What is the last line?",
                "options": [
                    "(A) IKKE TIL BORN UNDER",
                    "(B) 3 AR PGA SMADELE",
                    "(C) MINOS MENORES DE 36",
                    "(D) NO RECOMENDABLE PARA",
                    "(E) The image does not feature the content.",
                ],
            }
        },
        "translated": {
            "text_fields": {
                "question": "마지막 줄은 무엇인가요?",
                "options": [
                    "(A) 3세 미만 어린이에게 적합하지 않음",
                    "(B) 3세 이하",
                    "(C) 36세 미만",
                    "(D) 권장하지 않음",
                ],
            }
        },
        "translation_scope": ["question", "options"],
    }
    output.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

    result = validate_output(output)

    assert result["status"] == "failed"
    assert any("options cardinality mismatch" in error for error in result["errors"])


def test_missing_translated_options_are_invalid_when_source_has_options(tmp_path: Path) -> None:
    output = tmp_path / "translation_smoke.jsonl"
    row = {
        "benchmark_id": "mme_realworld",
        "source_id": "perception/remote_sensing/color/0002",
        "source": {"text_fields": {"question": "Which color?", "options": ["red", "blue"]}},
        "translated": {"text_fields": {"question": "어떤 색인가요?"}},
        "translation_scope": ["question", "options"],
    }
    output.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

    result = validate_output(output)

    assert result["status"] == "failed"
    assert any("missing translated options" in error for error in result["errors"])


def test_mme_ocr_cc_options_allow_source_literals_and_cjk(tmp_path: Path) -> None:
    output = tmp_path / "translation_smoke.jsonl"
    row = {
        "benchmark_id": "mme_realworld",
        "source_id": "perception/ocr_cc/adver_and_product/0069",
        "source": {"text_fields": {"question": "What text?", "options": ["京北", "PEKING 北京", "dean&david"]}},
        "translated": {
            "text_fields": {
                "question": "어떤 글자인가요?",
                "options": ["(A) 京北", "(B) PEKING 北京", "(C) dean&david"],
            }
        },
        "translation_scope": ["question", "options"],
    }
    output.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

    result = validate_output(output)

    assert result["status"] == "passed"


def test_mme_ocr_cc_question_allows_quoted_source_cjk_literal(tmp_path: Path) -> None:
    output = tmp_path / "translation_smoke.jsonl"
    row = {
        "benchmark_id": "mme_realworld",
        "source_id": "perception/ocr_cc/text_recog/0006",
        "source": {"text_fields": {"question": 'What is below "四川"?', "options": ["BURGER FABRIEK"]}},
        "translated": {
            "text_fields": {
                "question": '“四川”이라고 불리는 표시판 아래의 글자는 무엇인가요?',
                "options": ["(A) BURGER FABRIEK"],
            }
        },
        "translation_scope": ["question", "options"],
    }
    output.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

    result = validate_output(output)

    assert result["status"] == "passed"


def test_cjk_options_outside_mme_ocr_cc_still_fail(tmp_path: Path) -> None:
    output = tmp_path / "translation_smoke.jsonl"
    row = {
        "benchmark_id": "mme_realworld",
        "source_id": "reasoning/diagram_and_table/table/0001",
        "source": {"text_fields": {"question": "What text?", "options": ["北京"]}},
        "translated": {
            "text_fields": {
                "question": "어떤 글자인가요?",
                "options": ["(A) 北京"],
            }
        },
        "translation_scope": ["question", "options"],
    }
    output.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

    result = validate_output(output)

    assert result["status"] == "failed"
    assert any("CJK/Hanja" in error for error in result["errors"])
