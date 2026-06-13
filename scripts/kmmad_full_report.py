#!/usr/bin/env python3
"""Build a compact stats/QC report for full K-MMAD translation artifacts."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from kmmad_common import flatten_text, utc_now, write_json
from kmmad_hf_export import read_jsonl

KOREAN_RE = re.compile(r"[가-힣]")
ASSISTANT_ARTIFACT_RE = re.compile(
    r"(?i)(?:^|\n)\s*assistant\s*:|as an ai (?:language )?model|i(?:'m| am) (?:an )?ai"
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())


def option_count_from_json(value: Any) -> int | None:
    if value in (None, ""):
        return None
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    if isinstance(value, (list, tuple, dict)):
        return len(value)
    return None


def translation_stats(translation_dir: Path) -> dict[str, Any]:
    summary = load_json(translation_dir / "translation_smoke_summary.json")
    validation = load_json(translation_dir / "translation_validation.json")
    failures = count_jsonl(translation_dir / "translation_failures.jsonl")
    translated = count_jsonl(translation_dir / "translation_smoke.jsonl")
    source_count = int(summary.get("source_count") or translated + failures)
    return {
        "source_count": source_count,
        "translated_count": translated,
        "valid_translated_count": translated if validation.get("status") == "passed" else 0,
        "invalid_count": max(0, source_count - translated) + (0 if validation.get("status") == "passed" else translated),
        "failure_count": failures,
        "validation_status": validation.get("status", "missing"),
        "validation_errors": validation.get("errors", [])[:20],
        "summary_status": summary.get("status", "missing"),
    }


def package_stats(package_dir: Path) -> dict[str, Any]:
    manifest = load_json(package_dir / "hf_package_manifest.json")
    validation = load_json(package_dir / "hf_package_validation.json")
    split_counts = {
        str(split): int(info.get("num_rows", 0))
        for split, info in (manifest.get("splits") or {}).items()
        if isinstance(info, dict)
    }
    media_packaging = manifest.get("media_packaging") or {}
    review_candidates: list[dict[str, Any]] = []
    for metadata in sorted(package_dir.glob("*/metadata.jsonl")):
        for row_idx, row in enumerate(read_jsonl(metadata)):
            reasons = []
            if not row.get("question_ko") or not KOREAN_RE.search(str(row.get("question_ko"))):
                reasons.append("question_ko_missing_or_not_korean")
            if not row.get("file_name"):
                reasons.append("primary_media_missing")
            media_files = row.get("media_files")
            if not isinstance(media_files, list) or not media_files:
                reasons.append("media_files_empty")
            elif any(not (package_dir / str(media_file)).exists() for media_file in media_files):
                reasons.append("media_file_path_missing")
            if any(ASSISTANT_ARTIFACT_RE.search(text) for text in flatten_text(row)):
                reasons.append("assistant_artifact_text")
            original_options_count = option_count_from_json(row.get("options_original_json"))
            translated_options_count = option_count_from_json(row.get("options_ko_json"))
            if original_options_count is not None:
                if translated_options_count is None:
                    reasons.append("options_missing_or_unparseable")
                elif original_options_count != translated_options_count:
                    reasons.append("options_cardinality_mismatch")
            if reasons:
                review_candidates.append(
                    {
                        "record_id": row.get("record_id"),
                        "source_record_id": row.get("source_record_id"),
                        "reasons": reasons,
                        "row_index": row_idx,
                    }
                )
    return {
        "split_counts": split_counts,
        "validation_status": validation.get("status", "missing"),
        "validation_errors": validation.get("errors", [])[:20],
        "copied_media_files": int(media_packaging.get("copied_files") or 0),
        "missing_media_refs": int(media_packaging.get("missing_media_refs") or 0),
        "review_candidate_count": len(review_candidates),
        "review_candidates_sample": review_candidates[:50],
    }


def build_report(*, translation_root: Path, package_root: Path, visualizer_dir: Path, output_dir: Path) -> dict[str, Any]:
    benchmarks = ["mmad", "blink", "mme_realworld"]
    per_benchmark: dict[str, Any] = {}
    totals = {
        "source_count": 0,
        "translated_count": 0,
        "valid_translated_count": 0,
        "invalid_count": 0,
        "failure_count": 0,
        "review_candidate_count": 0,
        "copied_media_files": 0,
        "missing_media_refs": 0,
    }
    for benchmark in benchmarks:
        stats = {
            "translation": translation_stats(translation_root / benchmark),
            "package": package_stats(package_root / benchmark),
        }
        per_benchmark[benchmark] = stats
        for key in ("source_count", "translated_count", "valid_translated_count", "invalid_count", "failure_count"):
            totals[key] += int(stats["translation"].get(key) or 0)
        for key in ("review_candidate_count", "copied_media_files", "missing_media_refs"):
            totals[key] += int(stats["package"].get(key) or 0)
    visualizer_validation = load_json(visualizer_dir / "unified_visualizer_validation.json")
    status = "passed"
    for benchmark, stats in per_benchmark.items():
        if stats["translation"]["validation_status"] != "passed":
            status = "needs_review"
        if stats["package"]["validation_status"] != "passed":
            status = "needs_review"
        if stats["translation"]["invalid_count"]:
            status = "needs_review"
        if stats["package"]["review_candidate_count"]:
            status = "needs_review"
        if "test" not in stats["package"].get("split_counts", {}):
            status = "needs_review"
    if visualizer_validation.get("status") != "passed":
        status = "needs_review"
    report = {
        "created_at": utc_now(),
        "status": status,
        "translation_root": str(translation_root),
        "package_root": str(package_root),
        "visualizer_dir": str(visualizer_dir),
        "totals": totals,
        "benchmarks": per_benchmark,
        "visualizer_validation": visualizer_validation,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "full_translation_stats_qc.json", report)
    md_lines = [
        "# Full translation stats/QC report",
        "",
        f"Status: `{status}`",
        "",
        "| benchmark | source | translated | valid | invalid | failures | package split rows | review candidates | media missing |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |",
    ]
    for benchmark in benchmarks:
        stats = per_benchmark[benchmark]
        tr = stats["translation"]
        pkg = stats["package"]
        md_lines.append(
            f"| {benchmark} | {tr['source_count']} | {tr['translated_count']} | "
            f"{tr['valid_translated_count']} | {tr['invalid_count']} | {tr['failure_count']} | "
            f"{pkg['split_counts']} | {pkg['review_candidate_count']} | {pkg['missing_media_refs']} |"
        )
    md_lines.extend([
        "",
        f"Unified visualizer validation: `{visualizer_validation.get('status', 'missing')}`",
        "",
        "Review-needed samples are machine-detected candidates only; independent visual/sample review "
        "should add a separate human/agent QC note when required.",
    ])
    (output_dir / "full_translation_stats_qc.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--translation-root", type=Path, required=True)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--visualizer-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    report = build_report(
        translation_root=args.translation_root,
        package_root=args.package_root,
        visualizer_dir=args.visualizer_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
