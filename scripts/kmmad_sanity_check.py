#!/usr/bin/env python3
"""Sanity-check an MMAD dataset directory and write a JSON report."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path
from typing import Any

from kmmad_common import (
    ANSWER_KEYS,
    CAPTION_KEYS,
    OPTION_KEYS,
    PATH_KEYS,
    QUESTION_KEYS,
    configured_path,
    discover_image_paths,
    discover_text_fields,
    get_first,
    load_config,
    load_first_records,
    pick_sample,
    utc_now,
    write_json,
)


def inspect_archives(root: Path, expected: list[str]) -> dict[str, Any]:
    archive_reports = []
    for archive in sorted(root.glob("*.zip")):
        item = {"path": str(archive), "size_bytes": archive.stat().st_size, "valid_zip": False, "members": None, "image_members": None}
        try:
            with zipfile.ZipFile(archive) as zf:
                names = zf.namelist()
                item["valid_zip"] = True
                item["members"] = len(names)
                item["image_members"] = sum(1 for name in names if Path(name).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"})
        except zipfile.BadZipFile:
            item["error"] = "bad zip"
        archive_reports.append(item)
    present = {Path(item["path"]).name for item in archive_reports}
    return {
        "expected_archives": expected,
        "present_expected_archives": sorted(present.intersection(expected)),
        "missing_expected_archives": sorted(set(expected).difference(present)),
        "archives": archive_reports,
    }


def row_field_report(records: list[dict[str, Any]], sample_size: int, seed: int) -> dict[str, Any]:
    sample = pick_sample(records, sample_size, seed)
    missing = {"question": 0, "options": 0, "caption": 0, "answer": 0, "path": 0}
    text_fields = set()
    examples = []
    for row in sample:
        q_key, q = get_first(row, QUESTION_KEYS)
        o_key, o = get_first(row, OPTION_KEYS)
        c_key, c = get_first(row, CAPTION_KEYS)
        a_key, a = get_first(row, ANSWER_KEYS)
        p_key, p = get_first(row, PATH_KEYS)
        if q is None:
            missing["question"] += 1
        if o is None:
            missing["options"] += 1
        if c is None:
            missing["caption"] += 1
        if a is None:
            missing["answer"] += 1
        if p is None:
            missing["path"] += 1
        text_fields.update(discover_text_fields(row))
        examples.append({
            "question_key": q_key,
            "options_key": o_key,
            "caption_key": c_key,
            "answer_key": a_key,
            "path_key": p_key,
            "question_preview": str(q)[:160] if q is not None else None,
            "caption_preview": str(c)[:160] if c is not None else None,
        })
    return {"sample_size": len(sample), "missing_in_sample": missing, "text_fields": sorted(text_fields), "examples": examples[:5]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/kmmad_translation_smoke.toml")
    parser.add_argument("--dataset", type=Path, default=None, help="Override original/MMAD dataset directory")
    parser.add_argument("--report-out", type=Path, default=None)
    parser.add_argument("--expected-rows", type=int, default=None)
    parser.add_argument("--allow-partial", action="store_true", help="Do not fail on count/archive mismatch; useful for local fixtures")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    source = config["source"]
    smoke = config["smoke"]
    root = args.dataset or configured_path(config, "original_subdir")
    report_out = args.report_out or configured_path(config, "manifest_subdir") / f"sanity-{utc_now()}.json"
    expected_rows = args.expected_rows if args.expected_rows is not None else int(source.get("expected_rows", 0))

    report: dict[str, Any] = {
        "created_at": utc_now(),
        "dataset_root": str(root),
        "status": "failed",
        "checks": {},
        "errors": [],
    }

    if not root.exists():
        report["errors"].append(f"dataset root does not exist: {root}")
        write_json(report_out, report)
        print(f"sanity report written: {report_out}")
        return 1

    record_path, records = load_first_records(root)
    images = discover_image_paths(root)
    archive_report = inspect_archives(root, source.get("archive_files", []))
    field_report = row_field_report(records, int(smoke.get("sample_size", 8)), int(smoke.get("sample_seed", 6029))) if records else {}

    report["checks"] = {
        "record_file": str(record_path) if record_path else None,
        "row_count": len(records),
        "expected_rows": expected_rows,
        "image_file_count_unpacked": len(images),
        "archives": archive_report,
        "field_report": field_report,
    }

    if not records:
        report["errors"].append("no parseable records found")
    elif expected_rows and len(records) != expected_rows and not args.allow_partial:
        report["errors"].append(f"row count mismatch: got {len(records)}, expected {expected_rows}")

    if archive_report["missing_expected_archives"] and not args.allow_partial:
        report["errors"].append("missing expected archive/source files: " + ", ".join(archive_report["missing_expected_archives"]))

    if records and field_report:
        missing = field_report["missing_in_sample"]
        for required in ("question", "options", "caption"):
            if missing.get(required, 0) == field_report["sample_size"]:
                report["errors"].append(f"required translation field absent in sampled rows: {required}")

    report["status"] = "passed" if not report["errors"] else "failed"
    write_json(report_out, report)
    print(f"sanity report written: {report_out}")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
