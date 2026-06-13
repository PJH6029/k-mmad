#!/usr/bin/env python3
"""Create one benchmark-selectable HTML visualizer from HF packages.

The visualizer uses symlinks into package media directories instead of copying
large image trees, so it is suitable for full benchmark artifacts on MLXP.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from kmmad_hf_export import VISUALIZER_HTML, _is_relative_to, read_jsonl, write_json
from kmmad_common import utc_now


def load_metadata_rows(package_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    manifest_path = package_dir / "hf_package_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        splits = manifest.get("splits", {}) if isinstance(manifest, dict) else {}
        for split in sorted(splits):
            metadata = package_dir / str(split) / "metadata.jsonl"
            if metadata.exists():
                rows.extend(read_jsonl(metadata))
    else:
        for metadata in sorted(package_dir.glob("*/metadata.jsonl")):
            rows.extend(read_jsonl(metadata))
    return rows


def safe_link_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value).strip("._-") or "media"


def has_dir_entries(path: Path) -> bool:
    return path.exists() and path.is_dir() and any(path.iterdir())


def assert_output_dir_safe(*, output_dir: Path, package_dirs: list[Path], overwrite: bool) -> None:
    output_resolved = output_dir.resolve()
    for package_dir in package_dirs:
        package_resolved = package_dir.resolve()
        if output_resolved == package_resolved:
            raise ValueError(f"visualizer output_dir must not equal package_dir: {output_dir}")
        if _is_relative_to(output_resolved, package_resolved):
            raise ValueError(f"visualizer output_dir must not be inside package_dir: {output_dir}")
        if _is_relative_to(package_resolved, output_resolved):
            raise ValueError(f"visualizer output_dir must not contain package_dir: {output_dir}")
    if has_dir_entries(output_dir) and not overwrite:
        raise FileExistsError(f"refusing to overwrite non-empty visualizer output dir without --overwrite: {output_dir}")


def reset_dir(path: Path, *, overwrite: bool) -> None:
    if path.exists():
        if path.is_symlink() or path.is_file():
            if not overwrite:
                raise FileExistsError(f"refusing to overwrite existing visualizer output path without --overwrite: {path}")
            path.unlink()
        else:
            if not overwrite and has_dir_entries(path):
                raise FileExistsError(f"refusing to overwrite non-empty visualizer output dir without --overwrite: {path}")
            shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def ensure_media_link(*, visualizer_dir: Path, package_dir: Path, media_file: str, benchmark_id: str) -> str:
    package_root = package_dir.resolve()
    source = (package_dir / media_file).resolve()
    if not _is_relative_to(source, package_root):
        raise ValueError(f"media escapes package root: {media_file}")
    if not source.exists():
        raise FileNotFoundError(f"media file missing: {source}")
    rel_parts = Path(media_file).parts
    if len(rel_parts) < 4 or rel_parts[1] != "images":
        # Fallback: symlink a record-level file target.
        link_rel = Path("media") / safe_link_name(benchmark_id) / safe_link_name("__files__") / source.name
        target = visualizer_dir / link_rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.symlink_to(source)
        return link_rel.as_posix()
    split, _, bench_part = rel_parts[:3]
    source_dir = (package_dir / split / "images" / bench_part).resolve()
    if not _is_relative_to(source_dir, package_root):
        raise ValueError(f"media directory escapes package root: {source_dir}")
    link_rel_dir = Path("media") / safe_link_name(benchmark_id)
    link_path = visualizer_dir / link_rel_dir
    if not link_path.exists():
        link_path.parent.mkdir(parents=True, exist_ok=True)
        link_path.symlink_to(source_dir, target_is_directory=True)
    return (link_rel_dir / Path(*rel_parts[3:])).as_posix()


def row_to_viewer_record(*, row: dict[str, Any], package_dir: Path, visualizer_dir: Path) -> dict[str, Any]:
    benchmark_id = str(row.get("benchmark_id") or "benchmark")
    media = [
        ensure_media_link(
            visualizer_dir=visualizer_dir,
            package_dir=package_dir,
            media_file=str(media_file),
            benchmark_id=benchmark_id,
        )
        for media_file in row.get("media_files", [])
    ]
    return {
        "record_id": row.get("record_id"),
        "source_record_id": row.get("source_record_id") or row.get("record_id"),
        "benchmark_id": benchmark_id,
        "benchmark_name": row.get("benchmark_name") or benchmark_id,
        "split": row.get("split"),
        "source_split": row.get("source_split") or row.get("split"),
        "source_id": row.get("source_id"),
        "task": row.get("task"),
        "media": media,
        "question_original": row.get("question_original"),
        "question_ko": row.get("question_ko"),
        "options_original": json.loads(row.get("options_original_json") or "null"),
        "options_ko": json.loads(row.get("options_ko_json") or "null"),
        "caption_original": row.get("caption_original"),
        "caption_ko": row.get("caption_ko"),
        "instruction_original": row.get("instruction_original"),
        "instruction_ko": row.get("instruction_ko"),
        "answer": row.get("answer"),
    }


def write_unified_visualizer(*, package_dirs: list[Path], output_dir: Path, overwrite: bool = False) -> dict[str, Any]:
    assert_output_dir_safe(output_dir=output_dir, package_dirs=package_dirs, overwrite=overwrite)
    reset_dir(output_dir, overwrite=overwrite)
    records: list[dict[str, Any]] = []
    by_benchmark: dict[str, int] = {}
    media_refs = 0
    for package_dir in package_dirs:
        rows = load_metadata_rows(package_dir)
        for row in rows:
            viewer = row_to_viewer_record(row=row, package_dir=package_dir, visualizer_dir=output_dir)
            records.append(viewer)
            benchmark_id = str(viewer.get("benchmark_id") or "benchmark")
            by_benchmark[benchmark_id] = by_benchmark.get(benchmark_id, 0) + 1
            media_refs += len(viewer.get("media") or [])
    records.sort(key=lambda item: (str(item.get("benchmark_id")), str(item.get("record_id"))))
    write_json(output_dir / "viewer_data.json", {"records": records})
    (output_dir / "index.html").write_text(VISUALIZER_HTML, encoding="utf-8")
    (output_dir / "README.md").write_text(
        "# Unified K-MMAD visualizer\n\n"
        "Serve this directory with `python -m http.server 8000 -d <visualizer-dir>` "
        "or serve the parent export directory. Media paths are symlinks into the "
        "HF package media trees; do not move the visualizer without the packages.\n",
        encoding="utf-8",
    )
    summary = {
        "created_at": utc_now(),
        "status": "passed" if records else "failed",
        "output_dir": str(output_dir),
        "package_dirs": [str(path) for path in package_dirs],
        "records": len(records),
        "media_refs": media_refs,
        "by_benchmark": dict(sorted(by_benchmark.items())),
    }
    write_json(output_dir / "unified_visualizer_summary.json", summary)
    return summary


def validate_visualizer(path: Path) -> dict[str, Any]:
    errors: list[str] = []
    data_path = path / "viewer_data.json"
    index_path = path / "index.html"
    if not data_path.exists():
        errors.append(f"missing viewer_data.json: {data_path}")
        records: list[dict[str, Any]] = []
    else:
        records = json.loads(data_path.read_text(encoding="utf-8")).get("records", [])
    if not index_path.exists():
        errors.append(f"missing index.html: {index_path}")
    missing_media = []
    for row in records:
        for media in row.get("media") or []:
            if not (path / media).exists():
                missing_media.append({"record_id": row.get("record_id"), "media": media})
    if missing_media:
        errors.append(f"missing media refs: {missing_media[:10]}")
    return {
        "status": "passed" if not errors else "failed",
        "records": len(records),
        "missing_media_count": len(missing_media),
        "errors": errors,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--validate-only", type=Path)
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing an existing non-empty visualizer output directory")
    args = parser.parse_args(argv)
    if args.validate_only:
        result = validate_visualizer(args.validate_only)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["status"] == "passed" else 1
    if not args.package_dir or args.output_dir is None:
        parser.error("--package-dir and --output-dir are required unless --validate-only is used")
    summary = write_unified_visualizer(package_dirs=args.package_dir, output_dir=args.output_dir, overwrite=args.overwrite)
    validation = validate_visualizer(args.output_dir)
    write_json(args.output_dir / "unified_visualizer_validation.json", validation)
    print(json.dumps({"summary": summary, "validation": validation}, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "passed" and validation["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
