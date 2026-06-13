#!/usr/bin/env python3
"""Prepare full BLINK/MME-RealWorld source records for K-MMAD translation.

This script writes normalized JSON record files and concrete media files under a
persistent dataset path.  It is non-destructive: existing media files are reused
and output JSON is overwritten atomically only at the requested prepared root.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from kmmad_common import read_records, utc_now, write_json

IMAGE_MAGIC_EXTENSIONS = (
    (b"\xff\xd8\xff", ".jpg"),
    (b"\x89PNG\r\n\x1a\n", ".png"),
    (b"RIFF", ".webp"),
    (b"GIF87a", ".gif"),
    (b"GIF89a", ".gif"),
)


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2) + "\n"


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json_dumps(value), encoding="utf-8")
    tmp.replace(path)


def infer_image_suffix(path_value: str | None, payload: bytes) -> str:
    if path_value:
        suffix = Path(path_value).suffix.lower()
        if suffix:
            return suffix
    for magic, suffix in IMAGE_MAGIC_EXTENSIONS:
        if payload.startswith(magic):
            return suffix
    return ".bin"


def safe_part(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value).strip("._-") or "item"


def prepare_blink(*, source_root: Path, output_root: Path) -> dict[str, Any]:
    import pyarrow.parquet as pq

    benchmark_root = output_root / "blink"
    records: list[dict[str, Any]] = []
    task_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    image_count = 0
    reused = 0
    written = 0

    for parquet_path in sorted(source_root.rglob("*.parquet")):
        task_name = parquet_path.parent.name
        source_split = parquet_path.name.split("-", 1)[0]
        table = pq.read_table(parquet_path)
        for row_offset, row in enumerate(table.to_pylist()):
            source_id = str(row.get("idx") or f"{source_split}_{task_name}_{row_offset:06d}")
            row_media: list[str] = []
            for image_key in ("image_1", "image_2", "image_3", "image_4"):
                image = row.get(image_key)
                if not isinstance(image, dict) or not image.get("bytes"):
                    continue
                payload = bytes(image["bytes"])
                suffix = infer_image_suffix(image.get("path"), payload)
                image_name = f"{image_key}{suffix}"
                rel = Path("media") / safe_part(task_name) / safe_part(source_split) / safe_part(source_id) / image_name
                target = benchmark_root / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists() and target.stat().st_size == len(payload):
                    reused += 1
                else:
                    target.write_bytes(payload)
                    written += 1
                row_media.append(rel.as_posix())
                image_count += 1
            record = {
                "id": source_id,
                "idx": source_id,
                "split": source_split,
                "source_split": source_split,
                "source_hf_config": task_name,
                "task": task_name,
                "subtask": row.get("sub_task") or task_name,
                "question": row.get("question") or row.get("prompt") or "",
                "choices": row.get("choices") or [],
                "answer": row.get("answer") or "",
                "instruction": row.get("prompt") or "",
                "explanation": row.get("explanation") or "",
                "image_paths": row_media,
                "source_parquet": parquet_path.relative_to(source_root).as_posix(),
                "source_row_index": row_offset,
            }
            records.append(record)
            task_counts[task_name] += 1
            split_counts[source_split] += 1

    atomic_write_json(benchmark_root / "blink.json", records)
    manifest = {
        "created_at": utc_now(),
        "benchmark_id": "blink",
        "source_root": str(source_root),
        "record_file": str(benchmark_root / "blink.json"),
        "rows": len(records),
        "media_files_referenced": image_count,
        "media_files_written": written,
        "media_files_reused": reused,
        "source_splits": dict(sorted(split_counts.items())),
        "tasks": dict(sorted(task_counts.items())),
    }
    write_json(benchmark_root / "source_manifest.json", manifest)
    return manifest


def load_mme_records(source_root: Path) -> list[dict[str, Any]]:
    path = source_root / "MME_RealWorld.json"
    records = read_records(path)
    if not records:
        raise FileNotFoundError(f"no MME records found in {path}")
    return records


def tar_groups(source_root: Path) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = {}
    for path in sorted(source_root.glob("*.tar.gz")):
        groups[path.name[:-7]] = [path]
    grouped_parts: dict[str, list[Path]] = defaultdict(list)
    for part in sorted(source_root.glob("*.tar.gz.part_*")):
        grouped_parts[part.name.split(".part_", 1)[0][:-7]].append(part)
    groups.update({key: sorted(parts) for key, parts in grouped_parts.items()})
    return groups


def extract_with_tar(*, archive_parts: list[Path], list_file: Path, media_root: Path) -> None:
    media_root.mkdir(parents=True, exist_ok=True)
    if len(archive_parts) == 1 and archive_parts[0].suffixes[-2:] == [".tar", ".gz"]:
        subprocess.run(
            ["tar", "-xzf", str(archive_parts[0]), "-C", str(media_root), "-T", str(list_file)],
            check=True,
        )
        return
    cat_proc = subprocess.Popen(["cat", *map(str, archive_parts)], stdout=subprocess.PIPE)
    assert cat_proc.stdout is not None
    try:
        subprocess.run(
            ["tar", "-xzf", "-", "-C", str(media_root), "-T", str(list_file)],
            stdin=cat_proc.stdout,
            check=True,
        )
    finally:
        cat_proc.stdout.close()
        cat_proc.wait()


def validate_member_ref(ref: str) -> str:
    normalized = ref.strip().lstrip("./")
    path = Path(normalized)
    if not normalized or path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"unsafe archive member path: {ref!r}")
    return normalized


def extract_mme_media(*, source_root: Path, media_root: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    refs = sorted({validate_member_ref(str(row.get("Image") or "")) for row in records})
    refs_by_prefix: dict[str, list[str]] = defaultdict(list)
    for ref in refs:
        refs_by_prefix[ref.split("/", 1)[0]].append(ref)
    groups = tar_groups(source_root)
    extracted: dict[str, Any] = {}
    for prefix, prefix_refs in sorted(refs_by_prefix.items()):
        missing = [ref for ref in prefix_refs if not (media_root / ref).is_file()]
        archive_parts = groups.get(prefix)
        if not archive_parts:
            extracted[prefix] = {"status": "missing_archive", "needed": len(prefix_refs), "missing": len(missing)}
            continue
        if not missing:
            extracted[prefix] = {"status": "reused", "needed": len(prefix_refs), "missing": 0}
            continue
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as fh:
            list_file = Path(fh.name)
            for ref in missing:
                fh.write(ref + "\n")
        try:
            extract_with_tar(archive_parts=archive_parts, list_file=list_file, media_root=media_root)
        finally:
            list_file.unlink(missing_ok=True)
        still_missing = [ref for ref in prefix_refs if not (media_root / ref).is_file()]
        extracted[prefix] = {
            "status": "passed" if not still_missing else "missing_members",
            "needed": len(prefix_refs),
            "requested_extraction": len(missing),
            "missing": len(still_missing),
            "archive_parts": [path.name for path in archive_parts],
        }
    return extracted


def prepare_mme(*, source_root: Path, output_root: Path, extract_media: bool) -> dict[str, Any]:
    records = load_mme_records(source_root)
    benchmark_root = output_root / "mme_realworld"
    media_root = benchmark_root / "media"
    prepared: list[dict[str, Any]] = []
    prefix_counts: Counter[str] = Counter()
    task_counts: Counter[str] = Counter()
    for idx, row in enumerate(records):
        image_ref = validate_member_ref(str(row.get("Image") or ""))
        prefix_counts[image_ref.split("/", 1)[0]] += 1
        task_counts[str(row.get("Subtask") or row.get("Task") or "unknown")] += 1
        prepared.append(
            {
                "id": row.get("Question_id") or f"mme_realworld-{idx:06d}",
                "question_id": row.get("Question_id") or f"mme_realworld-{idx:06d}",
                "split": "test",
                "source_split": "test",
                "task": row.get("Task") or "",
                "subtask": row.get("Subtask") or "",
                "category": row.get("Category") or "",
                "source_dataset": row.get("Dataset") or "",
                "question_type": row.get("Question Type") or "",
                "question": row.get("Text") or "",
                "choices": row.get("Answer choices") or [],
                "answer": row.get("Ground truth") or "",
                "image_path": (Path("media") / image_ref).as_posix(),
                "source_image": image_ref,
                "source_record": row,
            }
        )
    extraction = extract_mme_media(source_root=source_root, media_root=media_root, records=records) if extract_media else {}
    atomic_write_json(benchmark_root / "mme_realworld.json", prepared)
    missing_media = sum(1 for item in prepared if not (benchmark_root / item["image_path"]).is_file())
    manifest = {
        "created_at": utc_now(),
        "benchmark_id": "mme_realworld",
        "source_root": str(source_root),
        "record_file": str(benchmark_root / "mme_realworld.json"),
        "rows": len(prepared),
        "source_image_prefixes": dict(sorted(prefix_counts.items())),
        "tasks": dict(sorted(task_counts.items())),
        "media_extraction": extraction,
        "missing_media_after_prepare": missing_media,
        "status": "passed" if missing_media == 0 else "needs_media",
    }
    write_json(benchmark_root / "source_manifest.json", manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blink-source", type=Path, required=True)
    parser.add_argument("--mme-source", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--skip-mme-media-extract", action="store_true")
    args = parser.parse_args(argv)

    args.output_root.mkdir(parents=True, exist_ok=True)
    blink = prepare_blink(source_root=args.blink_source, output_root=args.output_root)
    mme = prepare_mme(
        source_root=args.mme_source,
        output_root=args.output_root,
        extract_media=not args.skip_mme_media_extract,
    )
    summary = {
        "created_at": utc_now(),
        "status": "passed" if blink["rows"] > 0 and mme["status"] == "passed" else "failed",
        "output_root": str(args.output_root),
        "benchmarks": {"blink": blink, "mme_realworld": mme},
    }
    write_json(args.output_root / "full-source-preparation-summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
