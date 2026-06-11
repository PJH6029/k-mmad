#!/usr/bin/env python3
"""Shared helpers for K-MMAD smoke tooling.

The helpers intentionally use only Python's standard library so they can run via
`uv run python ...` before project dependencies are settled.
"""

from __future__ import annotations

import csv
import json
import os
import random
import re
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

TEXT_EXTENSIONS = {".json", ".jsonl", ".csv"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

QUESTION_KEYS = ("question", "Question", "query", "prompt", "text")
OPTION_KEYS = ("options", "Options", "choices", "Choices", "option", "answers")
CAPTION_KEYS = ("caption", "Caption", "captions", "image_caption", "description")
PATH_KEYS = ("image", "image_path", "img_path", "path", "filename", "file_name")
ANSWER_KEYS = ("answer", "Answer", "label", "correct_answer", "gt")


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_config(path: str | Path = "configs/kmmad_translation_smoke.toml") -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = repo_root() / config_path
    with config_path.open("rb") as fh:
        return tomllib.load(fh)


def configured_path(config: dict[str, Any], key: str) -> Path:
    paths = config.get("paths", {})
    root = Path(paths.get("dataset_root", "."))
    value = paths[key]
    path = Path(value)
    return path if path.is_absolute() else root / path


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, data: Any) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    ensure_parent(path)
    with path.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_json_like(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        records: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if isinstance(obj, dict):
                    records.append(obj)
        return records
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        for key in ("data", "records", "rows", "questions", "annotations"):
            value = data.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        return [data]
    return []


def read_csv_records(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8-sig") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def read_records(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix in {".json", ".jsonl"}:
        return read_json_like(path)
    if suffix == ".csv":
        return read_csv_records(path)
    return []


def candidate_record_files(root: Path) -> list[Path]:
    preferred = ["mmad.json", "metadata.csv", "MMAD.json", "annotations.json"]
    found: list[Path] = []
    for name in preferred:
        found.extend(root.rglob(name))
    if found:
        return sorted(set(found), key=lambda p: (preferred.index(p.name) if p.name in preferred else 99, str(p)))
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in TEXT_EXTENSIONS)


def load_first_records(root: Path) -> tuple[Path | None, list[dict[str, Any]]]:
    for path in candidate_record_files(root):
        try:
            records = read_records(path)
        except Exception as exc:  # pragma: no cover - defensive report path
            print(f"warning: could not parse {path}: {exc}", file=sys.stderr)
            continue
        if records:
            return path, records
    return None, []


def pick_sample(records: list[dict[str, Any]], size: int, seed: int) -> list[dict[str, Any]]:
    if len(records) <= size:
        return list(records)
    rng = random.Random(seed)
    indexes = sorted(rng.sample(range(len(records)), size))
    return [records[i] for i in indexes]


def get_first(row: dict[str, Any], keys: Iterable[str]) -> tuple[str | None, Any]:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return key, row[key]
    lower = {str(k).lower(): k for k in row}
    for key in keys:
        real = lower.get(key.lower())
        if real is not None and row[real] not in (None, ""):
            return str(real), row[real]
    return None, None


def flatten_text(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, (int, float, bool)):
        return [str(value)]
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            out.extend(flatten_text(item))
        return out
    if isinstance(value, dict):
        out = []
        for item in value.values():
            out.extend(flatten_text(item))
        return out
    return [str(value)]


def looks_textual(value: Any) -> bool:
    return any(re.search(r"[A-Za-z가-힣]", text) for text in flatten_text(value))


def discover_text_fields(row: dict[str, Any]) -> list[str]:
    return sorted(str(key) for key, value in row.items() if looks_textual(value))


def discover_image_paths(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)


def relative_or_str(path: Path, start: Path | None = None) -> str:
    try:
        return str(path.relative_to(start or repo_root()))
    except ValueError:
        return str(path)


def command_exists(command: str) -> bool:
    for directory in os.environ.get("PATH", "").split(os.pathsep):
        if (Path(directory) / command).exists():
            return True
    return False
