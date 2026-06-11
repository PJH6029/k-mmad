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
PATH_KEYS = ("query_image", "image", "image_path", "img_path", "path", "filename", "file_name")
ANSWER_KEYS = ("answer", "Answer", "label", "correct_answer", "gt")
STRUCTURAL_TEXT_KEYS = {
    "answer",
    "gt",
    "label",
    "id",
    "dataset",
    "class",
    "class_name",
    "category",
    "split",
    "query_image",
    "template_image",
    "mask",
    "mask_path",
    "image",
    "image_path",
    "img_path",
    "path",
    "filename",
    "file_name",
    "mmad_image_key",
    "similar_templates",
    "random_templates",
}


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


def flatten_mmad_mapping(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize MMAD's image-keyed JSON into one row per QA turn.

    The upstream `mmad.json` is a mapping:

    `{query_image_path: {"conversation": [{"Question": ..., "Options": ...}, ...]}}`

    Translation smoke tooling is simpler and safer when each conversation turn is
    a row while structural image/template/mask fields remain preserved.
    """

    rows: list[dict[str, Any]] = []
    for image_key, value in data.items():
        if not isinstance(value, dict):
            rows.append({"mmad_image_key": image_key, "value": value})
            continue

        conversations = value.get("conversation")
        base = {key: item for key, item in value.items() if key != "conversation"}
        base.setdefault("query_image", image_key)
        base.setdefault("mmad_image_key", image_key)

        if isinstance(conversations, list):
            for idx, turn in enumerate(conversations):
                row = dict(base)
                row["conversation_index"] = idx
                if isinstance(turn, dict):
                    row.update(turn)
                else:
                    row["conversation"] = turn
                rows.append(row)
        else:
            rows.append(base)
    return rows


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
        if data and all(isinstance(value, dict) for value in data.values()):
            flattened = flatten_mmad_mapping(data)
            if flattened:
                return flattened
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
    preferred = ["metadata.csv", "mmad.json", "MMAD.json", "annotations.json"]
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
        return []
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


def looks_path_like(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    suffix = Path(stripped.split("?", 1)[0]).suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return True
    return "/" in stripped and " " not in stripped


def looks_label_like(text: str) -> bool:
    stripped = text.strip()
    return bool(re.fullmatch(r"[A-Z]|\d+|true|false|yes|no", stripped, flags=re.IGNORECASE))


def looks_textual(value: Any) -> bool:
    for text in flatten_text(value):
        if looks_path_like(text) or looks_label_like(text):
            continue
        if re.search(r"[A-Za-z가-힣]", text):
            return True
    return False


def discover_text_fields(row: dict[str, Any]) -> list[str]:
    return sorted(
        str(key)
        for key, value in row.items()
        if str(key).lower() not in STRUCTURAL_TEXT_KEYS and looks_textual(value)
    )


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
