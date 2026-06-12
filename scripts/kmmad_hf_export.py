#!/usr/bin/env python3
"""Export translated K-MMAD artifacts into a Hugging Face datasets layout.

The exporter assumes translation and translation QC have happened upstream. It
turns existing translation JSONL bundles into a stable evaluation-friendly
``DatasetDict.save_to_disk`` artifact plus JSONL shards, a manifest, and a
minimal dataset card. No Hub upload or destructive dataset operation is
performed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

from kmmad_common import (
    ANSWER_KEYS,
    CAPTION_KEYS,
    OPTION_KEYS,
    PATH_KEYS,
    QUESTION_KEYS,
    find_sensitive_strings,
    flatten_text,
    get_first,
    load_config,
    sanitize_jsonable,
    utc_now,
    write_json,
)
from kmmad_translate_smoke import validate_output as validate_translation_output

EXPORT_SCHEMA_VERSION = 1
HF_DATASET_DIRNAME = "dataset"
MANIFEST_FILENAME = "hf_export_manifest.json"
DATASET_CARD_FILENAME = "README.md"
DATA_DIRNAME = "data"
REQUIRED_COLUMNS = {
    "record_id",
    "benchmark_id",
    "source_id",
    "split",
    "question_ko",
    "translated_text_fields_json",
    "source_record_json",
    "translation_qc_status",
}
TEXT_FIELD_ALIASES = {
    "question": ("question", *QUESTION_KEYS),
    "options": ("options", *OPTION_KEYS),
    "caption": ("caption", *CAPTION_KEYS),
    "instruction": ("instruction", "instructions", "visual_prompt", "prompt_instruction", "context", "lecture"),
}
SAFE_SPLIT_RE = re.compile(r"[^A-Za-z0-9_]+")


def import_datasets_module() -> Any:
    try:
        return importlib.import_module("datasets")
    except ImportError as exc:  # pragma: no cover - exercised by users without deps installed.
        raise SystemExit(
            "Hugging Face export requires the 'datasets' package. Install with `uv sync` "
            "or `uv add datasets` before running this exporter."
        ) from exc


def json_dumps(value: Any) -> str:
    return json.dumps(sanitize_jsonable(value), ensure_ascii=False, sort_keys=True)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                item = json.loads(line)
                if not isinstance(item, dict):
                    raise ValueError(f"{path}: JSONL row is not an object")
                rows.append(item)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def stable_json_hash(value: Any, length: int = 12) -> str:
    payload = json.dumps(sanitize_jsonable(value), ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:length]


def sanitize_export_record(record: dict[str, Any]) -> dict[str, Any]:
    sanitized = sanitize_jsonable(record)
    if not isinstance(sanitized, dict):
        raise TypeError("sanitized export record is not a mapping")
    return sanitized


def normalize_split(value: Any, default_split: str) -> str:
    raw = str(value if value not in (None, "") else default_split).strip() or default_split
    normalized = SAFE_SPLIT_RE.sub("_", raw).strip("_").lower()
    return normalized or "test"


def first_non_empty(mapping: dict[str, Any], keys: Iterable[str]) -> tuple[str | None, Any]:
    key, value = get_first(mapping, keys)
    if key and value not in (None, ""):
        return key, value
    lower = {str(item_key).lower(): item_key for item_key in mapping}
    for key in keys:
        real = lower.get(str(key).lower())
        if real is not None and mapping.get(real) not in (None, ""):
            return str(real), mapping[real]
    return None, None


def as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def extract_translated_text_fields(row: dict[str, Any]) -> dict[str, Any]:
    translated = as_dict(row.get("translated"))
    text_fields = translated.get("text_fields")
    if isinstance(text_fields, dict):
        return text_fields
    return {str(key): value for key, value in translated.items() if value not in (None, "")}


def extract_source_record(row: dict[str, Any]) -> dict[str, Any]:
    source = as_dict(row.get("source"))
    nested_source = source.get("source_record")
    return nested_source if isinstance(nested_source, dict) else source


def extract_source_text_fields(row: dict[str, Any]) -> dict[str, Any]:
    source = as_dict(row.get("source"))
    text_fields = source.get("text_fields")
    if isinstance(text_fields, dict):
        return text_fields
    source_record = extract_source_record(row)
    extracted: dict[str, Any] = {}
    for canonical, aliases in TEXT_FIELD_ALIASES.items():
        key, value = first_non_empty(source_record, aliases)
        if key and value not in (None, ""):
            extracted[canonical] = value
    return extracted


def extract_value(mapping: dict[str, Any], canonical: str) -> Any:
    for key in TEXT_FIELD_ALIASES[canonical]:
        if key in mapping and mapping[key] not in (None, ""):
            return mapping[key]
    lower = {str(key).lower(): key for key in mapping}
    for key in TEXT_FIELD_ALIASES[canonical]:
        real = lower.get(str(key).lower())
        if real is not None and mapping.get(real) not in (None, ""):
            return mapping[real]
    return None


def value_to_text(value: Any) -> str:
    if value in (None, ""):
        return ""
    texts = flatten_text(value)
    return "\n".join(texts) if texts else str(value)


def extract_media(row: dict[str, Any]) -> list[str]:
    values: list[Any] = []
    if isinstance(row.get("media"), list):
        values.extend(row["media"])
    source = as_dict(row.get("source"))
    if isinstance(source.get("media"), list):
        values.extend(source["media"])
    source_record = extract_source_record(row)
    for key in PATH_KEYS:
        if key in source_record and source_record[key] not in (None, ""):
            values.append(source_record[key])
    media: list[str] = []
    for value in values:
        if isinstance(value, str):
            media.append(value)
        elif isinstance(value, list):
            media.extend(str(item) for item in value if item not in (None, ""))
        elif isinstance(value, dict):
            for key in ("path", "image", "image_path", "url", "video", "video_path"):
                if value.get(key):
                    media.append(str(value[key]))
                    break
        elif value not in (None, ""):
            media.append(str(value))
    return sorted(dict.fromkeys(media))


def extract_preserve_fields(row: dict[str, Any]) -> dict[str, Any]:
    preserve = as_dict(row.get("preserve_fields"))
    if preserve:
        return preserve
    source = as_dict(row.get("source"))
    preserve = as_dict(source.get("preserve_fields"))
    if preserve:
        return preserve
    source_record = extract_source_record(row)
    extracted: dict[str, Any] = {}
    for key in (*ANSWER_KEYS, "target", "correct_answer", "category", "dataset", "split"):
        if key in source_record and source_record[key] not in (None, ""):
            extracted[key] = source_record[key]
    return extracted


def extract_answer(row: dict[str, Any], preserve_fields: dict[str, Any]) -> str:
    for mapping in (preserve_fields, extract_source_record(row)):
        key, value = first_non_empty(mapping, (*ANSWER_KEYS, "target", "correct_answer"))
        if key and value not in (None, ""):
            return value_to_text(value)
    return ""


def source_id_for(row: dict[str, Any], index: int) -> str:
    source_record = extract_source_record(row)
    turn = source_record.get("conversation_index")
    turn_suffix = f"#turn-{turn}" if turn not in (None, "") else ""
    if row.get("source_id") not in (None, ""):
        return f"{row['source_id']}{turn_suffix}"
    for key in ("id", "uid", "sample_id", "question_id", "mmad_image_key"):
        if source_record.get(key) not in (None, ""):
            return f"{source_record[key]}{turn_suffix}"
    return f"row-{stable_json_hash(source_record or row)}"


def normalized_hf_record(
    row: dict[str, Any],
    *,
    index: int,
    source_artifact_label: str,
    dataset_name: str,
    original_hf_dataset: str,
    original_hf_config: str | None,
    original_hf_revision: str | None,
    default_split: str,
    fallback_benchmark_id: str,
    translation_run_id: str | None,
    translation_qc_status: str,
    translation_qc_report: str | None,
    exported_at: str,
) -> dict[str, Any]:
    source_record = extract_source_record(row)
    source_text_fields = extract_source_text_fields(row)
    translated_text_fields = extract_translated_text_fields(row)
    preserve_fields = extract_preserve_fields(row)
    skip_fields = as_dict(row.get("skip_fields")) or as_dict(as_dict(row.get("source")).get("skip_fields"))
    benchmark_id = str(row.get("benchmark_id") or fallback_benchmark_id)
    source_id = source_id_for(row, index)
    split = normalize_split(row.get("split") or source_record.get("split"), default_split)
    task = value_to_text(row.get("task") or source_record.get("task") or source_record.get("category"))
    question_ko = value_to_text(extract_value(translated_text_fields, "question"))
    options_ko = extract_value(translated_text_fields, "options")
    caption_ko = value_to_text(extract_value(translated_text_fields, "caption"))
    instruction_ko = value_to_text(extract_value(translated_text_fields, "instruction"))
    question_original = value_to_text(extract_value(source_text_fields, "question"))
    options_original = extract_value(source_text_fields, "options")
    caption_original = value_to_text(extract_value(source_text_fields, "caption"))
    instruction_original = value_to_text(extract_value(source_text_fields, "instruction"))
    record_id = f"{benchmark_id}/{split}/{source_id}"
    record = {
        "record_id": record_id,
        "dataset_name": dataset_name,
        "benchmark_id": benchmark_id,
        "benchmark_name": str(row.get("benchmark_name") or benchmark_id),
        "source_id": source_id,
        "split": split,
        "task": task,
        "media": extract_media(row),
        "question_original": question_original,
        "options_original_json": json_dumps(options_original),
        "caption_original": caption_original,
        "instruction_original": instruction_original,
        "question_ko": question_ko,
        "options_ko_json": json_dumps(options_ko),
        "caption_ko": caption_ko,
        "instruction_ko": instruction_ko,
        "answer": extract_answer(row, preserve_fields),
        "translated_text_fields_json": json_dumps(translated_text_fields),
        "preserve_fields_json": json_dumps(preserve_fields),
        "skip_fields_json": json_dumps(skip_fields),
        "source_record_json": json_dumps(source_record),
        "translation_scope": [str(item) for item in row.get("translation_scope", [])],
        "translation_artifact_path": source_artifact_label,
        "translation_run_id": translation_run_id or "",
        "translation_qc_status": translation_qc_status,
        "translation_qc_report": translation_qc_report or "",
        "original_hf_dataset": original_hf_dataset,
        "original_hf_config": original_hf_config or "",
        "original_hf_revision": original_hf_revision or "",
        "exported_at": exported_at,
        "export_schema_version": EXPORT_SCHEMA_VERSION,
    }
    return sanitize_export_record(record)


def translation_jsonl_files(paths: Iterable[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_file():
            files.append(path)
            continue
        if not path.exists():
            raise FileNotFoundError(f"translation output does not exist: {path}")
        summary_path = path / "general_vqa_translation_summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            for benchmark_id in summary.get("benchmarks", []):
                candidate = path / str(benchmark_id) / "translation_smoke.jsonl"
                if not candidate.exists():
                    raise FileNotFoundError(f"summary-listed translation output missing: {candidate}")
                files.append(candidate)
            continue
        candidate = path / "translation_smoke.jsonl"
        if candidate.exists():
            files.append(candidate)
            continue
        raise FileNotFoundError(
            f"directory export requires general_vqa_translation_summary.json or direct translation_smoke.jsonl: {path}"
        )
    return list(dict.fromkeys(files))


def safe_label_part(value: str) -> str:
    label = SAFE_SPLIT_RE.sub("_", value).strip("_").lower()
    return label or "input"


def build_input_root_labels(roots: Iterable[Path]) -> dict[Path, str]:
    labels: dict[Path, str] = {}
    for idx, root in enumerate(roots):
        stem = root.name if root.name else str(root)
        labels[root] = f"input-{idx:02d}-{safe_label_part(stem)}"
    return labels


def artifact_label_for(jsonl_file: Path, root_labels: dict[Path, str]) -> str:
    for root, label in root_labels.items():
        if root.is_dir():
            try:
                return f"{label}/{jsonl_file.relative_to(root).as_posix()}"
            except ValueError:
                continue
        if jsonl_file == root:
            return label
    return f"unmapped-{safe_label_part(jsonl_file.name)}"


def validate_translation_artifact_path(path: Path) -> dict[str, Any]:
    if path.is_dir() and (path / "translation_smoke.jsonl").exists():
        report = path / "untranslated_fields.json"
        return validate_translation_output(path / "translation_smoke.jsonl", report if report.exists() else None)
    return validate_translation_output(path)


def assert_output_dir_safe(output_dir: Path) -> None:
    if not output_dir.exists():
        return
    contents = list(output_dir.iterdir())
    if contents:
        raise FileExistsError(
            f"refusing to write into non-empty output directory without deleting data: {output_dir}"
        )


def group_by_split(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    splits: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        splits.setdefault(str(record["split"]), []).append(record)
    return dict(sorted(splits.items()))


def write_dataset_card(
    output_dir: Path,
    *,
    dataset_name: str,
    original_hf_dataset: str,
    original_hf_config: str | None,
    original_hf_revision: str | None,
    split_counts: dict[str, int],
    translation_qc_status: str,
    hub_repo_id: str | None,
) -> None:
    split_lines = "\n".join(f"- `{split}`: {count} rows" for split, count in split_counts.items())
    hub_note = (
        f"\nFuture upload target (dry-run only in this repo task): `{hub_repo_id}`.\n"
        if hub_repo_id
        else "\nNo Hub upload target was requested for this export.\n"
    )
    card = f"""---
language:
- ko
task_categories:
- visual-question-answering
tags:
- korean
- translated-benchmark
- multimodal
pretty_name: {dataset_name}
---

# {dataset_name}

This is a K-MMAD/Korean VQA export artifact generated from translated benchmark records for later model evaluation.

## Source provenance

- Original HF dataset/repo: `{original_hf_dataset}`
- Original HF config: `{original_hf_config or 'not specified'}`
- Original HF revision: `{original_hf_revision or 'not specified'}`
- Translation QC status recorded at export: `{translation_qc_status}`

## Splits

{split_lines or '- no rows'}

## Evaluation contract

Rows preserve source IDs, benchmark IDs, split/task metadata, media references, answer/preserve fields, skipped fields, the original source record JSON, and translated Korean text fields. Nested source/translation structures are stored as JSON strings to keep Arrow columns stable across heterogeneous VQA benchmarks.

This export does not perform benchmark evaluation, leaderboard submission, translation, or translation-quality scoring. It assumes upstream translation and QC are complete and records the declared QC status for downstream consumers. Production release readiness requires `translation_qc_status=passed` plus a machine-readable QC report path in the manifest; otherwise treat the export as a non-release packaging artifact.
{hub_note}
"""
    (output_dir / DATASET_CARD_FILENAME).write_text(card, encoding="utf-8")


def dataset_features(datasets: Any) -> Any:
    return datasets.Features({
        "record_id": datasets.Value("string"),
        "dataset_name": datasets.Value("string"),
        "benchmark_id": datasets.Value("string"),
        "benchmark_name": datasets.Value("string"),
        "source_id": datasets.Value("string"),
        "split": datasets.Value("string"),
        "task": datasets.Value("string"),
        "media": datasets.List(datasets.Value("string")),
        "question_original": datasets.Value("string"),
        "options_original_json": datasets.Value("string"),
        "caption_original": datasets.Value("string"),
        "instruction_original": datasets.Value("string"),
        "question_ko": datasets.Value("string"),
        "options_ko_json": datasets.Value("string"),
        "caption_ko": datasets.Value("string"),
        "instruction_ko": datasets.Value("string"),
        "answer": datasets.Value("string"),
        "translated_text_fields_json": datasets.Value("string"),
        "preserve_fields_json": datasets.Value("string"),
        "skip_fields_json": datasets.Value("string"),
        "source_record_json": datasets.Value("string"),
        "translation_scope": datasets.List(datasets.Value("string")),
        "translation_artifact_path": datasets.Value("string"),
        "translation_run_id": datasets.Value("string"),
        "translation_qc_status": datasets.Value("string"),
        "translation_qc_report": datasets.Value("string"),
        "original_hf_dataset": datasets.Value("string"),
        "original_hf_config": datasets.Value("string"),
        "original_hf_revision": datasets.Value("string"),
        "exported_at": datasets.Value("string"),
        "export_schema_version": datasets.Value("int64"),
    })


def save_hf_dataset(output_dir: Path, split_records: dict[str, list[dict[str, Any]]]) -> Any:
    datasets = import_datasets_module()
    features = dataset_features(datasets)
    dataset_dict = datasets.DatasetDict(
        {split: datasets.Dataset.from_list(records, features=features) for split, records in split_records.items()}
    )
    dataset_dir = output_dir / HF_DATASET_DIRNAME
    dataset_dict.save_to_disk(str(dataset_dir))
    return dataset_dict


def loaded_split_counts(dataset_obj: Any) -> dict[str, int]:
    if hasattr(dataset_obj, "items"):
        return {str(split): int(len(dataset)) for split, dataset in dataset_obj.items()}
    return {"train": int(len(dataset_obj))}


def add_secret_errors(errors: list[str], label: str, value: Any) -> None:
    for finding in find_sensitive_strings(value):
        suffix = finding[1:] if finding.startswith("$") else f".{finding}"
        errors.append(f"secret-like value found in export artifact: {label}{suffix}")


def validate_record_ids(errors: list[str], record_ids: dict[str, str], record_id: Any, label: str) -> None:
    if record_id in (None, ""):
        errors.append(f"record_id missing: {label}")
        return
    key = str(record_id)
    if key in record_ids:
        errors.append(f"duplicate record_id: {key} at {label}; first seen at {record_ids[key]}")
    else:
        record_ids[key] = label


def validate_export(output_dir: Path) -> dict[str, Any]:
    errors: list[str] = []
    manifest_path = output_dir / MANIFEST_FILENAME
    dataset_dir = output_dir / HF_DATASET_DIRNAME
    card_path = output_dir / DATASET_CARD_FILENAME
    manifest: dict[str, Any] = {}
    if not manifest_path.exists():
        errors.append(f"manifest missing: {manifest_path}")
    else:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            errors.append(f"manifest is not valid JSON: {exc}")
    if not dataset_dir.exists():
        errors.append(f"HF dataset directory missing: {dataset_dir}")
    if not card_path.exists():
        errors.append(f"dataset card missing: {card_path}")
    split_counts: dict[str, int] = {}
    columns: set[str] = set()
    loaded_record_ids: dict[str, str] = {}
    if dataset_dir.exists():
        try:
            datasets = import_datasets_module()
            loaded = datasets.load_from_disk(str(dataset_dir))
            split_counts = loaded_split_counts(loaded)
            if hasattr(loaded, "items"):
                for split, dataset in loaded.items():
                    columns.update(str(name) for name in dataset.column_names)
                    for idx, row in enumerate(dataset):
                        label = f"dataset[{split}][{idx}]"
                        add_secret_errors(errors, label, row)
                        validate_record_ids(errors, loaded_record_ids, row.get("record_id"), label)
            else:
                columns.update(str(name) for name in loaded.column_names)
                for idx, row in enumerate(loaded):
                    label = f"dataset[train][{idx}]"
                    add_secret_errors(errors, label, row)
                    validate_record_ids(errors, loaded_record_ids, row.get("record_id"), label)
        except Exception as exc:  # noqa: BLE001 - validation should report all loader errors.
            errors.append(f"load_from_disk failed: {exc}")
    missing_columns = sorted(REQUIRED_COLUMNS - columns)
    errors.extend(f"required column missing: {column}" for column in missing_columns)
    manifest_counts = manifest.get("splits", {}) if isinstance(manifest.get("splits"), dict) else {}
    if split_counts and manifest_counts:
        expected = {str(split): int(item.get("num_rows", -1)) for split, item in manifest_counts.items() if isinstance(item, dict)}
        if split_counts != expected:
            errors.append(f"split counts mismatch: loaded={split_counts} manifest={expected}")
    jsonl_record_ids: dict[str, str] = {}
    data_dir = output_dir / DATA_DIRNAME
    discovered_jsonl = sorted(data_dir.glob("*.jsonl")) if data_dir.exists() else []
    expected_jsonl_paths = {output_dir / DATA_DIRNAME / f"{split}.jsonl" for split in split_counts}
    for expected_path in sorted(expected_jsonl_paths):
        if not expected_path.exists():
            errors.append(f"split JSONL missing: {expected_path}")
    for jsonl_path in discovered_jsonl:
        split = jsonl_path.stem
        count = split_counts.get(split)
        try:
            jsonl_rows = read_jsonl(jsonl_path)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            errors.append(f"split JSONL is not readable: {jsonl_path}: {exc}")
            continue
        if count is None:
            errors.append(f"unexpected split JSONL shard: {jsonl_path}")
        elif len(jsonl_rows) != count:
            errors.append(f"split JSONL count mismatch for {split}: jsonl={len(jsonl_rows)} dataset={count}")
        for idx, row in enumerate(jsonl_rows):
            label = f"data/{jsonl_path.name}[{idx}]"
            add_secret_errors(errors, label, row)
            validate_record_ids(errors, jsonl_record_ids, row.get("record_id"), label)
    add_secret_errors(errors, "manifest", manifest)
    if card_path.exists():
        add_secret_errors(errors, "card", card_path.read_text(encoding="utf-8"))
    validation_path = output_dir / "hf_export_validation.json"
    if validation_path.exists():
        try:
            add_secret_errors(errors, "validation", json.loads(validation_path.read_text(encoding="utf-8")))
        except json.JSONDecodeError as exc:
            errors.append(f"validation report is not valid JSON: {exc}")
    return {
        "status": "passed" if not errors else "failed",
        "output_dir": ".",
        "splits": split_counts,
        "columns": sorted(columns),
        "errors": errors,
    }


def export_translations(
    *,
    translation_outputs: list[Path],
    output_dir: Path,
    dataset_name: str,
    original_hf_dataset: str,
    original_hf_config: str | None,
    original_hf_revision: str | None,
    default_split: str,
    fallback_benchmark_id: str,
    translation_run_id: str | None,
    translation_qc_status: str,
    translation_qc_report: str | None,
    hub_repo_id: str | None,
    skip_translation_validation: bool = False,
) -> dict[str, Any]:
    assert_output_dir_safe(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    exported_at = utc_now()
    if not skip_translation_validation:
        for path in translation_outputs:
            validation = validate_translation_artifact_path(path)
            if validation.get("status") != "passed":
                raise ValueError(f"translation artifact validation failed for {path}: {validation.get('errors')}")
    jsonl_files = translation_jsonl_files(translation_outputs)
    input_root_labels = build_input_root_labels(translation_outputs)
    records: list[dict[str, Any]] = []
    for jsonl_file in jsonl_files:
        for row in read_jsonl(jsonl_file):
            records.append(
                normalized_hf_record(
                    row,
                    index=len(records),
                    source_artifact_label=artifact_label_for(jsonl_file, input_root_labels),
                    dataset_name=dataset_name,
                    original_hf_dataset=original_hf_dataset,
                    original_hf_config=original_hf_config,
                    original_hf_revision=original_hf_revision,
                    default_split=default_split,
                    fallback_benchmark_id=fallback_benchmark_id,
                    translation_run_id=translation_run_id,
                    translation_qc_status=translation_qc_status,
                    translation_qc_report=translation_qc_report,
                    exported_at=exported_at,
                )
            )
    if not records:
        raise ValueError("no translated rows found to export")
    seen_record_ids: dict[str, int] = {}
    duplicate_record_ids: list[str] = []
    for idx, record in enumerate(records):
        record_id = record.get("record_id")
        if record_id in (None, ""):
            raise ValueError(f"record_id missing before export at normalized_records[{idx}]")
        key = str(record_id)
        if key in seen_record_ids:
            duplicate_record_ids.append(key)
        else:
            seen_record_ids[key] = idx
    if duplicate_record_ids:
        raise ValueError(f"duplicate record_id values before export: {sorted(set(duplicate_record_ids))}")
    split_records = group_by_split(records)
    for split, rows in split_records.items():
        write_jsonl(output_dir / DATA_DIRNAME / f"{split}.jsonl", rows)
    dataset_dict = save_hf_dataset(output_dir, split_records)
    split_counts = {split: len(rows) for split, rows in split_records.items()}
    write_dataset_card(
        output_dir,
        dataset_name=dataset_name,
        original_hf_dataset=original_hf_dataset,
        original_hf_config=original_hf_config,
        original_hf_revision=original_hf_revision,
        split_counts=split_counts,
        translation_qc_status=translation_qc_status,
        hub_repo_id=hub_repo_id,
    )
    manifest = {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "created_at": exported_at,
        "dataset_name": dataset_name,
        "original_hf": {
            "dataset": original_hf_dataset,
            "config": original_hf_config,
            "revision": original_hf_revision,
        },
        "translation": {
            "input_roots": list(input_root_labels.values()),
            "jsonl_files": [artifact_label_for(path, input_root_labels) for path in jsonl_files],
            "run_id": translation_run_id,
            "qc_status": translation_qc_status,
            "qc_report": translation_qc_report,
            "qc_attestation_present": bool(translation_qc_report),
            "release_ready": translation_qc_status == "passed" and bool(translation_qc_report),
            "qc_boundary": "Exporter records declared QC status; production release requires an external machine-readable QC report path.",
        },
        "export": {
            "output_dir": ".",
            "dataset_dir": HF_DATASET_DIRNAME,
            "data_dir": DATA_DIRNAME,
            "dataset_card": DATASET_CARD_FILENAME,
            "hub_repo_id": hub_repo_id,
            "hub_upload_performed": False,
            "push_to_hub_command": f"load_from_disk('{HF_DATASET_DIRNAME}').push_to_hub('{hub_repo_id}')" if hub_repo_id else None,
        },
        "splits": {
            split: {
                "num_rows": count,
                "jsonl": f"{DATA_DIRNAME}/{split}.jsonl",
            }
            for split, count in split_counts.items()
        },
        "columns": list(next(iter(dataset_dict.values())).column_names) if split_records else [],
    }
    write_json(output_dir / MANIFEST_FILENAME, manifest)
    validation = validate_export(output_dir)
    write_json(output_dir / "hf_export_validation.json", validation)
    if validation["status"] != "passed":
        raise ValueError(f"HF export validation failed: {validation['errors']}")
    return manifest | {"validation": validation}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/kmmad_translation_smoke.toml")
    parser.add_argument("--translation-output", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--original-hf-dataset", default=None)
    parser.add_argument("--original-hf-config", default=None)
    parser.add_argument("--original-hf-revision", default=None)
    parser.add_argument("--default-split", default=None)
    parser.add_argument("--fallback-benchmark-id", default=None)
    parser.add_argument("--translation-run-id", default=None)
    parser.add_argument(
        "--translation-qc-status",
        choices=("unchecked", "passed", "failed", "needs_review"),
        default=None,
    )
    parser.add_argument("--translation-qc-report", default=None)
    parser.add_argument("--hub-repo-id", default=None, help="Record intended Hub target; upload is not performed")
    parser.add_argument("--skip-translation-validation", action="store_true")
    parser.add_argument("--validate-only", type=Path, help="Validate an existing HF export directory")
    args = parser.parse_args(argv)

    if args.validate_only:
        result = validate_export(args.validate_only)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["status"] == "passed" else 1

    if not args.translation_output:
        parser.error("--translation-output is required unless --validate-only is used")
    if args.output_dir is None:
        parser.error("--output-dir is required unless --validate-only is used")

    config = load_config(args.config)
    export_config = config.get("hf_export", {})
    manifest = export_translations(
        translation_outputs=args.translation_output,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name or str(export_config.get("dataset_name", "k-mmad-ko")),
        original_hf_dataset=args.original_hf_dataset or str(export_config.get("original_hf_dataset", "jiang-cc/MMAD")),
        original_hf_config=args.original_hf_config or export_config.get("original_hf_config"),
        original_hf_revision=args.original_hf_revision or export_config.get("original_hf_revision"),
        default_split=args.default_split or str(export_config.get("default_split", "test")),
        fallback_benchmark_id=args.fallback_benchmark_id or str(export_config.get("fallback_benchmark_id", "mmad")),
        translation_run_id=args.translation_run_id,
        translation_qc_status=args.translation_qc_status or str(export_config.get("translation_qc_status", "unchecked")),
        translation_qc_report=args.translation_qc_report,
        hub_repo_id=args.hub_repo_id,
        skip_translation_validation=args.skip_translation_validation,
    )
    print(json.dumps({"status": "passed", "manifest": manifest}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
