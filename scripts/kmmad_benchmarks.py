#!/usr/bin/env python3
"""Benchmark registry and adapters for K-MMAD general VQA translation smoke."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from kmmad_common import get_first, load_first_records, pick_sample, read_records


@dataclass(frozen=True)
class BenchmarkSpec:
    benchmark_id: str
    display_name: str
    venue: str
    source_urls: tuple[str, ...]
    leaderboard_checked: str
    record_files: tuple[str, ...]
    question_keys: tuple[str, ...]
    option_keys: tuple[str, ...]
    instruction_keys: tuple[str, ...]
    preserve_keys: tuple[str, ...]
    skip_keys: tuple[str, ...]
    media_keys: tuple[str, ...]
    task_keys: tuple[str, ...]
    split_keys: tuple[str, ...] = ("split", "Split", "subset")
    id_keys: tuple[str, ...] = ("id", "sample_id", "question_id", "uid", "index")


BENCHMARKS: dict[str, BenchmarkSpec] = {
    "mme_realworld": BenchmarkSpec(
        benchmark_id="mme_realworld",
        display_name="MME-RealWorld",
        venue="ICLR 2025",
        source_urls=(
            "https://mme-realworld.github.io/home_page.html",
            "https://openreview.net/forum?id=k5VHHgsRbi",
        ),
        leaderboard_checked="2026-06-11",
        record_files=("mme_realworld.json", "annotations.json", "questions.json", "data.json", "samples.jsonl"),
        question_keys=("question", "Question", "query"),
        option_keys=("options", "Options", "choices", "Choices"),
        instruction_keys=("instruction", "prompt", "hint"),
        preserve_keys=("answer", "Answer", "label", "category", "subtask", "scenario", "difficulty"),
        skip_keys=("explanation", "rationale"),
        media_keys=("image", "image_path", "img_path", "images"),
        task_keys=("task", "subtask", "category", "scenario"),
    ),
    "blink": BenchmarkSpec(
        benchmark_id="blink",
        display_name="BLINK",
        venue="ECCV 2024",
        source_urls=(
            "https://zeyofu.github.io/blink/",
            "https://eval.ai/api/challenges/challenge/2287/",
        ),
        leaderboard_checked="2026-06-11",
        record_files=("blink.json", "test.json", "val.json", "annotations.json", "samples.jsonl"),
        question_keys=("question", "Question", "prompt"),
        option_keys=("choices", "Choices", "options", "Options"),
        instruction_keys=("instruction", "visual_prompt", "prompt_instruction"),
        preserve_keys=("answer", "label", "task", "subtask", "source_dataset"),
        skip_keys=("explanation", "rationale"),
        media_keys=("image", "image_path", "images", "image_paths"),
        task_keys=("task", "subtask", "category"),
    ),
    "mmmu_pro": BenchmarkSpec(
        benchmark_id="mmmu_pro",
        display_name="MMMU-Pro",
        venue="ACL 2025",
        source_urls=(
            "https://aclanthology.org/2025.acl-long.736/",
            "https://mmmu-benchmark.github.io/",
        ),
        leaderboard_checked="2026-06-11",
        record_files=("mmmu_pro.json", "validation.json", "test.json", "annotations.json", "samples.jsonl"),
        question_keys=("question", "Question", "problem", "prompt"),
        option_keys=("options", "Options", "choices", "Choices"),
        instruction_keys=("instruction", "context", "lecture"),
        preserve_keys=("answer", "Answer", "subject", "discipline", "topic", "question_type", "input_type"),
        skip_keys=("ocr_text", "image_embedded_text", "solution", "explanation"),
        media_keys=("image", "image_path", "images", "image_paths"),
        task_keys=("subject", "discipline", "topic", "category"),
    ),
    "mega_bench": BenchmarkSpec(
        benchmark_id="mega_bench",
        display_name="MEGA-Bench",
        venue="ICLR 2025",
        source_urls=(
            "https://tiger-ai-lab.github.io/MEGA-Bench/",
            "https://huggingface.co/spaces/TIGER-Lab/MEGA-Bench",
        ),
        leaderboard_checked="2026-06-11",
        record_files=("mega_bench.json", "tasks.json", "samples.json", "annotations.json", "samples.jsonl"),
        question_keys=("question", "Question", "query", "task_prompt", "prompt"),
        option_keys=("options", "Options", "choices", "Choices"),
        instruction_keys=("instruction", "instructions", "input", "description"),
        preserve_keys=("answer", "target", "output_format", "metric", "evaluator", "task_name", "application"),
        skip_keys=("rubric", "scoring_rule", "expected_output", "reference_answer"),
        media_keys=("image", "image_path", "images", "image_paths", "video", "video_path", "media"),
        task_keys=("task", "task_name", "application", "skill", "category"),
    ),
}

BENCHMARK_ALIASES = {
    "mme-realworld": "mme_realworld",
    "mme_real_world": "mme_realworld",
    "mme": "mme_realworld",
    "blink-benchmark": "blink",
    "mmmu-pro": "mmmu_pro",
    "mmmu": "mmmu_pro",
    "mega-bench": "mega_bench",
    "megabench": "mega_bench",
}

STRUCTURAL_KEYS = {"source_record", "text_fields", "preserve_fields", "skip_fields", "media", "metadata"}


def normalize_benchmark_id(value: str) -> str:
    normalized = value.strip().lower().replace(" ", "_")
    normalized = BENCHMARK_ALIASES.get(normalized, normalized)
    if normalized not in BENCHMARKS:
        raise KeyError(f"unsupported benchmark id: {value}")
    return normalized


def parse_benchmark_ids(value: str | Iterable[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw = [item.strip() for item in value.split(",")]
    else:
        raw = [str(item).strip() for item in value]
    return [normalize_benchmark_id(item) for item in raw if item]


def registry_summary() -> list[dict[str, Any]]:
    return [
        {
            "benchmark_id": spec.benchmark_id,
            "display_name": spec.display_name,
            "venue": spec.venue,
            "source_urls": list(spec.source_urls),
            "leaderboard_checked": spec.leaderboard_checked,
            "record_files": list(spec.record_files),
        }
        for spec in BENCHMARKS.values()
    ]


def _first_present(row: dict[str, Any], keys: Iterable[str]) -> tuple[str | None, Any]:
    return get_first(row, keys)


def _copy_present(row: dict[str, Any], keys: Iterable[str]) -> dict[str, Any]:
    copied: dict[str, Any] = {}
    lower = {str(key).lower(): key for key in row}
    for key in keys:
        real: Any = key if key in row else lower.get(key.lower())
        if real is not None and row.get(real) not in (None, ""):
            copied[str(real)] = row[real]
    return copied


def _media_list(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        media: list[str] = []
        for item in value:
            if isinstance(item, str):
                media.append(item)
            elif isinstance(item, dict):
                for candidate in ("path", "image", "image_path", "url", "video", "video_path"):
                    if candidate in item and item[candidate]:
                        media.append(str(item[candidate]))
                        break
        return media
    if isinstance(value, dict):
        return [str(item) for item in value.values() if isinstance(item, str)]
    return [str(value)]


def _source_id(spec: BenchmarkSpec, row: dict[str, Any], index: int) -> str:
    _, value = _first_present(row, spec.id_keys)
    return str(value) if value not in (None, "") else f"{spec.benchmark_id}-{index:06d}"


def _text_fields(spec: BenchmarkSpec, row: dict[str, Any]) -> dict[str, Any]:
    text_fields: dict[str, Any] = {}
    for group, keys in (
        ("question", spec.question_keys),
        ("options", spec.option_keys),
        ("instruction", spec.instruction_keys),
    ):
        key, value = _first_present(row, keys)
        if key and value not in (None, ""):
            text_fields[group if group not in text_fields else key] = value
    return text_fields


def adapt_record(spec: BenchmarkSpec, row: dict[str, Any], index: int) -> dict[str, Any]:
    media: list[str] = []
    for value in _copy_present(row, spec.media_keys).values():
        media.extend(_media_list(value))
    split_key, split = _first_present(row, spec.split_keys)
    task_key, task = _first_present(row, spec.task_keys)
    text_fields = _text_fields(spec, row)
    preserve_fields = _copy_present(row, (*spec.preserve_keys, *spec.id_keys, *(split_key or "",), *(task_key or "",)))
    skip_fields = _copy_present(row, spec.skip_keys)
    return {
        "benchmark_id": spec.benchmark_id,
        "benchmark_name": spec.display_name,
        "source_id": _source_id(spec, row, index),
        "split": split if split not in (None, "") else None,
        "task": task if task not in (None, "") else None,
        "media": sorted(dict.fromkeys(media)),
        "text_fields": text_fields,
        "preserve_fields": preserve_fields,
        "skip_fields": skip_fields,
        "translation_scope": list(text_fields),
        "metadata": {
            "venue": spec.venue,
            "leaderboard_checked": spec.leaderboard_checked,
            "source_urls": list(spec.source_urls),
            "source_index": index,
        },
        "source_record": row,
    }


def benchmark_record_files(root: Path, spec: BenchmarkSpec) -> list[Path]:
    candidates: list[Path] = []
    for name in spec.record_files:
        candidates.extend(root.rglob(name))
    if candidates:
        return sorted(set(candidates), key=lambda p: (spec.record_files.index(p.name) if p.name in spec.record_files else 99, str(p)))
    return []


def load_benchmark_records(root: Path, benchmark_id: str) -> tuple[Path | None, list[dict[str, Any]]]:
    spec = BENCHMARKS[normalize_benchmark_id(benchmark_id)]
    for path in benchmark_record_files(root, spec):
        records = read_records(path)
        if records:
            return path, records
    # Allow direct file/directory fallback for fixtures named differently.
    if root.is_file():
        records = read_records(root)
        return (root, records) if records else (None, [])
    record_path, records = load_first_records(root)
    return record_path, records


def load_benchmark_sample(root: Path, benchmark_id: str, sample_size: int, seed: int) -> tuple[Path | None, list[dict[str, Any]]]:
    spec = BENCHMARKS[normalize_benchmark_id(benchmark_id)]
    record_path, records = load_benchmark_records(root, spec.benchmark_id)
    sample = pick_sample(records, sample_size, seed) if records else []
    return record_path, [adapt_record(spec, row, index) for index, row in enumerate(sample)]
