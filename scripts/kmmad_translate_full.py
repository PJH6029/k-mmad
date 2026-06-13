#!/usr/bin/env python3
"""Run resumable full Korean translation for MMAD/BLINK/MME records.

The translator batches multiple records into one OpenAI-compatible chat request
and writes only valid translated rows to translation_smoke.jsonl. Failed rows are
kept in translation_failures.jsonl so full-run stats can report invalid samples
without corrupting HF export inputs.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kmmad_benchmarks import BENCHMARKS, adapt_record, load_benchmark_records, normalize_benchmark_id
from kmmad_common import (
    CAPTION_KEYS,
    OPTION_KEYS,
    QUESTION_KEYS,
    flatten_text,
    get_first,
    load_first_records,
    sanitize_jsonable,
    utc_now,
    write_json,
)
from kmmad_translate_smoke import (
    build_benchmark_translation_row,
    preserve_leading_option_label,
    validate_output,
    validate_translated_value,
)

JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)
KOREAN_RE = re.compile(r"[가-힣]")
ASSISTANT_ARTIFACT_RE = re.compile(r"\bassistant\b", re.IGNORECASE)


@dataclass(frozen=True)
class WorkItem:
    index: int
    source_row: dict[str, Any]
    output_template: dict[str, Any]
    text_fields: dict[str, Any]
    mode: str


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def append_jsonl_locked(path: Path, rows: list[dict[str, Any]], lock: threading.Lock) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    with lock:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(payload)


def extract_json_payload(text: str) -> Any:
    stripped = JSON_FENCE_RE.sub("", text.strip()).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start = stripped.find(open_ch)
        end = stripped.rfind(close_ch)
        if start != -1 and end != -1 and end > start:
            return json.loads(stripped[start : end + 1])
    raise ValueError(f"translation response is not JSON: {text[:500]}")


def build_headers(api_key_env: str) -> dict[str, str]:
    token = os.environ.get(api_key_env)
    if not token:
        raise RuntimeError(f"missing required environment variable: {api_key_env}")
    return {"Content-Type": "application/json", "Authorization": f"Bearer {token}"}


def call_batch(
    *,
    base_url: str,
    api_key_env: str,
    model: str,
    items: list[WorkItem],
    max_tokens: int,
    timeout_seconds: int,
) -> dict[int, dict[str, Any]]:
    url = base_url.rstrip("/") + "/chat/completions"
    request_items = [
        {"index": item.index, "text_fields": sanitize_jsonable(item.text_fields)}
        for item in items
    ]
    payload = {
        "model": model,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You translate multimodal VQA benchmark text into natural Korean. "
                    "Return only strict JSON with schema {\"items\":[{\"index\":number,"
                    "\"translations\":object}]}. Translate every string value in each text_fields "
                    "object while preserving the original JSON shape, option labels, numbers, units, "
                    "line breaks, and answer-choice letters. Do not answer the questions, do not infer "
                    "from images, do not add explanations, and do not use Chinese/Hanja characters. "
                    "Use Hangul Korean for linguistic content. Preserve non-linguistic IDs or pure "
                    "symbols if there is nothing to translate."
                ),
            },
            {
                "role": "user",
                "content": json.dumps({"items": request_items}, ensure_ascii=False),
            },
        ],
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=build_headers(api_key_env), method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(f"endpoint call failed: {exc}") from exc
    content = body["choices"][0]["message"]["content"]
    parsed = extract_json_payload(content)
    raw_items = parsed.get("items") if isinstance(parsed, dict) else parsed
    if not isinstance(raw_items, list):
        raise ValueError(f"translation JSON missing items list: {parsed!r}")
    results: dict[int, dict[str, Any]] = {}
    for raw in raw_items:
        if not isinstance(raw, dict) or "index" not in raw or not isinstance(raw.get("translations"), dict):
            raise ValueError(f"invalid translated item: {raw!r}")
        results[int(raw["index"])] = raw["translations"]
    missing = sorted(item.index for item in items if item.index not in results)
    if missing:
        raise ValueError(f"translation response missing item indexes: {missing[:10]}")
    return results


def has_forbidden_translation_artifact(value: Any) -> bool:
    return any(ASSISTANT_ARTIFACT_RE.search(text) for text in flatten_text(value))


def translation_errors(index: int, benchmark_id: str, translated: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for key, value in translated.items():
        validate_translated_value(index, str(key), value, errors, benchmark_id=benchmark_id)
    if has_forbidden_translation_artifact(translated):
        errors.append(f"row {index} contains assistant artifact text")
    return errors


def apply_option_label_preservation(source: Any, translated: Any) -> Any:
    if isinstance(source, str) and isinstance(translated, str):
        return preserve_leading_option_label(source, translated)
    if isinstance(source, list) and isinstance(translated, list):
        return [
            apply_option_label_preservation(src, dst)
            for src, dst in zip(source, translated, strict=False)
        ]
    if isinstance(source, dict) and isinstance(translated, dict):
        return {
            key: apply_option_label_preservation(source.get(key), value)
            for key, value in translated.items()
        }
    return translated


def finish_item(item: WorkItem, translations: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    cleaned = {
        key: apply_option_label_preservation(item.text_fields.get(key), value)
        for key, value in translations.items()
    }
    benchmark_id = str(item.output_template.get("benchmark_id") or "")
    if item.mode == "benchmark":
        row = dict(item.output_template)
        row["translated"] = {"text_fields": cleaned}
        row["translation_scope"] = list(item.text_fields)
        validation_payload = {"text_fields": cleaned}
    else:
        row = dict(item.output_template)
        row["translated"] = cleaned
        validation_payload = cleaned
    row["full_translation_index"] = item.index
    errors = translation_errors(item.index, benchmark_id, validation_payload)
    return row, errors


def translate_items(
    *,
    items: list[WorkItem],
    base_url: str,
    api_key_env: str,
    model: str,
    max_tokens: int,
    timeout_seconds: int,
    retries: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    last_error: str | None = None
    for attempt in range(retries + 1):
        try:
            translated = call_batch(
                base_url=base_url,
                api_key_env=api_key_env,
                model=model,
                items=items,
                max_tokens=max_tokens,
                timeout_seconds=timeout_seconds,
            )
            ok_rows: list[dict[str, Any]] = []
            failures: list[dict[str, Any]] = []
            for item in items:
                row, errors = finish_item(item, translated[item.index])
                if errors:
                    failures.append(
                        {
                            "index": item.index,
                            "source_id": source_id_for_failure(item),
                            "errors": errors,
                            "attempt": attempt,
                            "translated": translated.get(item.index),
                        }
                    )
                else:
                    ok_rows.append(row)
            if not failures:
                return ok_rows, []
            last_error = json.dumps(failures[:3], ensure_ascii=False)
        except Exception as exc:  # noqa: BLE001 - batch fallback reports endpoint/parser failures.
            last_error = str(exc)
        if attempt < retries:
            time.sleep(min(10, 2**attempt))
    if len(items) > 1:
        mid = max(1, len(items) // 2)
        left_ok, left_fail = translate_items(
            items=items[:mid],
            base_url=base_url,
            api_key_env=api_key_env,
            model=model,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
            retries=retries,
        )
        right_ok, right_fail = translate_items(
            items=items[mid:],
            base_url=base_url,
            api_key_env=api_key_env,
            model=model,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
            retries=retries,
        )
        return left_ok + right_ok, left_fail + right_fail
    item = items[0]
    return [], [
        {
            "index": item.index,
            "source_id": source_id_for_failure(item),
            "errors": [last_error or "unknown translation failure"],
            "text_fields": sanitize_jsonable(item.text_fields),
        }
    ]


def source_id_for_failure(item: WorkItem) -> str:
    return str(item.output_template.get("source_id") or item.source_row.get("id") or item.index)


def mmad_text_fields(row: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    q_key, question = get_first(row, QUESTION_KEYS)
    o_key, options = get_first(row, OPTION_KEYS)
    c_key, caption = get_first(row, CAPTION_KEYS)
    text_fields: dict[str, Any] = {}
    missing: list[str] = []
    if q_key:
        text_fields[q_key] = question
    else:
        missing.append("question")
    if o_key:
        text_fields[o_key] = options
    else:
        missing.append("options")
    if c_key:
        text_fields[c_key] = caption
    else:
        missing.append("caption")
    return text_fields, missing


def load_work_items(*, benchmark: str, input_root: Path, limit: int | None) -> tuple[Path | None, list[WorkItem]]:
    if benchmark == "mmad":
        record_path, records = load_first_records(input_root)
        if limit is not None:
            records = records[:limit]
        items: list[WorkItem] = []
        for idx, row in enumerate(records):
            text_fields, missing = mmad_text_fields(row)
            template: dict[str, Any] = {
                "source": row,
                "translated": {},
                "translation_scope": list(text_fields),
            }
            if missing:
                template["missing_translation_scope"] = missing
            items.append(WorkItem(idx, row, template, text_fields, "mmad"))
        return record_path, items

    benchmark_id = normalize_benchmark_id(benchmark)
    spec = BENCHMARKS[benchmark_id]
    record_path, raw_records = load_benchmark_records(input_root, benchmark_id)
    if limit is not None:
        raw_records = raw_records[:limit]
    adapted = [adapt_record(spec, row, idx) for idx, row in enumerate(raw_records)]
    items = []
    empty_config: dict[str, Any] = {"inference": {"endpoint_provider": "openai", "model": ""}}
    for idx, record in enumerate(adapted):
        template, _, _ = build_benchmark_translation_row(record, empty_config, mock=True)
        template["translated"] = {}
        items.append(WorkItem(idx, record, template, dict(record.get("text_fields") or {}), "benchmark"))
    return record_path, items


def existing_indexes(output_path: Path, failure_path: Path) -> set[int]:
    indexes: set[int] = set()
    for row in read_jsonl(output_path):
        if "full_translation_index" in row:
            indexes.add(int(row["full_translation_index"]))
            continue
        if "source" in row and isinstance(row["source"], dict) and "metadata" in row["source"]:
            meta = row["source"].get("metadata") or {}
            if "source_index" in meta:
                indexes.add(int(meta["source_index"]))
    for row in read_jsonl(failure_path):
        if "index" in row:
            indexes.add(int(row["index"]))
    return indexes


def batched(items: list[WorkItem], batch_size: int) -> list[list[WorkItem]]:
    return [items[idx : idx + batch_size] for idx in range(0, len(items), batch_size)]


def summarize_output(
    *,
    benchmark: str,
    record_path: Path | None,
    output_dir: Path,
    source_count: int,
    model: str,
    base_url: str,
    concurrency: int,
    batch_size: int,
) -> dict[str, Any]:
    output_path = output_dir / "translation_smoke.jsonl"
    failure_path = output_dir / "translation_failures.jsonl"
    untranslated_path = output_dir / "untranslated_fields.json"
    validation = validate_output(output_path, untranslated_path if untranslated_path.exists() else None)
    translated_count = len(read_jsonl(output_path))
    failure_count = len(read_jsonl(failure_path))
    summary = {
        "created_at": utc_now(),
        "benchmark_id": benchmark,
        "record_file": str(record_path) if record_path else None,
        "source_count": source_count,
        "translated_count": translated_count,
        "valid_translated_count": translated_count if validation.get("status") == "passed" else 0,
        "invalid_count": failure_count + (0 if validation.get("status") == "passed" else translated_count),
        "failure_count": failure_count,
        "mock_sanity_only": False,
        "batch_mode": "chat_completions_record_batches",
        "batch_size": batch_size,
        "concurrency": concurrency,
        "inference": {
            "endpoint_provider": "openai_compatible",
            "base_url": base_url,
            "model": model,
            "auth_mode": "bearer_env",
            "api_key_env": "OPENAI_API_KEY",
        },
        "validation": validation,
        "artifacts": {
            "translation_output": str(output_path),
            "translation_failures": str(failure_path),
            "untranslated_field_report": str(untranslated_path),
            "inspection_examples": str(output_dir / "inspection_examples.json"),
            "validation_report": str(output_dir / "translation_validation.json"),
            "summary": str(output_dir / "translation_smoke_summary.json"),
        },
        "status": "passed" if translated_count + failure_count == source_count and validation.get("status") == "passed" else "failed",
    }
    write_json(output_dir / "translation_validation.json", validation)
    write_json(output_dir / "translation_smoke_summary.json", summary)
    return summary


def write_untranslated_report(output_dir: Path, benchmark: str, source_count: int, text_keys: list[str]) -> None:
    write_json(
        output_dir / "untranslated_fields.json",
        {
            "created_at": utc_now(),
            "benchmark_id": benchmark,
            "sample_size": source_count,
            "translated_scope": sorted(set(text_keys)),
            "untranslated_text_fields": {},
            "missing_configured_translation_fields": {},
        },
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True, choices=["mmad", "blink", "mme_realworld"])
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--base-url-env", default="BASE_URL")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)

    base_url = args.base_url or os.environ.get(args.base_url_env)
    if not base_url:
        raise SystemExit(f"missing --base-url or environment variable {args.base_url_env}")
    record_path, items = load_work_items(
        benchmark=args.benchmark,
        input_root=args.input_root,
        limit=args.limit,
    )
    if not items:
        raise SystemExit(f"no records found for {args.benchmark} under {args.input_root}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "translation_smoke.jsonl"
    failure_path = args.output_dir / "translation_failures.jsonl"
    if not args.resume:
        output_path.unlink(missing_ok=True)
        failure_path.unlink(missing_ok=True)
    done = existing_indexes(output_path, failure_path) if args.resume else set()
    pending = [item for item in items if item.index not in done]
    text_keys = [key for item in items for key in item.text_fields]
    write_untranslated_report(args.output_dir, args.benchmark, len(items), text_keys)
    lock = threading.Lock()
    batches = batched(pending, max(1, args.batch_size))
    failures_seen = 0
    ok_seen = 0
    started = time.time()
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as executor:
        futures = [
            executor.submit(
                translate_items,
                items=batch,
                base_url=base_url,
                api_key_env=args.api_key_env,
                model=args.model,
                max_tokens=args.max_tokens,
                timeout_seconds=args.timeout_seconds,
                retries=args.retries,
            )
            for batch in batches
        ]
        for future_idx, future in enumerate(as_completed(futures), 1):
            ok_rows, failures = future.result()
            ok_seen += len(ok_rows)
            failures_seen += len(failures)
            append_jsonl_locked(output_path, ok_rows, lock)
            append_jsonl_locked(failure_path, failures, lock)
            if future_idx % 25 == 0 or future_idx == len(futures):
                elapsed = max(1.0, time.time() - started)
                progress = {
                    "created_at": utc_now(),
                    "benchmark_id": args.benchmark,
                    "source_count": len(items),
                    "previously_done": len(done),
                    "pending_start": len(pending),
                    "batches_done": future_idx,
                    "batches_total": len(futures),
                    "ok_written_this_run": ok_seen,
                    "failures_this_run": failures_seen,
                    "rows_per_second_this_run": round((ok_seen + failures_seen) / elapsed, 4),
                }
                write_json(args.output_dir / "translation_progress.json", progress)
                print(json.dumps(progress, ensure_ascii=False), flush=True)
    rows = read_jsonl(output_path)
    write_json(args.output_dir / "inspection_examples.json", rows[: min(3, len(rows))])
    summary = summarize_output(
        benchmark=args.benchmark,
        record_path=record_path,
        output_dir=args.output_dir,
        source_count=len(items),
        model=args.model,
        base_url=base_url,
        concurrency=args.concurrency,
        batch_size=args.batch_size,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
