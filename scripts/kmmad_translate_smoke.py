#!/usr/bin/env python3
"""Run or validate a tiny Korean translation smoke over MMAD QA text + captions."""

from __future__ import annotations

import argparse
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from kmmad_common import (
    CAPTION_KEYS,
    OPTION_KEYS,
    QUESTION_KEYS,
    configured_path,
    discover_text_fields,
    flatten_text,
    get_first,
    load_config,
    load_first_records,
    pick_sample,
    sanitize_jsonable,
    sanitize_text,
    utc_now,
    write_json,
)

KOREAN_RE = re.compile(r"[가-힣]")
CJK_RE = re.compile(r"[\u4e00-\u9fff]")
ASSISTANT_ARTIFACT_RE = re.compile(r"\bassistant\b", re.IGNORECASE)


def mock_translate(text: str) -> str:
    return f"한국어 번역 초안: {text}"


def normalized_auth_mode(inference: dict[str, Any]) -> str:
    mode = str(inference.get("auth_mode", "none")).strip().lower().replace("-", "_")
    aliases = {
        "": "none",
        "no_auth": "none",
        "openai_oauth": "none",
        "oauth_proxy": "none",
        "api_key": "bearer_env",
        "openai_api_key": "bearer_env",
        "bearer": "bearer_env",
    }
    return aliases.get(mode, mode)


def build_request_headers(config: dict[str, Any]) -> dict[str, str]:
    inference = config["inference"]
    headers = {"Content-Type": "application/json"}
    mode = normalized_auth_mode(inference)
    if mode == "none":
        return headers
    if mode != "bearer_env":
        raise RuntimeError(f"unsupported inference auth_mode: {mode}")
    env_name = str(inference.get("api_key_env", "OPENAI_API_KEY"))
    token = os.environ.get(env_name)
    if not token:
        raise RuntimeError(f"inference auth_mode bearer_env requires environment variable: {env_name}")
    headers["Authorization"] = f"Bearer {token}"
    return headers


def sanitized_headers(headers: dict[str, str]) -> dict[str, str]:
    sanitized: dict[str, str] = {}
    for key, value in headers.items():
        if key.lower() == "authorization" and value.lower().startswith("bearer "):
            sanitized[key] = "Bearer <redacted>"
        else:
            sanitized[key] = sanitize_text(value)
    return sanitized


def sanitized_inference_summary(config: dict[str, Any]) -> dict[str, Any]:
    inference = config["inference"]
    return sanitize_jsonable({
        "endpoint_provider": inference.get("endpoint_provider", "openai_compatible"),
        "base_url": inference.get("openai_compatible_base_url"),
        "model": inference.get("model"),
        "auth_mode": normalized_auth_mode(inference),
        "api_key_env": inference.get("api_key_env", "OPENAI_API_KEY"),
    })


def call_openai_compatible(config: dict[str, Any], text: str) -> str:
    inference = config["inference"]
    url = inference["openai_compatible_base_url"].rstrip("/") + "/chat/completions"
    payload = {
        "model": inference["model"],
        "temperature": inference.get("temperature", 0.0),
        "max_tokens": inference.get("max_tokens", 1024),
        "messages": [
            {
                "role": "system",
                "content": (
                    "Translate the user's industrial anomaly-detection benchmark text into natural Korean only. "
                    "Return only the translated text, without notes or explanations. "
                    "Preserve option letters, labels, numbers, and line breaks. "
                    "Do not use Chinese/Hanja characters. "
                    "Use this glossary when relevant: defect=결함, appearance=외관, speckles=반점, "
                    "smudging=번짐, chip=깨짐, crack=균열, scratch=긁힘, deformation=변형, "
                    "discoloration=변색, pill=알약, edge=가장자리, whiteboard eraser=화이트보드 지우개."
                ),
            },
            {"role": "user", "content": text},
        ],
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=build_request_headers(config), method="POST")
    try:
        with urllib.request.urlopen(req, timeout=int(inference.get("timeout_seconds", 120))) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(f"translation endpoint call failed: {exc}") from exc
    try:
        return body["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"unexpected translation response: {body}") from exc


def translate_value(value: Any, config: dict[str, Any], mock: bool) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        if not value.strip():
            return value
        return mock_translate(value) if mock else call_openai_compatible(config, value)
    if isinstance(value, list):
        return [translate_value(item, config, mock) for item in value]
    if isinstance(value, dict):
        return {key: translate_value(item, config, mock) for key, item in value.items()}
    return value


def build_translation_row(row: dict[str, Any], config: dict[str, Any], mock: bool) -> tuple[dict[str, Any], list[str], list[str]]:
    q_key, q = get_first(row, QUESTION_KEYS)
    o_key, o = get_first(row, OPTION_KEYS)
    c_key, c = get_first(row, CAPTION_KEYS)
    missing_scope: list[str] = []
    translated: dict[str, Any] = {
        "source": row,
        "translated": {},
        "translation_scope": ["question", "options", "caption"],
    }
    if q_key:
        translated["translated"][q_key] = translate_value(q, config, mock)
    else:
        missing_scope.append("question")
    if o_key:
        translated["translated"][o_key] = translate_value(o, config, mock)
    else:
        missing_scope.append("options")
    if c_key:
        translated["translated"][c_key] = translate_value(c, config, mock)
    else:
        missing_scope.append("caption")
    all_text = set(discover_text_fields(row))
    translated_keys = {key for key in (q_key, o_key, c_key) if key}
    untranslated = sorted(all_text - translated_keys)
    if missing_scope:
        translated["missing_translation_scope"] = missing_scope
    return translated, untranslated, missing_scope


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def has_korean(value: Any) -> bool:
    return any(KOREAN_RE.search(text) for text in flatten_text(value))


def validate_translated_value(row_idx: int, key: str, value: Any, errors: list[str]) -> None:
    texts = flatten_text(value)
    if not texts:
        errors.append(f"row {row_idx} translated field {key!r} is empty")
        return
    for text_idx, text in enumerate(texts):
        if not KOREAN_RE.search(text):
            errors.append(f"row {row_idx} translated field {key!r} item {text_idx} has no Korean text")
        if CJK_RE.search(text):
            errors.append(f"row {row_idx} translated field {key!r} item {text_idx} contains CJK/Hanja characters")
        if ASSISTANT_ARTIFACT_RE.search(text):
            errors.append(f"row {row_idx} translated field {key!r} item {text_idx} contains assistant artifact text")


def validate_output(output: Path, report: Path | None = None) -> dict[str, Any]:
    rows = read_jsonl(output)
    errors: list[str] = []
    if not rows:
        errors.append("translation output has no rows")
    for idx, row in enumerate(rows):
        translated = row.get("translated")
        if not isinstance(translated, dict) or not translated:
            errors.append(f"row {idx} missing translated object")
            continue
        missing_scope = row.get("missing_translation_scope", [])
        if isinstance(missing_scope, list):
            for required in ("question", "options"):
                if required in missing_scope:
                    errors.append(f"row {idx} missing required translation scope: {required}")
        for key, value in translated.items():
            validate_translated_value(idx, str(key), value, errors)
        if "source" not in row:
            errors.append(f"row {idx} missing source field")
    if report is not None and not report.exists():
        errors.append(f"untranslated-field report missing: {report}")
    return {"status": "passed" if not errors else "failed", "rows": len(rows), "errors": errors}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/kmmad_translation_smoke.toml")
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--mock", action="store_true", help="Use deterministic local mock translations")
    parser.add_argument("--validate-only", type=Path, default=None, help="Validate an existing JSONL output")
    parser.add_argument("--untranslated-report", type=Path, default=None)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    if args.validate_only:
        result = validate_output(args.validate_only, args.untranslated_report)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["status"] == "passed" else 1

    dataset = args.dataset or configured_path(config, "original_subdir")
    sample_size = args.sample_size or int(config["smoke"].get("sample_size", 8))
    output_dir = args.output_dir or configured_path(config, "translated_subdir") / utc_now()
    output_dir.mkdir(parents=True, exist_ok=True)

    record_path, records = load_first_records(dataset)
    if not records:
        raise SystemExit(f"no parseable MMAD records found under {dataset}")
    sample = pick_sample(records, sample_size, int(config["smoke"].get("sample_seed", 6029)))
    output_path = output_dir / "translation_smoke.jsonl"
    untranslated_report_path = output_dir / "untranslated_fields.json"
    inspection_path = output_dir / "inspection_examples.json"

    all_untranslated: dict[str, int] = {}
    missing_configured: dict[str, int] = {}
    with output_path.open("w", encoding="utf-8") as fh:
        for row in sample:
            translated, untranslated, missing_scope = build_translation_row(row, config, args.mock)
            for field in untranslated:
                all_untranslated[field] = all_untranslated.get(field, 0) + 1
            for field in missing_scope:
                missing_configured[field] = missing_configured.get(field, 0) + 1
            fh.write(json.dumps(translated, ensure_ascii=False) + "\n")

    write_json(untranslated_report_path, {
        "created_at": utc_now(),
        "record_file": str(record_path),
        "sample_size": len(sample),
        "translated_scope": ["question", "options", "caption"],
        "untranslated_text_fields": all_untranslated,
        "missing_configured_translation_fields": missing_configured,
    })
    rows = read_jsonl(output_path)
    write_json(inspection_path, rows[: min(3, len(rows))])
    result = validate_output(output_path, untranslated_report_path)
    write_json(output_dir / "translation_validation.json", result)
    write_json(output_dir / "translation_smoke_summary.json", {
        "created_at": utc_now(),
        "record_file": str(record_path),
        "sample_size": len(sample),
        "inference": sanitized_inference_summary(config),
        "validation": result,
        "artifacts": {
            "translation_output": str(output_path),
            "untranslated_field_report": str(untranslated_report_path),
            "inspection_examples": str(inspection_path),
            "validation_report": str(output_dir / "translation_validation.json"),
        },
    })
    print(f"translation smoke output: {output_path}")
    print(f"untranslated-field report: {untranslated_report_path}")
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
