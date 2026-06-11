#!/usr/bin/env python3
"""Create and validate K-MMAD smoke run records."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from kmmad_common import configured_path, find_sensitive_strings, load_config, sanitize_jsonable, utc_now, write_json

REQUIRED_FIELDS = [
    "schema_version",
    "run_id",
    "git_sha",
    "commands",
    "dataset",
    "artifacts",
    "reservation",
    "image_runtime",
    "model",
    "status",
]


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "UNKNOWN"


def default_record(config: dict[str, Any], run_id: str) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "run_id": run_id,
        "created_at": utc_now(),
        "status": "initialized",
        "git_sha": git_sha(),
        "cluster_git_sha": None,
        "commands": [],
        "dataset": {
            "root": config["paths"]["dataset_root"],
            "source": config["source"],
            "general_vqa": {
                "benchmarks": config.get("general_vqa", {}).get("benchmarks", []),
                "sample_size_per_benchmark": None,
                "sample_root": config["paths"].get("general_vqa_original_subdir"),
                "full_download_required_for_this_smoke": config.get("general_vqa", {}).get("first_pass_full_download_required", False),
                "first_pass_full_download_required": config.get("general_vqa", {}).get("first_pass_full_download_required", False),
                "mock_is_completion_evidence": config.get("general_vqa", {}).get("mock_is_completion_evidence", False),
                "requires_local_gpu_llm_evidence": config.get("general_vqa", {}).get("requires_local_gpu_llm_evidence", True),
                "requires_openai_oauth_evidence": config.get("general_vqa", {}).get("requires_openai_oauth_evidence", True),
            },
            "download_manifest": None,
            "sanity_report": None,
        },
        "reservation": {
            "project": config["mlxp"]["project"],
            "member": config["mlxp"]["member"],
            "purpose_prefix": config["mlxp"]["purpose_prefix"],
            "payload": None,
            "id": None,
            "cancelled_at": None,
            "active_reason": None,
        },
        "image_runtime": {
            "image_path": None,
            "image_digest": None,
            "managed_image_key": None,
            "command": None,
            "health_check": None,
        },
        "translation": {
            "mode": "mmad_or_general_vqa",
            "benchmarks": config.get("general_vqa", {}).get("benchmarks", []),
            "mock_sanity_only": config.get("general_vqa", {}).get("mock_is_completion_evidence", False) is False,
            "local_gpu_llm_evidence": False,
            "openai_oauth_evidence": False,
            "api_key_contract_only": True,
            "parallelism": config.get("general_vqa", {}).get("parallelism", {}),
        },
        "model": {
            "identifier": config["inference"]["model"],
            "revision": None,
            "endpoint": config["inference"].get("openai_compatible_base_url"),
            "endpoint_provider": config["inference"].get("endpoint_provider", "openai_compatible"),
            "auth_mode": config["inference"].get("auth_mode", "none"),
            "api_key_env": config["inference"].get("api_key_env", "OPENAI_API_KEY"),
            "auth_secret_present": None,
            "local_gpu_llm": {
                "identifier": None,
                "revision": None,
                "endpoint": None,
                "endpoint_provider": "local_openai_compatible",
                "auth_mode": "none",
            },
            "openai_oauth": {
                "identifier": None,
                "endpoint": None,
                "endpoint_provider": "openai_oauth",
                "auth_mode": "none",
                "models_seen": [],
            },
        },
        "artifacts": {
            "translation_output": None,
            "untranslated_field_report": None,
            "validation_report": None,
            "inspection_examples": None,
            "checksums": {},
            "logs": [],
        },
        "failures_retries": [],
        "notes": [],
    }


def require_path(errors: list[str], obj: dict[str, Any], path: str) -> Any:
    current: Any = obj
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            errors.append(f"missing required field: {path}")
            return None
        current = current[part]
    return current


def validate_general_vqa_schema(record: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if record.get("schema_version") != 2:
        errors.append("schema_version must be 2")
        return errors

    for path in (
        "dataset.general_vqa.benchmarks",
        "dataset.general_vqa.first_pass_full_download_required",
        "dataset.general_vqa.mock_is_completion_evidence",
        "dataset.general_vqa.requires_local_gpu_llm_evidence",
        "dataset.general_vqa.requires_openai_oauth_evidence",
        "translation.mode",
        "translation.benchmarks",
        "translation.mock_sanity_only",
        "translation.local_gpu_llm_evidence",
        "translation.openai_oauth_evidence",
        "translation.api_key_contract_only",
        "translation.parallelism",
    ):
        require_path(errors, record, path)

    if record.get("status") != "passed" or require_path(errors, record, "translation.mode") != "general_vqa_dual_path_smoke":
        return errors

    expected = {
        "translation.mock_sanity_only": False,
        "translation.local_gpu_llm_evidence": True,
        "translation.openai_oauth_evidence": True,
        "translation.api_key_contract_only": True,
    }
    for path, expected_value in expected.items():
        value = require_path(errors, record, path)
        if value is not None and value is not expected_value:
            errors.append(f"{path} must be {expected_value!r} for passed general VQA dual-path records")

    for path in (
        "artifacts.translation_output.local_gpu_llm",
        "artifacts.translation_output.openai_oauth",
        "artifacts.checksums.local_gpu_llm",
        "artifacts.checksums.openai_oauth",
        "model.local_gpu_llm.identifier",
        "model.local_gpu_llm.endpoint_provider",
        "model.openai_oauth.identifier",
        "model.openai_oauth.endpoint_provider",
        "image_runtime.image_digest",
        "reservation.id",
        "reservation.cancelled_at",
    ):
        value = require_path(errors, record, path)
        if value in (None, "", [], {}):
            errors.append(f"{path} must be populated for passed general VQA dual-path records")
    return errors


def validate(record: dict[str, Any]) -> list[str]:
    errors = [f"missing required field: {field}" for field in REQUIRED_FIELDS if field not in record]
    if not isinstance(record.get("commands"), list):
        errors.append("commands must be a list")
    for section in ("dataset", "reservation", "image_runtime", "model", "artifacts"):
        if not isinstance(record.get(section), dict):
            errors.append(f"{section} must be an object")
    sensitive_paths = find_sensitive_strings(record)
    if sensitive_paths:
        errors.append("record contains unredacted secret-like values at: " + ", ".join(sensitive_paths[:10]))
    if "schema_version" in record:
        errors.extend(validate_general_vqa_schema(record))
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    init = sub.add_parser("init")
    init.add_argument("--config", default="configs/kmmad_translation_smoke.toml")
    init.add_argument("--run-id", default=None)
    init.add_argument("--out", type=Path, default=None)
    val = sub.add_parser("validate")
    val.add_argument("record", type=Path)
    args = parser.parse_args(argv)

    if args.cmd == "init":
        config = load_config(args.config)
        run_id = args.run_id or f"kmmad-smoke-{utc_now()}-{git_sha()[:8]}"
        out = args.out or configured_path(config, "run_subdir") / run_id / "run_record.json"
        record = sanitize_jsonable(default_record(config, run_id))
        write_json(out, record)
        print(f"run record initialized: {out}")
        return 0

    record = json.loads(args.record.read_text(encoding="utf-8"))
    errors = validate(record)
    result = {"status": "passed" if not errors else "failed", "errors": errors}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
