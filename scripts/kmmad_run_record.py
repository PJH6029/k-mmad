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
        "run_id": run_id,
        "created_at": utc_now(),
        "status": "initialized",
        "git_sha": git_sha(),
        "cluster_git_sha": None,
        "commands": [],
        "dataset": {
            "root": config["paths"]["dataset_root"],
            "source": config["source"],
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
        "model": {
            "identifier": config["inference"]["model"],
            "revision": None,
            "endpoint": config["inference"].get("openai_compatible_base_url"),
            "endpoint_provider": config["inference"].get("endpoint_provider", "openai_compatible"),
            "auth_mode": config["inference"].get("auth_mode", "none"),
            "api_key_env": config["inference"].get("api_key_env", "OPENAI_API_KEY"),
            "auth_secret_present": None,
        },
        "artifacts": {
            "translation_output": None,
            "untranslated_field_report": None,
            "validation_report": None,
            "inspection_examples": None,
            "logs": [],
        },
        "failures_retries": [],
        "notes": [],
    }


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
