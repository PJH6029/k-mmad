#!/usr/bin/env python3
"""Plan or execute full MMAD dataset acquisition.

Default mode is dry-run planning. Execution is intentionally explicit via
`--execute` and is designed to be non-destructive: it refuses to write into a
non-empty target unless `--allow-existing` is provided.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from urllib.parse import quote

from kmmad_common import command_exists, configured_path, load_config, utc_now, write_json


def archive_url(hf_url: str, filename: str) -> str:
    return f"{hf_url.rstrip('/')}/resolve/main/{quote(filename)}?download=true"


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def build_plan(config: dict, method: str, target: Path) -> dict:
    source = config["source"]
    archive_files = source.get("archive_files", [])
    commands: list[list[str]] = []
    notes: list[str] = []

    if method == "git_lfs_clone":
        commands = [
            ["git", "lfs", "install"],
            ["git", "clone", source["hf_url"], str(target)],
        ]
        notes.append("Requires git-lfs and may use Hugging Face Xet/LFS storage.")
    elif method == "huggingface_cli_download":
        commands = [["huggingface-cli", "download", source["hf_dataset"], "--repo-type", "dataset", "--local-dir", str(target)]]
        notes.append("Requires huggingface-cli to be installed in the environment.")
    elif method == "archive_urls":
        commands = [
            ["curl", "-L", "--fail", "--continue-at", "-", "-o", str(target / name), archive_url(source["hf_url"], name)]
            for name in archive_files
        ]
        notes.append("Downloads the current separate Hugging Face files; does not rely on the old ALL_DATA.zip path.")
    else:
        raise SystemExit(f"unknown method: {method}")

    return {
        "created_at": utc_now(),
        "status": "planned",
        "method": method,
        "target": str(target),
        "source": {
            "hf_dataset": source.get("hf_dataset"),
            "hf_url": source.get("hf_url"),
            "github_url": source.get("github_url"),
            "arxiv_url": source.get("arxiv_url"),
            "expected_rows": source.get("expected_rows"),
            "expected_images": source.get("expected_images"),
            "expected_total_size_gb": source.get("expected_total_size_gb"),
        },
        "archive_files": archive_files,
        "commands": commands,
        "notes": notes,
    }


def assert_can_execute(method: str) -> None:
    required = {
        "git_lfs_clone": ["git"],
        "huggingface_cli_download": ["huggingface-cli"],
        "archive_urls": ["curl"],
    }[method]
    missing = [cmd for cmd in required if not command_exists(cmd)]
    if missing:
        raise SystemExit(f"missing required command(s) for {method}: {', '.join(missing)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/kmmad_translation_smoke.toml")
    parser.add_argument("--method", choices=["git_lfs_clone", "huggingface_cli_download", "archive_urls"], default="git_lfs_clone")
    parser.add_argument("--target", type=Path, default=None, help="Override original/MMAD target directory")
    parser.add_argument("--manifest-out", type=Path, default=None)
    parser.add_argument("--execute", action="store_true", help="Actually run download commands")
    parser.add_argument("--allow-existing", action="store_true", help="Allow non-empty target directory")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    target = args.target or configured_path(config, "original_subdir")
    manifest_dir = configured_path(config, "manifest_subdir")
    manifest_out = args.manifest_out or manifest_dir / f"download-plan-{utc_now()}.json"
    plan = build_plan(config, args.method, target)

    if not args.execute:
        write_json(manifest_out, plan)
        print(f"dry-run download plan written: {manifest_out}")
        return 0

    if target.exists() and any(target.iterdir()) and not args.allow_existing:
        raise SystemExit(f"target is non-empty; refusing non-destructive write without --allow-existing: {target}")

    assert_can_execute(args.method)
    target.mkdir(parents=True, exist_ok=True)
    plan["status"] = "running"
    write_json(manifest_out, plan)
    try:
        if args.method == "git_lfs_clone":
            for cmd in plan["commands"]:
                run(cmd)
        elif args.method == "huggingface_cli_download":
            run(plan["commands"][0])
        else:
            for cmd in plan["commands"]:
                run(cmd)
        plan["status"] = "complete"
    except subprocess.CalledProcessError as exc:
        plan["status"] = "failed"
        plan["error"] = {"returncode": exc.returncode, "cmd": exc.cmd}
        write_json(manifest_out, plan)
        raise
    write_json(manifest_out, plan)
    print(f"download manifest written: {manifest_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
