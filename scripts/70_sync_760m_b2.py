#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


CHECKPOINT_TMP_EXCLUDES = (
    "*.orbax-checkpoint-tmp-*",
    "*.orbax-checkpoint-tmp-*/*",
    ".DS_Store",
)


@dataclass(frozen=True)
class SyncOperation:
    kind: str
    src: Path
    dst: str
    excludes: tuple[str, ...] = ()


def _env_default(name: str, fallback: str = "") -> str:
    value = str(os.environ.get(name, "")).strip()
    return value or fallback


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sync the 760M paper run checkpoints and artifacts to Backblaze B2 "
            "without uploading Orbax temporary checkpoint directories."
        )
    )
    parser.add_argument("--paper-run-id", required=True)
    parser.add_argument("--exp-folder", required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--exp-dir", type=Path, required=True)
    parser.add_argument("--reports-root", type=Path, required=True)
    parser.add_argument(
        "--run-log",
        action="append",
        default=[],
        help="Optional run log file to upload under run_logs/<paper_run_id>/",
    )
    parser.add_argument("--b2-bucket", default=_env_default("B2_BUCKET"))
    parser.add_argument("--endpoint-url", default=_env_default("B2_ENDPOINT_URL"))
    parser.add_argument("--region", default=_env_default("AWS_DEFAULT_REGION", "us-east-005"))
    parser.add_argument("--b2-prefix", default="ttt-e2e-artifacts")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _build_operations(args: argparse.Namespace) -> list[SyncOperation]:
    checkpoint_root = args.checkpoint_root.expanduser().resolve() / args.exp_folder
    experiments_root = args.exp_dir.expanduser().resolve() / args.paper_run_id
    reports_root = args.reports_root.expanduser().resolve() / args.paper_run_id

    prefix = str(args.b2_prefix).strip().strip("/")
    bucket = str(args.b2_bucket).strip()
    if not bucket:
        raise ValueError("Missing Backblaze bucket. Pass --b2-bucket or set B2_BUCKET.")
    if not str(args.endpoint_url).strip():
        raise ValueError("Missing Backblaze endpoint URL. Pass --endpoint-url or set B2_ENDPOINT_URL.")

    remote_root = f"s3://{bucket}/{prefix}"
    operations: list[SyncOperation] = []

    if checkpoint_root.exists():
        operations.append(
            SyncOperation(
                kind="sync",
                src=checkpoint_root,
                dst=f"{remote_root}/checkpoints/{args.paper_run_id}",
                excludes=CHECKPOINT_TMP_EXCLUDES,
            )
        )
    if experiments_root.exists():
        operations.append(
            SyncOperation(
                kind="sync",
                src=experiments_root,
                dst=f"{remote_root}/experiments/{args.paper_run_id}",
                excludes=(".DS_Store",),
            )
        )
    if reports_root.exists():
        operations.append(
            SyncOperation(
                kind="sync",
                src=reports_root,
                dst=f"{remote_root}/reports/paper/{args.paper_run_id}",
                excludes=(".DS_Store",),
            )
        )

    for raw_path in args.run_log:
        log_path = Path(raw_path).expanduser().resolve()
        if not log_path.exists():
            continue
        operations.append(
            SyncOperation(
                kind="cp",
                src=log_path,
                dst=f"{remote_root}/run_logs/{args.paper_run_id}/{log_path.name}",
            )
        )

    return operations


def _command_for_operation(op: SyncOperation, *, endpoint_url: str, region: str) -> list[str]:
    if op.kind == "sync":
        cmd = [
            "aws",
            "s3",
            "sync",
            str(op.src),
            op.dst,
            "--endpoint-url",
            endpoint_url,
            "--region",
            region,
            "--only-show-errors",
        ]
        for pattern in op.excludes:
            cmd.extend(["--exclude", pattern])
        return cmd

    if op.kind == "cp":
        return [
            "aws",
            "s3",
            "cp",
            str(op.src),
            op.dst,
            "--endpoint-url",
            endpoint_url,
            "--region",
            region,
            "--only-show-errors",
        ]

    raise ValueError(f"Unsupported sync operation kind: {op.kind}")


def _run(cmd: Sequence[str], *, dry_run: bool) -> int:
    print("$ " + " ".join(shlex.quote(part) for part in cmd), flush=True)
    if dry_run:
        return 0
    return subprocess.run(list(cmd), check=False).returncode


def main() -> int:
    args = parse_args()
    operations = _build_operations(args)
    if not operations:
        print("No existing 760M artifacts found to sync.", flush=True)
        return 0

    rc = 0
    for op in operations:
        cmd = _command_for_operation(
            op,
            endpoint_url=str(args.endpoint_url),
            region=str(args.region),
        )
        op_rc = _run(cmd, dry_run=args.dry_run)
        if op_rc != 0:
            rc = op_rc
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
