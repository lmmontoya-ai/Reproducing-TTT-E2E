#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def _run(cmd: list[str], *, check: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        check=check,
        text=True,
        capture_output=True,
    )


def _instance_record(instance_id: int) -> dict[str, Any]:
    proc = _run(["vastai", "show", "instance", str(instance_id), "--raw"], check=True)
    payload = json.loads(proc.stdout)
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected vastai payload for instance {instance_id}")
    return payload


def _direct_ssh_host_port(payload: dict[str, Any]) -> tuple[str, int]:
    host = str(payload.get("public_ipaddr", "")).strip()
    ports = payload.get("ports", {})
    if not isinstance(ports, dict):
        raise ValueError("Missing Vast ports mapping")
    ssh_entries = ports.get("22/tcp", [])
    if not isinstance(ssh_entries, list) or not ssh_entries:
        raise ValueError("Missing Vast direct SSH port mapping")
    port = int(ssh_entries[0]["HostPort"])
    if not host:
        raise ValueError("Missing Vast public_ipaddr")
    return host, port


def _ssh_cmd(*, ssh_key: Path, host: str, port: int, remote_cmd: str) -> list[str]:
    return [
        "ssh",
        "-o",
        "StrictHostKeyChecking=no",
        "-i",
        str(ssh_key),
        "-p",
        str(port),
        f"root@{host}",
        remote_cmd,
    ]


def _scp_cmd(*, ssh_key: Path, host: str, port: int, src: Path, dst: str) -> list[str]:
    return [
        "scp",
        "-P",
        str(port),
        "-i",
        str(ssh_key),
        "-o",
        "StrictHostKeyChecking=no",
        str(src),
        f"root@{host}:{dst}",
    ]


def _ensure_remote_sync_script(
    *,
    ssh_key: Path,
    host: str,
    port: int,
    local_repo_root: Path,
    remote_repo_root: Path,
) -> None:
    local_sync = (local_repo_root / "scripts" / "70_sync_760m_b2.py").resolve()
    remote_sync = remote_repo_root / "scripts" / "70_sync_760m_b2.py"
    proc = _run(
        _ssh_cmd(
            ssh_key=ssh_key,
            host=host,
            port=port,
            remote_cmd=f"test -f {shlex.quote(str(remote_sync))}",
        )
    )
    if proc.returncode == 0:
        return
    _run(
        _scp_cmd(
            ssh_key=ssh_key,
            host=host,
            port=port,
            src=local_sync,
            dst=str(remote_sync),
        ),
        check=True,
    )


def _remote_file_json(
    *,
    ssh_key: Path,
    host: str,
    port: int,
    path: Path,
) -> dict[str, Any] | None:
    cmd = (
        "python3 - <<'PY'\n"
        "import json, pathlib\n"
        f"path = pathlib.Path({path!r})\n"
        "if not path.exists():\n"
        "    print('')\n"
        "else:\n"
        "    print(path.read_text())\n"
        "PY"
    )
    proc = _run(_ssh_cmd(ssh_key=ssh_key, host=host, port=port, remote_cmd=cmd))
    if proc.returncode != 0:
        return None
    stdout = proc.stdout.strip()
    if not stdout:
        return None
    payload = json.loads(stdout)
    if not isinstance(payload, dict):
        return None
    return payload


def _remote_pgrep(
    *,
    ssh_key: Path,
    host: str,
    port: int,
    pattern: str,
) -> bool:
    proc = _run(
        _ssh_cmd(
            ssh_key=ssh_key,
            host=host,
            port=port,
            remote_cmd=f"pgrep -af {shlex.quote(pattern)} >/dev/null",
        )
    )
    return proc.returncode == 0


def _summary_succeeded(payload: dict[str, Any]) -> bool:
    rows = payload.get("rows", [])
    if not isinstance(rows, list) or not rows:
        return False

    train_rows = [row for row in rows if str(row.get("step_id", "")).startswith("train:")]
    if not train_rows:
        return False
    if any(str(row.get("status", "")) != "succeeded" for row in train_rows):
        return False

    eval_rows = [row for row in rows if str(row.get("step_id", "")) == "eval:books32k"]
    if not eval_rows:
        return False
    for row in eval_rows:
        try:
            rc = int(row.get("returncode", 1))
        except (TypeError, ValueError):
            return False
        if rc != 0:
            return False

    return True


def _remote_sync_cmd(
    *,
    payload: dict[str, Any],
    remote_repo_root: Path,
    paper_run_id: str,
    exp_folder: str,
) -> str:
    extra_env = payload.get("extra_env", {})
    if not isinstance(extra_env, dict):
        raise ValueError("Missing extra_env credentials on Vast instance")
    exports = []
    for name in (
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_DEFAULT_REGION",
        "B2_BUCKET",
        "B2_ENDPOINT_URL",
    ):
        value = str(extra_env.get(name, "")).strip()
        if not value:
            raise ValueError(f"Missing {name} in Vast instance environment")
        exports.append(f"export {name}={shlex.quote(value)}")
    exports_blob = "; ".join(exports)
    sync_script = remote_repo_root / "scripts" / "70_sync_760m_b2.py"
    return (
        f"{exports_blob}; "
        f"pkill -f {shlex.quote('c1_b2_sync_loop.sh')} || true; "
        f"pkill -f {shlex.quote('70_sync_760m_b2.py')} || true; "
        f"python3 {shlex.quote(str(sync_script))} "
        f"--paper-run-id {shlex.quote(paper_run_id)} "
        f"--exp-folder {shlex.quote(exp_folder)} "
        f"--checkpoint-root {shlex.quote(str(remote_repo_root / 'checkpoints'))} "
        f"--exp-dir {shlex.quote(str(remote_repo_root / 'experiments'))} "
        f"--reports-root {shlex.quote(str(remote_repo_root / 'reports' / 'paper'))} "
        f"--b2-bucket {shlex.quote(str(extra_env['B2_BUCKET']))} "
        f"--endpoint-url {shlex.quote(str(extra_env['B2_ENDPOINT_URL']))} "
        f"--region {shlex.quote(str(extra_env['AWS_DEFAULT_REGION']))} "
        f"--run-log /workspace/c1_gate.log "
        f"--run-log /workspace/c1_ladder.log "
        f"--run-log /workspace/c1_b2_sync.log"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Watch the live 760M C1 ladder, force one final B2 sync after successful "
            "completion, then stop the Vast H200 instance to end GPU billing."
        )
    )
    parser.add_argument("--instance-id", type=int, required=True)
    parser.add_argument("--paper-run-id", default="protocol_r_760m_author_seed_v1")
    parser.add_argument("--exp-folder", default="protocol_r_760m_author_seed_v1")
    parser.add_argument("--remote-repo-root", type=Path, default=Path("/workspace/Reproducing-TTT-E2E"))
    parser.add_argument("--local-repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--ssh-key", type=Path, default=Path("~/.ssh/runpod_ed25519"))
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--action", choices=("stop", "destroy"), default="stop")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ssh_key = args.ssh_key.expanduser().resolve()
    remote_summary = (
        args.remote_repo_root
        / "reports"
        / "paper"
        / args.paper_run_id
        / "launch"
        / "launcher_summary.json"
    )

    while True:
        payload = _instance_record(args.instance_id)
        actual_status = str(payload.get("actual_status", "")).strip().lower()
        if actual_status not in {"running", "loading"}:
            print(f"Instance {args.instance_id} no longer active: {actual_status}", flush=True)
            return 0

        host, port = _direct_ssh_host_port(payload)
        _ensure_remote_sync_script(
            ssh_key=ssh_key,
            host=host,
            port=port,
            local_repo_root=args.local_repo_root.expanduser().resolve(),
            remote_repo_root=args.remote_repo_root,
        )

        summary = _remote_file_json(
            ssh_key=ssh_key,
            host=host,
            port=port,
            path=remote_summary,
        )

        if summary and _summary_succeeded(summary):
            print("C1 summary indicates success. Running final B2 sync.", flush=True)
            sync_cmd = _remote_sync_cmd(
                payload=payload,
                remote_repo_root=args.remote_repo_root,
                paper_run_id=args.paper_run_id,
                exp_folder=args.exp_folder,
            )
            sync_proc = _run(_ssh_cmd(ssh_key=ssh_key, host=host, port=port, remote_cmd=sync_cmd))
            if sync_proc.returncode != 0:
                print(sync_proc.stdout, end="", file=sys.stdout)
                print(sync_proc.stderr, end="", file=sys.stderr)
                print("Final B2 sync failed; instance will not be stopped.", file=sys.stderr)
                return sync_proc.returncode

            action_cmd = ["vastai", f"{args.action}", "instance", str(args.instance_id)]
            if args.action == "destroy":
                action_cmd = ["vastai", "destroy", "instance", str(args.instance_id)]
            else:
                action_cmd = ["vastai", "stop", "instance", str(args.instance_id)]
            proc = _run(action_cmd)
            print(proc.stdout, end="", file=sys.stdout)
            print(proc.stderr, end="", file=sys.stderr)
            return proc.returncode

        if not _remote_pgrep(
            ssh_key=ssh_key,
            host=host,
            port=port,
            pattern="66_run_760m_author_seed_ladder.py",
        ):
            print("Ladder process ended without a successful summary yet; waiting for summary/update.", flush=True)

        time.sleep(max(int(args.poll_seconds), 30))


if __name__ == "__main__":
    raise SystemExit(main())
