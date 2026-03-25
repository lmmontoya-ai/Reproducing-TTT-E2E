from __future__ import annotations

import argparse
import importlib.util
import tempfile
import unittest
from pathlib import Path
import sys


def _load_sync_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "70_sync_760m_b2.py"
    spec = importlib.util.spec_from_file_location("sync_760m_b2", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Sync760MB2Test(unittest.TestCase):
    def setUp(self) -> None:
        self.module = _load_sync_module()

    def test_build_operations_includes_existing_roots_and_run_logs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            checkpoint_root = root / "checkpoints"
            exp_dir = root / "experiments"
            reports_root = root / "reports" / "paper"
            (checkpoint_root / "paper_run").mkdir(parents=True)
            (exp_dir / "paper_run").mkdir(parents=True)
            (reports_root / "paper_run").mkdir(parents=True)
            run_log = root / "c1.log"
            run_log.write_text("ok\n", encoding="utf-8")

            args = argparse.Namespace(
                paper_run_id="paper_run",
                exp_folder="paper_run",
                checkpoint_root=checkpoint_root,
                exp_dir=exp_dir,
                reports_root=reports_root,
                run_log=[str(run_log)],
                b2_bucket="bucket",
                endpoint_url="https://example.invalid",
                region="us-east-005",
                b2_prefix="ttt-e2e-artifacts",
            )

            operations = self.module._build_operations(args)
            self.assertEqual(len(operations), 4)
            self.assertEqual(operations[0].kind, "sync")
            self.assertEqual(operations[0].src, (checkpoint_root / "paper_run").resolve())
            self.assertIn("*.orbax-checkpoint-tmp-*", operations[0].excludes)
            self.assertEqual(operations[-1].kind, "cp")
            self.assertTrue(operations[-1].dst.endswith("/run_logs/paper_run/c1.log"))

    def test_command_for_sync_operation_includes_tmp_excludes(self) -> None:
        op = self.module.SyncOperation(
            kind="sync",
            src=Path("/tmp/source"),
            dst="s3://bucket/prefix",
            excludes=("*.orbax-checkpoint-tmp-*", "*.orbax-checkpoint-tmp-*/*"),
        )
        cmd = self.module._command_for_operation(
            op,
            endpoint_url="https://example.invalid",
            region="us-east-005",
        )
        self.assertEqual(cmd[:4], ["aws", "s3", "sync", "/tmp/source"])
        self.assertIn("--exclude", cmd)
        self.assertIn("*.orbax-checkpoint-tmp-*", cmd)


if __name__ == "__main__":
    unittest.main()
