from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
import json
import argparse
import sys

from ttt.research.orchestrator import OrchestratorOptions


def _load_launcher_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "66_run_760m_author_seed_ladder.py"
    spec = importlib.util.spec_from_file_location("run_760m_author_seed_ladder", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class LadderResumePlanTest(unittest.TestCase):
    def setUp(self) -> None:
        self.module = _load_launcher_module()

    def test_author_seed_stage_prefers_existing_stage_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            checkpoint_root = root / "checkpoints"
            artifact_root = root / "artifacts"
            self._materialize_author_seed(artifact_root, "760m_fa")
            stage_root = checkpoint_root / "protocol_r_760m_author_seed_v1" / "adapt-760m-e2e-8K-from-fa"
            stage_root.mkdir(parents=True)
            (stage_root / "latest.json").write_text('{"step": 600, "path": "600"}\n', encoding="utf-8")

            plan = self.module._resolve_resume_plan(
                stage_id="S2_ADAPT",
                run_id="adapt-760m-e2e-8K-from-fa",
                checkpoint_root=checkpoint_root,
                exp_folder="protocol_r_760m_author_seed_v1",
                artifact_root=artifact_root,
            )

            self.assertEqual(plan["resume_source"], "stage_checkpoint")
            self.assertEqual(plan["seed_source"], "760m_fa")
            self.assertEqual(
                plan["explicit_resume_checkpoint_path"],
                stage_root,
            )
            self.assertIn("training.load_part=all", plan["extra_overrides"])

    def test_author_seed_stage_falls_back_to_seed_when_no_stage_checkpoint_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            checkpoint_root = root / "checkpoints"
            artifact_root = root / "artifacts"
            self._materialize_author_seed(artifact_root, "760m_e2e")

            plan = self.module._resolve_resume_plan(
                stage_id="S3",
                run_id="ext-760m-e2e-32K",
                checkpoint_root=checkpoint_root,
                exp_folder="protocol_r_760m_author_seed_v1",
                artifact_root=artifact_root,
            )

            self.assertEqual(plan["resume_source"], "author_seed")
            self.assertEqual(plan["seed_source"], "760m_e2e")
            self.assertEqual(
                plan["explicit_resume_checkpoint_path"],
                artifact_root / "760m_e2e",
            )
            self.assertIn("training.load_part=params", plan["extra_overrides"])

    def test_build_b2_sync_command_passes_checkpoint_and_report_roots(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            opts = OrchestratorOptions(
                deploy="interactive",
                runtime_mode="jax_train",
                exp_dir=root / "experiments",
                checkpoint_root=root / "checkpoints",
                profile_root=root / "profiles",
                dclm_root=root / "dclm",
                books_root=root / "books",
                exp_folder="protocol_r_760m_author_seed_v1",
                wandb_entity="none",
                wandb_project="none",
                wandb_key="env",
            )
            args = argparse.Namespace(
                paper_run_id="protocol_r_760m_author_seed_v1",
                exp_folder="protocol_r_760m_author_seed_v1",
                reports_root=root / "reports" / "paper",
                b2_bucket="bucket",
                b2_endpoint_url="https://example.invalid",
                b2_region="us-east-005",
                b2_prefix="ttt-e2e-artifacts",
                run_log=["/tmp/c1.log"],
            )

            cmd = self.module._build_b2_sync_command(
                args=args,
                repo_root=root,
                opts=opts,
            )

            self.assertEqual(cmd[0], self.module.sys.executable)
            self.assertIn("70_sync_760m_b2.py", cmd[1])
            self.assertIn("--checkpoint-root", cmd)
            self.assertIn(str(opts.checkpoint_root), cmd)
            self.assertIn("--reports-root", cmd)
            self.assertIn(str((root / "reports" / "paper").resolve()), cmd)
            self.assertIn("--run-log", cmd)

    def _materialize_author_seed(self, artifact_root: Path, key: str) -> None:
        seed_root = artifact_root / key
        raw_step = seed_root / "raw_orbax" / "28999"
        raw_step.mkdir(parents=True)
        payload = {
            "checkpoint_key": key,
            "local_raw_step_dir": str(raw_step),
            "step": 28999,
        }
        (seed_root / "artifact_manifest.json").write_text(
            json.dumps(payload) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()
