from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


def _load_watch_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "71_watch_760m_c1_and_stop.py"
    spec = importlib.util.spec_from_file_location("watch_760m_c1_and_stop", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Watch760MC1Test(unittest.TestCase):
    def setUp(self) -> None:
        self.module = _load_watch_module()

    def test_summary_succeeded_accepts_core_success_payload(self) -> None:
        payload = {
            "rows": [
                {"step_id": "train:S2_ADAPT", "status": "succeeded"},
                {"step_id": "train:S2", "status": "succeeded"},
                {"step_id": "train:S3", "status": "succeeded"},
                {"step_id": "eval:books32k", "returncode": 0},
            ]
        }
        self.assertTrue(self.module._summary_succeeded(payload))

    def test_summary_succeeded_rejects_failed_train_or_eval(self) -> None:
        failed_train = {
            "rows": [
                {"step_id": "train:S2_ADAPT", "status": "failed"},
                {"step_id": "eval:books32k", "returncode": 0},
            ]
        }
        failed_eval = {
            "rows": [
                {"step_id": "train:S2_ADAPT", "status": "succeeded"},
                {"step_id": "eval:books32k", "returncode": 1},
            ]
        }
        self.assertFalse(self.module._summary_succeeded(failed_train))
        self.assertFalse(self.module._summary_succeeded(failed_eval))


if __name__ == "__main__":
    unittest.main()
