from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "28_fetch_b2_dataset.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("fetch_b2_dataset", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FetchB2DatasetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mod = _load_module()

    def test_s3_dataset_root_uri_matches_expected_layout(self) -> None:
        uri = self.mod._s3_dataset_root_uri("TTTE2E", "ttt-e2e-datasets/paper_budget_760m_val-full", "books3")
        self.assertEqual(uri, "s3://TTTE2E/ttt-e2e-datasets/paper_budget_760m_val-full/books3")

    def test_validate_local_requires_fingerprint_sidecars(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            dataset_root = Path(td) / "dclm_filter_8k"
            (dataset_root / "train" / "c").mkdir(parents=True)
            (dataset_root / "val" / "c").mkdir(parents=True)

            self.assertFalse(self.mod._validate_local(dataset_root, ["train", "val"]))

            (dataset_root / "train.fingerprint.json").write_text("{}", encoding="utf-8")
            (dataset_root / "val.fingerprint.json").write_text("{}", encoding="utf-8")

            self.assertTrue(self.mod._validate_local(dataset_root, ["train", "val"]))


if __name__ == "__main__":
    unittest.main()
