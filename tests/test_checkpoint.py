from __future__ import annotations

import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np
import pandas as pd

from ko_variation.checkpoint import CheckpointError, CheckpointStore
from ko_variation.scan import ScanConfig, ScanMetrics, scan_pairs_glmm
import ko_variation.scan as scan_module


def _matrix() -> np.ndarray:
    predictor = np.tile([1, 1, 0, 0], 20)
    response_one = np.tile([1, 0, 1, 0], 20)
    response_two = np.tile([1, 0, 0, 1, 0, 1, 1, 0], 10)
    return np.column_stack([predictor, response_one, response_two]).astype(np.uint8)


class CheckpointResumeTests(unittest.TestCase):
    def test_interrupted_score_stage_resumes_without_repeating_completed_task(self) -> None:
        matrix = _matrix(); pairs = pd.DataFrame({"u": [0, 0], "v": [1, 2]})
        kinship = np.zeros((len(matrix), len(matrix))); config = ScanConfig(progress=False, min_cell_count=0, spa_mode="off")
        identity = {"fixture": "interrupted-score", "version": 2}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint"
            first = CheckpointStore(path, identity, tool_version="test", every=1)
            original = scan_module._worker_response; calls = 0
            def interrupt_second(task):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise RuntimeError("simulated interruption")
                return original(task)
            with mock.patch("ko_variation.scan._worker_response", side_effect=interrupt_second), self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                scan_pairs_glmm(pairs, matrix, kinship, config, checkpoint=first)
            resumed = CheckpointStore(path, identity, tool_version="test", every=1, resume=True)
            metrics = ScanMetrics()
            with mock.patch("ko_variation.scan.fit_null_glmm", wraps=scan_module.fit_null_glmm) as spy:
                actual = scan_pairs_glmm(pairs, matrix, kinship, config, checkpoint=resumed, metrics=metrics)
            expected = scan_pairs_glmm(pairs, matrix, kinship, config)
            self.assertEqual(spy.call_count, 1)
            self.assertEqual(metrics.counts["score_tasks_recovered"], 1)
            pd.testing.assert_frame_equal(actual[0], expected[0], check_dtype=False)
            pd.testing.assert_frame_equal(actual[1], expected[1], check_dtype=False)

    def test_completed_score_stage_is_fully_reused(self) -> None:
        matrix = _matrix()[:, :2]; pairs = pd.DataFrame({"u": [0], "v": [1]}); kinship = np.zeros((len(matrix), len(matrix)))
        config = ScanConfig(progress=False, min_cell_count=0, spa_mode="off"); identity = {"fixture": "complete", "version": 2}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint"
            expected = scan_pairs_glmm(pairs, matrix, kinship, config, checkpoint=CheckpointStore(path, identity, tool_version="test", every=1))
            resumed = CheckpointStore(path, identity, tool_version="test", every=1, resume=True)
            with mock.patch("ko_variation.scan.fit_null_glmm") as spy:
                actual = scan_pairs_glmm(pairs, matrix, kinship, config, checkpoint=resumed)
            spy.assert_not_called()
            pd.testing.assert_frame_equal(actual[0], expected[0], check_dtype=False)

    def test_resume_rejects_changed_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint"; CheckpointStore(path, {"input": "first"}, tool_version="test")
            with self.assertRaisesRegex(CheckpointError, "do not match"):
                CheckpointStore(path, {"input": "changed"}, tool_version="test", resume=True)

    def test_successful_completion_can_remove_checkpoint_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint"; store = CheckpointStore(path, {"input": "cleanup"}, tool_version="test", every=1)
            store.record("score", 1, {"value": 2}); store.complete(cleanup=True)
            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
