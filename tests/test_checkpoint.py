from __future__ import annotations

import gzip
import hashlib
import json
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np
import pandas as pd

from ko_variation.checkpoint import CheckpointError, CheckpointStore
from ko_variation.scan import ScanConfig, ScanMetrics, scan_pairs_glmm
import ko_variation.scan as scan_module


def _checkpoint_matrix() -> np.ndarray:
    predictor = np.tile([1, 1, 0, 0], 20)
    response_one = np.tile([1, 0, 1, 0], 20)
    response_two = np.tile([1, 0, 0, 1, 0, 1, 1, 0], 10)
    return np.column_stack([predictor, response_one, response_two]).astype(np.uint8)


class CheckpointResumeTests(unittest.TestCase):
    def test_interrupted_score_stage_resumes_without_repeating_completed_task(self) -> None:
        matrix = _checkpoint_matrix()
        pairs = pd.DataFrame({"u": [0, 0], "v": [1, 2]})
        kinship = np.zeros((matrix.shape[0], matrix.shape[0]), dtype=np.float64)
        config = ScanConfig(
            progress=False,
            direction_mode="input",
            min_cell_count=0,
            full_refit_p=0,
        )
        identity = {"fixture": "interrupted-score", "version": 1}

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "checkpoint"
            first_store = CheckpointStore(
                checkpoint_path,
                identity,
                tool_version="test",
                every=1,
            )
            original_worker = scan_module._worker_response
            calls = 0

            def interrupt_second_task(task):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise RuntimeError("simulated interruption")
                return original_worker(task)

            with mock.patch(
                "ko_variation.scan._worker_response",
                side_effect=interrupt_second_task,
            ), self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                scan_pairs_glmm(
                    pairs,
                    matrix,
                    kinship,
                    config,
                    checkpoint=first_store,
                )

            resumed_store = CheckpointStore(
                checkpoint_path,
                identity,
                tool_version="test",
                every=1,
                resume=True,
            )
            resumed_metrics = ScanMetrics()
            with mock.patch(
                "ko_variation.scan.fit_null_glmm",
                wraps=scan_module.fit_null_glmm,
            ) as fit_spy:
                resumed_results, resumed_models = scan_pairs_glmm(
                    pairs,
                    matrix,
                    kinship,
                    config,
                    checkpoint=resumed_store,
                    metrics=resumed_metrics,
                )

            uninterrupted_results, uninterrupted_models = scan_pairs_glmm(
                pairs,
                matrix,
                kinship,
                config,
            )

            self.assertEqual(fit_spy.call_count, 1)
            self.assertEqual(resumed_metrics.counts["score_tasks_recovered"], 1)
            pd.testing.assert_frame_equal(
                resumed_results,
                uninterrupted_results,
                check_dtype=False,
            )
            pd.testing.assert_frame_equal(
                resumed_models,
                uninterrupted_models,
                check_dtype=False,
            )

    def test_completed_score_and_full_refit_stages_are_fully_reused(self) -> None:
        matrix = _checkpoint_matrix()[:, :2]
        pairs = pd.DataFrame({"u": [0], "v": [1]})
        kinship = np.zeros((matrix.shape[0], matrix.shape[0]), dtype=np.float64)
        config = ScanConfig(
            progress=False,
            direction_mode="input",
            min_cell_count=0,
            full_refit_p=1.0,
        )
        identity = {"fixture": "full-refit", "version": 1}

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "checkpoint"
            first_store = CheckpointStore(
                checkpoint_path,
                identity,
                tool_version="test",
                every=1,
            )
            expected_results, expected_models = scan_pairs_glmm(
                pairs,
                matrix,
                kinship,
                config,
                checkpoint=first_store,
            )

            resumed_store = CheckpointStore(
                checkpoint_path,
                identity,
                tool_version="test",
                every=1,
                resume=True,
            )
            with mock.patch("ko_variation.scan.fit_null_glmm") as null_spy, mock.patch(
                "ko_variation.scan.fit_full_glmm"
            ) as full_spy:
                actual_results, actual_models = scan_pairs_glmm(
                    pairs,
                    matrix,
                    kinship,
                    config,
                    checkpoint=resumed_store,
                )

            null_spy.assert_not_called()
            full_spy.assert_not_called()
            pd.testing.assert_frame_equal(actual_results, expected_results, check_dtype=False)
            pd.testing.assert_frame_equal(actual_models, expected_models, check_dtype=False)

    def test_interrupted_full_refit_stage_resumes_only_unfinished_refit(self) -> None:
        matrix = _checkpoint_matrix()
        pairs = pd.DataFrame({"u": [0, 0], "v": [1, 2]})
        kinship = np.zeros((matrix.shape[0], matrix.shape[0]), dtype=np.float64)
        config = ScanConfig(
            progress=False,
            direction_mode="input",
            min_cell_count=0,
            full_refit_p=1.0,
        )
        identity = {"fixture": "interrupted-full-refit", "version": 1}

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "checkpoint"
            first_store = CheckpointStore(
                checkpoint_path,
                identity,
                tool_version="test",
                every=1,
            )
            original_full_worker = scan_module._worker_full_refit
            calls = 0

            def interrupt_second_refit(task):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise RuntimeError("simulated full-refit interruption")
                return original_full_worker(task)

            with mock.patch(
                "ko_variation.scan._worker_full_refit",
                side_effect=interrupt_second_refit,
            ), self.assertRaisesRegex(RuntimeError, "full-refit interruption"):
                scan_pairs_glmm(
                    pairs,
                    matrix,
                    kinship,
                    config,
                    checkpoint=first_store,
                )

            resumed_store = CheckpointStore(
                checkpoint_path,
                identity,
                tool_version="test",
                every=1,
                resume=True,
            )
            with mock.patch(
                "ko_variation.scan.fit_null_glmm",
                wraps=scan_module.fit_null_glmm,
            ) as null_spy, mock.patch(
                "ko_variation.scan.fit_full_glmm",
                wraps=scan_module.fit_full_glmm,
            ) as full_spy:
                resumed_results, resumed_models = scan_pairs_glmm(
                    pairs,
                    matrix,
                    kinship,
                    config,
                    checkpoint=resumed_store,
                )

            expected_results, expected_models = scan_pairs_glmm(
                pairs,
                matrix,
                kinship,
                config,
            )
            null_spy.assert_not_called()
            self.assertEqual(full_spy.call_count, 1)
            pd.testing.assert_frame_equal(
                resumed_results, expected_results, check_dtype=False
            )
            pd.testing.assert_frame_equal(
                resumed_models, expected_models, check_dtype=False
            )

    def test_resume_rejects_changed_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "checkpoint"
            CheckpointStore(
                checkpoint_path,
                {"input": "first"},
                tool_version="test",
            )
            with self.assertRaisesRegex(CheckpointError, "do not match"):
                CheckpointStore(
                    checkpoint_path,
                    {"input": "changed"},
                    tool_version="test",
                    resume=True,
                )

    def test_successful_completion_can_remove_checkpoint_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "checkpoint"
            store = CheckpointStore(
                checkpoint_path,
                {"input": "cleanup"},
                tool_version="test",
                every=1,
            )
            store.record("score", 1, {"value": 2})
            store.complete(cleanup=True)
            self.assertFalse(checkpoint_path.exists())

    def test_resume_rejects_malformed_checkpoint_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "checkpoint"
            store = CheckpointStore(
                checkpoint_path,
                {"input": "malformed"},
                tool_version="test",
            )
            document = {
                "schema_version": 1,
                "run_fingerprint": store.fingerprint,
                "stage": "score",
                "records": [None],
            }
            compressed = gzip.compress(
                json.dumps(document).encode("utf-8"), compresslevel=1, mtime=0
            )
            shard_path = checkpoint_path / "score-000000.json.gz"
            shard_path.write_bytes(compressed)
            store._manifest["shards"] = [
                {
                    "stage": "score",
                    "file": shard_path.name,
                    "n_tasks": 1,
                    "sha256": hashlib.sha256(compressed).hexdigest(),
                    "size_bytes": len(compressed),
                }
            ]
            store._write_manifest()

            resumed = CheckpointStore(
                checkpoint_path,
                {"input": "malformed"},
                tool_version="test",
                resume=True,
            )
            with self.assertRaisesRegex(CheckpointError, "record.*invalid"):
                list(resumed.iter_stage("score"))


if __name__ == "__main__":
    unittest.main()
