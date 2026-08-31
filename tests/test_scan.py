from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from ko_variation.scan import ScanConfig, _bh_adjust, _pair_rows, _should_apply_spa, scan_pairs_glmm


def _associated_matrix() -> np.ndarray:
    return np.vstack([
        np.tile([1, 1], (30, 1)), np.tile([1, 0], (10, 1)),
        np.tile([0, 1], (10, 1)), np.tile([0, 0], (30, 1)),
    ]).astype(np.uint8)


class UnorderedScanTests(unittest.TestCase):
    def test_one_row_per_pair_and_reversed_input_is_invariant(self) -> None:
        matrix = _associated_matrix()
        kinship = np.zeros((len(matrix), len(matrix)), dtype=np.float64)
        config = ScanConfig(progress=False, spa_mode="off")
        forward, _ = scan_pairs_glmm(pd.DataFrame({"u": [0], "v": [1]}), matrix, kinship, config)
        reverse, _ = scan_pairs_glmm(pd.DataFrame({"u": [1], "v": [0]}), matrix, kinship, config)
        self.assertEqual(len(forward), 1)
        self.assertEqual(forward[["u", "v"]].values.tolist(), [[0, 1]])
        self.assertEqual(reverse[["u", "v"]].values.tolist(), [[0, 1]])
        self.assertAlmostEqual(forward.loc[0, "p_primary"], reverse.loc[0, "p_primary"], places=15)
        self.assertNotIn("direction", forward.columns)
        self.assertNotIn("odds_ratio", forward.columns)

    def test_duplicate_unordered_pairs_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate unordered"):
            _pair_rows(
                pd.DataFrame({"u": [0, 1], "v": [1, 0]}),
                _associated_matrix(), ScanConfig(progress=False),
            )

    def test_graph_count_filter_precedes_model_scheduling(self) -> None:
        matrix = _associated_matrix()
        rows = _pair_rows(
            pd.DataFrame({"u": [0], "v": [1], "count": [3]}),
            matrix, ScanConfig(progress=False),
        )
        self.assertEqual(int(rows.loc[0, "graph_count_threshold"]), 4)
        self.assertEqual(rows.loc[0, "status"], "LOW_GRAPH_COUNT")
        result, models = scan_pairs_glmm(
            pd.DataFrame({"u": [0], "v": [1], "count": [3]}), matrix,
            np.zeros((len(matrix), len(matrix))), ScanConfig(progress=False),
        )
        self.assertTrue(np.isnan(result.loc[0, "p_primary"]))
        self.assertTrue(models.empty)

    def test_graph_filter_is_not_applied_without_count_column(self) -> None:
        rows = _pair_rows(
            pd.DataFrame({"u": [0], "v": [1]}), _associated_matrix(),
            ScanConfig(progress=False, min_cell_count=1),
        )
        self.assertEqual(rows.loc[0, "status"], "ELIGIBLE")
        self.assertTrue(np.isnan(rows.loc[0, "graph_count_threshold"]))

    def test_maf_threshold_is_inclusive_and_default_cell_count_is_one(self) -> None:
        self.assertEqual(ScanConfig().min_cell_count, 1)
        n = 100
        u = np.zeros(n, dtype=np.uint8); u[:5] = 1
        v = np.zeros(n, dtype=np.uint8); v[:3] = 1; v[5:7] = 1
        rows = _pair_rows(pd.DataFrame({"u": [0], "v": [1]}), np.column_stack([u, v]), ScanConfig())
        np.testing.assert_allclose(rows[["predictor_maf", "response_maf"]], [[0.05, 0.05]])
        self.assertEqual(rows.loc[0, "status"], "ELIGIBLE")

    def test_bh_values_match_known_example(self) -> None:
        p = np.array([0.01, 0.04, 0.03, 0.002])
        np.testing.assert_allclose(_bh_adjust(p), [0.02, 0.04, 0.04, 0.008], atol=1e-15)

    def test_spa_auto_is_default_and_targets_sparse_tail_results(self) -> None:
        config = ScanConfig()
        self.assertEqual(config.spa_mode, "auto")
        sparse_tail = {"p_score": 0.01, "response_maf": 0.05, "predictor_maf": 0.30, "min_cell": 10}
        balanced_tail = dict(sparse_tail, response_maf=0.30)
        sparse_central = dict(sparse_tail, p_score=0.10)
        self.assertTrue(_should_apply_spa(sparse_tail, config))
        self.assertFalse(_should_apply_spa(balanced_tail, config))
        self.assertFalse(_should_apply_spa(sparse_central, config))


if __name__ == "__main__":
    unittest.main()
