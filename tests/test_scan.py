from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from ko_variation.scan import (
    ScanConfig,
    _bh_adjust,
    _directional_rows,
    _should_apply_spa,
    scan_pairs_glmm,
)


def _associated_matrix() -> np.ndarray:
    return np.vstack(
        [
            np.tile([1, 1], (30, 1)),
            np.tile([1, 0], (10, 1)),
            np.tile([0, 1], (10, 1)),
            np.tile([0, 0], (30, 1)),
        ]
    ).astype(np.uint8)


class DirectionalScanTests(unittest.TestCase):
    def test_both_expands_pair_and_reuses_response_nulls(self) -> None:
        matrix = _associated_matrix()
        pairs = pd.DataFrame({"u": [0], "v": [1], "distance": [1234]})
        result, response_models = scan_pairs_glmm(
            pairs,
            matrix,
            np.zeros((matrix.shape[0], matrix.shape[0]), dtype=np.float64),
            ScanConfig(progress=False, direction_mode="both", spa_mode="off", full_refit_p=0),
        )

        self.assertEqual(len(result), 2)
        self.assertEqual(result["direction"].tolist(), ["u_predicts_v", "v_predicts_u"])
        self.assertEqual(result["predictor_locus"].tolist(), [0, 1])
        self.assertEqual(result["response_locus"].tolist(), [1, 0])
        self.assertEqual(result["distance"].tolist(), [1234, 1234])
        self.assertEqual(result["status"].tolist(), ["OK", "OK"])
        self.assertTrue(np.all(np.isfinite(result["p_primary"])))
        self.assertEqual(response_models["response_locus"].tolist(), [0, 1])
        self.assertTrue(response_models["response_locus"].is_unique)
        self.assertTrue(np.all(result["n_directional_tests"] == 2))

    def test_input_mode_keeps_input_orientation(self) -> None:
        matrix = _associated_matrix()
        pairs = pd.DataFrame({"u": [0], "v": [1]})
        rows = _directional_rows(
            pairs,
            matrix,
            ScanConfig(progress=False, direction_mode="input", min_cell_count=0),
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(int(rows.loc[0, "predictor_locus"]), 0)
        self.assertEqual(int(rows.loc[0, "response_locus"]), 1)
        self.assertEqual(rows.loc[0, "direction"], "u_predicts_v")

    def test_reverse_direction_swaps_the_off_diagonal_cells(self) -> None:
        matrix = np.vstack(
            [
                np.tile([1, 1], (12, 1)),
                np.tile([1, 0], (8, 1)),
                np.tile([0, 1], (28, 1)),
                np.tile([0, 0], (52, 1)),
            ]
        ).astype(np.uint8)
        rows = _directional_rows(
            pd.DataFrame({"u": [0], "v": [1]}),
            matrix,
            ScanConfig(progress=False, direction_mode="both", min_cell_count=0),
        )

        self.assertEqual(rows[["n11", "n10", "n01", "n00"]].values.tolist(), [
            [12, 8, 28, 52],
            [12, 28, 8, 52],
        ])
        np.testing.assert_allclose(rows["predictor_prevalence"], [0.2, 0.4])
        np.testing.assert_allclose(rows["response_prevalence"], [0.4, 0.2])

    def test_reserved_metadata_cannot_overwrite_generated_direction_fields(self) -> None:
        matrix = _associated_matrix()
        pairs = pd.DataFrame({"u": [0], "v": [1], "predictor_locus": [99]})
        with self.assertRaisesRegex(ValueError, "reserved KOVAR output columns"):
            _directional_rows(
                pairs,
                matrix,
                ScanConfig(progress=False, direction_mode="both"),
            )

    def test_maf_is_minor_state_frequency_and_threshold_is_inclusive(self) -> None:
        n = 100
        five_percent = np.zeros(n, dtype=np.uint8)
        five_percent[:5] = 1
        ninety_five_percent = np.ones(n, dtype=np.uint8)
        ninety_five_percent[:5] = 0
        four_percent = np.zeros(n, dtype=np.uint8)
        four_percent[:4] = 1
        ninety_six_percent = 1 - four_percent
        balanced = np.arange(n, dtype=np.uint8) % 2
        matrix = np.column_stack(
            [five_percent, ninety_five_percent, four_percent, ninety_six_percent, balanced]
        )

        exact = _directional_rows(
            pd.DataFrame({"u": [0], "v": [1]}),
            matrix,
            ScanConfig(progress=False, direction_mode="both", min_maf=0.05, min_cell_count=1),
        )
        np.testing.assert_allclose(exact["predictor_maf"], [0.05, 0.05], atol=1e-12)
        np.testing.assert_allclose(exact["response_maf"], [0.05, 0.05], atol=1e-12)
        self.assertEqual(exact["status"].tolist(), ["LOW_CELL_COUNT", "LOW_CELL_COUNT"])
        np.testing.assert_allclose(exact["predictor_prevalence"], [0.05, 0.95])
        np.testing.assert_allclose(exact["response_prevalence"], [0.95, 0.05])

        low_predictor = _directional_rows(
            pd.DataFrame({"u": [2], "v": [4]}),
            matrix,
            ScanConfig(progress=False, direction_mode="input", min_maf=0.05, min_cell_count=0),
        )
        self.assertEqual(low_predictor.loc[0, "status"], "LOW_PREDICTOR_MAF")

        low_response = _directional_rows(
            pd.DataFrame({"u": [4], "v": [3]}),
            matrix,
            ScanConfig(progress=False, direction_mode="input", min_maf=0.05, min_cell_count=0),
        )
        self.assertEqual(low_response.loc[0, "status"], "LOW_RESPONSE_MAF")

    def test_cell_filter_retains_row_without_inference(self) -> None:
        n = 200
        predictor = np.zeros(n, dtype=np.uint8)
        response = np.zeros(n, dtype=np.uint8)
        predictor[:20] = 1
        response[20:40] = 1
        matrix = np.column_stack([predictor, response])
        result, response_models = scan_pairs_glmm(
            pd.DataFrame({"u": [0], "v": [1]}),
            matrix,
            np.zeros((n, n), dtype=np.float64),
            ScanConfig(
                progress=False,
                direction_mode="input",
                min_maf=0.05,
                min_cell_count=5,
                full_refit_p=0,
            ),
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(result.loc[0, "status"], "LOW_CELL_COUNT")
        self.assertEqual(int(result.loc[0, "eligible"]), 0)
        self.assertTrue(np.isnan(result.loc[0, "p_score"]))
        self.assertTrue(np.isnan(result.loc[0, "p_primary"]))
        self.assertEqual(int(result.loc[0, "n_directional_tests"]), 0)
        self.assertTrue(response_models.empty)

    def test_bh_values_match_known_example(self) -> None:
        p_values = np.array([0.01, 0.04, 0.03, 0.002], dtype=np.float64)
        expected = np.array([0.02, 0.04, 0.04, 0.008], dtype=np.float64)
        np.testing.assert_allclose(_bh_adjust(p_values), expected, rtol=0, atol=1e-15)

    def test_spa_off_and_always_contract(self) -> None:
        matrix = _associated_matrix()
        pairs = pd.DataFrame({"u": [0], "v": [1]})
        kinship = np.zeros((matrix.shape[0], matrix.shape[0]), dtype=np.float64)

        off, _ = scan_pairs_glmm(
            pairs,
            matrix,
            kinship,
            ScanConfig(
                progress=False,
                direction_mode="input",
                spa_mode="off",
                full_refit_p=0,
            ),
        )
        self.assertEqual(int(off.loc[0, "spa_applied"]), 0)
        self.assertEqual(off.loc[0, "spa_status"], "NOT_APPLIED")
        self.assertTrue(np.isnan(off.loc[0, "p_spa"]))
        self.assertEqual(off.loc[0, "primary_method"], "score_normal")
        self.assertAlmostEqual(off.loc[0, "p_primary"], off.loc[0, "p_score"], places=15)

        always, _ = scan_pairs_glmm(
            pairs,
            matrix,
            kinship,
            ScanConfig(
                progress=False,
                direction_mode="input",
                spa_mode="always",
                full_refit_p=0,
            ),
        )
        self.assertEqual(int(always.loc[0, "spa_applied"]), 1)
        self.assertIn(always.loc[0, "spa_status"], {"OK", "NORMAL_NEAR_MEAN"})
        self.assertTrue(np.isfinite(always.loc[0, "p_spa"]))
        self.assertEqual(always.loc[0, "primary_method"], "score_spa")
        self.assertAlmostEqual(always.loc[0, "p_primary"], always.loc[0, "p_spa"], places=15)

    def test_spa_auto_requires_both_a_tail_result_and_sparse_context(self) -> None:
        config = ScanConfig(spa_mode="auto")
        sparse_tail = {
            "p_score": 0.01,
            "response_maf": 0.05,
            "predictor_maf": 0.30,
            "min_cell": 10,
        }
        balanced_tail = {
            "p_score": 0.01,
            "response_maf": 0.30,
            "predictor_maf": 0.30,
            "min_cell": 10,
        }
        sparse_central = dict(sparse_tail, p_score=0.10)

        self.assertTrue(_should_apply_spa(sparse_tail, config))
        self.assertFalse(_should_apply_spa(balanced_tail, config))
        self.assertFalse(_should_apply_spa(sparse_central, config))


if __name__ == "__main__":
    unittest.main()
