from __future__ import annotations

import unittest
from unittest import mock

import numpy as np
import pandas as pd

from ko_variation.glmm import fit_full_glmm, fit_null_glmm, score_predictor
from ko_variation.scan import (
    ScanConfig,
    _canonical_response_pattern,
    scan_pairs_glmm,
)


def _copy_and_complement_matrix() -> np.ndarray:
    predictor_response = np.vstack(
        [
            np.tile([1, 1], (30, 1)),
            np.tile([1, 0], (10, 1)),
            np.tile([0, 1], (10, 1)),
            np.tile([0, 0], (30, 1)),
        ]
    ).astype(np.uint8)
    predictor = predictor_response[:, 0]
    response = predictor_response[:, 1]
    return np.column_stack([predictor, response, response.copy(), 1 - response])


class ResponsePatternCacheTests(unittest.TestCase):
    def test_packed_canonical_key_is_exact_and_complement_invariant(self) -> None:
        response = np.array([0, 1, 1, 0, 1, 0, 0, 1, 1], dtype=np.uint8)
        copied = response.copy()
        complement = 1 - response
        different = response.copy()
        different[3] = 1

        key, flipped = _canonical_response_pattern(response)
        copied_key, copied_flipped = _canonical_response_pattern(copied)
        complement_key, complement_flipped = _canonical_response_pattern(complement)
        different_key, _ = _canonical_response_pattern(different)

        self.assertEqual(key, copied_key)
        self.assertEqual(flipped, copied_flipped)
        self.assertEqual(key, complement_key)
        self.assertNotEqual(flipped, complement_flipped)
        self.assertNotEqual(key, different_key)

    def test_one_null_fit_serves_identical_and_complement_responses_exactly(self) -> None:
        matrix = _copy_and_complement_matrix()
        pairs = pd.DataFrame({"u": [0, 0, 0], "v": [1, 2, 3]})
        sample_order = np.arange(matrix.shape[0], dtype=np.float64)
        kinship = np.exp(
            -np.abs(sample_order[:, None] - sample_order[None, :]) / 8.0
        )
        config = ScanConfig(
            progress=False,
            direction_mode="input",
            spa_mode="off",
            full_refit_p=0,
        )

        with mock.patch(
            "ko_variation.scan.fit_null_glmm",
            wraps=fit_null_glmm,
        ) as fit_spy:
            cached, response_models = scan_pairs_glmm(pairs, matrix, kinship, config)

        self.assertEqual(fit_spy.call_count, 1)
        self.assertIsNotNone(fit_spy.call_args.kwargs["kinship_eigensystem"])
        self.assertEqual(len(cached), 3)
        self.assertEqual(cached["response_locus"].tolist(), [1, 2, 3])
        self.assertEqual(len(response_models), 3)
        self.assertEqual(response_models["response_pattern_id"].nunique(), 1)
        self.assertEqual(response_models["response_pattern_size"].tolist(), [3, 3, 3])
        self.assertEqual(int(response_models["null_fit_reused"].sum()), 2)
        flips = response_models.set_index("response_locus")["response_pattern_flipped"]
        self.assertEqual(int(flips.loc[1]), int(flips.loc[2]))
        self.assertNotEqual(int(flips.loc[1]), int(flips.loc[3]))

        for result_index, response_locus in enumerate([1, 2, 3]):
            fit, null_cache = fit_null_glmm(
                response_locus,
                matrix[:, response_locus],
                kinship,
            )
            separate = score_predictor(
                matrix[:, 0],
                null_cache,
                compute_spa_adjustment=False,
            )
            self.assertEqual(fit.status, "OK")
            self.assertEqual(cached.loc[result_index, "status"], "OK")
            self.assertAlmostEqual(
                cached.loc[result_index, "tau_phylogenetic"], fit.tau, places=9
            )
            self.assertAlmostEqual(
                cached.loc[result_index, "score_u"], separate.score_u, places=10
            )
            self.assertAlmostEqual(
                cached.loc[result_index, "score_variance"],
                separate.score_variance,
                places=10,
            )
            self.assertAlmostEqual(
                cached.loc[result_index, "score_z"], separate.score_z, places=10
            )
            self.assertAlmostEqual(
                cached.loc[result_index, "beta_score"], separate.beta_score, places=10
            )
            self.assertAlmostEqual(
                cached.loc[result_index, "p_primary"], separate.p_score, places=10
            )

    def test_complement_full_refits_use_the_original_response_orientation(self) -> None:
        matrix = _copy_and_complement_matrix()
        pairs = pd.DataFrame({"u": [0, 0], "v": [1, 3]})
        kinship = np.zeros((matrix.shape[0], matrix.shape[0]), dtype=np.float64)
        config = ScanConfig(
            progress=False,
            direction_mode="input",
            spa_mode="off",
            full_refit_p=1.0,
        )

        with mock.patch(
            "ko_variation.scan.fit_null_glmm",
            wraps=fit_null_glmm,
        ) as null_spy, mock.patch(
            "ko_variation.scan.fit_full_glmm",
            wraps=fit_full_glmm,
        ) as full_spy:
            cached, _ = scan_pairs_glmm(pairs, matrix, kinship, config)

        self.assertEqual(null_spy.call_count, 1)
        self.assertEqual(full_spy.call_count, 2)
        self.assertTrue(np.all(cached["full_refit_attempted"] == 1))
        for result_index, response_locus in enumerate([1, 3]):
            separate = fit_full_glmm(
                matrix[:, response_locus],
                matrix[:, 0],
                kinship,
            )
            self.assertEqual(separate.status, "OK")
            self.assertEqual(cached.loc[result_index, "full_refit_status"], "OK")
            self.assertAlmostEqual(
                cached.loc[result_index, "beta_log_odds"], separate.beta, places=11
            )
            self.assertAlmostEqual(
                cached.loc[result_index, "se_log_odds"], separate.se, places=11
            )
            self.assertAlmostEqual(
                cached.loc[result_index, "odds_ratio"], separate.odds_ratio, places=11
            )
            self.assertAlmostEqual(
                cached.loc[result_index, "p_wald"], separate.wald_p, places=13
            )

    def test_spa_is_complement_invariant_while_directional_scores_change_sign(self) -> None:
        matrix = _copy_and_complement_matrix()
        pairs = pd.DataFrame({"u": [0, 0], "v": [1, 3]})
        kinship = np.zeros((matrix.shape[0], matrix.shape[0]), dtype=np.float64)

        cached, _ = scan_pairs_glmm(
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

        self.assertEqual(cached["spa_status"].tolist(), ["OK", "OK"])
        self.assertAlmostEqual(cached.loc[0, "score_u"], -cached.loc[1, "score_u"], places=12)
        self.assertAlmostEqual(cached.loc[0, "score_z"], -cached.loc[1, "score_z"], places=12)
        self.assertAlmostEqual(
            cached.loc[0, "beta_score"], -cached.loc[1, "beta_score"], places=12
        )
        self.assertAlmostEqual(
            cached.loc[0, "score_variance"], cached.loc[1, "score_variance"], places=12
        )
        self.assertAlmostEqual(cached.loc[0, "p_score"], cached.loc[1, "p_score"], places=14)
        self.assertAlmostEqual(cached.loc[0, "p_spa"], cached.loc[1, "p_spa"], places=14)
        self.assertAlmostEqual(
            cached.loc[0, "p_primary"], cached.loc[1, "p_primary"], places=14
        )


if __name__ == "__main__":
    unittest.main()
