from __future__ import annotations

import math
import unittest

import numpy as np

from ko_variation.glmm import (
    fit_full_glmm,
    fit_null_glmm,
    score_predictor,
    score_predictor_block,
)


def _two_by_two_vectors(
    n11: int,
    n10: int,
    n01: int,
    n00: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return predictor/response vectors for the oriented four-cell table."""
    predictor = np.concatenate(
        [
            np.ones(n11 + n10, dtype=np.float64),
            np.zeros(n01 + n00, dtype=np.float64),
        ]
    )
    response = np.concatenate(
        [
            np.ones(n11, dtype=np.float64),
            np.zeros(n10, dtype=np.float64),
            np.ones(n01, dtype=np.float64),
            np.zeros(n00, dtype=np.float64),
        ]
    )
    return predictor, response


class NoRelatednessGLMMTests(unittest.TestCase):
    def test_score_and_full_fit_match_binary_regression_oracle(self) -> None:
        # For this table, the intercept-only logistic score statistic is exactly
        # chi-square=20 and the unrestricted coefficient is log(OR)=log(9).
        predictor, response = _two_by_two_vectors(30, 10, 10, 30)
        kinship = np.zeros((response.size, response.size), dtype=np.float64)

        null, cache = fit_null_glmm(1, response, kinship)
        score = score_predictor(predictor, cache)
        full = fit_full_glmm(response, predictor, kinship, tolerance=1e-10)

        self.assertEqual(null.status, "OK")
        self.assertEqual(null.tau, 0.0)
        self.assertEqual(score.status, "OK")
        self.assertAlmostEqual(score.score_u, 10.0, places=10)
        self.assertAlmostEqual(score.score_variance, 5.0, places=10)
        self.assertAlmostEqual(score.score_z**2, 20.0, places=10)
        self.assertAlmostEqual(score.p_score, 7.744216431044088e-6, places=15)

        self.assertEqual(full.status, "OK")
        self.assertAlmostEqual(full.beta, math.log(9.0), places=10)
        self.assertAlmostEqual(full.odds_ratio, 9.0, places=9)
        expected_se = math.sqrt(1 / 30 + 1 / 10 + 1 / 10 + 1 / 30)
        self.assertAlmostEqual(full.se, expected_se, places=10)

    def test_balanced_independence_and_predictor_complement(self) -> None:
        predictor, response = _two_by_two_vectors(20, 20, 20, 20)
        kinship = np.zeros((response.size, response.size), dtype=np.float64)
        null, cache = fit_null_glmm(1, response, kinship)

        score = score_predictor(predictor, cache)
        complement = score_predictor(1.0 - predictor, cache)

        self.assertEqual(null.status, "OK")
        self.assertEqual(score.status, "OK")
        self.assertAlmostEqual(score.score_u, 0.0, places=12)
        self.assertAlmostEqual(score.p_score, 1.0, places=12)
        self.assertAlmostEqual(score.beta_score, 0.0, places=12)
        self.assertEqual(complement.status, "OK")
        self.assertAlmostEqual(complement.p_score, score.p_score, places=12)
        self.assertAlmostEqual(complement.score_u, -score.score_u, places=12)

    def test_predictor_complement_preserves_p_and_reverses_effect(self) -> None:
        predictor, response = _two_by_two_vectors(30, 10, 10, 30)
        kinship = np.zeros((response.size, response.size), dtype=np.float64)
        _, cache = fit_null_glmm(1, response, kinship)

        score = score_predictor(predictor, cache)
        complement = score_predictor(1.0 - predictor, cache)
        full = fit_full_glmm(response, predictor, kinship)
        full_complement = fit_full_glmm(response, 1.0 - predictor, kinship)

        self.assertAlmostEqual(complement.p_score, score.p_score, places=12)
        self.assertAlmostEqual(complement.score_u, -score.score_u, places=12)
        self.assertAlmostEqual(complement.beta_score, -score.beta_score, places=12)
        self.assertEqual(full.status, "OK")
        self.assertEqual(full_complement.status, "OK")
        self.assertAlmostEqual(full_complement.beta, -full.beta, places=9)
        self.assertAlmostEqual(full_complement.odds_ratio, 1.0 / full.odds_ratio, places=9)

    def test_diagonal_only_kernel_is_treated_as_no_shared_relatedness(self) -> None:
        predictor, response = _two_by_two_vectors(30, 10, 10, 30)
        zero_fit, zero_cache = fit_null_glmm(
            1, response, np.zeros((response.size, response.size), dtype=np.float64)
        )
        identity_fit, identity_cache = fit_null_glmm(
            1, response, np.eye(response.size, dtype=np.float64)
        )
        zero_score = score_predictor(predictor, zero_cache)
        identity_score = score_predictor(predictor, identity_cache)

        self.assertEqual(zero_fit.tau, 0.0)
        self.assertEqual(identity_fit.tau, 0.0)
        self.assertAlmostEqual(identity_score.score_u, zero_score.score_u, places=12)
        self.assertAlmostEqual(identity_score.score_variance, zero_score.score_variance, places=12)
        self.assertAlmostEqual(identity_score.p_score, zero_score.p_score, places=15)

    def test_spa_predictor_is_working_weight_residualized(self) -> None:
        n = 80
        kinship = np.eye(n, dtype=np.float64)
        for block in range(8):
            indices = slice(10 * block, 10 * (block + 1))
            kinship[indices, indices] = 0.8
            np.fill_diagonal(kinship[indices, indices], 1.0)
        rng = np.random.default_rng(3)
        response_probability = np.repeat([0.1, 0.9] * 4, 10)
        response = rng.binomial(1, response_probability).astype(np.float64)
        predictor = rng.binomial(1, 0.3, n).astype(np.float64)
        fit, cache = fit_null_glmm(1, response, kinship)
        score = score_predictor(predictor, cache)

        self.assertEqual(fit.status, "OK")
        self.assertGreater(fit.tau, 0.0)
        self.assertLessEqual(fit.max_eta_change, 1e-7)
        self.assertEqual(score.status, "OK")
        self.assertAlmostEqual(
            float(np.sum(cache["weights"] * score.adjusted_predictor)),
            0.0,
            places=10,
        )

    def test_invalid_damping_is_rejected(self) -> None:
        _, response = _two_by_two_vectors(20, 20, 20, 20)
        kinship = np.zeros((response.size, response.size), dtype=np.float64)
        with self.assertRaisesRegex(ValueError, "damping"):
            fit_null_glmm(1, response, kinship, damping=0.0)

    def test_block_scores_equal_scalar_scores(self) -> None:
        predictor, response = _two_by_two_vectors(30, 10, 10, 30)
        second = np.roll(predictor, 7)
        kinship = np.zeros((response.size, response.size), dtype=np.float64)
        _, cache = fit_null_glmm(1, response, kinship)
        block = score_predictor_block(np.column_stack([predictor, second]), cache)

        for column, batched in zip((predictor, second), block):
            scalar = score_predictor(column, cache)
            self.assertEqual(batched.status, scalar.status)
            self.assertAlmostEqual(batched.score_u, scalar.score_u, places=14)
            self.assertAlmostEqual(batched.score_variance, scalar.score_variance, places=14)
            self.assertAlmostEqual(batched.p_score, scalar.p_score, places=15)

    def test_joint_sample_and_kinship_permutation_is_invariant(self) -> None:
        n = 40
        kinship = np.eye(n, dtype=np.float64)
        for block in range(4):
            indices = slice(10 * block, 10 * (block + 1))
            kinship[indices, indices] = 0.5
            np.fill_diagonal(kinship[indices, indices], 1.0)
        response = np.asarray([0, 1] * 20, dtype=np.float64)
        predictor = np.asarray([0, 0, 1, 1] * 10, dtype=np.float64)
        fit, cache = fit_null_glmm(1, response, kinship)
        score = score_predictor(predictor, cache)
        permutation = np.random.default_rng(7).permutation(n)
        permuted_fit, permuted_cache = fit_null_glmm(
            1,
            response[permutation],
            kinship[np.ix_(permutation, permutation)],
        )
        permuted_score = score_predictor(predictor[permutation], permuted_cache)

        self.assertAlmostEqual(permuted_fit.tau, fit.tau, places=12)
        self.assertAlmostEqual(permuted_score.score_u, score.score_u, places=12)
        self.assertAlmostEqual(
            permuted_score.score_variance, score.score_variance, places=12
        )
        self.assertAlmostEqual(permuted_score.p_score, score.p_score, places=12)


if __name__ == "__main__":
    unittest.main()
