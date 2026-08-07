from __future__ import annotations

import itertools
import unittest

import numpy as np

from ko_variation.spa import cumulants, spa_pvalue


class SaddlepointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.weights = np.array([-1.5, -0.5, 0.5, 2.0], dtype=np.float64)
        self.probabilities = np.array([0.1, 0.25, 0.6, 0.8], dtype=np.float64)
        self.variance = float(
            np.sum(self.weights**2 * self.probabilities * (1.0 - self.probabilities))
        )

    def test_cumulants_at_zero_match_centered_bernoulli_moments(self) -> None:
        k0, k1, k2 = cumulants(0.0, self.weights, self.probabilities)
        self.assertAlmostEqual(k0, 0.0, places=14)
        self.assertAlmostEqual(k1, 0.0, places=14)
        self.assertAlmostEqual(k2, self.variance, places=14)

    def test_spa_reports_solved_root_and_variance_ratio(self) -> None:
        weights = np.tile(np.array([-1.0, 1.0]), 50)
        probabilities = np.full(weights.size, 0.2, dtype=np.float64)
        variance = float(np.sum(weights**2 * probabilities * (1.0 - probabilities)))
        result = spa_pvalue(
            score=10.0,
            target_variance=variance,
            weights=weights,
            probabilities=probabilities,
        )

        self.assertEqual(result.status, "OK")
        self.assertGreater(result.p_value, 0.0)
        self.assertLessEqual(result.p_value, 1.0)
        self.assertAlmostEqual(result.variance_ratio, 1.0, places=14)
        self.assertLess(abs(result.root_residual), 1e-9)
        _, tilted_score, _ = cumulants(
            result.saddlepoint,
            weights,
            probabilities,
        )
        self.assertAlmostEqual(tilted_score, 10.0, places=9)

    def test_zero_score_uses_normal_near_mean_limit(self) -> None:
        result = spa_pvalue(
            score=0.0,
            target_variance=self.variance,
            weights=self.weights,
            probabilities=self.probabilities,
        )
        self.assertEqual(result.status, "NORMAL_NEAR_MEAN")
        self.assertAlmostEqual(result.p_value, 1.0, places=14)
        self.assertEqual(result.saddlepoint, 0.0)

    def test_skewed_two_tail_spa_agrees_with_small_exact_enumeration(self) -> None:
        weights = np.array([
            0.1391842354, -0.5726175879, -0.4629326898, -2.4913365290,
            1.7498382363, 1.0942967256, -0.3752919833, 0.7239374403,
            0.2313415234, -0.6036919828, 0.9276983047, -0.3604256931,
        ])
        probabilities = np.array([
            0.2014207768, 0.2842540545, 0.1979746356, 0.2716145397,
            0.3886025834, 0.2890726878, 0.1870686916, 0.1155383994,
            0.1710862330, 0.2288730907, 0.3619232933, 0.3214473799,
        ])
        observed = 3.5438313177
        variance = float(np.sum(weights**2 * probabilities * (1 - probabilities)))
        values: list[float] = []
        masses: list[float] = []
        for bits in itertools.product((0.0, 1.0), repeat=weights.size):
            y = np.asarray(bits)
            values.append(float(np.sum(weights * (y - probabilities))))
            masses.append(float(np.prod(np.where(y == 1, probabilities, 1 - probabilities))))
        values_array = np.asarray(values)
        masses_array = np.asarray(masses)
        exact = float(masses_array[
            (values_array >= abs(observed) - 1e-8)
            | (values_array <= -abs(observed) + 1e-8)
        ].sum())
        result = spa_pvalue(observed, variance, weights, probabilities)

        self.assertEqual(result.status, "OK")
        self.assertAlmostEqual(exact, 0.0270032653, places=9)
        self.assertLess(abs(result.p_value - exact), 0.01)


if __name__ == "__main__":
    unittest.main()
