from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from ko_variation.glmm import prepare_kinship
from ko_variation.kinship import build_background_grm, build_tree_covariance


class KinshipTests(unittest.TestCase):
    def test_tree_covariance_matches_root_to_mrca_oracle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tree = Path(tmp) / "tree.nwk"
            tree.write_text("((s1:1,s2:1):1,(s3:1,s4:1):1);\n", encoding="utf-8")
            result = build_tree_covariance(
                tree,
                sample_names=["s1", "s2", "s3", "s4"],
            )

        expected = np.array(
            [
                [1.0, 0.5, 0.0, 0.0],
                [0.5, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.5],
                [0.0, 0.0, 0.5, 1.0],
            ]
        )
        np.testing.assert_allclose(result.K, expected, rtol=0, atol=1e-12)
        self.assertAlmostEqual(result.mean_diag_before_norm, 2.0)
        self.assertEqual(result.source, "tree_mrca_covariance")

    def test_prepare_kinship_accepts_singular_psd_without_scientific_shrinkage(self) -> None:
        raw = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float64)
        prepared, diagnostics = prepare_kinship(raw)

        np.testing.assert_allclose(prepared, raw, rtol=0, atol=1e-12)
        self.assertEqual(diagnostics.rank, 1)
        self.assertEqual(diagnostics.roundoff_correction, 0.0)

    def test_prepare_kinship_accepts_explicit_no_relatedness_kernel(self) -> None:
        raw = np.zeros((4, 4), dtype=np.float64)
        prepared, diagnostics = prepare_kinship(raw)

        np.testing.assert_array_equal(prepared, raw)
        self.assertEqual(diagnostics.rank, 0)
        self.assertEqual(diagnostics.mean_diagonal, 0.0)

    def test_prepare_kinship_rejects_materially_indefinite_matrix(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive semidefinite"):
            prepare_kinship(np.array([[1.0, 2.0], [2.0, 1.0]], dtype=np.float64))

    def test_background_grm_masks_tested_loci_and_remains_psd(self) -> None:
        rng = np.random.default_rng(42)
        matrix = rng.binomial(1, 0.35, size=(100, 24)).astype(np.uint8)
        result = build_background_grm(
            matrix,
            target_loci=np.array([0, 1]),
            min_mac=2,
            progress=False,
        )
        prepared, diagnostics = prepare_kinship(result.K)

        self.assertEqual(result.details["n_direct_masked"], 2)
        self.assertGreater(result.n_loci_used, 0)
        self.assertAlmostEqual(float(np.mean(np.diag(prepared))), 1.0, places=12)
        self.assertGreaterEqual(diagnostics.eigen_min, -1e-12)


if __name__ == "__main__":
    unittest.main()
