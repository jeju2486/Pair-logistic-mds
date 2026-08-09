from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from ko_variation.diagnostics import (
    BLAS_ENVIRONMENT_VARIABLES,
    NativeLibrary,
    assess_parallel_runtime,
    guard_blas_environment,
)


class RuntimeDiagnosticsTests(unittest.TestCase):
    def test_guard_sets_missing_values_and_preserves_explicit_value(self) -> None:
        clean_environment = {variable: "" for variable in BLAS_ENVIRONMENT_VARIABLES}
        clean_environment["OPENBLAS_NUM_THREADS"] = "3"
        with patch.dict(os.environ, clean_environment, clear=True):
            configured = guard_blas_environment()
            self.assertEqual(configured["OPENBLAS_NUM_THREADS"], "3")
            self.assertEqual(configured["MKL_NUM_THREADS"], "1")
            self.assertEqual(configured["OMP_NUM_THREADS"], "1")

    def test_nested_native_threads_are_reported_as_oversubscribed(self) -> None:
        runtime = assess_parallel_runtime(
            8,
            logical_cpus=16,
            libraries=[NativeLibrary("blas", "openblas", "libopenblas", "1", 4)],
            environment={variable: "1" for variable in BLAS_ENVIRONMENT_VARIABLES},
        )
        self.assertEqual(runtime.estimated_native_threads, 32)
        self.assertTrue(runtime.oversubscribed)
        self.assertIn("nested_blas_parallelism", runtime.warnings)
        self.assertIn("native_thread_oversubscription", runtime.warnings)

    def test_loaded_blas_threads_take_priority_over_environment_hint(self) -> None:
        runtime = assess_parallel_runtime(
            4,
            logical_cpus=64,
            libraries=[NativeLibrary("blas", "mkl", "mkl_rt", "2025", 1)],
            environment={variable: "8" for variable in BLAS_ENVIRONMENT_VARIABLES},
        )
        self.assertEqual(runtime.effective_blas_threads, 1)
        self.assertEqual(runtime.estimated_native_threads, 4)
        self.assertFalse(runtime.warnings)
        self.assertEqual(runtime.summary_fields()["runtime_blas_oversubscribed"], 0)


if __name__ == "__main__":
    unittest.main()
