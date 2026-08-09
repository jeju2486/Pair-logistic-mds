from __future__ import annotations

import subprocess
import sys
from pathlib import Path
import tempfile
import unittest

import pandas as pd


class CommandLineContractTests(unittest.TestCase):
    def _run(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "ko_variation.cli", *arguments],
            check=False,
            capture_output=True,
            text=True,
        )

    def test_version(self) -> None:
        completed = self._run("--version")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn(
            completed.stdout.strip(),
            {"KOVAR 0.8.1", "KO-Variation 0.8.1"},
        )

    def test_help_exposes_directional_glmm_options_and_removes_lmm_controls(self) -> None:
        completed = self._run("--help")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        help_text = completed.stdout
        for option in ("--direction-mode", "--min-maf", "--min-cell-count", "--spa-mode"):
            self.assertIn(option, help_text)
        for removed in ("--h2-max", "--h2-grid-size", "--grm-shrinkage", "--grm-rank", "--pair-test"):
            self.assertNotIn(removed, help_text)

    def test_tree_backed_run_writes_directional_results_and_null_models(self) -> None:
        sequences = ["CC"] * 30 + ["CA"] * 10 + ["AC"] * 10 + ["AA"] * 30
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fasta = root / "matrix.fa"
            pairs = root / "pairs.tsv"
            tree = root / "tree.nwk"
            out = root / "out"
            fasta.write_text(
                "".join(f">s{i}\n{sequence}\n" for i, sequence in enumerate(sequences)),
                encoding="utf-8",
            )
            pairs.write_text("u\tv\n0\t1\n", encoding="utf-8")
            tree.write_text(
                "(" + ",".join(f"s{i}:1" for i in range(len(sequences))) + ");\n",
                encoding="utf-8",
            )
            completed = self._run(
                "--fasta", str(fasta),
                "--pairs", str(pairs),
                "--tree", str(tree),
                "--out", str(out),
                "--full-refit-p", "0",
                "--threads", "2",
                "--no-progress",
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            results = pd.read_csv(out / "ko_variation.tsv", sep="\t")
            models = pd.read_csv(out / "response_models.tsv", sep="\t")
            summary = (out / "run_summary.txt").read_text(encoding="utf-8")

        self.assertEqual(results["direction"].tolist(), ["u_predicts_v", "v_predicts_u"])
        self.assertEqual(len(models), 2)
        self.assertIn("model\tdirectional_logistic_mixed_model_pql_score", summary)
        self.assertIn("release_status\texperimental", summary)
        self.assertIn("solver_backend\tdense_weighted_eigen_pql", summary)
        self.assertIn("runtime_blas_threads\t", summary)
        self.assertIn("response_pattern_cache\texact_identical_complement", summary)


if __name__ == "__main__":
    unittest.main()
