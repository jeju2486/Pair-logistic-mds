from __future__ import annotations

import subprocess
import sys
from pathlib import Path
import tempfile
import unittest

import pandas as pd


class CommandLineContractTests(unittest.TestCase):
    def _run(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run([sys.executable, "-m", "ko_variation.cli", *arguments], check=False, capture_output=True, text=True)

    def test_version(self) -> None:
        completed = self._run("--version")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.strip(), "KO-Variation 0.8.3")

    def test_help_exposes_small_primary_interface(self) -> None:
        completed = self._run("--help")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        help_text = completed.stdout
        for option in ("--fasta", "--pairs", "--tree", "--out", "--min-maf", "--min-cell-count", "--spa-mode", "--threads", "--resume"):
            self.assertIn(option, help_text)
        for removed in ("--direction-mode", "--full-refit-p", "--max-pairs", "--no-checkpoint", "--grm-proxy-r2", "--predictor-batch-size"):
            self.assertNotIn(removed, help_text)

    def test_defaults_use_spa_auto_and_minimum_cell_one(self) -> None:
        from ko_variation.cli import parse_args
        args = parse_args(["--fasta", "x", "--pairs", "y", "--tree", "z", "--out", "o"])
        self.assertEqual(args.spa_mode, "auto")
        self.assertEqual(args.min_cell_count, 1)

    def test_tree_backed_run_writes_one_unordered_result(self) -> None:
        sequences = ["CC"] * 30 + ["CA"] * 10 + ["AC"] * 10 + ["AA"] * 30
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); fasta = root / "matrix.fa"; pairs = root / "pairs.tsv"; tree = root / "tree.nwk"; out = root / "out"
            fasta.write_text("".join(f">s{i}\n{sequence}\n" for i, sequence in enumerate(sequences)), encoding="utf-8")
            pairs.write_text("u\tv\tcount\n1\t0\t4\n", encoding="utf-8")
            tree.write_text("(" + ",".join(f"s{i}:1" for i in range(len(sequences))) + ");\n", encoding="utf-8")
            completed = self._run("--fasta", str(fasta), "--pairs", str(pairs), "--tree", str(tree), "--out", str(out), "--spa-mode", "off", "--threads", "2", "--no-progress")
            self.assertEqual(completed.returncode, 0, completed.stderr)
            results = pd.read_csv(out / "ko_variation.tsv", sep="\t")
            summary = (out / "run_summary.txt").read_text(encoding="utf-8")
            metadata = pd.read_csv(out / "execution_metadata.tsv", sep="\t")
            checkpoint_exists = (out / ".kovar_checkpoint").exists()
        self.assertEqual(results[["u", "v"]].values.tolist(), [[0, 1]])
        self.assertNotIn("direction", results.columns)
        self.assertNotIn("odds_ratio", results.columns)
        self.assertIn("model\tunordered_logistic_mixed_model_pql_score", summary)
        self.assertIn("checkpoint_enabled\t1", summary)
        self.assertFalse(checkpoint_exists)
        self.assertIn("scan_timing", set(metadata["category"]))
        self.assertIn("pair_table_construction", set(metadata.loc[metadata["category"] == "scan_timing", "metric"]))


if __name__ == "__main__":
    unittest.main()
