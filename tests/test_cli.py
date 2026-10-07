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
        self.assertEqual(args.tree_missing_samples, "error")

    def test_missing_tree_samples_fail_before_pair_loading(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fasta = root / "matrix.fa"
            tree = root / "tree.nwk"
            fasta.write_text(">a\nAC\n>b\nCA\n>missing\nCC\n", encoding="utf-8")
            tree.write_text("(a:1,b:1);", encoding="utf-8")
            completed = self._run("--fasta", str(fasta), "--pairs", str(root / "absent.tsv"),
                                  "--tree", str(tree), "--out", str(root / "out"))
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("Tree is missing 1 FASTA samples", completed.stderr)
        self.assertNotIn("step=read_pairs", completed.stderr)

    def test_subset_matches_explicit_filtered_fasta_and_recalculates_filters(self) -> None:
        sequences = ["CC"] * 30 + ["CA"] * 10 + ["AC"] * 10 + ["AA"] * 30
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fasta, filtered, pairs, tree = [root / name for name in
                                          ("matrix.fa", "filtered.fa", "pairs.tsv", "tree.nwk")]
            records = "".join(f">s{i}\n{seq}\n" for i, seq in enumerate(sequences))
            filtered.write_text(records, encoding="utf-8")
            fasta.write_text(records + "".join(f">missing{i}\nCC\n" for i in range(4)), encoding="utf-8")
            pairs.write_text("u\tv\tcount\n0\t1\t4\n", encoding="utf-8")
            tree.write_text("(" + ",".join(f"s{i}:1" for i in reversed(range(80))) + ");", encoding="utf-8")
            for name, source, options in (("subset", fasta, ["--tree-missing-samples", "drop"]),
                                           ("explicit", filtered, [])):
                completed = self._run("--fasta", str(source), "--pairs", str(pairs),
                                      "--tree", str(tree), "--out", str(root / name),
                                      "--spa-mode", "off", "--no-progress", *options)
                self.assertEqual(completed.returncode, 0, completed.stderr)
            subset = pd.read_csv(root / "subset" / "ko_variation.tsv", sep="\t")
            explicit = pd.read_csv(root / "explicit" / "ko_variation.tsv", sep="\t")
            pd.testing.assert_frame_equal(subset, explicit)
            self.assertEqual(subset.status.tolist(), ["OK"])
            self.assertEqual(subset[["n11", "n10", "n01", "n00"]].values.tolist(), [[30, 10, 10, 30]])
            summary = (root / "subset" / "run_summary.txt").read_text(encoding="utf-8")
            self.assertIn("n_samples\t80\n", summary)
            self.assertIn("n_input_samples\t84\n", summary)
            self.assertIn("n_excluded_samples\t4\n", summary)
            inclusion = pd.read_csv(root / "subset" / "sample_inclusion.tsv", sep="\t")
            self.assertEqual(inclusion.loc[inclusion.status == "excluded_missing_tree", "sample"].tolist(),
                             [f"missing{i}" for i in range(4)])

    def test_checkpoint_identity_separates_sample_policies(self) -> None:
        from ko_variation.cli import parse_args, _checkpoint_identity
        command = ["--fasta", "f", "--pairs", "p", "--tree", "t", "--out", "o"]
        strict = parse_args(command)
        subset = parse_args(command + ["--tree-missing-samples", "drop"])
        kwargs = dict(input_fingerprints={}, n_samples=80, n_loci=2, n_pairs=1, blas_libraries="test")
        strict_id = _checkpoint_identity(strict, **kwargs)
        subset_id = _checkpoint_identity(subset, **kwargs)
        self.assertNotEqual(strict_id, subset_id)
        self.assertEqual(subset_id["settings"]["tree_missing_samples"], "drop")

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
