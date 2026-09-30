from __future__ import annotations
import json
from importlib.util import find_spec
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd
from ko_variation.postprocess import SelectionConfig, select_distal_signals
from ko_variation.network import build_gene_network, export_gene_network
from ko_variation.scan import ScanConfig, scan_pairs_glmm

HAS_EXPORT_DEPS = find_spec("matplotlib") is not None and find_spec("networkx") is not None


def results():
    return pd.DataFrame(dict(u=[0, 0, 1, 2], v=[1, 2, 3, 3],
                             status=["OK", "OK", "OK_SPA_FAILED", "LOW_CELL_COUNT"],
                             q_bh=[0.01, 0.05, 0.001, np.nan], p_primary=[0.001, 0.005, 0.0001, np.nan],
                             distance=[20000, 10000, 30000, 90000],
                             n11=[30, 20, 10, 0], n10=[10, 20, 30, 40],
                             n01=[10, 20, 30, 40], n00=[30, 20, 10, 0]))


def annotation():
    return pd.DataFrame(dict(locus=[0, 1, 2, 3], gene=["A", "B", "A", "A"],
                             contig=["chr"] * 4, position=[0, 20000, 10000, 50000],
                             label=["alpha", "beta", "alpha", "alpha"], group=["one", "two", "one", "one"]))


class SelectionTests(unittest.TestCase):
    def test_boundaries_and_raw_effect(self):
        frame = results()
        selected = select_distal_signals(frame)
        self.assertEqual(selected[["u", "v"]].values.tolist(), [[0, 1], [1, 3]])
        self.assertAlmostEqual(selected.raw_odds_ratio.iloc[0], 9)
        self.assertAlmostEqual(selected.raw_odds_ratio.iloc[1], 1 / 9)
        self.assertAlmostEqual(selected.raw_log_or_se.iloc[0], np.sqrt(2/30 + 2/10))
        self.assertTrue((selected.raw_or_ci_low < selected.raw_odds_ratio).all())
        self.assertEqual(len(select_distal_signals(frame, SelectionConfig(ld_distance=9999))), 3)
        self.assertNotIn("raw_odds_ratio", frame)

    def test_zero_cells_and_uncorrected_boundary(self):
        frame = results().iloc[:1].copy()
        frame["n10"] = 0
        selected = select_distal_signals(frame)
        self.assertAlmostEqual(selected.raw_odds_ratio.iloc[0], 30.5*30.5/(0.5*10.5))
        self.assertEqual(selected.raw_or_correction.iloc[0], 0.5)
        selected = select_distal_signals(frame, SelectionConfig(zero_cell_correction=0))
        self.assertTrue(np.isinf(selected.raw_odds_ratio.iloc[0]))
        self.assertTrue(np.isnan(selected.raw_or_ci_low.iloc[0]))

    def test_missing_distance_and_coordinates_cross_contig(self):
        frame = results().drop(columns="distance")
        ann = annotation()
        self.assertEqual(len(select_distal_signals(frame, annotation=ann)), 2)
        ann.loc[1, "contig"] = "plasmid"
        self.assertTrue(select_distal_signals(frame, annotation=ann).empty)
        selected = select_distal_signals(frame, SelectionConfig(cross_contig="distal"), ann)
        self.assertEqual(len(selected), 2)
        self.assertTrue(selected.physical_distance.isna().all())
        self.assertTrue(selected.distance_class.eq("cross_contig").all())
        frame = results(); frame["distance"] = np.nan
        self.assertTrue(select_distal_signals(frame).empty)

    def test_invalid_inputs(self):
        for cfg in [SelectionConfig(ld_distance=-1), SelectionConfig(significance_threshold=2),
                    SelectionConfig(confidence=1), SelectionConfig(zero_cell_correction=-1)]:
            with self.assertRaises(ValueError):
                select_distal_signals(results(), cfg)
        for column, value in [("u", 0.5), ("n00", -1), ("distance", -1), ("q_bh", 2)]:
            frame = results().astype({column: float}); frame.loc[0, column] = value
            with self.assertRaises(ValueError):
                select_distal_signals(frame)
        with self.assertRaisesRegex(ValueError, "Missing columns"):
            select_distal_signals(results().drop(columns="n00"))
        with self.assertRaisesRegex(ValueError, "unique canonical"):
            select_distal_signals(pd.concat([results(), results()]))
        ann = pd.concat([annotation(), annotation()])
        with self.assertRaisesRegex(ValueError, "duplicate locus"):
            select_distal_signals(results(), annotation=ann)

    def test_actual_scanner_schema(self):
        matrix = np.array([[1, 1]]*30 + [[1, 0]]*10 + [[0, 1]]*10 + [[0, 0]]*30, dtype=np.uint8)
        frame, _ = scan_pairs_glmm(pd.DataFrame(dict(u=[0], v=[1], distance=[20001])), matrix,
                                  np.zeros((80, 80)), ScanConfig(progress=False, spa_mode="off"))
        selected = select_distal_signals(frame)
        self.assertEqual(len(selected), 1)
        self.assertAlmostEqual(selected.raw_odds_ratio.iloc[0], 9)
        self.assertNotIn("phylogeny_adjusted_odds_ratio", selected)


class NetworkTests(unittest.TestCase):
    def test_aggregation_mixed_direction_and_representative(self):
        signals = select_distal_signals(results(), annotation=annotation())
        nodes, edges = build_gene_network(signals)
        self.assertEqual(len(nodes), 2)
        self.assertEqual(len(edges), 1)
        edge = edges.iloc[0]
        self.assertEqual(edge.n_locus_pairs, 2)
        self.assertEqual(edge.raw_direction, "mixed")
        self.assertEqual(edge.representative_u, 1)
        self.assertAlmostEqual(edge.representative_raw_odds_ratio, 1 / 9)
        self.assertEqual(json.loads(edge.locus_pairs), [[1, 3], [0, 1]])
        self.assertEqual(nodes.degree.tolist(), [1, 1])
        self.assertEqual(nodes.n_loci.tolist(), [2, 1])

    def test_missing_annotations_and_self_edges(self):
        signals = select_distal_signals(results())
        ann = annotation().iloc[:2]
        with self.assertRaisesRegex(ValueError, "Missing gene"):
            build_gene_network(signals, ann)
        nodes, edges = build_gene_network(signals, ann, missing_genes="drop")
        self.assertEqual(len(edges), 1)
        ann = annotation(); ann["gene"] = "A"
        self.assertTrue(build_gene_network(signals, ann)[1].empty)
        self.assertEqual(len(build_gene_network(signals, ann, include_self=True)[1]), 1)

    @unittest.skipUnless(HAS_EXPORT_DEPS, "Optional network export dependencies not installed")
    def test_exports_empty_nonempty_and_html_escaping(self):
        ann = annotation(); ann.loc[0, "label"] = "</script><img src=x onerror=alert(1)>"
        signals = select_distal_signals(results(), annotation=ann)
        nodes, edges = build_gene_network(signals)
        with tempfile.TemporaryDirectory() as tmp:
            for empty in (False, True):
                paths = export_gene_network(nodes.iloc[:0] if empty else nodes, edges.iloc[:0] if empty else edges,
                                            Path(tmp) / str(empty), dpi=72)
                self.assertTrue(paths["png"].read_bytes().startswith(b"\x89PNG"))
                document = paths["html"].read_text(encoding="utf-8")
                self.assertNotIn("</script><img", document)
                self.assertNotIn('src="https://', document)
                self.assertIn("pointermove", document)
                self.assertIn("application/json", document)
                if not empty:
                    self.assertIn("\\u003c/script\\u003e", document)

    @unittest.skipUnless(HAS_EXPORT_DEPS, "Optional network export dependencies not installed")
    def test_cli_pipeline_and_invalid_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            results().to_csv(root / "results.tsv", sep="\t", index=False)
            annotation().to_csv(root / "annotation.tsv", sep="\t", index=False)
            command = [sys.executable, "-m", "ko_variation.annotation_cli", "--results", str(root / "results.tsv"),
                       "--annotation", str(root / "annotation.tsv"), "--out", str(root / "output"), "--network", "--dpi", "72"]
            completed = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(len(pd.read_csv(root / "output.distal.tsv", sep="\t")), 2)
            self.assertEqual(json.loads((root / "output.selection.json").read_text())["adjusted_effects"], "not_estimated")
            completed = subprocess.run(command + ["--ld-distance", "-1"], capture_output=True, text=True)
            self.assertEqual(completed.returncode, 2)


if __name__ == "__main__":
    unittest.main()
