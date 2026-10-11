"""Binary FASTA export, tree-aligned neighbourhood heatmap and held-out MIC ridge.

This downstream helper reads an already selected network. It never reruns KOVAR
or changes its significance/LD selection. Gene names identify loci, not mechanisms.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re

import numpy as np

from . import __version__
from .amr_report import gene_key, label_tokens, mic_record, present, read_tsv, write_tsv

# Increment when numerical fitting/feature selection changes, to invalidate old folds.
ALGORITHM = "mic-ridge-1"


def log(message):
    print(f"[phenotype] {message}", flush=True)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_write(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def npz_write(path, **arrays):
    temporary = Path(str(path) + ".partial")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def binary_fasta(path):
    """Stream records; repeated contig headers are never accepted as sample IDs."""
    names, rows, name, chunks = [], [], None, []

    def finish():
        sequence = "".join(chunks).upper().encode("ascii")
        values = np.frombuffer(sequence, dtype=np.uint8)
        if not len(values) or np.any((values != ord("A")) & (values != ord("C"))):
            raise ValueError(f"{name}: expected a nonempty A/C binary sequence")
        if rows and len(values) != len(rows[0]):
            raise ValueError(f"{name}: unequal FASTA record length")
        names.append(name)
        rows.append((values == ord("C")).astype(np.uint8))
        if len(rows) % 50 == 0:
            log(f"read {len(rows)} FASTA records")

    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    finish()
                name, chunks = line[1:].split()[0], []
            else:
                if name is None:
                    raise ValueError("Sequence precedes first FASTA header")
                chunks.append(line)
    if name is not None:
        finish()
    if not names or len(names) != len(set(names)):
        raise ValueError("Use sample_ids.fasta with unique, nonempty isolate IDs")
    return names, np.stack(rows)


def export_matrix(args, out, sources):
    identity = {"fasta_sha256": sources["fasta"]["sha256"],
                "locus_map_sha256": sources.get("locus_map", {}).get("sha256"),
                "column_contract": "explicit_map" if args.locus_map else "zero_based_unitig_id"}
    cache, manifest = out / "matrix.npz", out / "matrix.json"
    if cache.exists() and manifest.exists() and json.loads(manifest.read_text()) == identity:
        log("reuse completed binary matrix")
        with np.load(cache, allow_pickle=False) as data:
            samples, X, loci = data["samples"].tolist(), data["X"], data["loci"]
    else:
        samples, X = binary_fasta(args.fasta)
        loci = np.arange(X.shape[1], dtype=np.int64)
        if args.locus_map:
            records = read_tsv(args.locus_map, ["fasta_column", "locus"])
            columns = {int(row["fasta_column"]): int(row["locus"]) for row in records}
            if len(columns) != len(records) or set(columns) != set(range(X.shape[1])):
                raise ValueError("Locus map must cover every zero-based FASTA column exactly once")
            loci = np.array([columns[i] for i in range(X.shape[1])], dtype=np.int64)
        if np.any(loci < 0) or len(set(loci.tolist())) != len(loci):
            raise ValueError("Original locus IDs must be unique and nonnegative")
        npz_write(cache, samples=np.asarray(samples), X=X, loci=loci)
        # Remove a stale TSV before recording a newly converted matrix.
        (out / "unitig_presence_absence.tsv.gz").unlink(missing_ok=True)
        json_write(manifest, identity)
        write_tsv(out / "column_mapping.tsv",
                  ({"fasta_column": i, "locus": int(locus)} for i, locus in enumerate(loci)),
                  ["fasta_column", "locus"])
    if not args.no_full_tsv and not (out / "unitig_presence_absence.tsv.gz").exists():
        target = out / "unitig_presence_absence.tsv.gz"
        temporary = Path(str(target) + ".partial")
        # Low compression avoids spending hours exporting a large text matrix.
        with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=1) as handle:
            handle.write("sample\t" + "\t".join(f"unitig_{i}" for i in loci) + "\n")
            for i, (sample, row) in enumerate(zip(samples, X), 1):
                handle.write(sample + "\t" + "\t".join(np.where(row, "1", "0")) + "\n")
                if i % 50 == 0 or i == len(samples):
                    log(f"full TSV export {i}/{len(samples)}")
        temporary.replace(target)
    log(f"matrix: {len(samples)} isolates x {X.shape[1]} unitigs")
    return samples, X, loci


def neighbourhood(args, loci, out):
    """Join locus IDs to genes explicitly: sorted gene endpoints need not match u/v."""
    source = Path(args.network).read_text(encoding="utf-8")
    match = re.search(r'<script id="data" type="application/json">(.*?)</script>', source, re.S)
    network = json.loads(match[1] if match else source)
    nodes = {node["gene"]: node for node in network["nodes"]}
    target = [gene for gene, node in nodes.items()
              if gene == args.target or gene_key(args.target) in label_tokens(
                  node.get("display_label", node.get("label", gene)))]
    if len(target) != 1:
        raise ValueError(f"Target must resolve to one cluster; matches: {target}. Use its cluster ID if needed")
    target = target[0]
    adjacency = {gene: set() for gene in nodes}
    represented = set()
    for edge in network["edges"]:
        a, b = edge["gene_a"], edge["gene_b"]
        adjacency[a].add(b)
        adjacency[b].add(a)
        for pair in json.loads(edge["locus_pairs"]):
            represented.update(map(int, pair))
    first = adjacency[target] - {target}
    second = set().union(*(adjacency[g] for g in first)) - first - {target} if first else set()
    if not first or not second:
        raise ValueError("The four-family comparison requires both direct and second-order neighbours")
    order = {target: 0, **{g: 1 for g in first}, **{g: 2 for g in second}}
    available = {int(locus): column for column, locus in enumerate(loci)}
    annotation, features, pool, seen = read_tsv(args.annotation, ["locus", "gene", "annotation_class"]), [], [], set()
    for row in annotation:
        locus = int(row["locus"])
        if locus in seen:
            raise ValueError(f"Duplicate annotation for locus {locus}")
        seen.add(locus)
        if locus not in represented or locus not in available:
            continue
        if "coding" not in {c.strip() for c in row["annotation_class"].split(";")}:
            continue
        gene = row["gene"]
        if gene in order:
            features.append(dict(locus=locus, fasta_column=available[locus], gene=gene,
                                 label=nodes[gene].get("display_label", gene),
                                 neighbourhood_order=order[gene], annotation_class=row["annotation_class"]))
        elif gene in nodes:
            pool.append(available[locus])
    # Distinct IDs may later collapse to identical training-set genotype patterns.
    features.sort(key=lambda r: (r["neighbourhood_order"], r["label"], r["locus"]))
    for gene in order:
        if not any(row["gene"] == gene for row in features):
            raise ValueError(f"No coding-overlap, original-ID-matched unitigs for {gene}")
    write_tsv(out / "features.tsv", features, list(features[0]))
    if args.random_pool:
        requested = {int(row["locus"]) for row in read_tsv(args.random_pool, ["locus"])}
        if not requested.issubset(available):
            raise ValueError("Random pool includes locus IDs absent from the FASTA")
        excluded = {available[int(row["locus"])] for row in annotation if row["gene"] in order
                    and int(row["locus"]) in available}
        pool = sorted({available[locus] for locus in requested} - excluded)
    if not pool:
        raise ValueError("No random-control background unitigs remain")
    json_write(out / "neighbourhood.json", dict(target=target, first=sorted(first), second=sorted(second),
               unitig_definition="binary exact-sequence presence; not a named resistance mutation",
               pool_definition="supplied locus pool" if args.random_pool else
               "coding-overlap unitigs represented by other nodes in the selected distal map",
               distance_to_target="not certified by this helper; preserve upstream selection provenance"))
    return features, np.asarray(pool, dtype=int)


def phenotype_rows(args, samples, out):
    metadata = read_tsv(args.metadata, ["id", "isolate", args.mic_column, args.sign_column])
    if args.sample_map:
        mapping = read_tsv(args.sample_map, ["metadata_id", "sample"])
        lookup = {r["metadata_id"]: r["sample"] for r in mapping}
        if len(lookup) != len(mapping) or len(set(lookup.values())) != len(mapping):
            raise ValueError("Sample mapping must be one-to-one")
    else:
        # Matches the established ARC assembly basename; save the exact join below.
        lookup = {r["id"]: r["id"] + "_" + r["isolate"].replace(".", "_") for r in metadata}
    by_sample = {}
    for row in metadata:
        sample = lookup.get(row["id"])
        if sample is not None:
            if sample in by_sample:
                raise ValueError(f"Duplicate metadata mapping to {sample}")
            by_sample[sample] = row
    result = []
    for sample in samples:
        row = by_sample.get(sample)
        value = row.get(args.mic_column, "") if row else ""
        sign = row.get(args.sign_column, "") if row else ""
        number, status = mic_record(value, sign) if present(value) else (None, "missing_mic")
        if not row:
            status = "missing_metadata"
        result.append(dict(sample=sample, metadata_id=row["id"] if row else "", mic=value, sign=sign,
                           status=status, log2_mic=float(np.log2(number)) if status == "exact" else None))
    if not any(r["metadata_id"] for r in result):
        raise ValueError("No metadata/sample match; supply --sample-map")
    write_tsv(out / "phenotypes.tsv", result, list(result[0]))
    extra = [dict(sample=sample, metadata_id=row["id"], status="metadata_without_fasta")
             for sample, row in by_sample.items() if sample not in set(samples)]
    write_tsv(out / "unmatched_metadata.tsv", extra, ["sample", "metadata_id", "status"])
    return result


def tree_and_groups(args, samples, out):
    from Bio import Phylo
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    tree = Phylo.read(args.tree, "newick")
    tips = tree.get_terminals()
    names = [tip.name for tip in tips]
    if len(names) != len(set(names)) or any(name is None for name in names):
        raise ValueError("Tree tips must have unique isolate names")
    if set(samples) != set(names):
        raise ValueError(f"Tree/FASTA mismatch: {len(set(samples)-set(names))} FASTA-only, "
                         f"{len(set(names)-set(samples))} tree-only. Supply a deliberately matched tree/FASTA.")
    for node in tree.find_clades():
        if node is not tree.root and (node.branch_length is None or
                not np.isfinite(node.branch_length) or node.branch_length < 0):
            raise ValueError("Midpoint rooting requires finite nonnegative branch lengths")
    tree.root_at_midpoint()
    tree.ladderize()
    Phylo.write(tree, out / "midpoint.treefile", "newick", format_branch_length="%.12g")
    tips = tree.get_terminals()
    paths = {tip.name: [tree.root] + tree.get_path(tip) for tip in tips}
    depths = tree.depths()
    D = np.zeros((len(samples), len(samples)))
    for i, a in enumerate(samples):
        for j in range(i):
            b = samples[j]
            common = tree.root
            for u, v in zip(paths[a], paths[b]):
                if u is not v:
                    break
                common = u
            D[i, j] = D[j, i] = max(0., depths[paths[a][-1]] + depths[paths[b][-1]] - 2 * depths[common])
    # Complete-linkage groups depend only on patristic distances, never MIC.
    groups = fcluster(linkage(squareform(D), method="complete"),
                      t=min(args.groups, len(samples)), criterion="maxclust")
    write_tsv(out / "tree_order.tsv", ({"sample": tip.name, "tree_row": i,
              "phylogenetic_group": int(groups[samples.index(tip.name)])} for i, tip in enumerate(tips)),
              ["sample", "tree_row", "phylogenetic_group"])
    log(f"midpoint-rooted tree: {len(tips)} tips; {len(set(groups))} phylogenetic groups")
    return tree, groups


def save_figure(fig, prefix, dpi):
    fig.savefig(str(prefix) + ".png", dpi=dpi, facecolor="white", bbox_inches="tight")
    fig.savefig(str(prefix) + ".svg", facecolor="white", bbox_inches="tight")


def tree_plot(tree, samples, X, features, phenotypes, groups, out, dpi):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.patches import Patch

    plt.rcParams["svg.fonttype"] = "none"
    tips = tree.get_terminals()
    idx = {sample: i for i, sample in enumerate(samples)}
    rows = [idx[tip.name] for tip in tips]
    cols = [f["fasta_column"] for f in features]
    fig, axes = plt.subplots(1, 4, figsize=(12, max(5, len(tips) * .025)),
                             gridspec_kw={"width_ratios": [4, 5, .55, .55], "wspace": .12})
    y = {tip: i for i, tip in enumerate(tips)}
    for node in tree.get_nonterminals(order="postorder"):
        y[node] = (y[node.clades[0]] + y[node.clades[-1]]) / 2
    depths = tree.depths()
    for node in tree.find_clades():
        for child in node.clades:
            axes[0].plot([depths[node], depths[child]], [y[child], y[child]], color="black", lw=.45)
        if node.clades:
            axes[0].plot([depths[node]] * 2, [y[node.clades[0]], y[node.clades[-1]]], color="black", lw=.45)
    axes[0].set_xlabel("Substitutions per site")
    axes[0].set_title("Midpoint-rooted core tree", fontsize=10)
    # Vector cells keep the heatmap itself editable in SVG, not only its labels.
    boundaries = np.arange(len(tips)+1) - .5
    axes[1].pcolormesh(np.arange(len(cols)+1)-.5, boundaries, X[np.ix_(rows, cols)],
                       cmap=ListedColormap(["white", "#222222"]), vmin=0, vmax=1,
                       shading="flat", rasterized=False)
    axes[1].set_xticks(range(len(features)),
                      [f"{f['label']} | {f['locus']}" for f in features], rotation=90, fontsize=7)
    axes[1].xaxis.tick_top()
    axes[1].set_xlabel("0: mprF   1: direct   2: second-order", fontsize=8)
    for i in range(1, len(features)):
        if features[i]["neighbourhood_order"] != features[i-1]["neighbourhood_order"]:
            axes[1].axvline(i-.5, color="#C62828", lw=1)
    axes[1].legend(handles=[Patch(facecolor="white", edgecolor="black", label="Absent"),
                            Patch(facecolor="#222222", label="Present")], loc="lower center",
                   bbox_to_anchor=(.5, -.065), ncol=2, frameon=False, fontsize=8)
    values = np.array([p["log2_mic"] if p["status"] == "exact" else np.nan for p in phenotypes])
    cmap = plt.get_cmap("Reds").copy()
    cmap.set_bad("#CCCCCC")
    image = axes[2].pcolormesh([-.5, .5], boundaries, np.ma.masked_invalid(values[rows, None]),
                              cmap=cmap, shading="flat", rasterized=False)
    censored = [i for i, row in enumerate(rows) if phenotypes[row]["status"] == "censored"]
    if censored:
        axes[2].scatter(np.zeros(len(censored)), censored, s=3, marker="x", color="black", lw=.4)
    axes[2].set_title("log₂ MIC", fontsize=8)
    # An ordinary colourbar shrinks just the MIC axes and breaks tip/row alignment.
    bar = axes[2].inset_axes([0, -.045, 1, .012])
    fig.colorbar(image, cax=bar, orientation="horizontal")
    palette = plt.get_cmap("tab20", int(max(groups)))
    axes[3].pcolormesh([-.5, .5], boundaries, groups[rows, None], cmap=palette,
                      norm=BoundaryNorm(np.arange(.5, max(groups)+1.5), int(max(groups))),
                      shading="flat", rasterized=False)
    axes[3].set_title("Group", fontsize=8)
    for ax in axes:
        ax.set_ylim(len(tips)-.5, -.5)
        ax.set_yticks([])
        if ax is not axes[0]:
            ax.tick_params(bottom=False, labelbottom=False)
    fig.text(.5, .005, "Grey MIC track: missing or censored; ×: censored. Rows follow tree tips, not MIC order.",
             ha="center", fontsize=8)
    save_figure(fig, out / "tree_presence_absence", dpi)
    plt.close(fig)


def retained_columns(X, columns, minimum):
    """Fit the presence filter and exact-duplicate collapse on training rows only."""
    keep, seen = [], set()
    for column in columns:
        vector = X[:, column]
        count = int(vector.sum())
        key = vector.tobytes()
        if min(count, len(vector)-count) >= minimum and key not in seen:
            keep.append(int(column))
            seen.add(key)
    return keep


def ridge_fit(X, y, columns, alpha, minimum):
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    keep = retained_columns(X, columns, minimum)
    if not keep:
        return keep, None, float(y.mean())
    scaler = StandardScaler().fit(X[:, keep])
    ridge = Ridge(alpha=alpha).fit(scaler.transform(X[:, keep]), y)
    return keep, scaler, ridge


def ridge_predict(fitted, X):
    keep, scaler, model = fitted
    return np.full(len(X), model) if scaler is None else model.predict(scaler.transform(X[:, keep]))


def tune_fit(X, y, groups, columns, args):
    from sklearn.model_selection import GroupKFold
    n = min(3, len(set(groups)))
    if n < 2:
        raise ValueError("At least two training phylogenetic groups are needed for inner tuning")
    splits = list(GroupKFold(n_splits=n).split(X, y, groups))
    # Inner fitting touches only this small model, never copies the whole genome.
    local = X[:, columns]
    local_columns = list(range(len(columns)))
    losses = []
    for alpha in args.alphas:
        squared = []
        for train, test in splits:
            fit = ridge_fit(local[train], y[train], local_columns, alpha, args.min_count)
            squared.extend((y[test] - ridge_predict(fit, local[test])) ** 2)
        losses.append(float(np.mean(squared)))
    alpha = args.alphas[int(np.argmin(losses))]
    return ridge_fit(X, y, columns, alpha, args.min_count), alpha


def random_columns(X, base, added, pool, minimum, rng):
    """Greedy nearest minor-frequency match, without duplicate training patterns."""
    candidates = retained_columns(X, pool, minimum)
    used = {X[:, c].tobytes() for c in base}
    candidates = [c for c in candidates if X[:, c].tobytes() not in used]
    result, mismatches = [], []
    frequencies = X.mean(axis=0)
    frequencies = np.minimum(frequencies, 1-frequencies)
    for source in rng.permutation(added):
        if not candidates:
            raise ValueError("Random background has too few unique variable genotype patterns")
        differences = np.abs(frequencies[candidates] - frequencies[source])
        nearest = np.flatnonzero(np.isclose(differences, differences.min()))
        index = int(rng.choice(nearest))
        selected = candidates.pop(index)
        result.append(selected)
        mismatches.append(float(differences[index]))
    return result, mismatches


def scores(observed, predicted):
    error = observed - predicted
    return dict(rmse=float(np.sqrt(np.mean(error**2))),
                within_one_dilution=float(np.mean(np.abs(error) <= 1)))


def predict_mic(args, samples, X, loci, features, pool, phenotype, groups, out, identity):
    from sklearn.model_selection import GroupKFold
    from threadpoolctl import threadpool_limits
    indices = np.array([i for i, p in enumerate(phenotype) if p["status"] == "exact"])
    if len(indices) < 20:
        raise ValueError("Fewer than 20 exact MIC measurements; do not run this comparison")
    y = np.array([phenotype[i]["log2_mic"] for i in indices])
    # Keep the full exported matrix on disk, but fit only target/background columns.
    needed = sorted(set(pool.tolist()) | {f["fasta_column"] for f in features})
    remap = {column: i for i, column in enumerate(needed)}
    features = [dict(f, fasta_column=remap[f["fasta_column"]]) for f in features]
    pool = np.array([remap[c] for c in pool], dtype=int)
    loci = loci[needed]
    data, group = X[np.ix_(indices, needed)], groups[indices]
    nfolds = min(args.folds, len(set(group)))
    if nfolds < 3:
        raise ValueError("Need at least three phenotype-bearing phylogenetic groups")
    splits = list(GroupKFold(n_splits=nfolds).split(data, y, group))
    model_columns = {f"M{i}": [f["fasta_column"] for f in features if f["neighbourhood_order"] <= i]
                     for i in range(3)}
    labels = ["Mean_only", "M0", "M1", "M2"] + [f"Random{i}_{r:03d}" for i in (1, 2)
               for r in range(1, args.random_sets+1)]
    prediction = np.full((len(labels), len(y)), np.nan)
    fold_rows, training_rows, feature_rows, outer_fold = [], [], [], np.zeros(len(y), dtype=int)
    checkpoint = out / "ridge_checkpoints"
    checkpoint.mkdir(exist_ok=True)
    for number, (train, test) in enumerate(splits, 1):
        outer_fold[test] = number
        meta, arrays = checkpoint / f"fold_{number}.json", checkpoint / f"fold_{number}.npz"
        if meta.exists() and arrays.exists() and json.loads(meta.read_text())["identity"] == identity:
            saved = json.loads(meta.read_text())
            with np.load(arrays, allow_pickle=False) as cached:
                if not np.array_equal(cached["test"], test):
                    raise ValueError("Checkpoint partition differs from current split")
                prediction[:, test] = cached["predictions"]
            fold_rows.extend(saved["fold_rows"])
            training_rows.extend(saved["training_rows"])
            feature_rows.extend(saved["feature_rows"])
            log(f"reuse completed ridge fold {number}/{nfolds}")
            continue
        log(f"ridge fold {number}/{nfolds}: train={len(train)}, test={len(test)}")
        saved_fold, saved_train, saved_features = [], [], []
        training, target = data[train], y[train]
        keep = {model: retained_columns(training, columns, args.min_count)
                for model, columns in model_columns.items()}
        added = {i: [c for c in keep[f"M{i}"] if c not in keep["M0"]] for i in (1, 2)}
        with threadpool_limits(limits=1):
            for m, label in enumerate(labels):
                mismatch = []
                if label == "Mean_only":
                    columns, alpha = [], None
                    fitted = ([], None, float(target.mean()))
                else:
                    if label.startswith("Random"):
                        size, replicate = map(int, label.removeprefix("Random").split("_"))
                        rng = np.random.default_rng(np.random.SeedSequence([args.seed, number, size, replicate]))
                        extra, mismatch = random_columns(training, keep["M0"], added[size], pool, args.min_count, rng)
                        columns = keep["M0"] + extra
                    else:
                        columns = model_columns[label]
                    fitted, alpha = tune_fit(training, target, group[train], columns, args)
                prediction[m, test] = ridge_predict(fitted, data[test])
                nfeatures = len(fitted[0])
                saved_fold.append(dict(fold=number, model=label, n_test=len(test), n_features=nfeatures,
                                      alpha=alpha, max_frequency_mismatch=max(mismatch, default=0.),
                                      **scores(y[test], prediction[m, test])))
                saved_train.append(dict(fold=number, model=label, n_train=len(train),
                                        **scores(target, ridge_predict(fitted, training))))
                saved_features.append(dict(fold=number, model=label, alpha=alpha,
                    candidate_loci=";".join(str(int(loci[c])) for c in columns),
                    retained_loci=";".join(str(int(loci[c])) for c in fitted[0])))
                if label.startswith("Random") and int(label.split("_")[1]) % 20 == 0:
                    log(f"fold {number}: {label} complete")
        npz_write(arrays, test=test, predictions=prediction[:, test])
        json_write(meta, dict(identity=identity, fold_rows=saved_fold,
                              training_rows=saved_train, feature_rows=saved_features))
        fold_rows.extend(saved_fold)
        training_rows.extend(saved_train)
        feature_rows.extend(saved_features)
    if not np.isfinite(prediction).all():
        raise ValueError("Incomplete held-out predictions")
    baseline = scores(y, prediction[labels.index("M0")])["rmse"]
    performance = [dict(model=label, n_test=len(y), **scores(y, p),
                        rmse_reduction_vs_M0=baseline-scores(y, p)["rmse"])
                   for label, p in zip(labels, prediction)]
    for name, records in (("performance", performance), ("fold_performance", fold_rows),
                          ("training_performance", training_rows), ("model_features", feature_rows)):
        write_tsv(out / f"{name}.tsv", records, list(records[0]))
    write_tsv(out / "predictions.tsv", (dict(sample=samples[indices[i]], fold=int(outer_fold[i]),
              model=label, observed_log2_mic=float(y[i]), predicted_log2_mic=float(prediction[m, i]))
              for m, label in enumerate(labels) for i in range(len(y))),
              ["sample", "fold", "model", "observed_log2_mic", "predicted_log2_mic"])
    prediction_plots(args, y, prediction, labels, performance, fold_rows, out)
    log(f"ridge complete: {len(y)} exact MICs, {nfolds} grouped folds")


def prediction_plots(args, y, predictions, labels, performance, folds, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["svg.fonttype"] = "none"
    by_name = {r["model"]: r for r in performance}
    fig, ax = plt.subplots(figsize=(7, 4))
    for x, name, colour in ((0, "M0", "black"), (1, "M1", "#C62828"), (2, "M2", "#C62828")):
        ax.scatter(np.full(len([r for r in folds if r["model"] == name]), x),
                   [r["rmse"] for r in folds if r["model"] == name], s=15, c=colour, alpha=.35)
        ax.scatter(x, by_name[name]["rmse"], s=55, color=colour, marker="D", zorder=3)
    for x, size in ((3, 1), (4, 2)):
        values = [r["rmse"] for r in performance if r["model"].startswith(f"Random{size}_")]
        ax.boxplot(values, positions=[x], widths=.5, patch_artist=True,
                   boxprops=dict(facecolor="#CCCCCC"), medianprops=dict(color="black"))
    ax.axhline(by_name["Mean_only"]["rmse"], color="black", ls=":", label="Training-mean benchmark")
    ax.set_xticks(range(5), [args.target, "+ direct", "+ direct\n+ second-order", "Random\n(direct size)",
                            "Random\n(extended size)"])
    ax.set_ylabel("Held-out RMSE (log₂ MIC units)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save_figure(fig, out / "ridge_performance", args.dpi)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), sharex=True, sharey=True)
    jitter = np.random.default_rng(args.seed).uniform(-.025, .025, len(y))
    lower = min(float(y.min()), float(predictions[1:4].min())) - .15
    upper = max(float(y.max()), float(predictions[1:4].max())) + .15
    for ax, model, colour in zip(axes, ["M0", "M1", "M2"], ["black", "#C62828", "#C62828"]):
        ax.scatter(y+jitter, predictions[labels.index(model)], s=9, alpha=.35, color=colour)
        ax.plot([lower, upper], [lower, upper], "k--", lw=.8)
        ax.set(xlim=(lower, upper), ylim=(lower, upper), xlabel="Observed log₂ MIC")
        ax.text(.04, .96, f"{model}; RMSE={by_name[model]['rmse']:.3f}", va="top", transform=ax.transAxes, fontsize=8)
    axes[0].set_ylabel("Predicted log₂ MIC")
    fig.tight_layout()
    save_figure(fig, out / "observed_predicted", args.dpi)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("fasta", "tree", "metadata", "network", "annotation", "out"):
        parser.add_argument("--" + name, required=True, type=Path)
    contract = parser.add_mutually_exclusive_group(required=True)
    contract.add_argument("--fasta-columns-are-locus-ids", action="store_true",
                          help="Declare unfiltered FASTA column 0 == original unitig 0")
    contract.add_argument("--locus-map", type=Path, help="TSV: zero-based fasta_column, original locus")
    parser.add_argument("--target", default="mprF")
    parser.add_argument("--mic-column", default="daptomycin_mic")
    parser.add_argument("--sign-column", default="daptomycin_mic_sign")
    parser.add_argument("--sample-map", type=Path, help="Override id_isolate basename join: metadata_id, sample")
    parser.add_argument("--random-pool", type=Path, help="Optional independently defined eligible locus pool")
    parser.add_argument("--groups", type=int, default=10, help="Maximum complete-linkage patristic groups")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--random-sets", type=int, default=100)
    parser.add_argument("--min-count", type=int, default=5, help="Minimum present AND absent training isolates")
    parser.add_argument("--alphas", type=float, nargs="+", default=[.01, .1, 1, 10, 100, 1000])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--no-full-tsv", action="store_true", help="Cache full matrix; export only neighbourhood TSV")
    parser.add_argument("--stage", choices=["all", "tree"], default="all", help="tree skips ridge fitting")
    args = parser.parse_args(argv)
    if args.groups < 3 or args.folds < 3 or args.random_sets < 1 or args.min_count < 1 or args.dpi < 72:
        parser.error("groups/folds >=3, random-sets/min-count >=1, dpi >=72 required")
    if args.seed < 0 or any(not np.isfinite(a) or a <= 0 for a in args.alphas):
        parser.error("seed must be nonnegative; ridge alphas must be finite and positive")
    sources = {}
    for name in ("fasta", "tree", "metadata", "network", "annotation", "locus_map", "sample_map", "random_pool"):
        path = getattr(args, name)
        if path:
            if not path.is_file():
                parser.error(f"Missing {name}: {path}")
            sources[name] = dict(path=str(path.resolve()), sha256=sha256(path))
    args.out.mkdir(parents=True, exist_ok=True)
    versions = {p: importlib.metadata.version(p) for p in
                ("numpy", "scipy", "scikit-learn", "biopython", "matplotlib")}
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    # Package versions are part of restart identity, so a changed environment cannot reuse fits.
    analysis_settings = {k: v for k, v in settings.items()
                         if k not in {"out", "dpi", "stage", "no_full_tsv"}}
    identity = hashlib.sha256(json.dumps(dict(algorithm=ALGORITHM, sources=sources,
                  settings=analysis_settings, versions=versions), sort_keys=True).encode()).hexdigest()
    summary = dict(status="started", algorithm=ALGORITHM, version=__version__, sources=sources,
                   settings=settings, packages=versions, identity=identity,
                   evaluation="fixed cohort-derived network; exact-MIC grouped held-out prediction",
                   network_discovery="not repeated inside folds; not independent end-to-end discovery validation",
                   gene_variation="exact unitig-sequence presence; named substitutions not inferred",
                   censoring="only explicit = measurements used for ridge; boundaries never treated as exact")
    json_write(args.out / "run.json", summary)
    samples, X, loci = export_matrix(args, args.out, sources)
    features, pool = neighbourhood(args, loci, args.out)
    write_tsv(args.out / "neighbourhood_presence_absence.tsv",
              (dict(sample=sample, **{f"unitig_{f['locus']}": int(X[i, f['fasta_column']]) for f in features})
               for i, sample in enumerate(samples)), ["sample"] + [f"unitig_{f['locus']}" for f in features])
    phenotype = phenotype_rows(args, samples, args.out)
    tree, groups = tree_and_groups(args, samples, args.out)
    tree_plot(tree, samples, X, features, phenotype, groups, args.out, args.dpi)
    if args.stage == "all":
        predict_mic(args, samples, X, loci, features, pool, phenotype, groups, args.out, identity)
    summary.update(status="complete", n_samples=len(samples), n_unitigs=len(loci), n_features=len(features),
                   phenotype_counts={s: sum(p["status"] == s for p in phenotype)
                                     for s in sorted({p["status"] for p in phenotype})})
    json_write(args.out / "run.json", summary)
    (args.out / "figure_captions.txt").write_text(
        "Tree heatmap. A midpoint-rooted copy of the core-genome tree is aligned with binary "
        "unitig-sequence presence, exact log2 MIC and phylogenetic group tracks. Coding-overlap "
        "unitigs are assigned through the locus annotation table, not the order of gene-edge endpoints. "
        "Grey MIC cells are missing or censored; crosses identify censored records. "
        "Midpoint rooting is a display convention, not evidence of ancestral states.\n\n"
        "Ridge comparison. Diamonds show pooled held-out RMSE; small dots show individual grouped-fold "
        "RMSE and are not independent replicates. Grey boxes summarize random-set RMSE, not confidence "
        "intervals. The dotted line is the held-out training-mean benchmark. Each random family matches "
        "the number of added retained predictors and uses nearest training minor-frequency matching; "
        "deviations are recorded in fold_performance.tsv. Default controls are other coding unitigs "
        "represented in the selected map; their physical distance to mprF is not independently verified. "
        "All models use identical held-out isolates and explicit exact MIC measurements only.\n\n"
        "Observed versus predicted. Predictions are held out by phylogenetic group. Horizontal jitter "
        "is display-only. The diagonal marks equality. MICs and predictions are expressed in log2 units. "
        "The existing network was discovered from the cohort and frozen before modelling; this is "
        "not an independent end-to-end evaluation of network discovery. Improved prediction does not "
        "establish mechanistic epistasis.\n", encoding="utf-8")
    log(f"outputs: {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
