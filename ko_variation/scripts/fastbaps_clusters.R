#!/usr/bin/env Rscript
# Official fastbaps API: infer one partition from cleaned, genuine aligned DNA.
# No MIC input and no requested final cluster count. Defaults follow the upstream
# optimised-symmetric example; k.init remains the package's recorded default.
argv <- commandArgs(trailingOnly = TRUE)
if (length(argv) %% 2 != 0) stop("Expected --name value arguments")
opts <- list()
for (i in seq(1, length(argv), by = 2)) opts[[argv[i]]] <- argv[i + 1]
required <- c("--alignment", "--out", "--prior", "--threads", "--seed")
if (!all(required %in% names(opts))) stop("Missing required fastbaps arguments")
if (!requireNamespace("fastbaps", quietly = TRUE) || !requireNamespace("ape", quietly = TRUE)) {
  stop("Install the official R fastbaps package and ape in the Rscript environment")
}
out <- opts[["--out"]]
dir.create(out, recursive = TRUE, showWarnings = FALSE)
set.seed(as.integer(opts[["--seed"]]))
message("[fastbaps] importing cleaned nucleotide alignment")
sparse.data <- fastbaps::import_fasta_sparse_nt(opts[["--alignment"]])
if (nrow(sparse.data$snp.matrix) == 0) stop("No SNPs remain in the cleaned alignment")
message("[fastbaps] setting prior: ", opts[["--prior"]])
sparse.data <- fastbaps::optimise_prior(sparse.data, type = opts[["--prior"]],
                                      n.cores = as.integer(opts[["--threads"]]))
hierarchy <- fastbaps::fast_baps(sparse.data, n.cores = as.integer(opts[["--threads"]]))
partition <- fastbaps::best_baps_partition(sparse.data, hierarchy)
samples <- colnames(sparse.data$snp.matrix)
if (is.null(names(partition)) || !setequal(samples, names(partition))) {
  stop("Named partition does not match alignment sample IDs")
}
# Join by names, never by incidental hierarchy/tip order.
labels <- as.character(partition[match(samples, names(partition))])
if (anyNA(labels) || any(labels == "")) stop("Missing cluster labels")
result <- data.frame(sample = samples, cluster = labels, stringsAsFactors = FALSE)
write.table(result, file.path(out, "clusters.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
sizes <- as.data.frame(table(labels), stringsAsFactors = FALSE)
names(sizes) <- c("cluster", "n_isolates")
write.table(sizes, file.path(out, "cluster_sizes.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
saveRDS(hierarchy, file.path(out, "hierarchy.rds"))
capture.output(sessionInfo(), file = file.path(out, "sessionInfo.txt"))
remote.sha <- utils::packageDescription("fastbaps")$RemoteSha
if (is.null(remote.sha)) remote.sha <- "not_recorded"
metadata <- data.frame(
  field = c("R", "fastbaps_version", "fastbaps_RemoteSha", "prior", "seed", "threads",
            "initial_partition", "n_samples", "n_variable_sites", "n_clusters"),
  value = c(R.version.string, as.character(utils::packageVersion("fastbaps")),
            as.character(remote.sha),
            opts[["--prior"]], opts[["--seed"]], opts[["--threads"]], "package_default",
            length(samples), nrow(sparse.data$snp.matrix), length(unique(labels))))
write.table(metadata, file.path(out, "fastbaps_run.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
message("[fastbaps] inferred ", length(unique(labels)), " clusters from ", length(samples), " isolates")
