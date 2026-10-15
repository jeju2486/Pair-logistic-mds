#!/usr/bin/env Rscript
# Official caper D for requested binary patterns. Each completed pattern is atomic
# and resumable; no variant filter, phenotype model or association test is run.
argv <- commandArgs(trailingOnly=TRUE)
opts <- list()
i <- 1L
while (i <= length(argv)) {
    key <- sub("^--", "", argv[[i]])
    if (key == "resume") { opts[[key]] <- TRUE; i <- i+1L } else {
        if (i == length(argv)) stop("Missing value for ", key)
        opts[[key]] <- argv[[i+1L]]; i <- i+2L
    }
}
suppressPackageStartupMessages(library(ape))
suppressPackageStartupMessages(library(caper))
dir.create(opts$out, recursive=TRUE, showWarnings=FALSE)
tree <- read.tree(opts$tree)
if (anyDuplicated(tree$tip.label) || !is.rooted(tree)) stop("D requires unique tips and a rooted tree")
if (is.null(tree$edge.length) || any(!is.finite(tree$edge.length)) || any(tree$edge.length < 0)) stop("Invalid branch lengths")
# Repeated IQ-TREE support labels are not unique node IDs. Remove support labels
# from this working copy; caper creates its own internal node identifiers.
tree$node.label <- NULL
# caper rejects zero internal branches. ape uses length < tol, so the smallest
# positive length as tol collapses exactly zeros, retaining every positive edge.
zero_internal <- sum(tree$edge[,2] > Ntip(tree) & tree$edge.length == 0)
if (zero_internal) {
    positive <- tree$edge.length[tree$edge.length > 0]
    tree <- di2multi(tree, tol=if (length(positive)) min(positive) else 1)
    # A basal zero collapse can expose a rooted polytomy. A zero root edge tells
    # ape/caper to retain this existing root without adding evolutionary length.
    if (!is.rooted(tree)) tree$root.edge <- 0
}
tip_edges <- tree$edge[,2] <= Ntip(tree)
zero_tips <- which(tip_edges & tree$edge.length == 0)
if (length(zero_tips)) {
    if (opts[["zero-tip-policy"]] != "epsilon") stop("Zero-length terminal branches: rerun with explicit --zero-tip-policy epsilon, or provide a suitable tree")
    tree$edge.length[zero_tips] <- as.numeric(opts[["tip-epsilon"]])
}
write.tree(tree, file=file.path(opts$out,"D.treefile"), digits=17)
write.table(data.frame(zero_internal_collapsed=zero_internal, zero_tips_replaced=length(zero_tips),
    zero_tip_policy=opts[["zero-tip-policy"]], tip_epsilon=opts[["tip-epsilon"]]),
    file.path(opts$out,"tree_adjustments.tsv"), sep="\t", quote=FALSE, row.names=FALSE)
capture.output(sessionInfo(), file=file.path(opts$out,"sessionInfo.txt"))
dat <- read.delim(opts$patterns, check.names=FALSE, stringsAsFactors=FALSE)
samples <- names(dat)[-1]
if (anyDuplicated(samples) || !setequal(samples,tree$tip.label)) stop("Pattern/tree samples differ")
if (anyDuplicated(dat$pattern_id)) stop("Duplicate pattern ID")
permutations <- as.integer(opts$permutations)
base_seed <- as.numeric(opts$seed)
columns <- c("pattern_id","D","D_status","p_random_lower","p_brownian_upper",
             "normalization_denominator","D_error")
results <- vector("list",nrow(dat))
for (j in seq_len(nrow(dat))) {
    id <- dat$pattern_id[j]
    if (!grepl("^pattern_[0-9]+$",id)) stop("Unexpected pattern ID")
    checkpoint <- file.path(opts$out,paste0(id,".tsv"))
    old <- if (isTRUE(opts$resume) && file.exists(checkpoint))
        tryCatch(read.delim(checkpoint,check.names=FALSE,stringsAsFactors=FALSE,na.strings="NA"),error=function(e) NULL) else NULL
    if (!is.null(old) && nrow(old)==1 && identical(names(old),columns) && old$pattern_id==id && old$D_status != "error") {
        results[[j]] <- old
        cat(sprintf("[D] %d/%d reuse %s\n",j,nrow(dat),id)); flush.console(); next
    }
    row <- data.frame(pattern_id=id,D=NA_real_,D_status="not_estimable_monomorphic",
        p_random_lower=NA_real_,p_brownian_upper=NA_real_,normalization_denominator=NA_real_,D_error="",check.names=FALSE)
    states <- as.numeric(unlist(dat[j,-1],use.names=FALSE))
    if (anyNA(states) || !all(states %in% c(0,1))) stop("Invalid binary states for ",id)
    if (length(unique(states))==2) {
        # Seed depends on stable pattern identity, not loop position or resume order.
        set.seed(as.integer((base_seed+as.numeric(sub("pattern_","",id))) %% 2147483646)+1L)
        row <- tryCatch({
            input <- data.frame(sample=samples,state=states)
            fit <- phylo.d(input,phy=tree,names.col=sample,binvar=state,permut=permutations)
            denominator <- unname(fit$Parameters$MeanRandom-fit$Parameters$MeanBrownian)
            row$normalization_denominator <- denominator
            if (!is.finite(fit$DEstimate) || !is.finite(denominator) || denominator <= 1e-12) {
                row$D_status <- "degenerate_reference"
            } else {
                row$D <- unname(fit$DEstimate); row$D_status <- "ok"
                # These are the directional simulation-tail probabilities returned
                # by caper, not generic two-sided p-values or KOVAR p-values.
                row$p_random_lower <- fit$Pval1
                row$p_brownian_upper <- fit$Pval0
            }
            row
        }, error=function(e) {row$D_status <- "error"; row$D_error <- gsub("[\t\r\n]"," ",conditionMessage(e)); row})
    }
    temp <- paste0(checkpoint,".partial")
    write.table(row,temp,sep="\t",quote=TRUE,row.names=FALSE,na="NA")
    if (file.exists(checkpoint)) unlink(checkpoint)
    if (!file.rename(temp,checkpoint)) stop("Cannot publish pattern checkpoint")
    results[[j]] <- row
    cat(sprintf("[D] %d/%d %s: %s D=%s\n",j,nrow(dat),id,row$D_status,format(row$D))); flush.console()
}
output <- if (length(results)) do.call(rbind,results) else {
    empty <- as.data.frame(setNames(replicate(length(columns),character(),simplify=FALSE),columns)); empty
}
write.table(output,file.path(opts$out,"D.tsv"),sep="\t",quote=TRUE,row.names=FALSE,na="NA")
