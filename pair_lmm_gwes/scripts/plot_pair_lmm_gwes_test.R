#!/usr/bin/env Rscript

# pair-lmm-gwes simulation-test plotter
#
# Plot policy:
#   grey   = below threshold only
#   black  = above threshold, passes post-filter, not D/E hitchhiker
#   orange = above threshold D/E hitchhiker, regardless of post-filter/NA CTMC
#   red    = A/B/C/D/E true block pairs, plotted regardless of threshold/filter/NA CTMC
#   above-threshold filtered-out non-hitchhikers are removed, not drawn as grey

suppressPackageStartupMessages({
  if (!requireNamespace("data.table", quietly = TRUE)) {
    stop("Missing R package: data.table. Install with install.packages('data.table')", call. = FALSE)
  }
  library(data.table)
})

die <- function(...) stop(paste0(...), call. = FALSE)
num <- function(x) suppressWarnings(as.numeric(x))
neglog10 <- function(p) -log10(pmax(num(p), .Machine$double.xmin))

parse_args <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  o <- list(
    score=NULL, out=NULL, fasta=NA_character_, event_tree=NA_character_,
    blocks_tsv=NA_character_, block_positions=NA_character_,
    y_col="bidirectional_score", threshold=NA_real_, alpha=0.05, threshold_tests="pair",
    top_out=NA_character_, classified_out=NA_character_,
    distance_col="distance", x_mode="distance", ld_dist=0, min_dist=0, max_dist=Inf,
    D=NA_integer_, E=NA_integer_, true_pair_mode="all",
    hitch_r2_min=0.3, hitch_mismatch_max=0.1, hitch_near_window=1000,
    post_filter=TRUE, filter_min_cell_gt=10, filter_gain_gt=1.5, filter_loss_gt=1.5,
    max_points=0, seed=1L, title=NA_character_, width=2400, height=1200, dpi=300, write_svg=FALSE, no_deps=FALSE
  )
  if (length(args) == 0) die("Need --score --fasta --blocks-tsv/--block-positions --out")
  i <- 1
  while (i <= length(args)) {
    a <- args[i]
    if (a %in% c("-h","--help")) { quit(status=0)
    } else if (a %in% c("--score","--input")) { o$score <- args[i+1]; i <- i+1
    } else if (a %in% c("-o","--out","--output")) { o$out <- args[i+1]; i <- i+1
    } else if (a == "--fasta") { o$fasta <- args[i+1]; i <- i+1
    } else if (a == "--event-tree") { o$event_tree <- args[i+1]; i <- i+1
    } else if (a == "--blocks-tsv") { o$blocks_tsv <- args[i+1]; i <- i+1
    } else if (a == "--block-positions") { o$block_positions <- args[i+1]; i <- i+1
    } else if (a %in% c("-y","--y-col","--score-col")) { o$y_col <- args[i+1]; i <- i+1
    } else if (a == "--threshold") { o$threshold <- num(args[i+1]); i <- i+1
    } else if (a == "--alpha") { o$alpha <- num(args[i+1]); i <- i+1
    } else if (a == "--threshold-tests") { o$threshold_tests <- args[i+1]; i <- i+1
    } else if (a == "--top-out") { o$top_out <- args[i+1]; i <- i+1
    } else if (a == "--classified-out") { o$classified_out <- args[i+1]; i <- i+1
    } else if (a == "--distance-col") { o$distance_col <- args[i+1]; i <- i+1
    } else if (a == "--x-mode") { o$x_mode <- args[i+1]; i <- i+1
    } else if (a %in% c("-l","--ld-dist")) { o$ld_dist <- num(args[i+1]); i <- i+1
    } else if (a == "--min-dist") { o$min_dist <- num(args[i+1]); i <- i+1
    } else if (a == "--max-dist") { o$max_dist <- num(args[i+1]); i <- i+1
    } else if (a == "--D") { o$D <- as.integer(args[i+1]); i <- i+1
    } else if (a == "--E") { o$E <- as.integer(args[i+1]); i <- i+1
    } else if (a == "--true-pair-mode") { o$true_pair_mode <- args[i+1]; i <- i+1
    } else if (a == "--hitch-r2-min") { o$hitch_r2_min <- num(args[i+1]); i <- i+1
    } else if (a == "--hitch-mismatch-max") { o$hitch_mismatch_max <- num(args[i+1]); i <- i+1
    } else if (a == "--hitch-near-window") { o$hitch_near_window <- num(args[i+1]); i <- i+1
    } else if (a == "--filter-min-cell-gt") { o$filter_min_cell_gt <- num(args[i+1]); i <- i+1
    } else if (a == "--filter-gain-gt") { o$filter_gain_gt <- num(args[i+1]); i <- i+1
    } else if (a == "--filter-loss-gt") { o$filter_loss_gt <- num(args[i+1]); i <- i+1
    } else if (a == "--no-post-filter") { o$post_filter <- FALSE
    } else if (a == "--max-points") { o$max_points <- num(args[i+1]); i <- i+1
    } else if (a == "--seed") { o$seed <- as.integer(args[i+1]); i <- i+1
    } else if (a == "--title") { o$title <- args[i+1]; i <- i+1
    } else if (a == "--width") { o$width <- num(args[i+1]); i <- i+1
    } else if (a == "--height") { o$height <- num(args[i+1]); i <- i+1
    } else if (a == "--dpi") { o$dpi <- as.integer(args[i+1]); i <- i+1
    } else if (a == "--write-svg") { o$write_svg <- TRUE
    } else if (a == "--no-deps") { o$no_deps <- TRUE
    } else die("Unknown option: ", a)
    i <- i + 1
  }
  if (is.null(o$score) || is.null(o$out)) die("Need --score and --out")
  if (is.na(o$fasta) || !file.exists(o$fasta)) die("Need valid --fasta")
  if ((is.na(o$blocks_tsv) || !file.exists(o$blocks_tsv)) && (is.na(o$block_positions) || !nzchar(o$block_positions))) die("Need --blocks-tsv or --block-positions")
  o
}

make_pair_keys <- function(dt) {
  dt[, u := as.integer(u)]
  dt[, v := as.integer(v)]
  dt[, u0 := pmin(u, v)]
  dt[, v0 := pmax(u, v)]
  dt[, key_pair := paste(u0, v0, sep=":")]
  dt
}

score_values <- function(dt, y_col) {
  if (y_col == "bidirectional_score" && all(c("p_u_to_v","p_v_to_u") %in% names(dt))) {
    x <- if ("bidirectional_score" %in% names(dt)) num(dt[["bidirectional_score"]]) else rep(NA_real_, nrow(dt))
    xp <- pmin(neglog10(dt[["p_u_to_v"]]), neglog10(dt[["p_v_to_u"]]), na.rm=FALSE)
    x[!is.finite(x) & is.finite(xp)] <- xp[!is.finite(x) & is.finite(xp)]
    return(x)
  }
  if (!(y_col %in% names(dt))) die("Missing score column: ", y_col)
  x <- num(dt[[y_col]])
  p_like <- grepl("^p($|_)|_p$|p_value", y_col) && !grepl("neglog|score", y_col)
  if (p_like) x <- neglog10(x)
  x
}

add_x <- function(dt, o) {
  xlab <- o$distance_col
  if (o$x_mode == "distance") {
    if (o$distance_col %in% names(dt)) dt[, plot_x := num(get(o$distance_col))] else { dt[, plot_x := abs(v-u)]; xlab <- "abs(v-u)" }
  } else if (o$x_mode == "absdiff") { dt[, plot_x := abs(v-u)]; xlab <- "abs(v-u)"
  } else if (o$x_mode == "midpoint") { dt[, plot_x := (u+v)/2]; xlab <- "pair midpoint"
  } else if (o$x_mode == "u") { dt[, plot_x := u]; xlab <- "u"
  } else if (o$x_mode == "v") { dt[, plot_x := v]; xlab <- "v"
  } else die("Bad --x-mode")
  list(dt=dt, xlab=xlab)
}

filter_x_y <- function(dt, o) {
  z <- dt[is.finite(plot_x) & is.finite(plot_y)]
  if (o$x_mode == "distance") {
    z <- z[plot_x != -1 & plot_x >= o$min_dist]
    if (is.finite(o$max_dist)) z <- z[plot_x <= o$max_dist]
  }
  z
}

parse_block_positions <- function(o) {
  if (!is.na(o$block_positions) && nzchar(o$block_positions)) {
    x <- as.integer(trimws(unlist(strsplit(o$block_positions, ",", fixed=TRUE))))
  } else {
    tab <- fread(o$blocks_tsv, header=FALSE, fill=TRUE, showProgress=FALSE)
    tab <- tab[!grepl("^[[:space:]]*#", as.character(V1))]
    x <- as.integer(tab[[1]])
  }
  x <- x[is.finite(x)]
  if (length(x) < 5) die("Need at least five block positions for A/B/C/D/E")
  x[1:5]
}

true_pairs <- function(pos, mode="all") {
  labs <- c("A","B","C","D","E")
  idx <- if (mode == "de") list(c(4,5)) else utils::combn(seq_along(pos), 2, simplify=FALSE)
  rbindlist(lapply(idx, function(ii) {
    a <- min(pos[ii[1]], pos[ii[2]]); b <- max(pos[ii[1]], pos[ii[2]])
    data.table(key_pair=paste(a,b,sep=":"), true_label=paste(labs[ii], collapse="-"))
  }))
}

read_fasta <- function(path) {
  nm <- character(); seqs <- character(); cur <- NULL; parts <- character()
  con <- file(path, "r"); on.exit(close(con), add=TRUE)
  repeat {
    line <- readLines(con, n=1, warn=FALSE); if (!length(line)) break
    line <- trimws(line); if (!nzchar(line)) next
    if (startsWith(line, ">")) {
      if (!is.null(cur)) { nm <- c(nm, cur); seqs <- c(seqs, toupper(paste0(parts, collapse=""))) }
      cur <- strsplit(substring(line,2), "[[:space:]]+")[[1]][1]; parts <- character()
    } else parts <- c(parts, line)
  }
  if (!is.null(cur)) { nm <- c(nm, cur); seqs <- c(seqs, toupper(paste0(parts, collapse=""))) }
  if (!length(seqs)) die("No FASTA records: ", path)
  list(names=nm, seqs=seqs, n=length(seqs), L=nchar(seqs[1]))
}

binary_col <- function(fa, pos0) {
  p <- as.integer(pos0) + 1L
  as.integer(substr(fa$seqs, p, p) %in% c("C","1","T","G"))
}

phi_r2 <- function(x, y) {
  x <- as.integer(x != 0)
  y <- as.integer(y != 0)
  n11 <- as.numeric(sum(x == 1 & y == 1))
  n10 <- as.numeric(sum(x == 1 & y == 0))
  n01 <- as.numeric(sum(x == 0 & y == 1))
  n00 <- as.numeric(sum(x == 0 & y == 0))
  den <- (n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00)
  if (!is.finite(den) || den <= 0) return(NA_real_)
  ((n11 * n00 - n10 * n01)^2) / den
}

mismatch <- function(x,y) mean(as.integer(x != 0) != as.integer(y != 0))

pair_counts <- function(x,y) {
  x <- as.integer(x != 0); y <- as.integer(y != 0)
  n11 <- sum(x==1 & y==1); n10 <- sum(x==1 & y==0); n01 <- sum(x==0 & y==1); n00 <- sum(x==0 & y==0)
  list(n11=n11,n10=n10,n01=n01,n00=n00,min_cell=min(n11,n10,n01,n00),r2=phi_r2(x,y),same_mismatch=(n10+n01)/length(x))
}

find_event_tree <- function(o) {
  if (!is.na(o$event_tree) && nzchar(o$event_tree) && file.exists(o$event_tree)) return(o$event_tree)
  cand <- file.path(dirname(o$score), "auto_nj_event_tree.nwk")
  if (file.exists(cand)) return(cand)
  NA_character_
}

recompute_ctmc_python <- function(fasta, tree, u, v) {
  blank <- list(gain=NA_real_, loss=NA_real_, q01=NA_real_, q10=NA_real_, loglik=NA_real_, status="NOT_RUN")
  if (is.na(tree) || !nzchar(tree) || !file.exists(tree)) { blank$status <- "NO_EVENT_TREE"; return(blank) }
  py <- Sys.which("python"); if (!nzchar(py)) py <- Sys.which("python3")
  if (!nzchar(py)) { blank$status <- "NO_PYTHON"; return(blank) }
  code <- "
import sys, json
from pair_lmm_gwes.io_utils import read_fake_fasta
from pair_lmm_gwes.tree_events import Tree11EventCounter
fasta, tree, u, v = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
fm = read_fake_fasta(fasta)
X = fm.X
z11 = (X[:,u] == 1) & (X[:,v] == 1)
c = Tree11EventCounter(tree, fm.sample_names, missing_length='zero')
s = c.summarize_ctmc_taylor2(z11)
print(json.dumps({'gain':s.ctmc_11_gain_expected,'loss':s.ctmc_11_loss_expected,'q01':s.ctmc_11_q01,'q10':s.ctmc_11_q10,'loglik':s.ctmc_11_loglik,'status':s.ctmc_11_status}))
"
  tf <- tempfile(fileext=".py"); writeLines(code, tf); on.exit(unlink(tf), add=TRUE)
  out <- tryCatch(system2(py, c(tf, fasta, tree, as.character(u), as.character(v)), stdout=TRUE, stderr=TRUE), error=function(e) paste0("ERROR:", conditionMessage(e)))
  if (!is.null(attr(out, "status")) && attr(out, "status") != 0) { blank$status <- paste(out, collapse=" | "); return(blank) }
  line <- out[length(out)]
  getn <- function(k) as.numeric(sub(".*:", "", regmatches(line, regexpr(paste0('"',k,'"[[:space:]]*:[[:space:]]*[-+0-9.eE]+'), line))))
  gets <- function(k) sub('.*:[[:space:]]*"([^"]*)".*', '\\1', regmatches(line, regexpr(paste0('"',k,'"[[:space:]]*:[[:space:]]*"[^"]*"'), line)))
  list(gain=getn("gain"), loss=getn("loss"), q01=getn("q01"), q10=getn("q10"), loglik=getn("loglik"), status=gets("status"))
}

apply_filter <- function(dt, o) {
  dt[, passes_post_filter := TRUE]
  if (!o$post_filter) return(dt)
  if ("min_cell" %in% names(dt)) dt[!(is.finite(num(min_cell)) & num(min_cell) > o$filter_min_cell_gt), passes_post_filter := FALSE] else dt[, passes_post_filter := FALSE]
  if ("ctmc_11_gain_expected" %in% names(dt)) dt[!(is.finite(num(ctmc_11_gain_expected)) & num(ctmc_11_gain_expected) > o$filter_gain_gt), passes_post_filter := FALSE] else dt[, passes_post_filter := FALSE]
  if ("ctmc_11_loss_expected" %in% names(dt)) dt[!(is.finite(num(ctmc_11_loss_expected)) & num(ctmc_11_loss_expected) > o$filter_loss_gt), passes_post_filter := FALSE] else dt[, passes_post_filter := FALSE]
  dt
}

axis_breaks <- function(x) {
  mx <- max(x, na.rm=TRUE); if (!is.finite(mx) || mx <= 0) mx <- 1
  step <- if (mx <= 100000) 10000 else 50000
  seq(0, ceiling(mx/step)*step, by=step)
}

o <- parse_args()
if (!file.exists(o$score)) die("Missing score file: ", o$score)
if (file.exists(o$out)) die("Output already exists: ", o$out)

dt <- fread(o$score, sep="\t", showProgress=TRUE)
dt <- make_pair_keys(dt)
dt[, plot_y := score_values(.SD, o$y_col)]
ax <- add_x(dt, o); dt <- ax$dt; xlab <- ax$xlab
dt <- filter_x_y(dt, o)

n_tests <- if (o$threshold_tests == "directional") 2*nrow(dt) else nrow(dt)
if (!is.finite(o$threshold)) o$threshold <- -log10(o$alpha / max(1, n_tests))
dt[, below_threshold := plot_y < o$threshold]
dt[, above_threshold := plot_y >= o$threshold]

block_pos <- parse_block_positions(o)
truth <- true_pairs(block_pos, o$true_pair_mode)
dt[, `:=`(is_true_block=FALSE, true_label=NA_character_)]
dt[truth, `:=`(is_true_block=TRUE, true_label=i.true_label), on="key_pair"]
missing_truth <- truth[!(key_pair %in% dt$key_pair)]
if (nrow(missing_truth)) warning("True pairs absent from score table and cannot be plotted: ", paste(missing_truth$true_label, collapse=", "))

if (!is.finite(o$D)) o$D <- block_pos[4]
if (!is.finite(o$E)) o$E <- block_pos[5]

dt <- apply_filter(dt, o)

fa <- read_fasta(o$fasta)
Dcol <- binary_col(fa, o$D); Ecol <- binary_col(fa, o$E); DE11 <- as.integer(Dcol==1 & Ecol==1)

# Always print/recompute D-E diagnostics if D-E exists in score table.
de_key <- paste(min(o$D,o$E), max(o$D,o$E), sep=":")
de_row <- dt[key_pair == de_key]
if (nrow(de_row)) {
  cnt <- pair_counts(binary_col(fa, min(o$D,o$E)), binary_col(fa, max(o$D,o$E)))
  need_ctmc <- !("ctmc_11_gain_expected" %in% names(dt)) || !is.finite(num(de_row$ctmc_11_gain_expected[1])) || !is.finite(num(de_row$ctmc_11_loss_expected[1]))
  ctmc <- list(gain=if ("ctmc_11_gain_expected" %in% names(de_row)) num(de_row$ctmc_11_gain_expected[1]) else NA_real_,
               loss=if ("ctmc_11_loss_expected" %in% names(de_row)) num(de_row$ctmc_11_loss_expected[1]) else NA_real_,
               q01=NA_real_, q10=NA_real_, loglik=NA_real_, status="EXISTING_VALUES_USED")
  if (need_ctmc) {
    ev <- find_event_tree(o)
    cat("[D/E diagnostic] CTMC missing/NA; trying Python recomputation using event tree: ", ev, "\n", sep="")
    ctmc <- recompute_ctmc_python(o$fasta, ev, min(o$D,o$E), max(o$D,o$E))
    if (is.finite(ctmc$gain)) {
      for (cc in c("ctmc_11_gain_expected","ctmc_11_loss_expected")) if (!(cc %in% names(dt))) dt[, (cc) := NA_real_]
      dt[key_pair == de_key, `:=`(ctmc_11_gain_expected=ctmc$gain, ctmc_11_loss_expected=ctmc$loss)]
    }
  }
  de_pass <- (!o$post_filter) ||
    (cnt$min_cell > o$filter_min_cell_gt && is.finite(ctmc$gain) && ctmc$gain > o$filter_gain_gt && is.finite(ctmc$loss) && ctmc$loss > o$filter_loss_gt)
  dt[key_pair == de_key, passes_post_filter := de_pass]
  de_row <- dt[key_pair == de_key]
  cat("\n[D/E diagnostic]\n")
  cat("pair_key\t", de_key, "\n", sep="")
  cat("score\t", de_row$plot_y[1], "\n", sep="")
  cat("above_threshold\t", de_row$above_threshold[1], "\n", sep="")
  cat("passes_post_filter_after_recalc\t", de_pass, "\n", sep="")
  cat("counts\t", sprintf("n11=%d,n10=%d,n01=%d,n00=%d", cnt$n11,cnt$n10,cnt$n01,cnt$n00), "\n", sep="")
  cat("min_cell_recomputed\t", cnt$min_cell, "\n", sep="")
  cat("r2_recomputed\t", cnt$r2, "\n", sep="")
  cat("ctmc_gain\t", ctmc$gain, "\n", sep="")
  cat("ctmc_loss\t", ctmc$loss, "\n", sep="")
  cat("ctmc_status\t", ctmc$status, "\n\n", sep="")
} else {
  cat("\n[D/E diagnostic]\nD/E pair absent from score table: ", de_key, "\n\n", sep="")
}

# classify D/E hitchhikers among all above-threshold non-true rows, regardless of post-filter/NA CTMC
dt[, `:=`(DE_hitchhiker_flag=FALSE, DE_hitchhiker_reason="not_evaluated",
          nearest_DE_distance=NA_real_, max_locus_r2_to_D_E_DE11=NA_real_,
          min_locus_mismatch_to_D_E_DE11=NA_real_, pair11_r2_to_DE11=NA_real_)]
idx <- which(dt$above_threshold & !dt$is_true_block)
if (length(idx)) {
  cache <- new.env(parent=emptyenv())
  get_col <- function(p) {
    k <- as.character(p)
    if (!exists(k, envir=cache, inherits=FALSE)) assign(k, binary_col(fa, p), envir=cache)
    get(k, envir=cache, inherits=FALSE)
  }
  for (ii in idx) {
    xu <- get_col(dt$u0[ii]); xv <- get_col(dt$v0[ii]); pair11 <- as.integer(xu==1 & xv==1)
    r2s <- c(phi_r2(xu,Dcol), phi_r2(xu,Ecol), phi_r2(xu,DE11), phi_r2(xv,Dcol), phi_r2(xv,Ecol), phi_r2(xv,DE11))
    mism <- c(mismatch(xu,Dcol), mismatch(xu,Ecol), mismatch(xu,DE11), mismatch(xv,Dcol), mismatch(xv,Ecol), mismatch(xv,DE11))
    maxr <- suppressWarnings(max(r2s, na.rm=TRUE)); if (!is.finite(maxr)) maxr <- NA_real_
    minm <- suppressWarnings(min(mism, na.rm=TRUE)); if (!is.finite(minm)) minm <- NA_real_
    pr2 <- phi_r2(pair11, DE11)
    nd <- min(abs(dt$u0[ii]-o$D), abs(dt$u0[ii]-o$E), abs(dt$v0[ii]-o$D), abs(dt$v0[ii]-o$E))
    near <- is.finite(nd) && nd <= o$hitch_near_window
    patt <- (is.finite(maxr) && maxr >= o$hitch_r2_min) || (is.finite(minm) && minm <= o$hitch_mismatch_max) || (is.finite(pr2) && pr2 >= o$hitch_r2_min)
    flag <- near || patt
    reason <- if (dt$u0[ii] %in% c(o$D,o$E) || dt$v0[ii] %in% c(o$D,o$E)) "contains_D_or_E" else if (near) "near_D_or_E_coordinate" else if (patt) "locus_pattern_correlated_with_D_E_DE11" else "not_obvious_DE_hitchhiker"
    set(dt, ii, "DE_hitchhiker_flag", flag); set(dt, ii, "DE_hitchhiker_reason", reason)
    set(dt, ii, "nearest_DE_distance", nd); set(dt, ii, "max_locus_r2_to_D_E_DE11", maxr)
    set(dt, ii, "min_locus_mismatch_to_D_E_DE11", minm); set(dt, ii, "pair11_r2_to_DE11", pr2)
  }
}
dt[is_true_block == TRUE, DE_hitchhiker_reason := "true_block_pair"]

dt[, plot_class := "not_plotted"]
dt[below_threshold == TRUE & is_true_block == FALSE, plot_class := "background"]  # grey below threshold only
# Apply CTMC/min-cell filter first: rows failing it are removed and not plotted.
dt[above_threshold == TRUE & passes_post_filter == TRUE & is_true_block == FALSE & DE_hitchhiker_flag == TRUE, plot_class := "DE_hitchhiker"]
dt[above_threshold == TRUE & passes_post_filter == TRUE & is_true_block == FALSE & DE_hitchhiker_flag == FALSE, plot_class := "survivor"]
dt[is_true_block == TRUE, plot_class := "true_block"]  # red regardless of threshold/filter

if (!is.na(o$top_out) && nzchar(o$top_out)) fwrite(dt[above_threshold == TRUE][order(-plot_y)], o$top_out, sep="\t")
if (!is.na(o$classified_out) && nzchar(o$classified_out)) fwrite(dt[plot_class != "not_plotted" | above_threshold == TRUE][order(-plot_y)], o$classified_out, sep="\t")

bg <- dt[plot_class == "background"]
hitch <- dt[plot_class == "DE_hitchhiker"]
surv <- dt[plot_class == "survivor"]
true_dt <- dt[plot_class == "true_block"]

if (is.finite(o$max_points) && o$max_points > 0 && nrow(bg) > o$max_points) {
  set.seed(o$seed); bg <- bg[sample(seq_len(nrow(bg)), o$max_points)]
}
plot_dt <- rbindlist(list(bg,hitch,surv,true_dt), fill=TRUE)
if (!nrow(plot_dt)) die("No rows to plot")

cat("Rows after distance/finite filtering: ", nrow(dt), "\n", sep="")
cat("Threshold: ", sprintf("%.6f", o$threshold), "\n", sep="")
cat("Below-threshold grey rows plotted: ", nrow(bg), "\n", sep="")
cat("Above-threshold rows before post-filter: ", nrow(dt[above_threshold == TRUE]), "\n", sep="")
cat("Above-threshold non-hitchhiker rows removed and not plotted: ", nrow(dt[above_threshold == TRUE & passes_post_filter == FALSE & is_true_block == FALSE & DE_hitchhiker_flag == FALSE]), "\n", sep="")
cat("D/E hitchhiker rows plotted orange regardless of filter: ", nrow(hitch), "\n", sep="")
cat("Other survivor rows plotted black: ", nrow(surv), "\n", sep="")
cat("True block rows plotted red regardless of threshold/filter: ", nrow(true_dt), "\n", sep="")

xb <- axis_breaks(plot_dt$plot_x); xmax <- max(xb)
ymax <- max(plot_dt$plot_y, na.rm=TRUE); if (!is.finite(ymax) || ymax <= 0) ymax <- 1
ylim <- c(0, ymax*1.05); yticks <- pretty(ylim, n=8)
if (is.na(o$title) || !nzchar(o$title)) o$title <- "pair-lmm-gwes simulation benchmark"
ylab <- if (o$y_col == "bidirectional_score") "Bidirectional score: min(-log10 p in both directions)" else o$y_col


sidecar_svg_path <- function(path) {
  ext <- tolower(tools::file_ext(path))
  if (ext == "svg") return(path)
  sub("\\.[^.]*$", ".svg", path)
}
save_gg_all <- function(plot_obj, path, width_px, height_px, dpi, write_svg=FALSE) {
  ext <- tolower(tools::file_ext(path))
  w_in <- width_px / dpi
  h_in <- height_px / dpi
  if (ext == "pdf") {
    ggplot2::ggsave(path, plot_obj, width=w_in, height=h_in, units="in")
  } else if (ext == "svg") {
    grDevices::svg(filename=path, width=w_in, height=h_in)
    print(plot_obj)
    grDevices::dev.off()
  } else {
    ggplot2::ggsave(path, plot_obj, width=width_px, height=height_px, units="px", dpi=dpi)
  }
  if (isTRUE(write_svg)) {
    svg_path <- sidecar_svg_path(path)
    if (!identical(normalizePath(svg_path, mustWork=FALSE), normalizePath(path, mustWork=FALSE))) {
      grDevices::svg(filename=svg_path, width=w_in, height=h_in)
      print(plot_obj)
      grDevices::dev.off()
      cat("Wrote SVG: ", svg_path, "\n", sep="")
    }
  }
}
open_base_device <- function(path, width_px, height_px, dpi) {
  ext <- tolower(tools::file_ext(path))
  w_in <- width_px / dpi
  h_in <- height_px / dpi
  if (ext == "svg") grDevices::svg(filename=path, width=w_in, height=h_in)
  else if (ext == "pdf") grDevices::pdf(file=path, width=w_in, height=h_in)
  else grDevices::png(filename=path, width=width_px, height=height_px)
}

outdir <- dirname(o$out); if (!dir.exists(outdir) && outdir != ".") dir.create(outdir, recursive=TRUE)
use_gg <- (!o$no_deps) && requireNamespace("ggplot2", quietly=TRUE)
use_repel <- use_gg && requireNamespace("ggrepel", quietly=TRUE)

if (use_gg) {
  suppressPackageStartupMessages(library(ggplot2))
  if (use_repel) suppressPackageStartupMessages(library(ggrepel))
  p <- ggplot()
  if (nrow(bg)) p <- p + geom_point(data=bg, aes(plot_x, plot_y), colour="#8f8f8f", alpha=0.18, size=0.18)
  if (nrow(hitch)) p <- p + geom_point(data=hitch, aes(plot_x, plot_y), colour="#f16913", alpha=0.22, size=0.20)
  if (nrow(surv)) p <- p + geom_point(data=surv, aes(plot_x, plot_y), colour="#252525", alpha=0.88, size=0.34)
  if (nrow(true_dt)) p <- p + geom_point(data=true_dt, aes(plot_x, plot_y), colour="#de2d26", alpha=0.98, size=0.90)
  if (nrow(true_dt)) {
    if (use_repel) p <- p + ggrepel::geom_text_repel(data=true_dt, aes(plot_x, plot_y, label=true_label), colour="#de2d26", size=3.0, box.padding=0.25, point.padding=0.2, min.segment.length=0)
    else p <- p + geom_text(data=true_dt, aes(plot_x, plot_y, label=true_label), colour="#de2d26", vjust=-0.7, size=2.8)
  }
  if (is.finite(o$ld_dist) && o$ld_dist > 0) p <- p + geom_vline(xintercept=o$ld_dist, colour="black", linetype="dashed", linewidth=0.35)
  p <- p + geom_hline(yintercept=o$threshold, colour="#238b45", linetype="dashed", linewidth=0.35) +
    scale_x_continuous(breaks=function(x) pretty(x, n=10), limits=c(0,xmax)) +
    scale_y_continuous(breaks=yticks, limits=ylim) +
    xlab(ifelse(o$x_mode == "distance", paste0("Distance between SNP/unitig columns (", o$distance_col, ")"), xlab)) +
    ylab(ylab) + ggtitle(o$title) + theme_minimal(base_size=8) +
    theme(legend.position="none", plot.margin=margin(5,8,5,5)) +
    annotate("text", x=xmax*0.98, y=ylim[1]+0.94*diff(ylim), label=paste0("#Score rows: ", floor(nrow(dt)/1000), "K"), hjust=1, size=3.0) +
    annotate("text", x=xmax*0.98, y=ylim[1]+0.90*diff(ylim), label=paste0("#Threshold: ", sprintf("%.2f", o$threshold)), hjust=1, size=3.0) +
    annotate("text", x=xmax*0.98, y=ylim[1]+0.86*diff(ylim), label=paste0("#Hitchhikers: ", nrow(hitch)), hjust=1, size=3.0) +
    annotate("text", x=xmax*0.98, y=ylim[1]+0.82*diff(ylim), label=paste0("#Other survivors: ", nrow(surv)), hjust=1, size=3.0) +
    annotate("text", x=xmax*0.98, y=ylim[1]+0.78*diff(ylim), label=paste0("#True pairs: ", nrow(true_dt)), hjust=1, size=3.0)
  save_gg_all(p, o$out, o$width, o$height, o$dpi, o$write_svg)
} else {
  open_base_device(o$out, o$width, o$height, o$dpi)
  plot(0,0,type="n",xlim=c(0,xmax),ylim=ylim,xaxs="i",yaxs="i",xlab="",ylab="",xaxt="n",yaxt="n",bty="n")
  if (nrow(bg)) points(bg$plot_x,bg$plot_y,col=rgb(140,140,140,70,maxColorValue=255),pch=19,cex=0.25)
  if (nrow(hitch)) points(hitch$plot_x,hitch$plot_y,col=rgb(241,105,19,55,maxColorValue=255),pch=19,cex=0.25)
  if (nrow(surv)) points(surv$plot_x,surv$plot_y,col="#252525",pch=19,cex=0.38)
  if (nrow(true_dt)) { points(true_dt$plot_x,true_dt$plot_y,col="#de2d26",pch=19,cex=0.70); text(true_dt$plot_x,true_dt$plot_y,labels=true_dt$true_label,col="#de2d26",pos=3,cex=0.75) }
  abline(h=o$threshold,col="#238b45",lty=2)
  if (is.finite(o$ld_dist) && o$ld_dist > 0) abline(v=o$ld_dist,lty=2)
  axis(1, at=xb, labels=xb, tick=FALSE, line=-0.8); axis(2, at=yticks, labels=yticks, las=1)
  title(main=o$title, xlab=ifelse(o$x_mode == "distance", paste0("Distance between SNP/unitig columns (", o$distance_col, ")"), xlab), ylab=ylab)
  dev.off()
}
cat("Wrote: ", o$out, "\n", sep="")
