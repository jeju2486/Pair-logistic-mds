#!/usr/bin/env Rscript

# KO-Variation (KOVAR) general plotter
#
# Plot policy:
#   grey  = below threshold only
#   black = above threshold
#   red   = user-specified true/highlighted pairs
#   --n applies PAN-GWES graph-count filtering: count >= ceiling(0.05 * n) by default

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
    score = NULL, out = NULL,
    y_col = "neglog10_p",
    threshold = NA_real_, alpha = 0.05,
    top_out = NA_character_,
    pairs = NA_character_, n_samples = NA_real_, pair_count_col = "count", pair_count_frac = 0.05, pair_count_min = NA_real_,
    distance_col = "distance", x_mode = "distance", ld_dist = 0, ld_dist_alt = 0,
    min_dist = 0, max_dist = Inf, max_points = 0, seed = 1L,
    inblock = NA_character_, true_block_ids = NA_character_, pair_mode = "cross",
    highlight_pair = NA_character_,
    title = NA_character_, width = 3000, height = 1800, dpi = 300, write_svg = FALSE, no_deps = FALSE
  )
  i <- 1
  if (length(args) == 0) die("Need --score and --out")
  while (i <= length(args)) {
    a <- args[i]
    if (a %in% c("-h", "--help")) { quit(status = 0)
    } else if (a %in% c("--score", "--input")) { o$score <- args[i+1]; i <- i+1
    } else if (a %in% c("-o", "--out", "--output")) { o$out <- args[i+1]; i <- i+1
    } else if (a %in% c("-y", "--y-col", "--score-col")) { o$y_col <- args[i+1]; i <- i+1
    } else if (a == "--threshold") { o$threshold <- num(args[i+1]); i <- i+1
    } else if (a == "--alpha") { o$alpha <- num(args[i+1]); i <- i+1
    } else if (a == "--top-out") { o$top_out <- args[i+1]; i <- i+1
    } else if (a == "--pairs") { o$pairs <- args[i+1]; i <- i+1
    } else if (a == "--n") { o$n_samples <- num(args[i+1]); i <- i+1
    } else if (a == "--pair-count-col") { o$pair_count_col <- args[i+1]; i <- i+1
    } else if (a == "--pair-count-frac") { o$pair_count_frac <- num(args[i+1]); i <- i+1
    } else if (a == "--pair-count-min") { o$pair_count_min <- num(args[i+1]); i <- i+1
    } else if (a == "--distance-col") { o$distance_col <- args[i+1]; i <- i+1
    } else if (a == "--x-mode") { o$x_mode <- args[i+1]; i <- i+1
    } else if (a %in% c("-l", "--ld-dist")) { o$ld_dist <- num(args[i+1]); i <- i+1
    } else if (a == "--ld-dist-alt") { o$ld_dist_alt <- num(args[i+1]); i <- i+1
    } else if (a == "--min-dist") { o$min_dist <- num(args[i+1]); i <- i+1
    } else if (a == "--max-dist") { o$max_dist <- num(args[i+1]); i <- i+1
    } else if (a == "--max-points") { o$max_points <- num(args[i+1]); i <- i+1
    } else if (a == "--seed") { o$seed <- as.integer(args[i+1]); i <- i+1
    } else if (a == "--inblock") { o$inblock <- args[i+1]; i <- i+1
    } else if (a == "--true-block-ids") { o$true_block_ids <- args[i+1]; i <- i+1
    } else if (a == "--pair-mode") { o$pair_mode <- args[i+1]; i <- i+1
    } else if (a == "--highlight-pair") { o$highlight_pair <- args[i+1]; i <- i+1
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
  o
}

make_pair_keys <- function(dt) {
  if (!all(c("u","v") %in% names(dt))) die("Input must contain u and v columns")
  dt[, u := as.integer(u)]
  dt[, v := as.integer(v)]
  dt[, u0 := pmin(u, v)]
  dt[, v0 := pmax(u, v)]
  dt[, key_pair := paste(u0, v0, sep = ":")]
  dt
}

read_pangwes_pairs <- function(path) {
  if (is.na(path) || !nzchar(path)) return(NULL)
  if (!file.exists(path)) die("Missing pair file: ", path)
  first <- readLines(path, n=100, warn=FALSE)
  first <- first[nzchar(trimws(first)) & !grepl("^#", trimws(first))]
  if (!length(first)) die("No pair rows found in ", path)
  toks <- strsplit(gsub(",", "\t", trimws(first[1])), "[[:space:]]+")[[1]]
  no_header <- suppressWarnings(!any(is.na(as.integer(toks[1:2]))))
  if (isTRUE(no_header)) {
    z <- fread(path, header=FALSE, sep="auto", showProgress=FALSE)
    default_cols <- c("u", "v", "distance", "ARACNE", "MI", "count", "M2", "min_distance", "max_distance")
    if (ncol(z) <= length(default_cols)) setnames(z, default_cols[seq_len(ncol(z))])
    else setnames(z, c(default_cols, paste0("extra_", seq_len(ncol(z) - length(default_cols)))))
  } else {
    z <- fread(path, header=TRUE, sep="auto", showProgress=FALSE)
    lower <- tolower(names(z))
    u_idx <- match(TRUE, lower %in% c("u", "unitig_i", "i", "locus_i", "marker_i", "variant_i"))
    v_idx <- match(TRUE, lower %in% c("v", "unitig_j", "j", "locus_j", "marker_j", "variant_j"))
    if (is.na(u_idx)) u_idx <- 1
    if (is.na(v_idx)) v_idx <- 2
    setnames(z, c(u_idx, v_idx), c("u", "v"))
  }
  z <- make_pair_keys(z)
  meta_cols <- setdiff(names(z), c("u", "v", "u0", "v0"))
  z <- unique(z[, ..meta_cols], by="key_pair")
  z
}

merge_pair_metadata <- function(dt, o) {
  z <- read_pangwes_pairs(o$pairs)
  if (is.null(z)) return(dt)
  add_cols <- setdiff(names(z), "key_pair")
  keep_cols <- c("key_pair", add_cols[!(add_cols %in% names(dt))])
  # If the score table lacks PAN-GWES metadata, add it from --pairs. Existing
  # columns are left unchanged so result-table metadata remains authoritative.
  if (length(keep_cols) > 1) dt <- merge(dt, z[, ..keep_cols], by="key_pair", all.x=TRUE, sort=FALSE)
  dt
}

apply_pair_count_filter <- function(dt, o) {
  dt[, pair_count_filter_applied := FALSE]
  dt[, pair_count_filter_threshold := NA_real_]
  dt[, pair_count_filter_pass := TRUE]
  if (!is.finite(o$n_samples) && !is.finite(o$pair_count_min)) return(dt)
  if (!(o$pair_count_col %in% names(dt))) {
    die("--n/--pair-count-min requested but count column is missing: ", o$pair_count_col,
        ". Supply a KOVAR result table or use --pairs to merge PAN-GWES metadata.")
  }
  min_count <- if (is.finite(o$pair_count_min)) ceiling(o$pair_count_min) else ceiling(o$pair_count_frac * o$n_samples)
  if (!is.finite(min_count) || min_count < 0) die("Bad pair graph-count threshold")
  cnt <- num(dt[[o$pair_count_col]])
  dt[, pair_count_filter_applied := TRUE]
  dt[, pair_count_filter_threshold := min_count]
  dt[, pair_count_filter_pass := is.finite(cnt) & cnt >= min_count]
  dt
}

score_values <- function(dt, y_col) {
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

parse_highlight_pairs <- function(x) {
  if (is.na(x) || !nzchar(x)) return(data.table(key_pair=character(), true_label=character()))
  parts <- unlist(strsplit(x, ";", fixed=TRUE))
  rbindlist(lapply(parts, function(p) {
    xy <- as.integer(trimws(unlist(strsplit(p, ",", fixed=TRUE))))
    if (length(xy) != 2 || any(is.na(xy))) die("Bad --highlight-pair entry: ", p)
    a <- min(xy); b <- max(xy)
    data.table(key_pair=paste(a,b,sep=":"), true_label=paste0(a,"-",b))
  }), fill=TRUE)
}

find_block_file <- function(prefix, id) {
  cand <- unique(c(
    Sys.glob(sprintf("%s.block_%d.txt", prefix, id)),
    Sys.glob(sprintf("%s_block_%d.txt", prefix, id)),
    Sys.glob(sprintf("%s.block_%05d.txt", prefix, id)),
    Sys.glob(sprintf("%s_block_%05d.txt", prefix, id))
  ))
  if (file.exists(prefix) && isTRUE(file.info(prefix)$isdir)) {
    cand <- c(cand, list.files(prefix, pattern=sprintf("[._]block_0*%d\\.txt$", id), full.names=TRUE))
  }
  cand <- cand[file.exists(cand)]
  if (!length(cand)) die("Cannot find block file for block ", id)
  cand[1]
}

mark_true_pairs <- function(dt, o) {
  lab <- parse_highlight_pairs(o$highlight_pair)
  if (!is.na(o$inblock) && nzchar(o$inblock) && !is.na(o$true_block_ids) && nzchar(o$true_block_ids) && o$pair_mode != "none") {
    ids <- as.integer(trimws(unlist(strsplit(o$true_block_ids, ",", fixed=TRUE))))
    ids <- ids[is.finite(ids)]
    block_sets <- lapply(ids, function(id) as.character(fread(find_block_file(o$inblock, id), header=FALSE, sep="\n", showProgress=FALSE)[[1]]))
    names(block_sets) <- as.character(ids)
    pair_ids <- if (o$pair_mode == "de" && length(ids) >= 5) list(ids[4:5]) else if (o$pair_mode == "all") utils::combn(ids, 2, simplify=FALSE) else utils::combn(ids, 2, simplify=FALSE)
    labels <- LETTERS[seq_along(ids)]
    ui <- as.character(dt$u); vi <- as.character(dt$v)
    out <- list()
    for (bp in pair_ids) {
      s1 <- block_sets[[as.character(bp[1])]]; s2 <- block_sets[[as.character(bp[2])]]
      hit <- (ui %in% s1 & vi %in% s2) | (ui %in% s2 & vi %in% s1)
      if (any(hit)) {
        lab1 <- labels[match(bp[1], ids)]; lab2 <- labels[match(bp[2], ids)]
        out[[length(out)+1]] <- data.table(key_pair=dt$key_pair[hit], true_label=paste0(lab1,"-",lab2))
      }
    }
    if (length(out)) lab <- unique(rbind(lab, rbindlist(out), fill=TRUE), by="key_pair")
  }
  unique(lab, by="key_pair")
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
dt <- merge_pair_metadata(dt, o)
dt[, plot_y := score_values(.SD, o$y_col)]
ax <- add_x(dt, o); dt <- ax$dt; xlab <- ax$xlab
dt <- filter_x_y(dt, o)
n_before_pair_count_filter <- nrow(dt)
dt <- apply_pair_count_filter(dt, o)
pair_count_removed <- nrow(dt[pair_count_filter_pass == FALSE])
dt <- dt[pair_count_filter_pass == TRUE]

n_tests <- nrow(dt)
if (!is.finite(o$threshold)) o$threshold <- -log10(o$alpha / max(1, n_tests))

dt[, below_threshold := plot_y < o$threshold]
dt[, above_threshold := plot_y >= o$threshold]

true_map <- mark_true_pairs(dt, o)
dt[, `:=`(is_true=FALSE, true_label=NA_character_)]
if (nrow(true_map)) dt[true_map, `:=`(is_true=TRUE, true_label=i.true_label), on="key_pair"]
dt[is_true == TRUE & (is.na(true_label) | true_label == ""), true_label := paste0(u0,"-",v0)]

bg <- dt[below_threshold == TRUE & is_true == FALSE]                         # only below threshold is grey
hits <- dt[above_threshold == TRUE & is_true == FALSE]
true_dt <- dt[is_true == TRUE]                                                # always plotted

if (!is.na(o$top_out) && nzchar(o$top_out)) fwrite(dt[above_threshold == TRUE][order(-plot_y)], o$top_out, sep="\t")

if (is.finite(o$max_points) && o$max_points > 0 && nrow(bg) > o$max_points) {
  set.seed(o$seed); bg <- bg[sample(seq_len(nrow(bg)), o$max_points)]
}

plot_dt <- rbindlist(list(bg[, plot_class := "below_threshold"], hits[, plot_class := "survivor"], true_dt[, plot_class := "true_pair"]), fill=TRUE)
if (!nrow(plot_dt)) die("No rows to plot")

cat("Rows after distance/finite/pair-count filtering: ", nrow(dt), "\n", sep="")
if (is.finite(o$n_samples) || is.finite(o$pair_count_min)) {
  min_count_label <- if (nrow(dt)) unique(dt$pair_count_filter_threshold)[1] else if (is.finite(o$pair_count_min)) ceiling(o$pair_count_min) else ceiling(o$pair_count_frac * o$n_samples)
  cat("Pair graph-count filter: ", o$pair_count_col, " >= ", min_count_label, " removed ", pair_count_removed, " of ", n_before_pair_count_filter, " rows\n", sep="")
}
cat("Threshold: ", sprintf("%.6f", o$threshold), "\n", sep="")
cat("Below-threshold grey rows plotted: ", nrow(bg), "\n", sep="")
cat("Above-threshold rows plotted black: ", nrow(hits), "\n", sep="")
cat("True/highlight rows plotted red: ", nrow(true_dt), "\n", sep="")

xb <- axis_breaks(plot_dt$plot_x); xmax <- max(xb)
ymax <- max(plot_dt$plot_y, na.rm=TRUE); if (!is.finite(ymax) || ymax <= 0) ymax <- 1
ylim <- c(0, ymax*1.05); yticks <- pretty(ylim, n=8)

if (is.na(o$title) || !nzchar(o$title)) o$title <- "KOVAR pangenome covariation"
ylab <- if (o$y_col == "neglog10_p") "-log10(primary p-value)" else o$y_col


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
  if (nrow(hits)) p <- p + geom_point(data=hits, aes(plot_x, plot_y), colour="#252525", alpha=0.85, size=0.32)
  if (nrow(true_dt)) p <- p + geom_point(data=true_dt, aes(plot_x, plot_y), colour="#de2d26", alpha=0.98, size=0.85)
  if (nrow(true_dt)) {
    if (use_repel) p <- p + ggrepel::geom_text_repel(data=true_dt, aes(plot_x, plot_y, label=true_label), colour="#de2d26", size=4.2, box.padding=0.25, point.padding=0.2, min.segment.length=0)
    else p <- p + geom_text(data=true_dt, aes(plot_x, plot_y, label=true_label), colour="#de2d26", vjust=-0.7, size=4.0)
  }
  if (is.finite(o$ld_dist) && o$ld_dist > 0) p <- p + geom_vline(xintercept=o$ld_dist, colour="black", linetype="dashed", linewidth=0.35)
  if (is.finite(o$ld_dist_alt) && o$ld_dist_alt > 0) p <- p + geom_vline(xintercept=o$ld_dist_alt, colour="hotpink1", linetype="dashed", linewidth=0.35)
  p <- p + geom_hline(yintercept=o$threshold, colour="#238b45", linetype="dashed", linewidth=0.35) +
    scale_x_continuous(breaks=function(x) pretty(x, n=10), limits=c(0, xmax)) +
    scale_y_continuous(breaks=yticks, limits=ylim) +
    xlab(ifelse(o$x_mode == "distance", paste0("Distance between SNP/unitig columns (", o$distance_col, ")"), xlab)) +
    ylab(ylab) + ggtitle(o$title) + theme_minimal(base_size=14) +
    theme(legend.position="none", plot.margin=margin(8,12,8,8), plot.title=element_text(size=19), axis.title=element_text(size=16), axis.text=element_text(size=12)) +
    annotate("text", x=xmax*0.98, y=ylim[1]+0.94*diff(ylim), label=paste0("#Score rows: ", floor(nrow(dt)/1000), "K"), hjust=1, size=5.0) +
    annotate("text", x=xmax*0.98, y=ylim[1]+0.90*diff(ylim), label=paste0("#Threshold: ", sprintf("%.2f", o$threshold)), hjust=1, size=5.0) +
    annotate("text", x=xmax*0.98, y=ylim[1]+0.86*diff(ylim), label=paste0("#Significant pairs: ", nrow(hits)), hjust=1, size=5.0)
  save_gg_all(p, o$out, o$width, o$height, o$dpi, o$write_svg)
} else {
  open_base_device(o$out, o$width, o$height, o$dpi)
  plot(0,0,type="n",xlim=c(0,xmax),ylim=ylim,xaxs="i",yaxs="i",xlab="",ylab="",xaxt="n",yaxt="n",bty="n")
  if (nrow(bg)) points(bg$plot_x,bg$plot_y,col=rgb(140,140,140,70,maxColorValue=255),pch=19,cex=0.25)
  if (nrow(hits)) points(hits$plot_x,hits$plot_y,col="#252525",pch=19,cex=0.35)
  if (nrow(true_dt)) { points(true_dt$plot_x,true_dt$plot_y,col="#de2d26",pch=19,cex=0.70); text(true_dt$plot_x,true_dt$plot_y,labels=true_dt$true_label,col="#de2d26",pos=3,cex=0.75) }
  abline(h=o$threshold,col="#238b45",lty=2)
  if (is.finite(o$ld_dist) && o$ld_dist > 0) abline(v=o$ld_dist,lty=2)
  axis(1, at=xb, labels=xb, tick=FALSE, line=-0.8); axis(2, at=yticks, labels=yticks, las=1)
  title(main=o$title, xlab=ifelse(o$x_mode == "distance", paste0("Distance between SNP/unitig columns (", o$distance_col, ")"), xlab), ylab=ylab)
  dev.off()
}
cat("Wrote: ", o$out, "\n", sep="")
