#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  has_dt <- requireNamespace("data.table", quietly = TRUE)
  has_gg <- requireNamespace("ggplot2", quietly = TRUE)
})

die <- function(...) {
  cat("ERROR: ", paste0(...), "\n", sep = "", file = stderr())
  quit(status = 1)
}

usage <- function() {
  cat("
plot_pair_mds_gwes.R

Required:
  --input FILE          pair_mds_logistic.tsv
  --out FILE            output plot path (.png, .pdf, .jpg)

Options:
  --score-col COL       score column [bidirectional_score]
  --x-mode MODE         distance|absdiff|midpoint|u|v [distance]
  --distance-col COL    distance column name [distance]
  --title TEXT          plot title [auto]
  --highlight-pair P    unordered pair(s), e.g. 50000,90000 or 50000,90000;20000,35000
  --threshold VALUE     horizontal threshold line; 0 disables [0]
  --label-top-n N       label top N points; 0 disables [0]
  --top-out FILE        optional TSV of top plotted rows
  --top-n N             rows to write to --top-out [100]
  --max-y VALUE         y-axis upper limit; 0 disables [0]
  --width VALUE         plot width in inches [8]
  --height VALUE        plot height in inches [5]
  --dpi VALUE           PNG/JPG dpi [300]
  --point-size VALUE    point size [0.55]
  --point-alpha VALUE   point alpha [0.55]
  --help                show this help
", sep = "")
}

parse_args <- function(argv) {
  opts <- list(input=NULL, out=NULL, score_col="bidirectional_score",
               x_mode="distance", distance_col="distance", title=NULL,
               highlight_pair="", threshold=0, label_top_n=0,
               top_out="", top_n=100, max_y=0, width=8, height=5,
               dpi=300, point_size=0.55, point_alpha=0.55)
  if (length(argv) == 0 || any(argv %in% c("--help", "-h"))) {
    usage(); quit(status = 0)
  }
  i <- 1
  while (i <= length(argv)) {
    key <- argv[[i]]
    if (!startsWith(key, "--")) die("Unexpected argument: ", key)
    name <- gsub("-", "_", sub("^--", "", key))
    if (!name %in% names(opts)) die("Unknown option: ", key)
    if (i == length(argv)) die("Missing value after ", key)
    opts[[name]] <- argv[[i + 1]]
    i <- i + 2
  }
  num <- c("threshold","label_top_n","top_n","max_y","width","height","dpi",
           "point_size","point_alpha")
  for (f in num) {
    opts[[f]] <- as.numeric(opts[[f]])
    if (is.na(opts[[f]])) die("--", gsub("_","-",f), " must be numeric")
  }
  opts$label_top_n <- as.integer(opts$label_top_n)
  opts$top_n <- as.integer(opts$top_n)
  opts$dpi <- as.integer(opts$dpi)
  opts
}

read_table <- function(path) {
  if (!file.exists(path)) die("Input file not found: ", path)
  if (has_dt) return(as.data.frame(data.table::fread(path, sep="\t", data.table=FALSE)))
  read.delim(path, sep="\t", header=TRUE, stringsAsFactors=FALSE, check.names=FALSE)
}

parse_pairs <- function(s) {
  if (is.null(s) || s == "") return(matrix(numeric(0), ncol=2))
  pieces <- unlist(strsplit(s, ";", fixed=TRUE))
  out <- list()
  for (p in pieces) {
    xy <- suppressWarnings(as.numeric(unlist(strsplit(p, ",", fixed=TRUE))))
    if (length(xy) != 2 || any(is.na(xy))) die("Bad --highlight-pair entry: ", p)
    out[[length(out)+1]] <- sort(xy)
  }
  do.call(rbind, out)
}

main <- function() {
  opts <- parse_args(commandArgs(trailingOnly=TRUE))
  if (is.null(opts$input)) die("--input is required")
  if (is.null(opts$out)) die("--out is required")
  if (!has_gg) die("R package 'ggplot2' is required. Install with install.packages('ggplot2')")

  df <- read_table(opts$input)
  req <- c("u", "v", opts$score_col)
  miss <- setdiff(req, names(df))
  if (length(miss)) die("Missing column(s): ", paste(miss, collapse=", "))

  df$u <- as.numeric(df$u)
  df$v <- as.numeric(df$v)
  df$score <- as.numeric(df[[opts$score_col]])

  if (opts$x_mode == "distance") {
    if (opts$distance_col %in% names(df)) {
      df$x_value <- as.numeric(df[[opts$distance_col]])
      x_label <- opts$distance_col
    } else {
      df$x_value <- abs(df$v - df$u)
      x_label <- "abs(v - u)"
      warning("distance column not found; using abs(v - u)")
    }
  } else if (opts$x_mode == "absdiff") {
    df$x_value <- abs(df$v - df$u); x_label <- "abs(v - u)"
  } else if (opts$x_mode == "midpoint") {
    df$x_value <- (df$u + df$v) / 2; x_label <- "pair midpoint"
  } else if (opts$x_mode == "u") {
    df$x_value <- df$u; x_label <- "u"
  } else if (opts$x_mode == "v") {
    df$x_value <- df$v; x_label <- "v"
  } else {
    die("--x-mode must be distance, absdiff, midpoint, u, or v")
  }

  plot_df <- df[is.finite(df$x_value) & is.finite(df$score), , drop=FALSE]
  if (!nrow(plot_df)) die("No finite points to plot")

  plot_df$highlight <- FALSE
  hp <- parse_pairs(opts$highlight_pair)
  if (nrow(hp)) {
    uv <- cbind(pmin(plot_df$u, plot_df$v), pmax(plot_df$u, plot_df$v))
    for (i in seq_len(nrow(hp))) {
      plot_df$highlight <- plot_df$highlight | (uv[,1] == hp[i,1] & uv[,2] == hp[i,2])
    }
  }

  plot_df <- plot_df[order(plot_df$score, decreasing=TRUE), , drop=FALSE]

  if (!is.null(opts$top_out) && opts$top_out != "") {
    top_n <- min(opts$top_n, nrow(plot_df))
    write.table(plot_df[seq_len(top_n), , drop=FALSE], file=opts$top_out,
                sep="\t", quote=FALSE, row.names=FALSE)
  }

  if (is.null(opts$title) || opts$title == "") {
    opts$title <- paste0(opts$score_col, " by ", x_label)
  }

  p <- ggplot2::ggplot(plot_df, ggplot2::aes(x=x_value, y=score)) +
    ggplot2::geom_point(ggplot2::aes(shape=highlight),
                        alpha=opts$point_alpha, size=opts$point_size, na.rm=TRUE) +
    ggplot2::scale_shape_manual(values=c(`FALSE`=16, `TRUE`=17), guide="none") +
    ggplot2::labs(title=opts$title, x=x_label, y=opts$score_col) +
    ggplot2::theme_bw(base_size=11) +
    ggplot2::theme(panel.grid.minor=ggplot2::element_blank(),
                   plot.title=ggplot2::element_text(face="bold"))

  if (any(plot_df$highlight)) {
    p <- p + ggplot2::geom_point(data=plot_df[plot_df$highlight, , drop=FALSE],
                                 ggplot2::aes(x=x_value, y=score),
                                 size=max(2.0, opts$point_size * 3),
                                 shape=17, inherit.aes=FALSE)
  }
  if (opts$threshold > 0) p <- p + ggplot2::geom_hline(yintercept=opts$threshold, linetype="dashed")

  if (opts$label_top_n > 0) {
    lab <- head(plot_df, opts$label_top_n)
    lab$label <- paste0(lab$u, "-", lab$v)
    p <- p + ggplot2::geom_text(data=lab, ggplot2::aes(x=x_value, y=score, label=label),
                                size=2.5, vjust=-0.4, check_overlap=TRUE,
                                inherit.aes=FALSE)
  }

  if (opts$max_y > 0) p <- p + ggplot2::coord_cartesian(ylim=c(0, opts$max_y))

  out_dir <- dirname(opts$out)
  if (!dir.exists(out_dir)) dir.create(out_dir, recursive=TRUE)
  ggplot2::ggsave(filename=opts$out, plot=p, width=opts$width,
                  height=opts$height, dpi=opts$dpi)

  cat("Wrote plot:", opts$out, "\n")
  cat("Plotted rows:", nrow(plot_df), "\n")
  if (any(plot_df$highlight)) cat("Highlighted rows:", sum(plot_df$highlight), "\n")
}

main()
