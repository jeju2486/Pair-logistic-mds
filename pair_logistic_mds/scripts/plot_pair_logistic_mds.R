#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  if (!requireNamespace("data.table", quietly = TRUE)) {
    stop("Missing R package: data.table", call. = FALSE)
  }
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("Missing R package: ggplot2", call. = FALSE)
  }

  library(data.table)
  library(ggplot2)
})

print_usage <- function() {
  cat("
Usage:
  Rscript plot_pair_logistic_mds.R \\
    --score score_all.tsv \\
    --top top_score_or_exact.tsv \\
    --out plot.png \\
    [options]

Required:
  --score          Full quick score-scan TSV
  --top            Top-score TSV or exact-refit TSV
  --out            Output plot file: .png or .pdf

Optional:
  --y-col          P-value column for background score scan
                   default: score_p_pair_max

  --top-y-col      P-value column for top-pair overlay
                   default: same as --y-col

  --max-distance   Maximum distance to plot
                   default: no maximum

  --keep-negative-distance
                   Keep distance < 0 rows
                   default: remove distance < 0

  --width          Figure width
                   default: 8

  --height         Figure height
                   default: 5

  --dpi            PNG resolution
                   default: 300

  --title          Plot title
                   default: MDS-adjusted pairwise locus association

Example:
  Rscript plot_pair_logistic_mds.R \\
    --score results/pair_logistic_mds.score_all.tsv \\
    --top results/pair_logistic_mds.score_top.tsv \\
    --out plots/pair_logistic_mds_score_distance.png

Exact overlay example:
  Rscript plot_pair_logistic_mds.R \\
    --score results/pair_logistic_mds.score_all.tsv \\
    --top results/pair_logistic_mds.exact_top.tsv \\
    --out plots/pair_logistic_mds_exact_overlay_distance.png \\
    --y-col score_p_pair_max \\
    --top-y-col p_pair_max
\n")
}

parse_args <- function(args) {
  opts <- list(
    score = NULL,
    top = NULL,
    out = NULL,
    y_col = "score_p_pair_max",
    top_y_col = NULL,
    max_distance = NA_real_,
    keep_negative_distance = FALSE,
    bins = 250,
    width = 8,
    height = 5,
    dpi = 300,
    title = "MDS-adjusted pairwise locus association"
  )
  i <- 1
  while (i <= length(args)) {
    a <- args[[i]]

    if (a %in% c("-h", "--help")) {
      print_usage()
      quit(save = "no", status = 0)
    } else if (a == "--score") {
      opts$score <- args[[i + 1]]
      i <- i + 2
    } else if (a == "--top") {
      opts$top <- args[[i + 1]]
      i <- i + 2
    } else if (a == "--out") {
      opts$out <- args[[i + 1]]
      i <- i + 2
    } else if (a == "--y-col") {
      opts$y_col <- args[[i + 1]]
      i <- i + 2
    } else if (a == "--top-y-col") {
      opts$top_y_col <- args[[i + 1]]
      i <- i + 2
    } else if (a == "--max-distance") {
      opts$max_distance <- as.numeric(args[[i + 1]])
      i <- i + 2
    } else if (a == "--keep-negative-distance") {
      opts$keep_negative_distance <- TRUE
      i <- i + 1
    } else if (a == "--width") {
      opts$width <- as.numeric(args[[i + 1]])
      i <- i + 2
    } else if (a == "--height") {
      opts$height <- as.numeric(args[[i + 1]])
      i <- i + 2
    } else if (a == "--dpi") {
      opts$dpi <- as.numeric(args[[i + 1]])
      i <- i + 2
    } else if (a == "--title") {
      opts$title <- args[[i + 1]]
      i <- i + 2
    } else if (a == "--bins") {
      opts$bins <- as.integer(args[[i + 1]])
      i <- i + 2
    } else {
      stop("Unknown argument: ", a, call. = FALSE)
    }
  }

  if (is.null(opts$score) || is.null(opts$top) || is.null(opts$out)) {
    print_usage()
    stop("Missing required argument: --score, --top, and --out are required.", call. = FALSE)
  }

  if (is.null(opts$top_y_col)) {
    opts$top_y_col <- opts$y_col
  }

  opts
}

make_pair_id <- function(dt) {
  if (!all(c("u", "v") %in% names(dt))) {
    stop("Input file must contain columns: u and v", call. = FALSE)
  }

  u <- as.integer(dt$u)
  v <- as.integer(dt$v)

  paste(pmin(u, v), pmax(u, v), sep = "__")
}

prepare_plot_data <- function(dt, p_col, label, keep_negative_distance, max_distance) {
  if (!("distance" %in% names(dt))) {
    stop("Input file must contain column: distance", call. = FALSE)
  }
  if (!(p_col %in% names(dt))) {
    stop("P-value column not found: ", p_col, call. = FALSE)
  }

  out <- copy(dt)

  out[, distance := as.numeric(distance)]
  out[, p_value := as.numeric(get(p_col))]

  if (!keep_negative_distance) {
    out <- out[is.finite(distance) & distance >= 0]
  } else {
    out <- out[is.finite(distance)]
  }

  if (is.finite(max_distance)) {
    out <- out[distance <= max_distance]
  }

  out <- out[is.finite(p_value) & p_value > 0]

  if (nrow(out) == 0) {
    warning("No valid rows remained for: ", label)
    return(out)
  }

  out[, p_value := pmax(p_value, .Machine$double.xmin)]
  out[, neglog10p := -log10(p_value)]
  out[, layer := label]

  out
}

args <- commandArgs(trailingOnly = TRUE)
opts <- parse_args(args)

message("[1/5] Reading input files")
score_dt <- fread(opts$score)
top_dt <- fread(opts$top)

message("  score rows: ", nrow(score_dt))
message("  top rows:   ", nrow(top_dt))

message("[2/5] Checking columns")

required_score <- c("u", "v", "distance", opts$y_col)
missing_score <- setdiff(required_score, names(score_dt))
if (length(missing_score) > 0) {
  stop("Score file missing columns: ", paste(missing_score, collapse = ", "), call. = FALSE)
}

required_top <- c("u", "v", "distance", opts$top_y_col)
missing_top <- setdiff(required_top, names(top_dt))
if (length(missing_top) > 0) {
  stop("Top file missing columns: ", paste(missing_top, collapse = ", "), call. = FALSE)
}

message("[3/5] Removing top pairs from background to avoid double plotting")

score_dt[, pair_id := make_pair_id(score_dt)]
top_dt[, pair_id := make_pair_id(top_dt)]

top_ids <- unique(top_dt$pair_id)

background_dt <- score_dt[!(pair_id %in% top_ids)]

message("  background rows after removing top pairs: ", nrow(background_dt))
message("  overlay top rows: ", nrow(top_dt))

message("[4/5] Preparing plot data")

background_plot <- prepare_plot_data(
  background_dt,
  p_col = opts$y_col,
  label = "All score-scan pairs",
  keep_negative_distance = opts$keep_negative_distance,
  max_distance = opts$max_distance
)

top_label <- if (opts$top_y_col == opts$y_col) {
  "Top selected pairs"
} else {
  "Top pairs, exact/refit p-value"
}

top_plot <- prepare_plot_data(
  top_dt,
  p_col = opts$top_y_col,
  label = top_label,
  keep_negative_distance = opts$keep_negative_distance,
  max_distance = opts$max_distance
)

message("  background plotted rows: ", nrow(background_plot))
message("  top plotted rows:        ", nrow(top_plot))

plot_dt <- rbindlist(
  list(background_plot, top_plot),
  use.names = TRUE,
  fill = TRUE
)

if (nrow(plot_dt) == 0) {
  stop("No rows available for plotting after filtering.", call. = FALSE)
}

message("[5/5] Making plot")

p <- ggplot() +
  geom_bin2d(
    data = background_plot,
    aes(x = distance, y = neglog10p),
    bins = opts$bins
  ) +
  scale_fill_gradient(
    low = "grey90",
    high = "grey25",
    name = "Pair count"
  ) +
  geom_point(
    data = top_plot,
    aes(x = distance, y = neglog10p),
    size = 1.05,
    alpha = 0.85,
    colour = "red"
  ) +
  labs(
    title = opts$title,
    subtitle = NULL,
    caption = paste0(
      "Background: ", opts$y_col,
      " | Overlay: ", opts$top_y_col,
      " | top pairs removed from background"
    ),
    x = "Distance",
    y = expression(-log[10](p))
  ) +
  theme_bw(base_size = 12) +
  theme(
    panel.grid.minor = element_blank(),
    plot.title = element_text(face = "bold", size = 14),
    plot.caption = element_text(size = 8, hjust = 0),
    plot.margin = margin(8, 12, 8, 8),
    legend.position = "right"
  )

out_path <- opts$out
out_dir <- dirname(out_path)
if (!dir.exists(out_dir) && out_dir != ".") {
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
}

if (grepl("\\.pdf$", out_path, ignore.case = TRUE)) {
  ggsave(
    filename = out_path,
    plot = p,
    width = opts$width,
    height = opts$height,
    device = cairo_pdf
  )
} else {
  ggsave(
    filename = out_path,
    plot = p,
    width = opts$width,
    height = opts$height,
    dpi = opts$dpi
  )
}

message("Done: ", out_path)
