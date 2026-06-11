#!/usr/bin/env Rscript

# pair-mds-gwes-plot
#
# Previous-tool-style distance plot for pair_mds_gwes output.
#
# Plot scheme:
#   grey   = background score rows
#   green  = high-confidence selected rows
#   orange = high-score non-OK/suspicious rows
#   red    = true/real epistasis rows from block files or --highlight-pair
#
# This is intentionally similar to pair_saige_glmm_new_plotting.R,
# but adapted to pair_mds_gwes compact columns.

suppressPackageStartupMessages({
  if (!requireNamespace("data.table", quietly = TRUE)) {
    stop("Missing R package: data.table. Install with install.packages('data.table')", call. = FALSE)
  }
  library(data.table)
})

log_step <- function(msg) {
  cat(sprintf("[%s] %s\n", format(Sys.time(), "%H:%M:%S"), msg))
  flush.console()
}

die <- function(...) {
  stop(paste0(...), call. = FALSE)
}

print_usage <- function() {
  cat("Usage:\n")
  cat("  pair-mds-gwes-plot --score pair_mds_logistic.tsv --out out.png [options]\n")
  cat("  pair-mds-gwes-plot --input pair_mds_logistic.tsv --out out.png [options]\n\n")
  cat("Required:\n")
  cat("      --score, --input       pair_mds_logistic.tsv\n")
  cat("      --out, -o              Output PNG/PDF/JPG\n\n")
  cat("Optional data/score options:\n")
  cat("  -y, --y-col                Y column. Default: bidirectional_score\n")
  cat("      --top                  Optional high-confidence/top TSV. Default: derive from score file\n")
  cat("      --top-y-col            Top overlay Y column. Default: same as --y-col\n")
  cat("      --top-min-y            Minimum y for green high-confidence rows. Default: 5\n")
  cat("      --suspicious-min-y     Minimum y for orange non-OK rows. Default: 5\n")
  cat("      --top-n                If --top absent, select top N OK rows above --top-min-y. Default: 100\n")
  cat("      --top-out              Optional TSV of selected high-confidence rows\n")
  cat("      --n-rows               Read first N rows from score file. Default: all\n\n")
  cat("Optional x-axis/distance options:\n")
  cat("      --distance-col         Distance column. Default: distance\n")
  cat("      --x-mode               distance|absdiff|midpoint|u|v. Default: distance\n")
  cat("  -l, --ld-dist              LD-distance vertical line. Default: 0 off\n")
  cat("      --ld-dist-alt          Second vertical line. Default: 0 off\n")
  cat("      --min-dist             Filter distance >= min-dist. Default: 0\n")
  cat("      --max-dist             Filter distance <= max-dist. Default: Inf\n")
  cat("      --max-points           Downsample grey background only. Default: 0 off\n")
  cat("      --seed                 RNG seed for downsampling. Default: 1\n\n")
  cat("Real-epistasis highlighting:\n")
  cat("      --inblock              Prefix/dir for true block files. Example prefix: true_snp_blocks\n")
  cat("      --true-block-ids       Comma-separated true block ids, e.g. 0,1,2,3,4\n")
  cat("      --pair-mode            cross|all|de|none. Default: cross\n")
  cat("      --highlight-pair       Extra unordered pair(s): 50000,90000 or 50000,90000;20000,30000\n\n")
  cat("Plot options:\n")
  cat("      --title                Plot title. Default: auto\n")
  cat("      --width                Width in pixels for PNG/JPG, inches for PDF. Default: 2400\n")
  cat("      --height               Height in pixels for PNG/JPG, inches for PDF. Default: 1200\n")
  cat("      --dpi                  DPI for ggplot PNG/JPG. Default: 300\n")
  cat("      --no-deps              Force base plotting. Default: use ggplot2 if available\n")
  cat("  -h, --help                 Help\n")
}

parse_args <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  o <- list(
    score = NULL,
    top = NA_character_,
    output = NULL,
    y_col = "bidirectional_score",
    top_y_col = NA_character_,
    top_min_y = 5,
    suspicious_min_y = 5,
    top_n = 100,
    top_out = NA_character_,
    n_rows = 0,
    distance_col = "distance",
    x_mode = "distance",
    ld_dist = 0,
    ld_dist_alt = 0,
    min_dist = 0,
    max_dist = Inf,
    max_points = 0,
    seed = 1L,
    inblock = NA_character_,
    true_block_ids = NA_character_,
    pair_mode = "cross",
    highlight_pair = NA_character_,
    title = NA_character_,
    width = 2400,
    height = 1200,
    dpi = 300,
    no_deps = FALSE
  )

  if (length(args) == 0) {
    print_usage(); quit(status = 0)
  }

  i <- 1
  while (i <= length(args)) {
    a <- args[i]
    if (a %in% c("-h", "--help")) {
      print_usage(); quit(status = 0)
    } else if (a %in% c("--score", "--input")) { o$score <- args[i + 1]; i <- i + 1
    } else if (a == "--top") { o$top <- args[i + 1]; i <- i + 1
    } else if (a %in% c("-o", "--out", "--output")) { o$output <- args[i + 1]; i <- i + 1
    } else if (a %in% c("-y", "--y-col", "--score-col")) { o$y_col <- args[i + 1]; i <- i + 1
    } else if (a == "--top-y-col") { o$top_y_col <- args[i + 1]; i <- i + 1
    } else if (a == "--top-min-y" || a == "--threshold") { o$top_min_y <- suppressWarnings(as.numeric(args[i + 1])); i <- i + 1
    } else if (a == "--suspicious-min-y") { o$suspicious_min_y <- suppressWarnings(as.numeric(args[i + 1])); i <- i + 1
    } else if (a == "--top-n") { o$top_n <- suppressWarnings(as.integer(args[i + 1])); i <- i + 1
    } else if (a == "--top-out") { o$top_out <- args[i + 1]; i <- i + 1
    } else if (a == "--n-rows") { o$n_rows <- suppressWarnings(as.numeric(args[i + 1])); i <- i + 1
    } else if (a == "--distance-col") { o$distance_col <- args[i + 1]; i <- i + 1
    } else if (a == "--x-mode") { o$x_mode <- args[i + 1]; i <- i + 1
    } else if (a %in% c("-l", "--ld-dist")) { o$ld_dist <- suppressWarnings(as.numeric(args[i + 1])); i <- i + 1
    } else if (a == "--ld-dist-alt") { o$ld_dist_alt <- suppressWarnings(as.numeric(args[i + 1])); i <- i + 1
    } else if (a == "--min-dist") { o$min_dist <- suppressWarnings(as.numeric(args[i + 1])); i <- i + 1
    } else if (a == "--max-dist") { o$max_dist <- suppressWarnings(as.numeric(args[i + 1])); i <- i + 1
    } else if (a == "--max-points") { o$max_points <- suppressWarnings(as.numeric(args[i + 1])); i <- i + 1
    } else if (a == "--seed") { o$seed <- suppressWarnings(as.integer(args[i + 1])); i <- i + 1
    } else if (a == "--inblock") { o$inblock <- args[i + 1]; i <- i + 1
    } else if (a == "--true-block-ids") { o$true_block_ids <- args[i + 1]; i <- i + 1
    } else if (a == "--pair-mode") { o$pair_mode <- args[i + 1]; i <- i + 1
    } else if (a == "--highlight-pair") { o$highlight_pair <- args[i + 1]; i <- i + 1
    } else if (a == "--title") { o$title <- args[i + 1]; i <- i + 1
    } else if (a == "--width") { o$width <- suppressWarnings(as.numeric(args[i + 1])); i <- i + 1
    } else if (a == "--height") { o$height <- suppressWarnings(as.numeric(args[i + 1])); i <- i + 1
    } else if (a == "--dpi") { o$dpi <- suppressWarnings(as.integer(args[i + 1])); i <- i + 1
    } else if (a == "--no-deps") { o$no_deps <- TRUE
    } else {
      die("Unknown option: ", a, " (use --help)")
    }
    i <- i + 1
  }

  if (is.null(o$score) || is.null(o$output)) {
    print_usage(); die("Missing required: --score/--input and --out")
  }
  if (is.na(o$top_y_col) || !nzchar(o$top_y_col)) o$top_y_col <- o$y_col
  if (!is.finite(o$top_min_y)) o$top_min_y <- 5
  if (!is.finite(o$suspicious_min_y)) o$suspicious_min_y <- 5
  if (!is.finite(o$top_n) || o$top_n < 0) o$top_n <- 100
  if (!o$pair_mode %in% c("cross", "all", "de", "none")) {
    die("--pair-mode must be one of: cross, all, de, none")
  }
  if (!o$x_mode %in% c("distance", "absdiff", "midpoint", "u", "v")) {
    die("--x-mode must be one of: distance, absdiff, midpoint, u, v")
  }
  o
}

safe_neglog10 <- function(p) {
  p <- suppressWarnings(as.numeric(p))
  p <- pmax(p, .Machine$double.xmin)
  -log10(p)
}

make_key_cols <- function(dt) {
  if (!all(c("u", "v") %in% names(dt))) die("Input must contain columns: u and v")
  dt[, u := suppressWarnings(as.integer(u))]
  dt[, v := suppressWarnings(as.integer(v))]
  dt[, u0 := pmin(u, v)]
  dt[, v0 := pmax(u, v)]
  dt[, key_pair := paste(u0, v0, sep = ":")]
  dt
}

get_y_values <- function(dt, y_col) {
  if (y_col %in% names(dt)) {
    x <- suppressWarnings(as.numeric(dt[[y_col]]))
    p_like <- grepl("^p($|_)|_p$|p_value", y_col) && !grepl("neglog|score", y_col)
    if (p_like) return(safe_neglog10(x))
    return(x)
  }
  if (y_col == "neglog10_p_u_to_v" && "p_u_to_v" %in% names(dt)) return(safe_neglog10(dt[["p_u_to_v"]]))
  if (y_col == "neglog10_p_v_to_u" && "p_v_to_u" %in% names(dt)) return(safe_neglog10(dt[["p_v_to_u"]]))
  die("Missing y column: ", y_col)
}

add_distance <- function(dt, opts) {
  if (opts$x_mode == "distance") {
    if (opts$distance_col %in% names(dt)) {
      dt[, plot_x := suppressWarnings(as.numeric(get(opts$distance_col)))]
      attr(dt, "x_label") <- opts$distance_col
    } else {
      dt[, plot_x := abs(v - u)]
      attr(dt, "x_label") <- "abs(v - u)"
      warning("distance column not found; using abs(v - u)")
    }
  } else if (opts$x_mode == "absdiff") {
    dt[, plot_x := abs(v - u)]
    attr(dt, "x_label") <- "abs(v - u)"
  } else if (opts$x_mode == "midpoint") {
    dt[, plot_x := (u + v) / 2]
    attr(dt, "x_label") <- "pair midpoint"
  } else if (opts$x_mode == "u") {
    dt[, plot_x := u]
    attr(dt, "x_label") <- "u"
  } else if (opts$x_mode == "v") {
    dt[, plot_x := v]
    attr(dt, "x_label") <- "v"
  }
  dt
}

prep_dt <- function(dt, opts, require_ok = FALSE, min_y = -Inf, y_col = opts$y_col) {
  if (nrow(dt) == 0) return(data.table(plot_x = numeric(), y = numeric()))
  out <- copy(dt)
  out <- add_distance(out, opts)
  if (!("status" %in% names(out))) out[, status := "OK"]
  if (require_ok) out <- out[status == "OK"]
  out[, y := if ("plot_y" %in% names(out)) suppressWarnings(as.numeric(plot_y)) else get_y_values(.SD, y_col)]
  out <- out[is.finite(plot_x) & is.finite(y)]
  if (opts$x_mode == "distance") {
    out <- out[plot_x != -1]
    out <- out[plot_x >= opts$min_dist]
    if (is.finite(opts$max_dist)) out <- out[plot_x <= opts$max_dist]
  }
  out <- out[y >= min_y]
  out
}

is_suspicious_status <- function(x) {
  x <- as.character(x)
  !is.na(x) & nzchar(x) & x != "OK"
}

parse_true_block_ids <- function(x) {
  parts <- trimws(strsplit(x, ",", fixed = TRUE)[[1]])
  ids <- suppressWarnings(as.integer(parts))
  if (length(ids) < 2 || any(is.na(ids))) die("--true-block-ids must contain at least two comma-separated integers")
  unique(ids)
}

find_block_file <- function(inblock_path, block_id) {
  patt <- sprintf("[._]block_(0*%d)\\.txt$", block_id)
  if (file.exists(inblock_path) && file.info(inblock_path)$isdir) {
    hits <- list.files(inblock_path, pattern = patt, full.names = TRUE)
  } else {
    hits <- unique(c(
      Sys.glob(sprintf("%s.block_%d.txt", inblock_path, block_id)),
      Sys.glob(sprintf("%s_block_%d.txt", inblock_path, block_id)),
      Sys.glob(sprintf("%s.block_%05d.txt", inblock_path, block_id)),
      Sys.glob(sprintf("%s_block_%05d.txt", inblock_path, block_id))
    ))
  }
  if (length(hits) == 0) die(sprintf("Could not find block file for block %d under --inblock %s", block_id, inblock_path))
  hits[1]
}

read_block_unitigs <- function(fn) {
  u <- tryCatch(fread(fn, header = FALSE, sep = "\n", col.names = "unitig", showProgress = FALSE)[["unitig"]],
                error = function(e) character(0))
  as.character(u)
}

parse_highlight_pairs <- function(x) {
  if (is.na(x) || !nzchar(x)) return(data.table(u0 = integer(), v0 = integer(), key_pair = character()))
  pieces <- unlist(strsplit(x, ";", fixed = TRUE))
  rows <- list()
  for (p in pieces) {
    xy <- suppressWarnings(as.integer(unlist(strsplit(p, ",", fixed = TRUE))))
    if (length(xy) != 2 || any(is.na(xy))) die("Bad --highlight-pair entry: ", p)
    a <- min(xy); b <- max(xy)
    rows[[length(rows) + 1]] <- data.table(u0 = a, v0 = b, key_pair = paste(a, b, sep = ":"))
  }
  unique(rbindlist(rows))
}

mark_true_pairs <- function(dt, opts) {
  if (nrow(dt) == 0) return(rep(FALSE, 0))
  hp <- rep(FALSE, nrow(dt))

  # Direct explicit pairs.
  explicit <- parse_highlight_pairs(opts$highlight_pair)
  if (nrow(explicit) > 0) {
    setkey(explicit, key_pair)
    tmp <- copy(dt); setkey(tmp, key_pair)
    idx <- tmp[explicit, on = "key_pair", which = TRUE, nomatch = 0L]
    hp[idx] <- TRUE
  }

  # Block-based real-epistasis pairs.
  if (is.na(opts$inblock) || !nzchar(opts$inblock) ||
      is.na(opts$true_block_ids) || !nzchar(opts$true_block_ids) ||
      opts$pair_mode == "none") {
    return(hp)
  }

  log_step("Loading true-pair block unitig lists")
  block_ids <- parse_true_block_ids(opts$true_block_ids)
  block_sets <- vector("list", length(block_ids))
  names(block_sets) <- as.character(block_ids)

  for (bid in block_ids) {
    fn <- find_block_file(opts$inblock, bid)
    block_sets[[as.character(bid)]] <- read_block_unitigs(fn)
    cat("  block ", bid, ": ", length(block_sets[[as.character(bid)]]), " unitigs from ", fn, "\n", sep = "")
  }

  if (opts$pair_mode == "de") {
    if (length(block_ids) < 5) {
      warning("--pair-mode de requested but fewer than five block ids are available; using last two ids")
      selected_pairs <- list(tail(block_ids, 2))
    } else {
      selected_pairs <- list(block_ids[4:5])
    }
  } else if (opts$pair_mode == "all") {
    selected_pairs <- list()
    for (i in seq_along(block_ids)) {
      for (j in i:seq_along(block_ids)) {
        selected_pairs[[length(selected_pairs) + 1]] <- c(block_ids[i], block_ids[j])
      }
    }
  } else {
    selected_pairs <- utils::combn(block_ids, 2, simplify = FALSE)
  }

  ui_chr <- as.character(dt$u)
  uj_chr <- as.character(dt$v)

  for (bp in selected_pairs) {
    b1 <- as.character(bp[1]); b2 <- as.character(bp[2])
    s1 <- block_sets[[b1]]; s2 <- block_sets[[b2]]
    if (b1 == b2) {
      hp <- hp | (ui_chr %in% s1 & uj_chr %in% s1)
    } else {
      hp <- hp | ((ui_chr %in% s1 & uj_chr %in% s2) | (ui_chr %in% s2 & uj_chr %in% s1))
    }
  }

  cat("  true rows highlighted: ", sum(hp), "\n", sep = "")
  hp
}

compute_axis_breaks <- function(x, step = 50000) {
  max_x <- max(x, na.rm = TRUE)
  if (!is.finite(max_x) || max_x <= 0) max_x <- 1
  max_tick <- ceiling(max_x / step) * step
  seq(0, max_tick, by = step)
}

make_top_from_score <- function(score_dt, opts) {
  ok <- if ("status" %in% names(score_dt)) score_dt[status == "OK"] else score_dt
  ok <- ok[is.finite(plot_y) & plot_y >= opts$top_min_y]
  if (opts$top_n > 0 && nrow(ok) > opts$top_n) {
    ok <- ok[order(-plot_y)][seq_len(opts$top_n)]
  } else {
    ok <- ok[order(-plot_y)]
  }
  ok
}

opts <- parse_args()
t0 <- proc.time()[[3]]

if (!file.exists(opts$score)) die("Score file missing: ", opts$score)
if (!is.na(opts$top) && nzchar(opts$top) && !file.exists(opts$top)) die("Top file missing: ", opts$top)
if (file.exists(opts$output)) die("Output exists, delete or change --out: ", opts$output)

log_step("Reading score file")
n_to_read <- if (!is.finite(opts$n_rows) || opts$n_rows <= 0) Inf else as.integer(opts$n_rows)
score_dt <- fread(opts$score, sep = "\t", nrows = n_to_read, showProgress = TRUE)
if (!("status" %in% names(score_dt))) score_dt[, status := "OK"]
score_dt <- make_key_cols(score_dt)
score_dt[, plot_y := get_y_values(.SD, opts$y_col)]
score_dt <- add_distance(score_dt, opts)
x_label <- attr(score_dt, "x_label")
cat("Score rows: ", nrow(score_dt), "\n", sep = "")

if (!is.na(opts$top) && nzchar(opts$top)) {
  log_step("Reading top/high-confidence file")
  top_dt <- fread(opts$top, sep = "\t", showProgress = TRUE)
  if (!("status" %in% names(top_dt))) top_dt[, status := "OK"]
  top_dt <- make_key_cols(top_dt)
  top_dt[, plot_y := get_y_values(.SD, opts$top_y_col)]
  top_dt <- add_distance(top_dt, opts)
  top_dt <- top_dt[is.finite(plot_y) & plot_y >= opts$top_min_y]
} else {
  log_step("Selecting top/high-confidence rows from score file")
  top_dt <- make_top_from_score(score_dt, opts)
}
cat("High-confidence candidate rows: ", nrow(top_dt), "\n", sep = "")

if (!is.na(opts$top_out) && nzchar(opts$top_out)) {
  outdir_top <- dirname(opts$top_out)
  if (!dir.exists(outdir_top) && outdir_top != ".") dir.create(outdir_top, recursive = TRUE, showWarnings = FALSE)
  fwrite(top_dt, file = opts$top_out, sep = "\t", na = "NA")
  cat("Wrote top rows: ", opts$top_out, "\n", sep = "")
}

log_step("Marking true/real epistasis rows")
true_score_flag <- mark_true_pairs(score_dt, opts)
true_score_raw <- score_dt[true_score_flag]
cat("True/real epistasis score rows: ", nrow(true_score_raw), "\n", sep = "")
true_keys <- if (nrow(true_score_raw) > 0) unique(true_score_raw[, list(key_pair)]) else data.table(key_pair = character())

log_step("Selecting suspicious high-score rows")
susp_raw <- score_dt[is_suspicious_status(status) & is.finite(plot_y) & plot_y >= opts$suspicious_min_y]
cat("Suspicious high-score rows: ", nrow(susp_raw), "\n", sep = "")
susp_keys <- if (nrow(susp_raw) > 0) unique(susp_raw[, list(key_pair)]) else data.table(key_pair = character())

log_step("Splitting high-confidence rows into selected and true rows")
top_true_flag <- if (nrow(true_keys) > 0 && nrow(top_dt) > 0) {
  tmp_top <- copy(top_dt); setkey(tmp_top, key_pair)
  tmp_true <- copy(true_keys); setkey(tmp_true, key_pair)
  idx <- tmp_top[tmp_true, on = "key_pair", which = TRUE, nomatch = 0L]
  flag <- rep(FALSE, nrow(top_dt)); flag[idx] <- TRUE; flag
} else rep(FALSE, nrow(top_dt))
top_true_raw <- top_dt[top_true_flag]
top_other_raw <- top_dt[!top_true_flag]
cat("Top selected rows excluding true pairs: ", nrow(top_other_raw), "\n", sep = "")
cat("True-pair top rows: ", nrow(top_true_raw), "\n", sep = "")

log_step("Removing highlighted pairs from grey background")
top_keys <- if (nrow(top_dt) > 0) unique(top_dt[, list(key_pair)]) else data.table(key_pair = character())
remove_keys <- unique(rbindlist(list(top_keys, true_keys, susp_keys), use.names = TRUE, fill = TRUE))
score_bg <- copy(score_dt)
if (nrow(remove_keys) > 0) {
  setkey(score_bg, key_pair)
  setkey(remove_keys, key_pair)
  score_bg <- score_bg[!remove_keys, on = "key_pair"]
}

log_step("Preparing plot data")
bg <- prep_dt(score_bg, opts, require_ok = FALSE, min_y = -Inf, y_col = opts$y_col)
hi_top <- prep_dt(top_other_raw, opts, require_ok = TRUE, min_y = opts$top_min_y, y_col = opts$top_y_col)
hi_susp <- prep_dt(susp_raw, opts, require_ok = FALSE, min_y = opts$suspicious_min_y, y_col = opts$y_col)
hi_true <- prep_dt(true_score_raw, opts, require_ok = FALSE, min_y = -Inf, y_col = opts$y_col)

cat("Background plotted rows: ", nrow(bg), "\n", sep = "")
cat("High-confidence plotted rows: ", nrow(hi_top), "\n", sep = "")
cat("Suspicious plotted rows: ", nrow(hi_susp), "\n", sep = "")
cat("True/real epistasis plotted rows: ", nrow(hi_true), "\n", sep = "")
if (nrow(bg) == 0 && nrow(hi_top) == 0 && nrow(hi_susp) == 0 && nrow(hi_true) == 0) die("No rows left after filtering.")

if (is.finite(opts$max_points) && opts$max_points > 0 && nrow(bg) > opts$max_points) {
  log_step("Downsampling grey background")
  set.seed(opts$seed)
  bg <- bg[sample(seq_len(nrow(bg)), size = opts$max_points, replace = FALSE)]
}

all_x <- c(bg$plot_x, hi_top$plot_x, hi_susp$plot_x, hi_true$plot_x)
all_y <- c(bg$y, hi_top$y, hi_susp$y, hi_true$y)
x_breaks <- compute_axis_breaks(all_x, step = 50000)
x_max_tick <- max(x_breaks)
y_max <- max(all_y, na.rm = TRUE)
if (!is.finite(y_max) || y_max <= 0) y_max <- 1
y_lim <- c(0, y_max * 1.05)
y_ticks <- pretty(y_lim, n = 8)

n_score_k <- floor(nrow(score_dt) / 1000)
n_bg_k <- floor(nrow(bg) / 1000)

outdir <- dirname(opts$output)
if (!dir.exists(outdir) && outdir != ".") dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

have_ggplot2 <- (!opts$no_deps) && requireNamespace("ggplot2", quietly = TRUE)
have_hexbin <- have_ggplot2 && requireNamespace("hexbin", quietly = TRUE)
have_ggrastr <- have_ggplot2 && requireNamespace("ggrastr", quietly = TRUE)

log_step("Drawing plot")
col_bg <- "#8f8f8f"
col_top <- "#31a354"
col_susp <- "#f16913"
col_true <- "#de2d26"
col_ld <- "black"
col_ld_alt <- "hotpink1"

y_label <- opts$y_col
if (opts$y_col == "bidirectional_score") y_label <- "Bidirectional score: min(-log10 p in both directions)"

if (is.na(opts$title) || !nzchar(opts$title)) {
  opts$title <- "pair-mds-gwes candidate pangenome covariation"
}

if (have_ggplot2) {
  suppressPackageStartupMessages(library(ggplot2))
  if (have_hexbin) suppressPackageStartupMessages(library(hexbin))
  if (have_ggrastr) suppressPackageStartupMessages(library(ggrastr))

  p <- ggplot()
  if (nrow(bg) > 0) {
    if (have_hexbin && nrow(bg) > 50000) {
      bg_layer <- geom_hex(data = bg, aes(x = plot_x, y = y), bins = 300, fill = col_bg, alpha = 0.75)
      if (have_ggrastr) bg_layer <- ggrastr::rasterise(bg_layer, dpi = 1000)
      p <- p + bg_layer
    } else {
      p <- p + geom_point(data = bg, aes(x = plot_x, y = y), colour = col_bg, alpha = 0.20, size = 0.18)
    }
  }

  if (is.finite(opts$ld_dist) && opts$ld_dist > 0) p <- p + geom_vline(xintercept = opts$ld_dist, col = col_ld, linetype = "dashed", linewidth = 0.35)
  if (is.finite(opts$ld_dist_alt) && opts$ld_dist_alt > 0) p <- p + geom_vline(xintercept = opts$ld_dist_alt, col = col_ld_alt, linetype = "dashed", linewidth = 0.35)

  if (nrow(hi_top) > 0) p <- p + geom_point(data = hi_top, aes(x = plot_x, y = y), colour = col_top, size = 0.34, alpha = 0.85)
  if (nrow(hi_true) > 0) p <- p + geom_point(data = hi_true, aes(x = plot_x, y = y), colour = col_true, size = 0.48, alpha = 0.95)
  if (nrow(hi_susp) > 0) p <- p + geom_point(data = hi_susp, aes(x = plot_x, y = y), colour = col_susp, size = 0.42, alpha = 0.95)

  p <- p +
    scale_x_continuous(breaks = function(x) pretty(x, n = 10), limits = c(0, x_max_tick)) +
    scale_y_continuous(breaks = y_ticks, limits = y_lim) +
    xlab(ifelse(opts$x_mode == "distance", paste0("Distance between SNP/unitig columns (", opts$distance_col, ")"), x_label)) +
    ylab(y_label) +
    ggtitle(opts$title) +
    theme_minimal(base_size = 8) +
    theme(legend.position = "none", plot.margin = margin(5, 8, 5, 5)) +
    annotate("text", x = x_max_tick * 0.98, y = y_lim[1] + 0.94 * diff(y_lim), label = paste0("#Score rows: ", n_score_k, "K"), hjust = 1, size = 3.0) +
    annotate("text", x = x_max_tick * 0.98, y = y_lim[1] + 0.90 * diff(y_lim), label = paste0("#Background plotted: ", n_bg_k, "K"), hjust = 1, size = 3.0) +
    annotate("text", x = x_max_tick * 0.98, y = y_lim[1] + 0.86 * diff(y_lim), label = paste0("#High-confidence: ", nrow(hi_top)), hjust = 1, size = 3.0) +
    annotate("text", x = x_max_tick * 0.98, y = y_lim[1] + 0.82 * diff(y_lim), label = paste0("#Suspicious high-score: ", nrow(hi_susp)), hjust = 1, size = 3.0) +
    annotate("text", x = x_max_tick * 0.98, y = y_lim[1] + 0.78 * diff(y_lim), label = paste0("#Real epistasis highlighted: ", nrow(hi_true)), hjust = 1, size = 3.0)

  ext <- tolower(tools::file_ext(opts$output))
  if (ext == "pdf") {
    ggsave(filename = opts$output, plot = p, width = opts$width, height = opts$height, units = "in")
  } else {
    ggsave(filename = opts$output, plot = p, width = opts$width, height = opts$height, units = "px", dpi = opts$dpi)
  }
} else {
  png(opts$output, width = opts$width, height = opts$height, pointsize = 16)
  if (nrow(bg) > 0) {
    plot(bg$plot_x, bg$y, col = rgb(140, 140, 140, alpha = 80, maxColorValue = 255), type = "p", pch = 19, cex = 0.25, xlim = c(0, x_max_tick), ylim = y_lim, xaxs = "i", yaxs = "i", xlab = "", ylab = "", xaxt = "n", yaxt = "n", bty = "n")
  } else {
    plot(0, 0, type = "n", xlim = c(0, x_max_tick), ylim = y_lim, xaxs = "i", yaxs = "i", xlab = "", ylab = "", xaxt = "n", yaxt = "n", bty = "n")
  }
  if (is.finite(opts$ld_dist) && opts$ld_dist > 0) segments(opts$ld_dist, y_lim[1], opts$ld_dist, y_lim[2], col = col_ld, lty = 2, lwd = 2)
  if (is.finite(opts$ld_dist_alt) && opts$ld_dist_alt > 0) segments(opts$ld_dist_alt, y_lim[1], opts$ld_dist_alt, y_lim[2], col = col_ld_alt, lty = 2, lwd = 2)
  if (nrow(hi_top) > 0) points(hi_top$plot_x, hi_top$y, col = col_top, pch = 19, cex = 0.35)
  if (nrow(hi_true) > 0) points(hi_true$plot_x, hi_true$y, col = col_true, pch = 19, cex = 0.50)
  if (nrow(hi_susp) > 0) points(hi_susp$plot_x, hi_susp$y, col = col_susp, pch = 19, cex = 0.45)
  axis(1, at = x_breaks, tick = FALSE, labels = x_breaks, line = -0.8)
  title(xlab = ifelse(opts$x_mode == "distance", paste0("Distance between SNP/unitig columns (", opts$distance_col, ")"), x_label), line = 1.2)
  axis(2, at = y_ticks, labels = FALSE, tcl = -0.5)
  axis(2, at = y_ticks, labels = y_ticks, las = 1, tcl = -0.5)
  title(ylab = y_label, line = 2.5)
  text(0.90 * x_max_tick, y_lim[1] + 0.94 * diff(y_lim), paste0("#Score rows: ", n_score_k, "K"), cex = 1, adj = 0)
  text(0.90 * x_max_tick, y_lim[1] + 0.90 * diff(y_lim), paste0("#Background plotted: ", n_bg_k, "K"), cex = 1, adj = 0)
  text(0.90 * x_max_tick, y_lim[1] + 0.86 * diff(y_lim), paste0("#High-confidence: ", nrow(hi_top)), cex = 1, adj = 0)
  text(0.90 * x_max_tick, y_lim[1] + 0.82 * diff(y_lim), paste0("#Suspicious high-score: ", nrow(hi_susp)), cex = 1, adj = 0)
  text(0.90 * x_max_tick, y_lim[1] + 0.78 * diff(y_lim), paste0("#Real epistasis highlighted: ", nrow(hi_true)), cex = 1, adj = 0)
  dev.off()
}

cat(sprintf("Wrote: %s\n", opts$output))
cat(sprintf("Total time: %.2fs\n", proc.time()[[3]] - t0))
