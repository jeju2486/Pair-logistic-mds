from __future__ import annotations

import shutil
import subprocess
import sys
from importlib import resources


def main(argv: list[str] | None = None) -> int:
    """Console entry point for the packaged R plotter.

    This exposes the R plotting script as:

        pair-mds-gwes-plot --input results/pair_mds_logistic.tsv --out plot.png

    It still requires Rscript plus the R packages ggplot2 and optionally data.table
    to be available in the user's environment.
    """
    if argv is None:
        argv = sys.argv[1:]

    rscript = shutil.which("Rscript")
    if rscript is None:
        sys.stderr.write(
            "ERROR: Rscript was not found in PATH.\n"
            "Install R, then install the R dependencies:\n"
            "  install.packages(c('ggplot2', 'data.table'))\n"
        )
        return 127

    try:
        script_path = resources.files("pair_mds_gwes").joinpath("scripts", "plot_pair_mds_gwes.R")
    except Exception as exc:
        sys.stderr.write(f"ERROR: could not locate packaged R plotting script: {exc}\n")
        return 1

    cmd = [rscript, str(script_path), *argv]
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
