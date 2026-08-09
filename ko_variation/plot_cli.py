from __future__ import annotations

import shutil
import subprocess
import sys
from importlib import resources


def main(argv: list[str] | None = None) -> int:
    """Wrapper for the packaged R plotting script.

    The core KOVAR scanner is standalone Python. Plotting remains an optional
    R-backed command so the statistical core has no plotting dependencies.
    """
    if argv is None:
        argv = sys.argv[1:]
    rscript = shutil.which("Rscript")
    if rscript is None:
        sys.stderr.write(
            "Missing command: Rscript\n"
            "Install R, then install required R packages if needed: data.table; ggplot2 is optional.\n"
        )
        return 127
    script = resources.files("ko_variation").joinpath("scripts/plot_ko_variation.R")
    cmd = [rscript, str(script), *argv]
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
