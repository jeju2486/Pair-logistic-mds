from __future__ import annotations

import shutil
import subprocess
import sys
from importlib import resources


def main(argv: list[str] | None = None) -> int:
    """Wrapper for the packaged simulation-test R plotting script."""
    if argv is None:
        argv = sys.argv[1:]
    rscript = shutil.which("Rscript")
    if rscript is None:
        sys.stderr.write(
            "Missing command: Rscript\n"
            "Install R, then install required R packages if needed: data.table; ggplot2 is optional.\n"
        )
        return 127
    script = resources.files("pair_lmm_gwes").joinpath("scripts/plot_pair_lmm_gwes_test.R")
    cmd = [rscript, str(script), *argv]
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
