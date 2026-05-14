#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from importlib.resources import files


def main() -> int:
    r_script = files("pair_logistic_mds").joinpath(
        "scripts/plot_pair_logistic_mds.R"
    )

    if not r_script.is_file():
        sys.stderr.write(f"ERROR: R plotting script not found: {r_script}\n")
        return 1

    cmd = ["Rscript", str(r_script)] + sys.argv[1:]
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())