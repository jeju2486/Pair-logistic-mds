# Package architecture

KOVAR keeps production analysis code in the installed `ko_variation` package.
Validation simulations and benchmarks belong outside that package so they do
not become runtime accessory functions.

## Version 0.8.0 production structure

```text
KOVAR/
|-- README.md
|-- LICENSE
|-- pyproject.toml
|-- requirements.txt
|-- .gitignore
|-- docs/
|   |-- architecture.md
|   |-- interpretation.md
|   |-- model.md
|   |-- research_basis.md
|   `-- validation.md
|-- ko_variation/
|   |-- __init__.py       # Version metadata
|   |-- cli.py            # Workflow, arguments and outputs
|   |-- io_utils.py       # FASTA and candidate-pair input
|   |-- kinship.py        # Tree covariance and fallback GRM
|   |-- glmm.py           # PQL fit, score test and full refit
|   |-- spa.py            # Optional experimental SPA
|   |-- scan.py           # Directions, filters and correction
|   |-- plot_cli.py       # Optional R plot wrapper
|   `-- scripts/
|       `-- plot_ko_variation.R
`-- tests/                # Numerical, workflow and CLI tests
```

The earlier LMM, CTMC/event filtration, ancestral-state reconstruction,
neighbor-joining construction, and test-only plotter are not part of the 0.8.0
package.

## Planned validation-only structure

```text
simulations/
|-- generate_phylogenetic_binary.py
|-- scenarios.yaml
|-- evaluate_calibration.py
`-- evaluate_power.py

benchmarks/
|-- compare_v070_lmm.py
|-- compare_glmm_reference.py
|-- compare_glmm_spa.py
`-- benchmark_runtime.py
```

These files are planned validation infrastructure, not implemented runtime
features.

## Planned optional lineage diagnostics

A later `lineage.py` module may provide leave-one-lineage-out influence and
effect-heterogeneity summaries. It should remain a diagnostic rather than a
CTMC or ancestral-event filter. Classification thresholds require simulation,
so version 0.8.0 does not claim an implemented globality classifier.

## Dependency direction

```text
cli.py
|-- io_utils.py
|-- kinship.py
`-- scan.py
    |-- glmm.py
    `-- spa.py

plot_cli.py
`-- scripts/plot_ko_variation.R
```

`glmm.py` and `spa.py` contain statistical calculations and do not perform file
I/O. `scan.py` organizes hypotheses and applies filters and corrections. The
CLI owns paths and file creation.
