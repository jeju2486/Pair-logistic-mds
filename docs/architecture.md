# Package architecture

KOVAR keeps production analysis code in the installed `ko_variation` package.
Validation simulations and benchmarks belong outside that package so they do
not become runtime accessory functions.

## Version 0.8.1 production structure

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
|   |-- diagnostics.py    # Native-thread and process diagnostics
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
neighbor-joining construction, and test-only plotter are not part of the 0.8.1
package.

## Response-pattern-first execution

The 0.8.1 scanner keeps the public directional table unchanged while scheduling
the expensive null-model stage by exact binary response pattern:

```text
validated candidate pairs
`-- original directional rows
    `-- exact canonical response patterns
        |-- one null PQL fit per pattern
        |-- batched predictor score tests
        `-- transformed diagnostics for identical/complement response loci
```

Canonicalization chooses one deterministic representative from a response and
its bitwise complement. It does not discard a locus or pair. The scan expands
the cached fit back to all original response loci before writing output, and
records both the representative and the complement transformation.

The first PQL iteration uses the prepared eigensystem of the tree/GRM because
its initial working weights are constant. Later iterations retain the exact
response-specific weighted eigendecomposition. Within each iteration, trial
variance-component objectives are evaluated in spectral coordinates; only the
accepted solution is reconstructed in sample space.

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
so version 0.8.1 does not claim an implemented globality classifier.

## Dependency direction

```text
cli.py
|-- diagnostics.py
|-- io_utils.py
|-- kinship.py
`-- scan.py
    |-- glmm.py
    `-- spa.py

plot_cli.py
`-- scripts/plot_ko_variation.R
```

`glmm.py` and `spa.py` contain statistical calculations and do not perform file
I/O. `scan.py` organizes hypotheses, exact response-pattern reuse, filters, and
corrections. The CLI owns paths and file creation. `diagnostics.py` inspects
native library/process settings but does not alter an explicit user setting.

## Deliberately deferred work

Version 0.8.1 does not include GPU execution, hard pair filtering for runtime,
or response-level checkpoint/resume. A safe checkpoint implementation must be
integrated with response scheduling, configuration/input fingerprints, atomic
shard writes, and deterministic final multiple-testing correction; a
write-at-program-end wrapper would not provide reliable resume behavior.
