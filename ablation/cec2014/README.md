# CEC2014 DMGSO Ablation

## Scope

This directory contains the revision-stage CEC2014 ablation experiment for
DMGSO variants V0--V4.

## Experimental design

- Benchmark: CEC2014
- Functions: F1--F30
- Dimension: D=30
- Variants: V0, V1, V2, V3, V4
- Deterministic configurations per variant/function: 10
- Maximum function-evaluation budget: 300,000
- Total run-level records: 1,500
- Recorded status: all runs `ok`

The 1,500 records correspond to:

5 variants x 30 functions x 10 deterministic configurations.

## Variant definitions

- V0: directional sensing only
- V1: V0 + adaptive step regulation
- V2: V1 + long-range sensing
- V3: V1 + memory guidance
- V4: full architecture combining adaptive step regulation,
  long-range sensing, memory guidance, and relocation

## Execution

`execution/dmgso_core_revision.py`
contains the frozen DMGSO implementation.

`execution/run_ablation_dmgso_cec2014_safe.py`
is the safe revision-stage runner. It validates an existing run-level
dataset before reuse/resume, checks the expected experimental design and
function-evaluation budget, and completes missing experiment keys without
silently replacing validated records.

## Data provenance

The primary run-level experimental record is:

`data/raw/ablation_cec2014_run_summary_revision.csv`

The frozen copy is:

`data/frozen/ablation_cec2014_run_summary_revision_FROZEN.csv`

These two files were verified to be byte-for-byte identical.

A frozen function-level summary generated from the revision experiment is
also retained in `data/frozen/`.

## Final summaries and figures

The `summaries/` directory contains the final version-, function-, and
category-level summaries used for the revised manuscript.

The `figures/` directory contains the eight final CEC2014 ablation figures.

`execution/plot_ablation_cec2014_figures_REVISION.py`
is the final plotting script associated with these manuscript outputs.

## Scope boundary

Earlier pilot, historical, architecture-development, and superseded
visualization branches are intentionally excluded. They are not required
to reproduce the revision-stage CEC2014 ablation results reported here.
