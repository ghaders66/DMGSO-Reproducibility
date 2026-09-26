# COCO/BBOB DMGSO Ablation

## Scope

This directory contains the audited revision-stage COCO/BBOB ablation
experiment for DMGSO variants V0--V4.

## Experimental design

- Benchmark: COCO/BBOB
- Functions: F1--F24
- Dimensions: D=2, 5, 10, 20, 40
- Instances: 1, 2, 3, 4, 5, 71--80
- Variants: V0, V1, V2, V3, V4
- Run/configuration IDs: 0--4
- Function-evaluation budget: 1000 x D
- Total run-level records: 45,000
- Recorded status: all runs `ok`

The 45,000 records correspond to:

5 variants x 24 functions x 5 dimensions x 15 instances x 5
deterministic run/configuration IDs.

The same run ID is paired across V0--V4.

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

`execution/run_ablation_dmgso_coco_bbob_revision.py`
is the audited revision-stage experiment runner.

The runner uses the official frozen COCO/BBOB optimum-value table for the
revision grid rather than assuming an optimum of zero. It performs
problem-ID, dimension, function, and instance consistency checks and fails
fast on missing or inconsistent optimum values.

The reported error is based on the benchmark optimum associated with each
COCO/BBOB problem.

## Audited run-level data

The primary revision-stage dataset is:

`data/audited/ablation_coco_bbob_audited_run_summary.csv`

It contains exactly 45,000 validated experiment records.

This audited dataset represents a revision-stage execution using the
frozen revision DMGSO core. It must not be interpreted as merely a
post-processing correction of the earlier historical ablation dataset.

The earlier historical experiment used a different execution branch and
is intentionally excluded from the main reproducibility chain.

## Final summaries and figures

The `summaries/` directory contains the final audited function-,
dimension-, and version-level summaries.

The `figures/` directory contains the ten final COCO/BBOB ablation figures.

`execution/plot_ablation_coco_bbob_figures_SUBMITTED_STYLE_AUDITED_FONT_FIXED.py`
is the final plotting script associated with these manuscript outputs.

## Scope boundary

Historical, exploratory, superseded visualization, and pre-audit ablation
files are intentionally excluded. The materials included here define the
revision-stage reproducibility chain used for the revised manuscript.
