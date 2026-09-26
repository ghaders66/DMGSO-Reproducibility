# Experimental Protocol

This document records the benchmark settings, observation semantics, function-
evaluation budgets, aggregation rules, and statistical units used in the
revised DMGSO experiments.

## CEC2014 external comparison

- Benchmark: CEC2014 F1--F30
- Dimension: D=30
- Maximum FE budget: 300,000 per observation
- Algorithms: DMGSO, CMA-ES, DE, PSO
- Observations per algorithm/function: 30
- Total run-level records: 3,600
- Final status: all 3,600 records `ok`

### Observation semantics

For DMGSO, `run_id=0,...,29` identifies deterministic configurations and
controls the deterministic Halton initialization/relocation. It is not a
stochastic-replication seed.

For CMA-ES, DE, and PSO, `run_id=0,...,29` identifies stochastic runs.
Their explicit random seeds are `1000 + run_id`, giving seeds 1000--1029.

### Aggregation and inference

The 30 observations within each algorithm/function pair are first aggregated
using the median error. Statistical inference then uses the 30 CEC2014
functions as paired inferential units (`n=30`). Individual run records are
not treated as independent inferential units.

The analysis includes Friedman testing, average ranks, paired Wilcoxon
signed-rank comparisons of DMGSO against each baseline, Holm correction,
paired rank-biserial effect sizes, bootstrap confidence intervals, and
descriptive realized-FE summaries.

## COCO/BBOB external comparison

- Functions: F1--F24
- Dimensions: D=2, 5, 10, 20, 40
- Instances: 1--5 and 71--80 (15 instances)
- Run/configuration IDs: 0--4
- Maximum FE budget: 1000 x D
- Algorithms: DMGSO, Nelder-Mead, Powell, COBYLA, Compass Search
- D2--D20 records: 36,000
- D40 records: 9,000
- Total run-level records: 45,000
- Historical execution status: all records `ok`

The D2--D20 and D40 experiments were executed separately.

### Official optimum correction

The revision-stage corrected error is

`error_corrected = best_f - f_opt_official`

where `f_opt_official` is obtained from the frozen official COCO/BBOB optimum
table generated with cocoex 2.8.2.

This correction is post-processing: the optimizer experiments were not rerun.
Execution fields were preserved, and the historical and frozen `best_f`
values were numerically identical under the documented strict tolerance.

Statistical analyses operate on the frozen corrected datasets and perform
paired comparisons separately by dimension.

## CEC2014 ablation experiment

- Functions: F1--F30
- Dimension: D=30
- Variants: V0, V1, V2, V3, V4
- Deterministic configurations per variant/function: 10
- Maximum FE budget: 300,000
- Total run-level records: 1,500
- Recorded status: all runs `ok`

The 1,500 records correspond to 5 variants x 30 functions x 10 deterministic
configurations.

## COCO/BBOB ablation experiment

- Functions: F1--F24
- Dimensions: D=2, 5, 10, 20, 40
- Instances: 1--5 and 71--80 (15 instances)
- Variants: V0, V1, V2, V3, V4
- Run/configuration IDs: 0--4
- Maximum FE budget: 1000 x D
- Total run-level records: 45,000
- Recorded status: all runs `ok`

The same deterministic run/configuration ID is paired across V0--V4.

The audited COCO/BBOB ablation dataset is a revision-stage execution using
the frozen revision DMGSO core. It is not merely a post-processing correction
of an earlier historical ablation experiment.

## Ablation variant semantics

- V0: directional sensing only
- V1: V0 + adaptive step regulation
- V2: V1 + long-range sensing
- V3: V1 + memory guidance
- V4: full architecture with adaptive step regulation, long-range sensing,
  memory guidance, and relocation

V2 and V3 are parallel extensions of V1. Consequently, V2-to-V3 is not an
isolated one-component transition.

## Sensing-topology experiment

- Benchmark: COCO/BBOB F1--F24
- Dimensions: D=20 and D=40
- Instances: 15 official instances
- Deterministic run/configuration IDs: 0--4
- Maximum FE budget: 1000 x D
- Topologies: `HISTORICAL`, `AXIS_ONLY`, `FIXED8_UNIQUE`, `DIM_ADAPTIVE`
- Records per topology: 3,600
- Combined observations: 14,400

The historical topology reuses the frozen historical DMGSO results and is not
rerun. The other three controlled topologies are retained in a separate
frozen revision dataset.

Inference is performed separately for D=20 and D=40 using the BBOB function
as the inferential unit (`n=24`). Each function-level value is the median
corrected error across 15 instances x 5 deterministic configurations,
i.e. 75 run-level observations.

## Main DMGSO parameterization

The audited main implementation uses the following default configuration:

| Parameter | Value |
| --- | ---: |
| `k_diag` | 8 |
| `s0_frac` | 0.02 |
| `delta0_frac` | 0.005 |
| `Delta0_frac` | 0.05 |
| `s_min_frac` | 1e-6 |
| `s_max_frac` | 0.2 |
| `c_plus` | 1.05 |
| `c_minus` | 0.70 |
| `stag_trigger` | 60 |
| `max_relocs` | 20 |
| `mem_best_B` | 8 |
| `mem_trap_B` | 10 |
| `mem_pull` | 0.35 |
| `mem_push` | 0.25 |
| `gate_stag_scale` | 80.0 |
| `eps` | 1e-12 |

Problem dimension and FE budget are supplied by the benchmark protocol.
The configuration defaults shown in the source (`D=30`, `FE_budget=300000`)
correspond to the CEC2014 setting and are overridden where required by the
COCO/BBOB dimensional experiments.

The optional refinement branch is disabled for the main manuscript protocol
with `refine_enable=False`.

## Function-evaluation accounting

The stated FE values are maximum budgets, not an assertion that every
algorithm necessarily consumes the full allowance.

For the CEC2014 external comparison, the frozen realized-FE summary reports:

- DMGSO: median 300,000 FE; all 900 observations reached 300,000
- PSO: median 300,000 FE; all 900 observations reached 300,000
- DE: median 299,700 FE
- CMA-ES: median 17,710 FE

Realized FE usage is therefore retained as a descriptive computational-cost
quantity rather than silently replacing the common maximum-budget protocol.

## Statistical settings

The external-comparison statistical scripts use:

- Friedman omnibus testing on paired benchmark units
- planned paired Wilcoxon signed-rank comparisons
- Holm family-wise multiplicity correction
- paired rank-biserial correlation
- percentile bootstrap confidence intervals for median paired improvement
- win/tie/loss counts using fixed numerical tolerances

Bootstrap settings are 20,000 resamples with base reproducibility seed
`20260911`. Numerical ties use `rtol=1e-9` and `atol=1e-12`.

For CEC2014, inference uses 30 paired function-level median errors. For the
COCO/BBOB external comparison, inference is performed separately at each
dimension using 360 paired problem units (24 functions x 15 instances).
Holm correction is applied within the corresponding planned-comparison family.

## Audited software and execution environment

- Python 3.11.15
- GCC 13.3.0
- Ubuntu 24.04.4 LTS under WSL2
- x86_64 architecture
- Intel Core i9-13980HX
- 16 physical cores / 32 logical CPUs visible to WSL2
- approximately 15 GiB memory visible to WSL2
- 4.0 GiB WSL swap

Pinned direct Python dependencies:

- numpy==1.26.4
- scipy==1.13.1
- pandas==2.2.3
- matplotlib==3.11.1
- coco-experiment==2.8.2
- cma==4.4.2
- opfunu==1.0.4

The physical workstation contains 32 GB RAM; the approximately 15 GiB value
above is the memory visible inside the recorded WSL2 execution environment.

Exact dependency pins are retained in `environment/requirements.txt`, and
the recorded platform information is retained in
`environment/environment_info.txt`.
