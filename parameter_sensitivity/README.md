# DMGSO Parameter Sensitivity Study

This directory contains the reproducibility materials for the one-factor-at-a-time (OFAT) parameter-sensitivity analysis added during manuscript revision.

## Purpose

The experiment characterizes the sensitivity of the frozen DMGSO V4 configuration. It is not a parameter-tuning experiment, and the baseline configuration was not retuned.

## Experimental protocol

- Benchmark: CEC2014, F1-F30
- Dimension: D = 30
- Maximum function evaluations: 300,000
- Run IDs: 0-4
- DMGSO version: V4
- Refinement: disabled
- Baseline configurations: 1
- Perturbed configurations: 16
- Total configurations: 17
- Runs per function/configuration: 5
- Total optimization runs: 2,550
- Python: 3.11.15
- Virtual environment: `.venv_revision_py311`

Eight control parameters were examined independently using two perturbations per parameter. Only one sensitivity parameter was changed from the frozen baseline in each perturbed configuration.

The canonical DMGSO core was not modified. Its expected SHA256 is:

`cadf35f0ae664619527707caf2deea4a163e79eb970cf273e53f5490f9cd9191`

## Statistical analysis

The inferential unit is the CEC2014 benchmark function (n = 30). For each function and configuration, corrected final error is summarized by the median across the five run IDs.

Each perturbation is compared with the common baseline using:

- paired Wilcoxon signed-rank test;
- Holm correction across 16 comparisons;
- rank-biserial correlation;
- median paired difference;
- bootstrap 95% confidence interval;
- Win/Tie/Loss counts.

## Main reproducibility result

Fifteen of the 16 tested perturbations were not significantly different from the baseline after Holm correction. Increasing the local sensing radius from `delta0_frac = 0.005` to `0.010` significantly degraded performance.

This result characterizes sensitivity within the examined perturbation ranges and does not establish global parameter insensitivity or optimality of the baseline configuration.

## Directory contents

- `code/` - experiment runner and analysis script
- `configs/` - frozen experimental protocol
- `raw/` - run-level results
- `processed/` - function-level and paired summaries
- `statistics/` - statistical results
- `audit/` - integrity checks, manifests, and SHA256 records
