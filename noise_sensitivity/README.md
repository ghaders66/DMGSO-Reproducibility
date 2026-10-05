# DMGSO Observational-Noise Sensitivity Study

This directory contains the reproducibility materials for the controlled observational-noise experiment added during manuscript revision.

## Purpose

The experiment evaluates the sensitivity of the frozen DMGSO configuration to additive Gaussian observational noise without parameter retuning or modification of the DMGSO core.

## Noise model

The optimizer observes

`f_tilde(x) = f(x) + eta * S_f * z`, with `z ~ N(0,1)`.

The tested noise levels are:

- eta = 0
- eta = 1e-4
- eta = 1e-3
- eta = 1e-2

`S_f` is the IQR of noiseless objective values evaluated at 128 fixed unscrambled Halton reference points over the domain of each benchmark function.

## Experimental protocol

- Benchmark: CEC2014
- Functions: F1, F2, F3, F4, F8, F12, F17, F20, F22, F23, F26, F30
- Dimension: D = 30
- Maximum function evaluations: 300,000
- Run IDs: 0-4
- Noise levels: 4
- Total optimization runs: 240
- Parameter retuning: none

For each function and run ID, the nonzero noise levels use the same standard-normal stream. The run ID controls the deterministic DMGSO configuration, while the noise seed controls observational noise.

The canonical DMGSO core was not modified. Its expected SHA256 is:

`cadf35f0ae664619527707caf2deea4a163e79eb970cf273e53f5490f9cd9191`

## Primary outcome

The primary outcome is the true noiseless error evaluated at the final `best_x`. This final noiseless assessment is not supplied to the optimizer and is not counted in the optimization function-evaluation budget.

## Main reproducibility result

Under the tested additive Gaussian noise model, all three nonzero noise levels significantly degraded final noiseless solution quality relative to the noiseless control after multiplicity correction.

The experiment therefore does not support a general claim of noise robustness. Its conclusions are limited to the tested noise model, levels, benchmark functions, and protocol.

## Directory contents

- `code/` - experiment and analysis scripts
- `configs/` - frozen protocol and function-specific noise scales
- `raw/` - run-level results and frozen raw dataset
- `processed/` - function-level summaries
- `statistics/` - inferential and descriptive results
- `diagnostics/` - supplementary mechanism diagnostics
- `audit/` - integrity checks and provenance manifests
