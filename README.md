# DMGSO Reproducibility Package

This repository provides the audited computational materials supporting the
revised DMGSO study. Its purpose is to enable independent inspection and
reproduction of the algorithm implementation, benchmark protocols, statistical
analyses, ablation experiments, sensing-topology analysis, and publication
figures reported in the manuscript.

DMGSO (Deterministic Memory-Guided Sensing Optimization) is a deterministic,
single-agent, derivative-free optimization framework. The repository preserves
the historical audited execution artifacts used to generate the reported
results while also documenting the experimental protocol and implementation
details required for independent reconstruction.

## Reproducibility map

| Reproducibility requirement | Repository evidence |
| --- | --- |
| DMGSO source implementation | `cec2014/execution/dmgso_core_revision.py`, `coco_bbob/execution/dmgso_core.py` |
| Algorithm-to-code consistency | `docs/ALGORITHM_TO_CODE.md` |
| Experimental parameters and benchmark settings | `docs/EXPERIMENTAL_PROTOCOL.md` |
| Independent reproduction guidance | `docs/REPRODUCIBILITY.md` |
| CEC2014 external comparison | `cec2014/` |
| COCO/BBOB external comparison | `coco_bbob/` |
| V0--V4 ablation experiments | `ablation/` |
| Reviewer-driven sensing-topology experiment | `sensing_topology/` |
| Frozen benchmark results | benchmark-specific `data/frozen/` directories |
| Statistical analyses | benchmark-specific `statistics/` directories |
| Publication figure-generation scripts | benchmark-specific `figures/` or ablation `execution/` directories |
| Exact Python environment | `environment/requirements.txt`, `environment/environment_info.txt` |
| File-integrity verification | benchmark-specific `provenance/SHA256SUMS.txt` manifests |

## Repository structure

- `cec2014/` -- CEC2014 external comparison.
- `coco_bbob/` -- COCO/BBOB external comparison, including standard target-attainment analysis.
- `coco_bbob/coco_target_attainment/` -- native COCO logging and cocopp target-attainment/ECDF reproducibility materials.
- `ablation/` -- CEC2014 and COCO/BBOB V0--V4 ablation studies.
- `sensing_topology/` -- reviewer-driven sensing-topology experiment.
- `environment/` -- pinned dependencies and execution-environment information.
- `docs/` -- reproducibility, algorithm-to-code, and experimental-protocol documentation.

## CEC2014 external comparison

The `cec2014/` directory contains the audited CEC2014 external-comparison
workflow, including the DMGSO implementation, benchmark wrapper, baseline
implementations, execution scripts, raw and frozen run-level data, statistical
analysis, publication figures, and SHA-256 provenance information.

The reported protocol uses CEC2014 functions F1--F30 at D=30 with a maximum
budget of 300,000 function evaluations. DMGSO is compared with CMA-ES, DE,
and PSO.

## COCO/BBOB external comparison

The `coco_bbob/` directory contains the audited deterministic COCO/BBOB
comparison for functions F1--F24 at dimensions D=2, 5, 10, 20, and 40.
The comparison includes DMGSO, Nelder-Mead, Powell, COBYLA, and Compass Search.

The repository retains the historical raw execution summaries and the frozen
corrected datasets based on the official BBOB optimum values. The correction
is a post-processing operation on the error reference and does not rerun the
optimization experiments.

The `coco_bbob/coco_target_attainment/` directory provides the supplementary
native COCO logging workflow used for standard cocopp target-attainment and
ECDF reporting. It includes the reproducible runner, run-level summary,
provenance manifest, and figure-export script. Generated native COCO and
cocopp artifacts are excluded from version control because they can be
regenerated from the supplied workflow.

## Ablation studies

The `ablation/` directory contains the CEC2014 and COCO/BBOB experiments used
to examine DMGSO variants V0--V4.

The implemented revision semantics are:

- V0: directional sensing baseline
- V1: V0 + adaptive step
- V2: V1 + long-range sensing
- V3: V1 + memory
- V4: full DMGSO architecture with adaptive step, long-range sensing, memory,
  adaptive fusion, and deterministic relocation

V2 and V3 are parallel extensions of V1 rather than sequential additions.
Therefore, V2-to-V3 is not interpreted as an isolated single-component
ablation.

## Sensing-topology analysis

The `sensing_topology/` directory contains the reviewer-driven controlled
analysis comparing `HISTORICAL`, `AXIS_ONLY`, `FIXED8_UNIQUE`, and
`DIM_ADAPTIVE` configurations at D=20 and D=40.

The historical DMGSO results are reused from the frozen COCO/BBOB experiment.
The three controlled revision topologies are supplied as a separate frozen
run-level dataset.

Original audited scripts and repository-portable entry points are retained
side by side. Portable reproduction outputs are written under
`sensing_topology/reproduction_output/`, which is excluded from version
control so that newly generated results cannot overwrite the frozen evidence.

## Parameter-sensitivity analysis

The `parameter_sensitivity/` directory contains the frozen one-factor-at-a-time
sensitivity study of eight principal DMGSO control parameters on CEC2014
(F1--F30, D=30, 300,000 evaluations, five deterministic run IDs). The study
characterizes the frozen V4 configuration without retuning or modifying the
DMGSO core. Full protocol, run-level data, statistical outputs, and provenance
records are included.

## Observational-noise sensitivity

The `noise_sensitivity/` directory contains the controlled additive-Gaussian
observational-noise study on 12 CEC2014 functions at D=30 using four noise
levels and five run IDs (240 optimization runs). No parameter retuning or core
modification was performed. The directory includes the frozen protocol,
function-specific noise scales, run-level data, statistical outputs,
diagnostics, and provenance records.

## Frozen evidence and provenance

Frozen datasets and reported outputs are retained as evidence of the analyses
used in the revised manuscript. SHA-256 manifests are supplied in the
corresponding `provenance/` directories.

For example, integrity can be checked from a benchmark directory with:

```bash
sha256sum -c provenance/SHA256SUMS.txt
```

Historical execution scripts are retained rather than silently rewritten for
presentation. Where repository portability requires path or import changes,
a separate portable entry point is provided alongside the historical script.

## Reproducing the reported analyses

The recorded Python environment is provided in `environment/requirements.txt`.

The frozen datasets permit the statistical analyses and publication figures
to be regenerated without rerunning every computationally expensive benchmark
experiment.

For detailed guidance, see:

- `docs/REPRODUCIBILITY.md` -- reproduction workflow and integrity checks
- `docs/EXPERIMENTAL_PROTOCOL.md` -- benchmark settings, budgets, run semantics,
  aggregation, and statistical procedures
- `docs/ALGORITHM_TO_CODE.md` -- mapping between manuscript Algorithm 1 and
  the implementation

## Reproducibility design

The repository distinguishes three forms of evidence:

1. **Historical execution provenance** -- scripts and raw outputs associated
   with the reported computational experiments.
2. **Frozen analysis evidence** -- audited datasets and statistical outputs
   used for the revised manuscript.
3. **Portable reproduction support** -- repository-local entry points, where
   required, that remove workstation-specific paths without changing the
   scientific protocol.

This separation preserves the provenance of the reported results while
providing a practical route for independent verification.

## Software environment

The recorded revision environment uses Python 3.11.15. Exact dependency pins,
platform information, and execution-environment details are provided in
`environment/`.

## Citation and archival release

Citation metadata and the persistent archival identifier will be supplied with
the final versioned archival release of this repository.

## License

License information will be supplied with the final release.
