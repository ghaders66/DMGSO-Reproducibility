# DMGSO COCO/BBOB External-Comparison Reproducibility Package

This directory contains the audited reproducibility materials for the
COCO/BBOB external-comparison experiments of DMGSO
(Deterministic Memory-Guided Sensing Optimization).

## 1. Experimental design

The external comparison contains:

- COCO/BBOB functions: F1-F24
- Dimensions: D = 2, 5, 10, 20, 40
- Instances per function: 15
- Instance IDs: 1-5, 71-75, 76-80
- Observations (run IDs): 0-4
- Function-evaluation budget: 1000 x D
- Algorithms:
  - DMGSO
  - Nelder-Mead
  - Powell
  - COBYLA
  - Compass Search

The historical datasets contain:

- D2-D20: 36,000 optimization records
- D40: 9,000 optimization records
- Total: 45,000 optimization records

All historical runs have status = "ok".

## 2. Software environment

The audited revision environment used:

- Python 3.11.15
- NumPy 1.26.4
- SciPy 1.13.1
- pandas 2.2.3
- cocoex 2.8.2

## 3. Directory structure

### execution/

Contains the historical execution code used for the COCO/BBOB comparison:

- `dmgso_core.py`
- `run_coco_bbob_deterministic_compare.py`
- `run_coco_bbob_deterministic_compare_D40.py`

The D2-D20 and D40 experiments were executed separately.

The SHA-256 hash of the historical `dmgso_core.py` is identical to the
audited revision DMGSO core, establishing byte-for-byte identity of the
DMGSO implementation.

### data/raw/

Contains the preserved historical optimization results:

- `coco_bbob_deterministic_run_summary_D_20.csv`
- `coco_bbob_deterministic_run_summary_D40.csv`

These files are preserved as historical source data.

### correction/

Contains the revision-stage scripts used to obtain and apply the official
COCO/BBOB optimum values:

- `extract_bbob_fopt_revision.py`
- `audit_correct_coco_external_revision.py`

The optimizer experiments were not rerun during this correction stage.

### data/frozen/

Contains the audited frozen revision datasets:

- `bbob_fopt_cocoex_2.8.2_FROZEN.csv`
- `coco_bbob_deterministic_D2_D20_corrected_FROZEN.csv`
- `coco_bbob_deterministic_D40_corrected_FROZEN.csv`

The corrected error measure is based on the official COCO/BBOB optimum:

    error_corrected = best_f - f_opt_official

A provenance audit confirmed one-to-one correspondence between all
45,000 historical optimization records and the frozen corrected records.

The following execution fields were unchanged:

- FE_budget
- budget_mult
- fe_used
- n_accept
- n_reloc
- status

All 45,000 `best_f` values were also numerically identical under a strict
comparison tolerance (`rtol=1e-12`, `atol=1e-14`).

Small exact-text/floating-point representation differences occurred in
558 values (441 for D2-D20 and 117 for D40). The maximum absolute
difference was approximately 3.638e-12. No numerically different
`best_f` values were detected under the stated tolerance.

## 4. Statistical analysis

The final external-comparison statistical analysis is:

`statistics/analyze_coco_bbob_external_comparison_revision.py`

The script validates the frozen inputs before statistical inference and
does not modify the frozen source datasets.

Derived statistical outputs are stored in:

`statistics/outputs/`

They include:

- problem-level median corrected errors
- dimension/algorithm descriptive statistics
- Friedman tests by dimension
- average ranks by dimension
- planned paired Wilcoxon comparisons
- FE-usage summaries
- DMGSO dimension-trend summaries
- statistical-analysis manifest

## 5. Publication figures

The final audited publication plotting script is:

`figures/plot_coco_publication_figures_REVISION_v5.py`

It reads the frozen corrected datasets and generates the eight
publication figures stored in:

`figures/outputs/`

The plotting configuration uses 300 dpi output.

For the robustness visualization, the metric is the within-problem IQR
of log10 corrected error across the five observations. A numerical floor
of 1e-12 is used only for exact-zero corrected errors in this log-scale
dispersion calculation.

## 6. Provenance and integrity

`provenance/SHA256SUMS.txt` contains SHA-256 checksums for the files in
this reproducibility package.

All copied execution code, datasets, analysis outputs, and publication
figures were verified byte-for-byte against their audited source files
before packaging.

The historical source files remain unchanged. This package was created
by copying the verified source materials rather than moving or modifying
the originals.

## 7. Reproducibility chain

The documented workflow is:

Historical execution
    -> Historical raw results
    -> Official COCO/BBOB optimum extraction
    -> Revision correction and audit
    -> Frozen corrected datasets
    -> Statistical analysis
    -> Publication figures

This separation preserves the provenance of the original optimization
experiments while documenting the revision-stage correction and analysis
pipeline.
