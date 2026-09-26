# DMGSO CEC2014 External-Comparison Reproducibility Package

This directory contains the audited reproducibility materials for the
CEC2014 external-comparison experiments of DMGSO
(Deterministic Memory-Guided Sensing Optimization).

## 1. Experimental design

The external comparison contains:

- Benchmark: CEC2014
- Functions: F1-F30
- Dimension: D = 30
- Maximum function-evaluation budget: 300,000 evaluations per observation
- Observations per algorithm/function: 30
- Algorithms:
  - DMGSO
  - CMA-ES
  - DE
  - PSO
- Records per algorithm: 900
- Total run-level records: 3,600
- Final dataset status: all 3,600 records are `ok`

The inferential statistical unit is the CEC2014 benchmark function
(n = 30), not the individual run/configuration record.

## 2. Observation semantics

The 30 observations per function have different interpretations for
DMGSO and the stochastic baseline algorithms.

For DMGSO:

- `run_id = 0,...,29` indexes deterministic configurations.
- `run_id` controls deterministic Halton-based initialization/relocation.
- The DMGSO seed field is therefore not interpreted as a stochastic seed.

For CMA-ES, DE, and PSO:

- `run_id = 0,...,29` indexes stochastic runs.
- Explicit seeds are `1000 + run_id`, i.e. 1000-1029.

This distinction is preserved in the statistical analysis and in
`08_run_semantics_summary.csv`.

## 3. Execution code

The `execution/` directory contains the audited execution components:

- `dmgso_core_revision.py`
- `wrapper_cec2014_revision.py`
- `baseline_algorithms_revision.py`
- `run_cec2014_all_algorithms_revision.py`
- `resume_cec2014_external_revision.py`

The main runner writes:

`cec2014_run_summary_revision.csv`

The resume script was used to complete an interrupted external-comparison
execution without rerunning already completed algorithm/function/run
combinations.

The pre-resume checkpoint contained 3,176 records and covered functions
through F27. The completed dataset contains all 3,600 expected records
for F1-F30.

The historical pre-resume checkpoint is retained in the source project
for provenance but is not required to reproduce the final reported
results and is therefore not included in this package.

## 4. Final and frozen datasets

`data/raw/cec2014_run_summary_revision.csv`

is the completed run-level external-comparison dataset.

`data/frozen/cec2014_run_summary_revision_EXTERNAL_FROZEN.csv`

is the frozen reference copy used to preserve the final revision-stage
dataset.

The two files are byte-for-byte identical.

Their SHA-256 checksum is:

`c68607bdb0ff218deba460f1d810a85021585068df890bd78660d010ff04d020`

The final dataset contains:

- 3,600 rows
- 17 columns
- F1-F30
- D = 30
- run IDs 0-29
- FE budget = 300,000
- four algorithms
- 3,600 `ok` status records

## 5. Statistical analysis

The final external-comparison analysis is implemented in:

`statistics/analyze_cec2014_external_comparison_revision.py`

The analysis first aggregates the 30 observations within each
algorithm/function pair using the median error.

Inferential comparisons are then performed at the benchmark-function
level using the resulting 30 paired function-level observations.

The analysis includes:

- function-level descriptive statistics
- function-level median-error matrix
- Friedman omnibus test
- average ranks
- paired Wilcoxon signed-rank comparisons of DMGSO against the baselines
- Holm multiplicity correction
- paired rank-biserial effect sizes
- bootstrap confidence intervals
- category-level descriptive summaries
- realized function-evaluation usage summaries
- explicit run-semantics documentation

The final derived outputs are stored in `statistics/outputs/`:

- `00_external_comparison_statistics_manifest.txt`
- `01_function_level_descriptive.csv`
- `02_function_level_median_error_matrix.csv`
- `03_friedman_summary.csv`
- `04_average_ranks.csv`
- `05_planned_pairwise_wilcoxon_effects.csv`
- `06_category_descriptive.csv`
- `07_fe_usage_summary.csv`
- `08_run_semantics_summary.csv`

## 6. Publication figures

The final publication plotting script is:

`figures/plot_cec2014_publication_figures_REVISION_FONT_READY.py`

It reads the completed CEC2014 external-comparison dataset and validates
the expected algorithms, functions, and benchmark dimension before
generating the publication figures.

The final figures are:

- `fig_cec2014_rank_heatmap.png`
- `fig_cec2014_log_error_heatmap.png`
- `fig_cec2014_performance_profile.png`
- `fig_cec2014_category_heatmap.png`
- `fig_cec2014_category_grouped_bar.png`
- `fig_cec2014_category_radar_rank.png`
- `fig_cec2014_robustness_boxplot.png`
- `fig_cec2014_dmgso_search_activity.png`

The publication plotting configuration uses 300 dpi output.

## 7. Scope separation

This directory contains only the CEC2014 external-comparison
reproducibility chain.

The following materials are intentionally excluded from this directory
and are handled separately where required:

- CEC2014 ablation experiments
- pilot experiments
- R3-27 trajectory experiments
- historical superseded scripts
- intermediate visualization scripts
- Python cache files

This prevents revision-stage auxiliary experiments from being mixed with
the frozen external-comparison results.

## 8. Provenance and integrity

All 26 copied research files in this directory were verified
byte-for-byte against their audited source files using `cmp`.

The original project files were not moved, deleted, renamed, or modified.
This package was created only from copies of the verified source
materials.

`provenance/SHA256SUMS.txt` provides cryptographic checksums for the
packaged files.

## 9. Reproducibility chain

The documented workflow is:

CEC2014 wrapper and benchmark configuration
    -> DMGSO and baseline implementations
    -> Main external-comparison runner
    -> Safe resume of incomplete execution
    -> Completed 3,600-record dataset
    -> Frozen byte-identical reference dataset
    -> Function-level statistical analysis
    -> Publication figures

This separation preserves the provenance of the optimization experiments
while providing the exact code, frozen data, statistical outputs, and
publication figures used in the revision-stage external comparison.
