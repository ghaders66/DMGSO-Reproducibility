# -*- coding: utf-8 -*-
"""
Reviewer-Driven Statistical Analysis of the CEC2014 External Comparison
=======================================================================

Purpose
-------
This script performs only the statistical analyses required for the
Information Sciences revision of the DMGSO manuscript for the CEC2014 external
comparison against CMA-ES, DE, and PSO.

The analysis is intentionally restricted to reviewer-relevant questions. It
does not rerun any optimizer, modify the frozen benchmark results, or introduce
new statistical claims beyond the finalized experimental methodology.

Frozen experimental design
--------------------------
Benchmark suite              : CEC2014
Dimension                    : D = 30
Functions                    : F1-F30
Algorithms                   : DMGSO, CMA-ES, DE, PSO
Maximum FE budget            : 300,000
Records per algorithm        : 900
Total raw records            : 3,600

Run/configuration semantics
---------------------------
The 30 observations per function do NOT have the same interpretation for all
methods:

DMGSO
    run_id = 0,...,29 indexes deterministic low-discrepancy configurations.
    In the frozen implementation, run_id controls Halton-based initialization
    and relocation. Therefore these observations are not independent stochastic
    replicates.

CMA-ES, DE, PSO
    run_id = 0,...,29 indexes stochastic runs with explicit seeds
    BASE_SEED + run_id (1000,...,1029 in the frozen runner).

Statistical unit
----------------
The benchmark function is the inferential statistical unit.

For each algorithm/function pair, the 30 run/configuration-level errors are
first summarized by their median. Friedman and planned paired Wilcoxon tests
are then performed on the resulting 30 paired function-level observations.

This avoids pseudo-replication and directly addresses the reviewer concern that
benchmark functions/problems, rather than repeated runs, should constitute the
paired inferential units.

Reviewer-driven analyses
------------------------
The script produces only:

1. Function-level descriptive statistics for all four algorithms.
2. A 30 x 4 function-level median-error matrix.
3. Friedman omnibus test across the four algorithms.
4. Average ranks and Kendall's W effect size.
5. Planned paired Wilcoxon comparisons:
       DMGSO vs CMA-ES
       DMGSO vs DE
       DMGSO vs PSO
6. Holm correction across these three planned comparisons.
7. Paired rank-biserial correlation.
8. Median paired improvement and reproducible 95% bootstrap confidence
   interval.
9. Function-level win/tie/loss outcomes.
10. Category-wise descriptive summaries only.
11. Function-evaluation usage summary documenting actual FE consumption and
    early termination relative to the common maximum budget.

Direction convention for pairwise comparisons
---------------------------------------------
CEC2014 error is minimized.

For DMGSO versus a baseline:

    paired_improvement = error_baseline - error_DMGSO

Therefore:

    paired_improvement > 0  : DMGSO is better
    paired_improvement = 0  : tie
    paired_improvement < 0  : baseline is better

Rank-biserial correlation follows the same convention:
positive values favor DMGSO.

Important FE-budget interpretation
----------------------------------
The common experimental constraint is a MAXIMUM budget of 300,000 objective
function evaluations, not equal realized FE consumption.

The frozen data show algorithm-specific stopping behavior. In particular,
CMA-ES can terminate substantially before the maximum budget, whereas DMGSO
and PSO consume the full budget in the frozen experiment. This script reports
the realized FE usage but does not equate early stopping with general
computational superiority.

Scope of claims
---------------
Statistical significance must always be interpreted together with:

    - rank direction,
    - win/tie/loss outcomes,
    - effect-size direction,
    - practical paired differences.

A significant p-value is NOT interpreted as evidence that DMGSO is superior
when the direction of the effect favors the comparator.

Data-integrity policy
---------------------
The raw external-comparison dataset is frozen and MUST NOT be modified.

Expected SHA-256:
    c68607bdb0ff218deba460f1d810a85021585068df890bd78660d010ff04d020

All outputs are written only to:
    cec2014/statistics/external_comparison/

No file is written to raw/, backup/, or processed/.

Reproducibility
---------------
Bootstrap confidence intervals use a fixed random seed solely for resampling
the 30 benchmark functions. This does not alter or rerun any optimization
experiment.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, wilcoxon


# =============================================================================
# 1. PROJECT PATHS AND FROZEN INPUT
# =============================================================================

CODE_DIR = Path(__file__).resolve().parent
CEC_DIR = CODE_DIR.parents[1]

RAW_PATH = CEC_DIR / "raw" / "cec2014_run_summary_revision.csv"
OUTPUT_DIR = CEC_DIR / "statistics" / "external_comparison"

EXPECTED_SHA256 = (
    "c68607bdb0ff218deba460f1d810a850"
    "21585068df890bd78660d010ff04d020"
)

ALGORITHMS = ["DMGSO", "CMA-ES", "DE", "PSO"]
FUNCTIONS = list(range(1, 31))
RUN_IDS = list(range(30))
DIMENSION = 30
FE_BUDGET = 300_000

PLANNED_COMPARISONS = [
    ("DMGSO", "CMA-ES"),
    ("DMGSO", "DE"),
    ("DMGSO", "PSO"),
]

ALPHA = 0.05
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED = 20260911

# Numerical tolerance used only to classify function-level W/T/L outcomes.
TIE_RTOL = 1e-9
TIE_ATOL = 1e-12


# =============================================================================
# 2. FROZEN-DATA HASH AND STRUCTURE VALIDATION
# =============================================================================

def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file without modifying it."""
    h = hashlib.sha256()

    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def validate_frozen_input(df: pd.DataFrame) -> None:
    """
    Validate that the dataset matches the frozen external-comparison protocol.

    Statistical analysis stops immediately if any structural, budget, status,
    or numerical-integrity check fails.
    """
    required = {
        "suite",
        "algorithm",
        "function_id",
        "function_name",
        "category",
        "dimension",
        "run_id",
        "seed",
        "FE_budget_max",
        "fe_used",
        "best_f",
        "f_opt",
        "error",
        "n_accept",
        "n_reloc",
        "wall_time_sec",
        "status",
    }

    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(
            f"Frozen dataset is missing required columns: {missing}"
        )

    if len(df) != 3600:
        raise RuntimeError(
            f"Expected 3600 records, found {len(df)}."
        )

    if set(df["algorithm"].astype(str).unique()) != set(ALGORITHMS):
        raise RuntimeError(
            "Algorithm set does not match DMGSO/CMA-ES/DE/PSO."
        )

    if set(pd.to_numeric(df["function_id"]).astype(int).unique()) != set(FUNCTIONS):
        raise RuntimeError(
            "Function IDs do not match F1-F30."
        )

    if set(pd.to_numeric(df["run_id"]).astype(int).unique()) != set(RUN_IDS):
        raise RuntimeError(
            "run_id values do not match 0-29."
        )

    if set(pd.to_numeric(df["dimension"]).astype(int).unique()) != {DIMENSION}:
        raise RuntimeError(
            "Unexpected benchmark dimension."
        )

    if set(pd.to_numeric(df["FE_budget_max"]).astype(int).unique()) != {FE_BUDGET}:
        raise RuntimeError(
            "Unexpected FE budget."
        )

    key = ["algorithm", "function_id", "dimension", "run_id"]
    if df.duplicated(key).any():
        raise RuntimeError(
            "Duplicate external-comparison records detected."
        )

    # Every algorithm/function combination must contain all 30 run_id values.
    coverage = (
        df.groupby(["algorithm", "function_id"])["run_id"]
        .nunique()
    )

    if len(coverage) != 120 or not (coverage == 30).all():
        raise RuntimeError(
            "Incomplete algorithm/function run coverage."
        )

    if not (df["status"].astype(str).str.lower() == "ok").all():
        raise RuntimeError(
            "Non-ok status detected."
        )

    if (pd.to_numeric(df["fe_used"]) > pd.to_numeric(df["FE_budget_max"])).any():
        raise RuntimeError(
            "FE-budget violation detected."
        )

    numeric_cols = [
        "best_f",
        "f_opt",
        "error",
        "fe_used",
        "wall_time_sec",
    ]

    for col in numeric_cols:
        values = pd.to_numeric(
            df[col],
            errors="coerce",
        ).to_numpy(dtype=float)

        if not np.isfinite(values).all():
            raise RuntimeError(
                f"Non-finite values detected in {col}."
            )

    if (pd.to_numeric(df["error"]) < -1e-8).any():
        raise RuntimeError(
            "Materially negative benchmark error detected."
        )

    # Seed semantics are intentionally asymmetric.
    dmgso = df["algorithm"].astype(str) == "DMGSO"
    baselines = ~dmgso

    if not df.loc[dmgso, "seed"].isna().all():
        raise RuntimeError(
            "DMGSO seed column is expected to be empty because run_id "
            "indexes deterministic configurations."
        )

    if df.loc[baselines, "seed"].isna().any():
        raise RuntimeError(
            "Missing stochastic baseline seed detected."
        )


# =============================================================================
# 3. FUNCTION-LEVEL DESCRIPTIVE STATISTICS
# =============================================================================

def make_function_descriptive(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize the 30 run/configuration observations per algorithm/function.

    These descriptive statistics preserve within-function variability while
    keeping inferential testing at the benchmark-function level.
    """
    out = (
        df.groupby(
            [
                "algorithm",
                "function_id",
                "function_name",
                "category",
                "dimension",
            ],
            as_index=False,
            dropna=False,
        )
        .agg(
            observations=("run_id", "count"),
            error_mean=("error", "mean"),
            error_median=("error", "median"),
            error_std=("error", "std"),
            error_q25=("error", lambda x: x.quantile(0.25)),
            error_q75=("error", lambda x: x.quantile(0.75)),
            error_min=("error", "min"),
            error_max=("error", "max"),
            best_f_mean=("best_f", "mean"),
            best_f_median=("best_f", "median"),
            fe_used_mean=("fe_used", "mean"),
            fe_used_median=("fe_used", "median"),
            fe_used_min=("fe_used", "min"),
            fe_used_max=("fe_used", "max"),
            wall_time_mean=("wall_time_sec", "mean"),
            wall_time_median=("wall_time_sec", "median"),
        )
    )

    out["error_iqr"] = (
        out["error_q75"] - out["error_q25"]
    )

    out = out.sort_values(
        ["function_id", "algorithm"]
    ).reset_index(drop=True)

    if len(out) != 120:
        raise RuntimeError(
            f"Expected 120 algorithm/function rows, found {len(out)}."
        )

    return out


def make_function_median_matrix(
    function_desc: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the 30 x 4 median-error matrix used for benchmark-level inference.

    Rows are CEC2014 benchmark functions; columns are algorithms.
    """
    wide = (
        function_desc
        .pivot(
            index="function_id",
            columns="algorithm",
            values="error_median",
        )
        .reindex(
            index=FUNCTIONS,
            columns=ALGORITHMS,
        )
    )

    if wide.shape != (30, 4):
        raise RuntimeError(
            f"Expected matrix shape (30, 4), found {wide.shape}."
        )

    if wide.isna().any().any():
        raise RuntimeError(
            "Missing value detected in function-level median-error matrix."
        )

    return wide


# =============================================================================
# 4. FRIEDMAN OMNIBUS TEST AND AVERAGE RANKS
# =============================================================================

def friedman_and_ranks(
    median_matrix: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform Friedman analysis across the four algorithms on 30 functions.

    Lower median error is better. Kendall's W is reported as the omnibus effect
    size so interpretation does not rely on the p-value alone.
    """
    arrays = [
        median_matrix[a].to_numpy(dtype=float)
        for a in ALGORITHMS
    ]

    statistic, p_value = friedmanchisquare(*arrays)

    n = median_matrix.shape[0]
    k = median_matrix.shape[1]

    kendall_w = float(statistic) / (n * (k - 1))

    rank_rows = []

    for fid, row in median_matrix.iterrows():
        ranks = rankdata(
            row.to_numpy(dtype=float),
            method="average",
        )

        for algorithm, rank in zip(ALGORITHMS, ranks):
            rank_rows.append(
                {
                    "function_id": int(fid),
                    "algorithm": algorithm,
                    "rank": float(rank),
                }
            )

    rank_df = pd.DataFrame(rank_rows)

    average_ranks = (
        rank_df.groupby("algorithm", as_index=False)["rank"]
        .mean()
        .rename(columns={"rank": "average_rank"})
    )

    average_ranks["rank_order"] = (
        average_ranks["average_rank"]
        .rank(method="min", ascending=True)
        .astype(int)
    )

    average_ranks = (
        average_ranks
        .sort_values(["average_rank", "algorithm"])
        .reset_index(drop=True)
    )

    summary = pd.DataFrame(
        [
            {
                "statistical_unit": "CEC2014 benchmark function",
                "n_functions": n,
                "n_algorithms": k,
                "friedman_statistic": float(statistic),
                "p_value": float(p_value),
                "alpha": ALPHA,
                "significant": bool(p_value < ALPHA),
                "kendall_W": kendall_w,
            }
        ]
    )

    return summary, average_ranks


# =============================================================================
# 5. PLANNED DMGSO-VS-BASELINE PAIRED COMPARISONS
# =============================================================================

def holm_adjust(
    p_values: Iterable[float],
) -> np.ndarray:
    """Apply Holm's step-down family-wise-error correction."""
    p = np.asarray(
        list(p_values),
        dtype=float,
    )

    m = len(p)
    order = np.argsort(p)
    ordered = p[order]

    adjusted_ordered = np.empty(
        m,
        dtype=float,
    )

    running_max = 0.0

    for i, value in enumerate(ordered):
        candidate = (m - i) * value
        running_max = max(
            running_max,
            candidate,
        )
        adjusted_ordered[i] = min(
            running_max,
            1.0,
        )

    adjusted = np.empty(
        m,
        dtype=float,
    )

    adjusted[order] = adjusted_ordered

    return adjusted


def paired_rank_biserial(
    improvement: np.ndarray,
) -> float:
    """
    Compute paired rank-biserial correlation.

    Positive values favor DMGSO because improvement is defined as
    baseline error minus DMGSO error.
    """
    d = np.asarray(
        improvement,
        dtype=float,
    )

    d = d[d != 0.0]

    if d.size == 0:
        return 0.0

    ranks = rankdata(
        np.abs(d),
        method="average",
    )

    w_plus = float(
        ranks[d > 0].sum()
    )

    w_minus = float(
        ranks[d < 0].sum()
    )

    denominator = w_plus + w_minus

    if denominator == 0.0:
        return 0.0

    return (
        w_plus - w_minus
    ) / denominator


def bootstrap_median_ci(
    values: np.ndarray,
    *,
    confidence: float = 0.95,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> Tuple[float, float]:
    """
    Percentile bootstrap CI for the median function-level paired improvement.

    Benchmark functions, not run/configuration rows, are resampled.
    """
    x = np.asarray(
        values,
        dtype=float,
    )

    if x.ndim != 1 or x.size == 0:
        raise ValueError(
            "Bootstrap input must be a non-empty 1-D array."
        )

    rng = np.random.default_rng(seed)

    indices = rng.integers(
        low=0,
        high=x.size,
        size=(replicates, x.size),
    )

    medians = np.median(
        x[indices],
        axis=1,
    )

    tail = (
        1.0 - confidence
    ) / 2.0

    lower = float(
        np.quantile(
            medians,
            tail,
        )
    )

    upper = float(
        np.quantile(
            medians,
            1.0 - tail,
        )
    )

    return lower, upper


def win_tie_loss(
    dmgso_error: np.ndarray,
    baseline_error: np.ndarray,
) -> Tuple[int, int, int]:
    """
    Count function-level outcomes from the DMGSO perspective.

    Win  : DMGSO median error < baseline median error
    Tie  : numerically equal within the fixed tolerance
    Loss : DMGSO median error > baseline median error
    """
    dmgso = np.asarray(
        dmgso_error,
        dtype=float,
    )

    baseline = np.asarray(
        baseline_error,
        dtype=float,
    )

    tied = np.isclose(
        dmgso,
        baseline,
        rtol=TIE_RTOL,
        atol=TIE_ATOL,
    )

    wins = int(
        (
            (dmgso < baseline)
            & (~tied)
        ).sum()
    )

    losses = int(
        (
            (dmgso > baseline)
            & (~tied)
        ).sum()
    )

    ties = int(
        tied.sum()
    )

    return wins, ties, losses


def planned_pairwise_tests(
    median_matrix: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare DMGSO only against the three pre-specified external baselines.

    No unnecessary all-pairs baseline-vs-baseline tests are introduced.
    """
    rows: List[Dict] = []

    dmgso = median_matrix[
        "DMGSO"
    ].to_numpy(dtype=float)

    for comparison_index, (_, baseline_name) in enumerate(
        PLANNED_COMPARISONS
    ):
        baseline = median_matrix[
            baseline_name
        ].to_numpy(dtype=float)

        improvement = (
            baseline - dmgso
        )

        if np.all(
            improvement == 0.0
        ):
            wilcoxon_statistic = 0.0
            p_value = 1.0
        else:
            test = wilcoxon(
                improvement,
                zero_method="wilcox",
                alternative="two-sided",
                correction=False,
                method="auto",
            )

            wilcoxon_statistic = float(
                test.statistic
            )

            p_value = float(
                test.pvalue
            )

        rbc = paired_rank_biserial(
            improvement
        )

        median_improvement = float(
            np.median(
                improvement
            )
        )

        ci_low, ci_high = bootstrap_median_ci(
            improvement,
            seed=BOOTSTRAP_SEED + comparison_index,
        )

        wins, ties, losses = win_tie_loss(
            dmgso,
            baseline,
        )

        rows.append(
            {
                "comparison": f"DMGSO vs {baseline_name}",
                "reference_algorithm": "DMGSO",
                "baseline_algorithm": baseline_name,
                "statistical_unit": "CEC2014 benchmark function",
                "n_functions": len(improvement),
                "wilcoxon_statistic": wilcoxon_statistic,
                "p_value_raw": p_value,
                "rank_biserial_correlation": rbc,
                "effect_direction": (
                    "favors DMGSO"
                    if rbc > 0
                    else f"favors {baseline_name}"
                    if rbc < 0
                    else "no directional effect"
                ),
                "median_paired_improvement_baseline_minus_dmgso":
                    median_improvement,
                "bootstrap_95ci_low": ci_low,
                "bootstrap_95ci_high": ci_high,
                "wins_dmgso_better": wins,
                "ties": ties,
                "losses_dmgso_worse": losses,
            }
        )

    result = pd.DataFrame(rows)

    result["p_value_holm"] = holm_adjust(
        result["p_value_raw"].to_numpy(dtype=float)
    )

    result["significant_holm_0.05"] = (
        result["p_value_holm"] < ALPHA
    )

    return result


# =============================================================================
# 6. CATEGORY-WISE DESCRIPTIVE SUMMARY
# =============================================================================

def category_descriptive(
    function_desc: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize function-level median errors by CEC2014 category.

    This is descriptive only. No additional category-specific hypothesis tests
    are introduced unless separately justified by the revision.
    """
    out = (
        function_desc
        .groupby(
            ["category", "algorithm"],
            as_index=False,
            dropna=False,
        )
        .agg(
            n_functions=("function_id", "nunique"),
            median_of_function_medians=("error_median", "median"),
            mean_of_function_medians=("error_median", "mean"),
            q25_of_function_medians=("error_median", lambda x: x.quantile(0.25)),
            q75_of_function_medians=("error_median", lambda x: x.quantile(0.75)),
        )
    )

    out["iqr_of_function_medians"] = (
        out["q75_of_function_medians"]
        - out["q25_of_function_medians"]
    )

    return (
        out
        .sort_values(
            ["category", "algorithm"]
        )
        .reset_index(drop=True)
    )


# =============================================================================
# 7. FUNCTION-EVALUATION USAGE SUMMARY
# =============================================================================

def fe_usage_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Document realized FE usage under the common maximum FE budget.

    This table is descriptive and is intended to support transparent discussion
    of algorithm-specific stopping behavior.
    """
    work = df.copy()

    work["reached_exact_budget"] = (
        work["fe_used"]
        == work["FE_budget_max"]
    )

    work["terminated_below_budget"] = (
        work["fe_used"]
        < work["FE_budget_max"]
    )

    out = (
        work.groupby(
            "algorithm",
            as_index=False,
        )
        .agg(
            observations=("fe_used", "size"),
            fe_min=("fe_used", "min"),
            fe_q25=("fe_used", lambda x: x.quantile(0.25)),
            fe_median=("fe_used", "median"),
            fe_mean=("fe_used", "mean"),
            fe_q75=("fe_used", lambda x: x.quantile(0.75)),
            fe_max=("fe_used", "max"),
            exact_budget_count=("reached_exact_budget", "sum"),
            below_budget_count=("terminated_below_budget", "sum"),
        )
    )

    out["exact_budget_percent"] = (
        100.0
        * out["exact_budget_count"]
        / out["observations"]
    )

    out["below_budget_percent"] = (
        100.0
        * out["below_budget_count"]
        / out["observations"]
    )

    return out


# =============================================================================
# 8. RUN-SEMANTICS SUMMARY
# =============================================================================

def run_semantics_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Record the interpretation of the 30 observations per function.

    This supports reproducibility and prevents deterministic DMGSO
    configurations from being mislabeled as independent stochastic runs.
    """
    records = []

    for algorithm in ALGORITHMS:
        sub = df[
            df["algorithm"] == algorithm
        ]

        if algorithm == "DMGSO":
            semantics = (
                "30 deterministic run_id configurations; "
                "run_id controls deterministic Halton initialization/relocation"
            )
            seed_policy = (
                "No stochastic seed; seed column intentionally empty"
            )
        else:
            semantics = (
                "30 stochastic runs per function"
            )
            seed_policy = (
                "Explicit seed recorded for each run"
            )

        records.append(
            {
                "algorithm": algorithm,
                "observations_per_function": int(
                    sub.groupby("function_id")["run_id"]
                    .nunique()
                    .min()
                ),
                "run_semantics": semantics,
                "seed_policy": seed_policy,
                "unique_seed_values": int(
                    sub["seed"].nunique(
                        dropna=True
                    )
                ),
            }
        )

    return pd.DataFrame(records)


# =============================================================================
# 9. REVISION MANIFEST
# =============================================================================

def write_manifest(
    raw_hash: str,
    df: pd.DataFrame,
    friedman_summary: pd.DataFrame,
) -> None:
    """
    Write a concise provenance and methodology record for this analysis.
    """
    manifest_path = (
        OUTPUT_DIR
        / "00_external_comparison_statistics_manifest.txt"
    )

    fr = friedman_summary.iloc[0]

    content = f"""CEC2014 EXTERNAL COMPARISON — REVISION STATISTICS MANIFEST
======================================================================
Input file:
{RAW_PATH}

Verified SHA-256:
{raw_hash}

Frozen design:
- CEC2014 functions: F1-F30
- Dimension: D=30
- Algorithms: DMGSO, CMA-ES, DE, PSO
- Observations per algorithm/function: 30
- Total raw records: {len(df)}
- Maximum FE budget: {FE_BUDGET}

Run/configuration semantics:
- DMGSO: deterministic run_id configurations controlling Halton
  initialization/relocation.
- CMA-ES, DE, PSO: stochastic runs with explicit seeds.

Inferential statistical unit:
CEC2014 benchmark function (n=30)

Function-level aggregation before inference:
Median error across the 30 run/configuration observations.

Omnibus analysis:
Friedman test across the four algorithms.

Overall effect size:
Kendall's W.

Planned paired comparisons:
DMGSO vs CMA-ES
DMGSO vs DE
DMGSO vs PSO

Pairwise analysis:
Wilcoxon signed-rank test on 30 paired function-level median errors.
Holm correction across the three planned comparisons.

Pairwise effect size:
Paired rank-biserial correlation.
Positive values favor DMGSO.

Practical-significance summaries:
- Median paired improvement = baseline error - DMGSO error
- 95% bootstrap CI for the median paired improvement
- Function-level win/tie/loss from the DMGSO perspective

Bootstrap:
{BOOTSTRAP_REPLICATES} percentile resamples of the 30 benchmark functions.
Base reproducibility seed: {BOOTSTRAP_SEED}

FE-budget interpretation:
300,000 is the common maximum budget, not equal realized FE consumption.
Algorithm-specific stopping behavior is reported separately and must not be
interpreted as general computational superiority without additional evidence.

Friedman result:
statistic = {float(fr['friedman_statistic']):.16g}
p-value   = {float(fr['p_value']):.16g}
Kendall W = {float(fr['kendall_W']):.16g}
======================================================================
"""

    manifest_path.write_text(
        content,
        encoding="utf-8",
    )


# =============================================================================
# 10. MAIN ANALYSIS WORKFLOW
# =============================================================================

def main() -> None:
    """
    Execute the reviewer-driven external-comparison statistics workflow.

    The frozen raw dataset is read-only. All derived outputs are written only
    to the dedicated external_comparison statistics directory.
    """
    print("=" * 82)
    print("CEC2014 DMGSO EXTERNAL COMPARISON — REVIEWER-DRIVEN STATISTICAL ANALYSIS")
    print("=" * 82)

    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Frozen raw dataset not found:\n{RAW_PATH}"
        )

    actual_hash = sha256_file(
        RAW_PATH
    )

    print(f"Frozen input : {RAW_PATH}")
    print(f"SHA-256      : {actual_hash}")

    if actual_hash != EXPECTED_SHA256:
        raise RuntimeError(
            "Frozen input SHA-256 does not match the provenance-verified "
            "external-comparison dataset. Analysis stopped."
        )

    print(
        "[PASS] Frozen raw-data SHA-256 verified."
    )

    df = pd.read_csv(
        RAW_PATH
    )

    validate_frozen_input(
        df
    )

    print(
        "[PASS] Frozen experimental structure verified."
    )

    print(
        "[PASS] Statistical unit: CEC2014 benchmark function (n=30)."
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    function_desc = make_function_descriptive(
        df
    )

    median_matrix = make_function_median_matrix(
        function_desc
    )

    friedman_summary, average_ranks = friedman_and_ranks(
        median_matrix
    )

    pairwise = planned_pairwise_tests(
        median_matrix
    )

    category_summary = category_descriptive(
        function_desc
    )

    fe_summary = fe_usage_summary(
        df
    )

    semantics_summary = run_semantics_summary(
        df
    )

    # -------------------------------------------------------------------------
    # Derived outputs only; frozen raw data are never rewritten.
    # -------------------------------------------------------------------------

    function_desc.to_csv(
        OUTPUT_DIR / "01_function_level_descriptive.csv",
        index=False,
    )

    median_matrix.reset_index().to_csv(
        OUTPUT_DIR / "02_function_level_median_error_matrix.csv",
        index=False,
    )

    friedman_summary.to_csv(
        OUTPUT_DIR / "03_friedman_summary.csv",
        index=False,
    )

    average_ranks.to_csv(
        OUTPUT_DIR / "04_average_ranks.csv",
        index=False,
    )

    pairwise.to_csv(
        OUTPUT_DIR / "05_planned_pairwise_wilcoxon_effects.csv",
        index=False,
    )

    category_summary.to_csv(
        OUTPUT_DIR / "06_category_descriptive.csv",
        index=False,
    )

    fe_summary.to_csv(
        OUTPUT_DIR / "07_fe_usage_summary.csv",
        index=False,
    )

    semantics_summary.to_csv(
        OUTPUT_DIR / "08_run_semantics_summary.csv",
        index=False,
    )

    write_manifest(
        actual_hash,
        df,
        friedman_summary,
    )

    # -------------------------------------------------------------------------
    # Concise console report for auditability.
    # -------------------------------------------------------------------------

    print()
    print("[OK] Statistical outputs written to:")
    print(f"     {OUTPUT_DIR}")

    print()
    print("Friedman omnibus result:")
    print(
        friedman_summary.to_string(
            index=False
        )
    )

    print()
    print("Average ranks (lower is better):")
    print(
        average_ranks.to_string(
            index=False
        )
    )

    print()
    print("Planned DMGSO-vs-baseline comparisons:")

    display_cols = [
        "comparison",
        "p_value_raw",
        "p_value_holm",
        "significant_holm_0.05",
        "rank_biserial_correlation",
        "effect_direction",
        "median_paired_improvement_baseline_minus_dmgso",
        "bootstrap_95ci_low",
        "bootstrap_95ci_high",
        "wins_dmgso_better",
        "ties",
        "losses_dmgso_worse",
    ]

    print(
        pairwise[display_cols].to_string(
            index=False
        )
    )

    print()
    print("FE usage summary:")
    print(
        fe_summary.to_string(
            index=False
        )
    )

    print()
    print("Run/configuration semantics:")
    print(
        semantics_summary.to_string(
            index=False
        )
    )

    print()
    print(
        "[PASS] Reviewer-driven external-comparison statistics "
        "completed successfully."
    )

    print("=" * 82)


if __name__ == "__main__":
    main()
