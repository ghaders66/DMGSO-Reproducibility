# -*- coding: utf-8 -*-
"""
Reviewer-Driven Statistical Analysis of the COCO/BBOB External Comparison
=========================================================================

Purpose
-------
This script performs only the statistical analyses needed for the
Information Sciences revision of the DMGSO manuscript for the COCO/BBOB
external comparison.

The analysis is intentionally limited to reviewer-relevant questions:
overall comparative performance, dimension-wise scalability, paired
significance, effect-size direction, and transparent FE-budget reporting.

No optimizer is rerun. No frozen source file is modified.

Frozen corrected inputs
-----------------------
The analysis uses the provenance-verified corrected COCO/BBOB datasets:

1. D = 2, 5, 10, 20
   coco_bbob_deterministic_D2_D20_corrected_FROZEN.csv

2. D = 40
   coco_bbob_deterministic_D40_corrected_FROZEN.csv

Corrected performance measure
-----------------------------
Only:

    error_corrected = best_f - f_opt_official

is used downstream.

The historical columns "f_opt" and "error" are retained in the corrected
files for provenance only and MUST NOT be used for inference.

Experimental design
-------------------
Algorithms:
    DMGSO
    NELDER-MEAD
    POWELL
    COBYLA
    COMPASS

Dimensions:
    D = {2, 5, 10, 20, 40}

For each algorithm × function × instance × dimension block:
    5 run_id configurations are available.

The five observations are first summarized using their median corrected error.

Dimension-wise inferential unit
-------------------------------
At each dimension, the inferential paired unit is:

    one BBOB function × instance problem

There are:

    24 functions × 15 instances = 360 paired problem units per dimension.

All five algorithms are present for every paired unit.

This design was verified before the present script was created.

Reviewer-driven analyses
------------------------
For each dimension separately, this script produces:

1. Algorithm-level descriptive summaries based on the 360 problem-level
   median corrected errors.
2. Friedman omnibus test across the five algorithms.
3. Average ranks (lower is better).
4. Kendall's W as omnibus effect size.
5. Planned paired Wilcoxon comparisons:
       DMGSO vs NELDER-MEAD
       DMGSO vs POWELL
       DMGSO vs COBYLA
       DMGSO vs COMPASS
6. Holm correction across the four planned comparisons within each dimension.
7. Paired rank-biserial correlation.
8. Median paired improvement and reproducible 95% bootstrap CI.
9. Win/tie/loss counts from the DMGSO perspective.
10. FE-usage summaries by algorithm and dimension.
11. A compact dimension-wise DMGSO trend table based on average rank.

No baseline-vs-baseline pairwise hypothesis tests are introduced.

Direction convention
--------------------
COCO/BBOB corrected error is minimized.

For DMGSO versus a baseline:

    paired_improvement = error_baseline - error_DMGSO

Therefore:

    paired_improvement > 0  : DMGSO is better
    paired_improvement = 0  : tie
    paired_improvement < 0  : baseline is better

Positive paired rank-biserial correlation also favors DMGSO.

Multiplicity convention
-----------------------
Holm correction is applied within each dimension across the four planned
DMGSO-vs-baseline tests. The correction family is therefore the set of four
pre-specified comparisons answering the same dimension-specific question.

Interpretation limits
---------------------
- Statistical significance is not equivalent to practical superiority.
- Average ranks and effect-size direction must be interpreted together.
- The 360 paired units per dimension are 24 functions × 15 instances;
  they must not be described as 360 distinct benchmark functions.
- Dimension-wise results are analyzed separately to avoid mixing search spaces
  of different dimensionality.
- No cross-dimension hypothesis test is introduced here; scalability is
  summarized descriptively through dimension-wise ranks and paired outcomes.

FE-budget interpretation
------------------------
The protocol uses:

    FE_budget = 1000 × D

This is a maximum allowed budget. Realized FE consumption may differ among
algorithms due to their stopping behavior. FE usage is therefore reported
descriptively and is not, by itself, interpreted as computational superiority.

Frozen-input SHA-256
--------------------
D2-D20 corrected:
68c2ae2b4776714d2a3fcd4202232db1d745e1d71fe6465cff6102c9ab14d229

D40 corrected:
9e67d3bbe0f8d8c19f382235107abba7698f2eec59068eb5f51da60c95a3877d

Official f* reference:
ff273dc9c7cd97c15f3abf70e7b0816ca76137b57cc7907fb7aea814357b5dc0

Output policy
-------------
All statistical outputs are written only to:

    coco_bbob/statistics/external_comparison/

No raw, processed, or frozen backup file is rewritten.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, wilcoxon


# =============================================================================
# 1. PROJECT PATHS AND FROZEN INPUTS
# =============================================================================

CODE_DIR = Path(__file__).resolve().parent
COCO_DIR = CODE_DIR.parents[1]

FROZEN_DIR = COCO_DIR / "backup" / "frozen_20260911"

D2_D20_PATH = (
    FROZEN_DIR
    / "coco_bbob_deterministic_D2_D20_corrected_FROZEN.csv"
)

D40_PATH = (
    FROZEN_DIR
    / "coco_bbob_deterministic_D40_corrected_FROZEN.csv"
)

FOPT_PATH = (
    FROZEN_DIR
    / "bbob_fopt_cocoex_2.8.2_FROZEN.csv"
)

OUTPUT_DIR = (
    COCO_DIR
    / "statistics"
    / "external_comparison"
)

EXPECTED_HASH_D2_D20 = (
    "68c2ae2b4776714d2a3fcd4202232db1"
    "d745e1d71fe6465cff6102c9ab14d229"
)

EXPECTED_HASH_D40 = (
    "9e67d3bbe0f8d8c19f382235107abba7"
    "698f2eec59068eb5f51da60c95a3877d"
)

EXPECTED_HASH_FOPT = (
    "ff273dc9c7cd97c15f3abf70e7b0816c"
    "a76137b57cc7907fb7aea814357b5dc0"
)

ALGORITHMS = [
    "DMGSO",
    "NELDER-MEAD",
    "POWELL",
    "COBYLA",
    "COMPASS",
]

BASELINES = [
    "NELDER-MEAD",
    "POWELL",
    "COBYLA",
    "COMPASS",
]

DIMENSIONS = [2, 5, 10, 20, 40]

EXPECTED_FUNCTIONS = set(range(1, 25))
EXPECTED_INSTANCES = {
    1, 2, 3, 4, 5,
    71, 72, 73, 74, 75,
    76, 77, 78, 79, 80,
}
EXPECTED_RUN_IDS = set(range(5))

BUDGET_MULT = 1000
ALPHA = 0.05

BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED = 20260911

TIE_RTOL = 1e-9
TIE_ATOL = 1e-12


# =============================================================================
# 2. HASH AND FROZEN-DATA VALIDATION
# =============================================================================

def sha256_file(path: Path) -> str:
    """Return SHA-256 digest without modifying the file."""
    h = hashlib.sha256()

    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def verify_hash(path: Path, expected: str, label: str) -> str:
    """Verify one provenance-locked input file."""
    if not path.exists():
        raise FileNotFoundError(
            f"{label} not found:\n{path}"
        )

    actual = sha256_file(path)

    if actual != expected:
        raise RuntimeError(
            f"{label} SHA-256 mismatch.\n"
            f"Expected: {expected}\n"
            f"Actual:   {actual}"
        )

    return actual


def validate_combined_input(df: pd.DataFrame) -> None:
    """
    Re-check the frozen corrected experimental structure before inference.

    The script intentionally repeats critical design checks so that statistics
    cannot be generated from an incomplete or altered file.
    """
    required = {
        "suite",
        "algorithm",
        "function_id",
        "instance_id",
        "function_name",
        "dimension",
        "run_id",
        "FE_budget",
        "budget_mult",
        "fe_used",
        "best_f",
        "f_opt_official",
        "error_corrected",
        "wall_time_sec",
        "status",
    }

    missing = sorted(required - set(df.columns))

    if missing:
        raise RuntimeError(
            f"Corrected dataset is missing required columns: {missing}"
        )

    if len(df) != 45000:
        raise RuntimeError(
            f"Expected 45000 corrected records, found {len(df)}."
        )

    if set(df["algorithm"].astype(str).unique()) != set(ALGORITHMS):
        raise RuntimeError(
            "Unexpected algorithm set."
        )

    if set(pd.to_numeric(df["function_id"]).astype(int).unique()) != EXPECTED_FUNCTIONS:
        raise RuntimeError(
            "Unexpected/incomplete function IDs."
        )

    if set(pd.to_numeric(df["instance_id"]).astype(int).unique()) != EXPECTED_INSTANCES:
        raise RuntimeError(
            "Unexpected/incomplete instance IDs."
        )

    if set(pd.to_numeric(df["dimension"]).astype(int).unique()) != set(DIMENSIONS):
        raise RuntimeError(
            "Unexpected/incomplete dimensions."
        )

    if set(pd.to_numeric(df["run_id"]).astype(int).unique()) != EXPECTED_RUN_IDS:
        raise RuntimeError(
            "Unexpected/incomplete run_id values."
        )

    if set(pd.to_numeric(df["budget_mult"]).astype(int).unique()) != {BUDGET_MULT}:
        raise RuntimeError(
            "Unexpected budget multiplier."
        )

    key = [
        "algorithm",
        "function_id",
        "instance_id",
        "dimension",
        "run_id",
    ]

    if df.duplicated(key).any():
        raise RuntimeError(
            "Duplicate complete experiment keys detected."
        )

    expected_budget = (
        pd.to_numeric(df["dimension"])
        * pd.to_numeric(df["budget_mult"])
    )

    if (
        pd.to_numeric(df["FE_budget"])
        != expected_budget
    ).any():
        raise RuntimeError(
            "FE_budget != 1000 × D for at least one record."
        )

    if (
        pd.to_numeric(df["fe_used"])
        > pd.to_numeric(df["FE_budget"])
    ).any():
        raise RuntimeError(
            "FE-budget violation detected."
        )

    if not (
        df["status"].astype(str).str.lower() == "ok"
    ).all():
        raise RuntimeError(
            "Non-ok status detected."
        )

    for col in [
        "best_f",
        "f_opt_official",
        "error_corrected",
        "fe_used",
        "wall_time_sec",
    ]:
        x = pd.to_numeric(
            df[col],
            errors="coerce",
        ).to_numpy(dtype=float)

        if not np.isfinite(x).all():
            raise RuntimeError(
                f"Non-finite values detected in {col}."
            )

    if (
        pd.to_numeric(df["error_corrected"])
        < -1e-8
    ).any():
        raise RuntimeError(
            "Materially negative corrected error detected."
        )

    # Five run/configuration observations per algorithm × problem.
    counts = (
        df.groupby(
            [
                "algorithm",
                "function_id",
                "instance_id",
                "dimension",
            ],
            observed=True,
        )["run_id"]
        .nunique()
    )

    if len(counts) != 9000 or not (counts == 5).all():
        raise RuntimeError(
            "Five-run/configuration block completeness failed."
        )


# =============================================================================
# 3. PROBLEM-LEVEL MEDIAN AGGREGATION
# =============================================================================

def make_problem_level_medians(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Collapse five run/configuration observations to one robust value per:

        algorithm × function × instance × dimension

    This table is the sole inferential input.
    """
    out = (
        df.groupby(
            [
                "algorithm",
                "function_id",
                "instance_id",
                "dimension",
            ],
            as_index=False,
            observed=True,
        )
        .agg(
            configurations=("run_id", "count"),
            error_median=("error_corrected", "median"),
            error_mean=("error_corrected", "mean"),
            error_std=("error_corrected", "std"),
            error_q25=("error_corrected", lambda x: x.quantile(0.25)),
            error_q75=("error_corrected", lambda x: x.quantile(0.75)),
            error_min=("error_corrected", "min"),
            error_max=("error_corrected", "max"),
            fe_median=("fe_used", "median"),
            wall_time_median=("wall_time_sec", "median"),
        )
    )

    out["error_iqr"] = (
        out["error_q75"] - out["error_q25"]
    )

    if len(out) != 9000:
        raise RuntimeError(
            f"Expected 9000 algorithm-problem rows, found {len(out)}."
        )

    if not (out["configurations"] == 5).all():
        raise RuntimeError(
            "At least one algorithm-problem row does not summarize 5 observations."
        )

    return (
        out
        .sort_values(
            ["dimension", "function_id", "instance_id", "algorithm"]
        )
        .reset_index(drop=True)
    )


# =============================================================================
# 4. DIMENSION-ALGORITHM DESCRIPTIVE SUMMARY
# =============================================================================

def dimension_algorithm_descriptive(
    problem_level: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize the 360 problem-level medians for each algorithm and dimension.

    This is descriptive. Because corrected errors are heterogeneous across BBOB
    functions, median and IQR are emphasized over the mean.
    """
    out = (
        problem_level
        .groupby(
            ["dimension", "algorithm"],
            as_index=False,
            observed=True,
        )
        .agg(
            paired_problem_units=("error_median", "size"),
            median_of_problem_medians=("error_median", "median"),
            mean_of_problem_medians=("error_median", "mean"),
            q25_of_problem_medians=("error_median", lambda x: x.quantile(0.25)),
            q75_of_problem_medians=("error_median", lambda x: x.quantile(0.75)),
            min_problem_median=("error_median", "min"),
            max_problem_median=("error_median", "max"),
        )
    )

    out["iqr_of_problem_medians"] = (
        out["q75_of_problem_medians"]
        - out["q25_of_problem_medians"]
    )

    return (
        out
        .sort_values(["dimension", "algorithm"])
        .reset_index(drop=True)
    )


# =============================================================================
# 5. FRIEDMAN TESTS AND AVERAGE RANKS BY DIMENSION
# =============================================================================

def build_dimension_matrix(
    problem_level: pd.DataFrame,
    dimension: int,
) -> pd.DataFrame:
    """
    Build a 360 × 5 paired matrix for one dimension.

    Row identity:
        function_id × instance_id

    Columns:
        five algorithms
    """
    sub = problem_level[
        problem_level["dimension"] == dimension
    ]

    wide = (
        sub.pivot(
            index=["function_id", "instance_id"],
            columns="algorithm",
            values="error_median",
        )
        .reindex(columns=ALGORITHMS)
        .sort_index()
    )

    if wide.shape != (360, 5):
        raise RuntimeError(
            f"D={dimension}: expected matrix (360, 5), found {wide.shape}."
        )

    if wide.isna().any().any():
        raise RuntimeError(
            f"D={dimension}: incomplete algorithm pairing."
        )

    return wide


def friedman_and_ranks_by_dimension(
    problem_level: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform dimension-specific Friedman tests and compute average ranks.
    """
    friedman_rows = []
    rank_rows = []

    for d in DIMENSIONS:
        wide = build_dimension_matrix(
            problem_level,
            d,
        )

        arrays = [
            wide[a].to_numpy(dtype=float)
            for a in ALGORITHMS
        ]

        statistic, p_value = friedmanchisquare(
            *arrays
        )

        n = wide.shape[0]  # 360 paired function-instance problems
        k = wide.shape[1]  # 5 algorithms

        kendall_w = float(statistic) / (
            n * (k - 1)
        )

        friedman_rows.append(
            {
                "dimension": d,
                "statistical_unit":
                    "BBOB function-instance problem",
                "n_paired_problem_units": n,
                "n_algorithms": k,
                "friedman_statistic": float(statistic),
                "p_value": float(p_value),
                "alpha": ALPHA,
                "significant": bool(p_value < ALPHA),
                "kendall_W": kendall_w,
            }
        )

        local_rank_rows = []

        for (fid, iid), row in wide.iterrows():
            ranks = rankdata(
                row.to_numpy(dtype=float),
                method="average",
            )

            for algorithm, rank in zip(
                ALGORITHMS,
                ranks,
            ):
                local_rank_rows.append(
                    {
                        "dimension": d,
                        "function_id": int(fid),
                        "instance_id": int(iid),
                        "algorithm": algorithm,
                        "rank": float(rank),
                    }
                )

        local_rank_df = pd.DataFrame(
            local_rank_rows
        )

        avg = (
            local_rank_df
            .groupby(
                ["dimension", "algorithm"],
                as_index=False,
            )["rank"]
            .mean()
            .rename(
                columns={"rank": "average_rank"}
            )
        )

        avg["rank_order"] = (
            avg["average_rank"]
            .rank(
                method="min",
                ascending=True,
            )
            .astype(int)
        )

        rank_rows.append(avg)

    return (
        pd.DataFrame(friedman_rows),
        pd.concat(rank_rows, ignore_index=True)
        .sort_values(
            ["dimension", "average_rank", "algorithm"]
        )
        .reset_index(drop=True),
    )


# =============================================================================
# 6. PLANNED DMGSO-VS-BASELINE TESTS BY DIMENSION
# =============================================================================

def holm_adjust(
    p_values: Iterable[float],
) -> np.ndarray:
    """Apply Holm step-down family-wise-error correction."""
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
    Paired rank-biserial correlation.

    Positive values favor DMGSO because:
        improvement = baseline - DMGSO.
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
    Percentile bootstrap CI for median paired improvement.

    The resampling unit is the dimension-specific function-instance problem.
    """
    x = np.asarray(
        values,
        dtype=float,
    )

    if x.ndim != 1 or x.size == 0:
        raise ValueError(
            "Bootstrap input must be a non-empty 1-D array."
        )

    rng = np.random.default_rng(
        seed
    )

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

    return (
        float(
            np.quantile(
                medians,
                tail,
            )
        ),
        float(
            np.quantile(
                medians,
                1.0 - tail,
            )
        ),
    )


def win_tie_loss(
    dmgso_error: np.ndarray,
    baseline_error: np.ndarray,
) -> Tuple[int, int, int]:
    """
    Count paired problem-level W/T/L from the DMGSO perspective.
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
        ((dmgso < baseline) & (~tied)).sum()
    )

    losses = int(
        ((dmgso > baseline) & (~tied)).sum()
    )

    ties = int(
        tied.sum()
    )

    return wins, ties, losses


def planned_pairwise_by_dimension(
    problem_level: pd.DataFrame,
) -> pd.DataFrame:
    """
    Run only the four planned DMGSO-vs-baseline tests at each dimension.

    Holm correction is applied separately within each dimension.
    """
    all_rows: List[Dict] = []

    for dim_index, d in enumerate(DIMENSIONS):
        wide = build_dimension_matrix(
            problem_level,
            d,
        )

        dmgso = wide[
            "DMGSO"
        ].to_numpy(dtype=float)

        dimension_rows = []

        for baseline_index, baseline_name in enumerate(BASELINES):
            baseline = wide[
                baseline_name
            ].to_numpy(dtype=float)

            improvement = (
                baseline - dmgso
            )

            if np.all(
                improvement == 0.0
            ):
                statistic = 0.0
                p_value = 1.0
            else:
                test = wilcoxon(
                    improvement,
                    zero_method="wilcox",
                    alternative="two-sided",
                    correction=False,
                    method="auto",
                )

                statistic = float(
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
                seed=(
                    BOOTSTRAP_SEED
                    + dim_index * 100
                    + baseline_index
                ),
            )

            wins, ties, losses = win_tie_loss(
                dmgso,
                baseline,
            )

            dimension_rows.append(
                {
                    "dimension": d,
                    "comparison": f"DMGSO vs {baseline_name}",
                    "reference_algorithm": "DMGSO",
                    "baseline_algorithm": baseline_name,
                    "statistical_unit":
                        "BBOB function-instance problem",
                    "n_paired_problem_units": len(improvement),
                    "wilcoxon_statistic": statistic,
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

        dim_df = pd.DataFrame(
            dimension_rows
        )

        dim_df["p_value_holm"] = holm_adjust(
            dim_df["p_value_raw"].to_numpy(dtype=float)
        )

        dim_df["significant_holm_0.05"] = (
            dim_df["p_value_holm"] < ALPHA
        )

        all_rows.extend(
            dim_df.to_dict(orient="records")
        )

    return (
        pd.DataFrame(all_rows)
        .sort_values(
            ["dimension", "baseline_algorithm"]
        )
        .reset_index(drop=True)
    )


# =============================================================================
# 7. FE-USAGE SUMMARY
# =============================================================================

def fe_usage_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Report realized function-evaluation usage by algorithm and dimension.

    This is descriptive only.
    """
    work = df.copy()

    work["reached_exact_budget"] = (
        work["fe_used"]
        == work["FE_budget"]
    )

    work["terminated_below_budget"] = (
        work["fe_used"]
        < work["FE_budget"]
    )

    out = (
        work.groupby(
            ["dimension", "algorithm"],
            as_index=False,
            observed=True,
        )
        .agg(
            observations=("fe_used", "size"),
            fe_budget=("FE_budget", "first"),
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

    return (
        out
        .sort_values(["dimension", "algorithm"])
        .reset_index(drop=True)
    )


# =============================================================================
# 8. DMGSO DIMENSION TREND
# =============================================================================

def dmgso_dimension_trend(
    average_ranks: pd.DataFrame,
    pairwise: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a compact descriptive scalability summary for DMGSO.

    No cross-dimension hypothesis test is introduced.
    """
    rank_part = (
        average_ranks.loc[
            average_ranks["algorithm"] == "DMGSO",
            [
                "dimension",
                "average_rank",
                "rank_order",
            ],
        ]
        .copy()
    )

    pair_part = (
        pairwise.groupby(
            "dimension",
            as_index=False,
        )
        .agg(
            total_pairwise_wins=(
                "wins_dmgso_better",
                "sum",
            ),
            total_pairwise_ties=(
                "ties",
                "sum",
            ),
            total_pairwise_losses=(
                "losses_dmgso_worse",
                "sum",
            ),
            comparisons_significant_after_holm=(
                "significant_holm_0.05",
                "sum",
            ),
        )
    )

    return (
        rank_part
        .merge(
            pair_part,
            on="dimension",
            how="left",
            validate="one_to_one",
        )
        .sort_values("dimension")
        .reset_index(drop=True)
    )


# =============================================================================
# 9. REVISION MANIFEST
# =============================================================================

def write_manifest(
    hashes: Dict[str, str],
    friedman: pd.DataFrame,
) -> None:
    """
    Write a concise provenance and methodological record.
    """
    path = (
        OUTPUT_DIR
        / "00_coco_bbob_external_statistics_manifest.txt"
    )

    lines = [
        "COCO/BBOB EXTERNAL COMPARISON — REVISION STATISTICS MANIFEST",
        "======================================================================",
        "",
        "FROZEN INPUTS",
        f"D2-D20: {D2_D20_PATH}",
        f"SHA-256: {hashes['D2_D20']}",
        "",
        f"D40: {D40_PATH}",
        f"SHA-256: {hashes['D40']}",
        "",
        f"Official f*: {FOPT_PATH}",
        f"SHA-256: {hashes['FOPT']}",
        "",
        "CORRECTED PERFORMANCE MEASURE",
        "error_corrected = best_f - f_opt_official",
        "Historical error/f_opt columns are provenance-only.",
        "",
        "EXPERIMENTAL DESIGN",
        "Algorithms: DMGSO, NELDER-MEAD, POWELL, COBYLA, COMPASS",
        "Dimensions: 2, 5, 10, 20, 40",
        "Functions: 24",
        "Instances per function: 15",
        "run_id observations per algorithm-problem block: 5",
        "",
        "INFERENTIAL AGGREGATION",
        "Median corrected error across the five run_id observations.",
        "",
        "DIMENSION-WISE STATISTICAL UNIT",
        "One BBOB function-instance problem.",
        "Paired units per dimension: 24 x 15 = 360.",
        "",
        "OMNIBUS ANALYSIS",
        "Friedman test separately at each dimension.",
        "Overall effect size: Kendall's W.",
        "",
        "PLANNED PAIRED COMPARISONS",
        "DMGSO vs NELDER-MEAD",
        "DMGSO vs POWELL",
        "DMGSO vs COBYLA",
        "DMGSO vs COMPASS",
        "",
        "PAIRWISE ANALYSIS",
        "Wilcoxon signed-rank test on 360 paired problem units per dimension.",
        "Holm correction across four planned comparisons within each dimension.",
        "Paired effect size: rank-biserial correlation.",
        "Positive effect size favors DMGSO.",
        "",
        "PRACTICAL-SIGNIFICANCE SUMMARIES",
        "Median paired improvement = baseline error - DMGSO error.",
        "95% bootstrap CI for the median paired improvement.",
        "Problem-level win/tie/loss.",
        "",
        "FE PROTOCOL",
        "Maximum budget = 1000 x D.",
        "Realized FE use is reported descriptively.",
        "",
        "DIMENSION-WISE FRIEDMAN RESULTS",
    ]

    for _, row in friedman.iterrows():
        lines.append(
            f"D={int(row['dimension'])}: "
            f"statistic={row['friedman_statistic']:.16g}, "
            f"p={row['p_value']:.16g}, "
            f"Kendall_W={row['kendall_W']:.16g}"
        )

    lines.extend(
        [
            "",
            "INTERPRETATION LIMITS",
            "The 360 units per dimension are function-instance problems, not 360 functions.",
            "No cross-dimension hypothesis test is introduced.",
            "Statistical significance is interpreted with rank/effect direction.",
            "",
            "======================================================================",
        ]
    )

    path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


# =============================================================================
# 10. MAIN WORKFLOW
# =============================================================================

def main() -> None:
    """
    Execute the complete reviewer-driven COCO/BBOB external analysis.
    """
    print("=" * 86)
    print("COCO/BBOB DMGSO EXTERNAL COMPARISON — REVIEWER-DRIVEN STATISTICS")
    print("=" * 86)

    hashes = {
        "D2_D20": verify_hash(
            D2_D20_PATH,
            EXPECTED_HASH_D2_D20,
            "Frozen D2-D20 corrected dataset",
        ),
        "D40": verify_hash(
            D40_PATH,
            EXPECTED_HASH_D40,
            "Frozen D40 corrected dataset",
        ),
        "FOPT": verify_hash(
            FOPT_PATH,
            EXPECTED_HASH_FOPT,
            "Frozen official f* reference",
        ),
    }

    print("[PASS] Frozen SHA-256 verification completed.")

    df = pd.concat(
        [
            pd.read_csv(D2_D20_PATH),
            pd.read_csv(D40_PATH),
        ],
        ignore_index=True,
        sort=False,
    )

    validate_combined_input(
        df
    )

    print("[PASS] Corrected experimental structure verified.")
    print(
        "[PASS] Dimension-wise statistical unit: "
        "BBOB function-instance problem (n=360 per D)."
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    problem_level = make_problem_level_medians(
        df
    )

    descriptive = dimension_algorithm_descriptive(
        problem_level
    )

    friedman, average_ranks = friedman_and_ranks_by_dimension(
        problem_level
    )

    pairwise = planned_pairwise_by_dimension(
        problem_level
    )

    fe_summary = fe_usage_summary(
        df
    )

    trend = dmgso_dimension_trend(
        average_ranks,
        pairwise,
    )

    # -------------------------------------------------------------------------
    # Derived outputs only.
    # -------------------------------------------------------------------------
    problem_level.to_csv(
        OUTPUT_DIR / "01_problem_level_median_corrected_error.csv",
        index=False,
    )

    descriptive.to_csv(
        OUTPUT_DIR / "02_dimension_algorithm_descriptive.csv",
        index=False,
    )

    friedman.to_csv(
        OUTPUT_DIR / "03_friedman_by_dimension.csv",
        index=False,
    )

    average_ranks.to_csv(
        OUTPUT_DIR / "04_average_ranks_by_dimension.csv",
        index=False,
    )

    pairwise.to_csv(
        OUTPUT_DIR / "05_planned_pairwise_wilcoxon_by_dimension.csv",
        index=False,
    )

    fe_summary.to_csv(
        OUTPUT_DIR / "06_fe_usage_by_dimension.csv",
        index=False,
    )

    trend.to_csv(
        OUTPUT_DIR / "07_dmgso_dimension_trend.csv",
        index=False,
    )

    write_manifest(
        hashes,
        friedman,
    )

    # -------------------------------------------------------------------------
    # Console summary.
    # -------------------------------------------------------------------------
    print()
    print("[OK] Statistical outputs written to:")
    print(f"     {OUTPUT_DIR}")

    print()
    print("Friedman omnibus results by dimension:")
    print(
        friedman.to_string(
            index=False
        )
    )

    print()
    print("Average ranks by dimension:")
    print(
        average_ranks.to_string(
            index=False
        )
    )

    print()
    print("Planned DMGSO-vs-baseline comparisons:")

    display_cols = [
        "dimension",
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
    print("DMGSO dimension trend:")
    print(
        trend.to_string(
            index=False
        )
    )

    print()
    print(
        "[PASS] Reviewer-driven COCO/BBOB external statistics "
        "completed successfully."
    )

    print("=" * 86)


if __name__ == "__main__":
    main()
