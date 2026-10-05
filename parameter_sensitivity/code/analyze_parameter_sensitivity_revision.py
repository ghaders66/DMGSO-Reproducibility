#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Reviewer 5 - Comment 6
Confirmatory OFAT parameter-sensitivity analysis for frozen DMGSO V4.

DESIGN
------
- CEC2014 F1-F30
- D = 30
- FE budget = 300,000
- run_id = 0,...,4
- 17 configurations:
    baseline + 16 one-factor-at-a-time perturbations
- 2550 expected runs
- Primary outcome: corrected_error
- Five run_ids are aggregated by median within each
  function x configuration.
- Inferential unit: benchmark function (n = 30).
- Each of the 16 perturbations is compared with the common baseline.
- Planned inference:
    two-sided paired Wilcoxon signed-rank test
    Holm correction across all 16 comparisons
    paired rank-biserial effect size
    median paired difference (perturbation - baseline)
    percentile bootstrap 95% CI
    Win/Tie/Loss from perturbation perspective
- Lower corrected_error is better.

IMPORTANT
---------
This analysis characterizes sensitivity.
It does NOT tune or modify the frozen DMGSO baseline.
No algorithm/core/source file is modified.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.stats import wilcoxon


# =====================================================================
# Frozen paths and protocol expectations
# =====================================================================

ROOT = Path(
    "/mnt/d/PHD/DMGSO_Information Scince/Revision/parameter_sensitivity"
)

RAW = ROOT / "raw" / "parameter_sensitivity_runs.csv"
COMPLETION = ROOT / "audit" / "full_run_completion.json"

PROCESSED = ROOT / "processed"
STATISTICS = ROOT / "statistics"
AUDIT = ROOT / "audit"
LOGS = ROOT / "logs"

EXPECTED_CORE_SHA = (
    "cadf35f0ae664619527707caf2deea4a163e79eb970cf273e53f5490f9cd9191"
)

EXPECTED_PROTOCOL_SHA = (
    "5760a3082cfaa7be9e249484863aff908061156e9d5c76565ad7f1a40223526f"
)

EXPECTED_RUNS = 2550
EXPECTED_FUNCTIONS = list(range(1, 31))
EXPECTED_RUN_IDS = list(range(5))
EXPECTED_D = 30
EXPECTED_FE = 300000
EXPECTED_CONFIGS = 17

PRIMARY_OUTCOME = "corrected_error"

BOOTSTRAP_SEED = 20261002
BOOTSTRAP_REPS = 100000

TIE_ATOL = 1e-12
TIE_RTOL = 1e-12


# =====================================================================
# Utilities
# =====================================================================

def ensure_dirs():
    for p in (PROCESSED, STATISTICS, AUDIT, LOGS):
        p.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj):
    path.write_text(
        json.dumps(
            obj,
            indent=2,
            sort_keys=True,
            default=str,
        ) + "\n",
        encoding="utf-8",
    )


def holm_adjust(p_values):
    """
    Holm step-down adjusted p-values.
    """
    p = np.asarray(p_values, dtype=float)
    m = len(p)

    order = np.argsort(p)
    adjusted_sorted = np.empty(m, dtype=float)

    running_max = 0.0

    for rank, idx in enumerate(order):
        adj = (m - rank) * p[idx]
        running_max = max(running_max, adj)
        adjusted_sorted[rank] = min(running_max, 1.0)

    adjusted = np.empty(m, dtype=float)

    for rank, idx in enumerate(order):
        adjusted[idx] = adjusted_sorted[rank]

    return adjusted


def paired_rank_biserial(diff):
    """
    Paired rank-biserial correlation for:
        diff = perturbation - baseline

    Positive value:
        perturbation tends to have larger error (worse).

    Negative value:
        perturbation tends to have smaller error (better).

    Exact zero differences are removed.
    """
    d = np.asarray(diff, dtype=float)
    d = d[d != 0.0]

    if len(d) == 0:
        return 0.0

    abs_d = np.abs(d)

    ranks = pd.Series(abs_d).rank(
        method="average"
    ).to_numpy(dtype=float)

    w_plus = float(ranks[d > 0].sum())
    w_minus = float(ranks[d < 0].sum())

    denom = w_plus + w_minus

    if denom == 0:
        return 0.0

    return (w_plus - w_minus) / denom


def bootstrap_median_ci(
    diff,
    rng,
    reps=BOOTSTRAP_REPS,
):
    """
    Percentile bootstrap 95% CI for the median paired difference.
    Resampling unit = benchmark function.
    """
    d = np.asarray(diff, dtype=float)
    n = len(d)

    idx = rng.integers(
        0,
        n,
        size=(reps, n),
    )

    medians = np.median(d[idx], axis=1)

    lo, hi = np.quantile(
        medians,
        [0.025, 0.975],
    )

    return float(lo), float(hi)


# =====================================================================
# Input audit
# =====================================================================

def load_and_audit():

    if not RAW.exists():
        raise FileNotFoundError(RAW)

    if not COMPLETION.exists():
        raise FileNotFoundError(COMPLETION)

    completion = json.loads(
        COMPLETION.read_text(encoding="utf-8")
    )

    problems = []

    # ---------------------------------------------------------------
    # Completion-record guardrails
    # ---------------------------------------------------------------

    if completion.get("status") != "complete":
        problems.append(
            "Completion record status is not 'complete'."
        )

    if completion.get("expected_runs") != EXPECTED_RUNS:
        problems.append(
            "Completion-record expected_runs mismatch."
        )

    if completion.get("observed_unique_runs") != EXPECTED_RUNS:
        problems.append(
            "Completion-record observed_unique_runs mismatch."
        )

    if completion.get("core_sha256_before") != EXPECTED_CORE_SHA:
        problems.append(
            "Unexpected pre-run core SHA256."
        )

    if completion.get("core_sha256_after") != EXPECTED_CORE_SHA:
        problems.append(
            "Unexpected post-run core SHA256."
        )

    if (
        completion.get("core_sha256_before")
        != completion.get("core_sha256_after")
    ):
        problems.append(
            "Frozen core changed during the full run."
        )

    if completion.get("protocol_sha256") != EXPECTED_PROTOCOL_SHA:
        problems.append(
            "Protocol SHA256 mismatch."
        )

    # ---------------------------------------------------------------
    # Raw-data guardrails
    # ---------------------------------------------------------------

    df = pd.read_csv(RAW)

    required = {
        "case_id",
        "parameter",
        "level",
        "parameter_value",
        "function_id",
        "category",
        "dimension",
        "run_id",
        "fe_budget",
        "fe_used",
        "best_f",
        "f_opt",
        "corrected_error",
        "wall_seconds",
        "cfg_version",
        "cfg_D",
        "cfg_FE_budget",
    }

    missing = required - set(df.columns)

    if missing:
        problems.append(
            f"Missing required columns: {sorted(missing)}"
        )

    # ---------------------------------------------------------------
    # Exact deterministic duplicate handling
    # ---------------------------------------------------------------

    run_key = ["case_id", "function_id", "run_id"]
    physical_rows = len(df)
    unique_run_keys = df[run_key].drop_duplicates().shape[0]

    duplicate_rows = df[
        df.duplicated(run_key, keep=False)
    ].copy()
    duplicate_key_count = (
        duplicate_rows[run_key].drop_duplicates().shape[0]
    )

    # Repeated deterministic executions may differ only in timestamp
    # and wall-clock duration. Every scientific/configuration field
    # must be exactly identical before a duplicate can be excluded.
    scientific_columns = [
        c for c in df.columns
        if c not in {"timestamp", "wall_seconds"}
    ]
    nonidentical_duplicate_keys = []

    for key_values, group in duplicate_rows.groupby(run_key, sort=True):
        reference = group.iloc[0][scientific_columns]
        for i in range(1, len(group)):
            candidate = group.iloc[i][scientific_columns]
            equal = True
            for col in scientific_columns:
                a, b = reference[col], candidate[col]
                if pd.isna(a) and pd.isna(b):
                    continue
                if pd.isna(a) != pd.isna(b):
                    equal = False
                    break
                if isinstance(a, (float, np.floating)) or isinstance(b, (float, np.floating)):
                    try:
                        if not np.isclose(float(a), float(b), rtol=0.0, atol=0.0, equal_nan=True):
                            equal = False
                            break
                    except (TypeError, ValueError):
                        if str(a) != str(b):
                            equal = False
                            break
                elif str(a) != str(b):
                    equal = False
                    break
            if not equal:
                nonidentical_duplicate_keys.append(tuple(key_values))
                break

    if nonidentical_duplicate_keys:
        problems.append(
            "Duplicate run keys with non-identical scientific/configuration "
            f"content detected: {nonidentical_duplicate_keys}"
        )

    # Never modify the frozen raw CSV. Build a canonical in-memory view
    # containing one record per predefined run key.
    df_analysis = (
        df.sort_values(run_key + ["timestamp"])
        .drop_duplicates(subset=run_key, keep="first")
        .reset_index(drop=True)
    )

    if unique_run_keys != EXPECTED_RUNS:
        problems.append(
            f"Expected {EXPECTED_RUNS} unique run keys; found {unique_run_keys}."
        )
    if len(df_analysis) != EXPECTED_RUNS:
        problems.append(
            f"Expected {EXPECTED_RUNS} canonical unique runs; found {len(df_analysis)}."
        )

    # All subsequent completeness/statistical checks use the canonical
    # unique-run view.
    df = df_analysis

    if set(df["function_id"].astype(int)) != set(EXPECTED_FUNCTIONS):
        problems.append(
            "Function set is not exactly F1-F30."
        )

    if set(df["run_id"].astype(int)) != set(EXPECTED_RUN_IDS):
        problems.append(
            "run_id set is not exactly 0-4."
        )

    if not (df["dimension"].astype(int) == EXPECTED_D).all():
        problems.append(
            "Unexpected dimension detected."
        )

    if not (df["fe_budget"].astype(int) == EXPECTED_FE).all():
        problems.append(
            "Unexpected FE budget detected."
        )

    if not (df["fe_used"].astype(int) == EXPECTED_FE).all():
        problems.append(
            "At least one run did not consume exactly 300000 FEs."
        )

    if not (df["cfg_D"].astype(int) == EXPECTED_D).all():
        problems.append(
            "cfg_D mismatch."
        )

    if not (df["cfg_FE_budget"].astype(int) == EXPECTED_FE).all():
        problems.append(
            "cfg_FE_budget mismatch."
        )

    if not (df["cfg_version"].astype(str) == "V4").all():
        problems.append(
            "Unexpected cfg_version detected."
        )

    if not np.isfinite(
        df[PRIMARY_OUTCOME].astype(float)
    ).all():
        problems.append(
            "Non-finite corrected_error detected."
        )

    # corrected_error consistency
    recomputed = (
        df["best_f"].astype(float)
        - df["f_opt"].astype(float)
    )

    if not np.allclose(
        recomputed,
        df["corrected_error"].astype(float),
        rtol=1e-10,
        atol=1e-10,
    ):
        problems.append(
            "corrected_error != best_f - f_opt for some rows."
        )

    case_ids = sorted(
        df["case_id"].astype(str).unique()
    )

    if len(case_ids) != EXPECTED_CONFIGS:
        problems.append(
            f"Expected {EXPECTED_CONFIGS} unique case_id values; "
            f"found {len(case_ids)}."
        )

    if "baseline" not in case_ids:
        problems.append(
            "Baseline case is missing."
        )

    # 30 functions x 5 run_ids = 150 rows per configuration
    case_counts = (
        df.groupby("case_id")
        .size()
        .sort_index()
    )

    bad_case_counts = case_counts[
        case_counts != 150
    ]

    if len(bad_case_counts):
        problems.append(
            "Some configurations do not contain exactly 150 runs: "
            + bad_case_counts.to_dict().__repr__()
        )

    # each function x configuration must contain five run_ids
    cell_counts = (
        df.groupby(
            ["case_id", "function_id"]
        )
        .size()
    )

    if not (cell_counts == 5).all():
        problems.append(
            "At least one case/function cell does not contain five runs."
        )

    # exactly 16 non-baseline OFAT perturbations
    perturbations = [
        x for x in case_ids
        if x != "baseline"
    ]

    if len(perturbations) != 16:
        problems.append(
            f"Expected 16 perturbations; found {len(perturbations)}."
        )

    audit = {
        "pass": len(problems) == 0,
        "problems": problems,
        "raw_path": str(RAW),
        "raw_sha256": sha256_file(RAW),
        "completion_record": str(COMPLETION),
        "completion_record_sha256": sha256_file(COMPLETION),
        "core_sha256_before":
            completion.get("core_sha256_before"),
        "core_sha256_after":
            completion.get("core_sha256_after"),
        "protocol_sha256":
            completion.get("protocol_sha256"),
        "physical_raw_rows": int(physical_rows),
        "unique_run_keys": int(unique_run_keys),
        "canonical_analysis_rows": int(len(df_analysis)),
        "duplicate_keys_detected": int(duplicate_key_count),
        "duplicate_extra_rows_excluded": int(physical_rows - len(df_analysis)),
        "duplicate_scientific_content_identical": bool(len(nonidentical_duplicate_keys) == 0),
        "duplicate_handling": (
            "Repeated deterministic records remain unchanged in the frozen raw CSV; "
            "one canonical record per run key is used in memory only after all "
            "scientific/configuration fields are verified identical. timestamp and "
            "wall_seconds are allowed to differ."
        ),
        "unique_cases": int(len(case_ids)),
        "case_ids": case_ids,
        "primary_outcome": PRIMARY_OUTCOME,
        "inferential_unit": "CEC2014 benchmark function",
        "n_functions": 30,
        "runs_per_function_case": 5,
        "analysis_role":
            "sensitivity characterization, not parameter tuning",
    }

    write_json(
        AUDIT / "parameter_sensitivity_analysis_input_audit.json",
        audit,
    )

    if problems:
        raise RuntimeError(
            "INPUT AUDIT FAILED:\n- "
            + "\n- ".join(problems)
        )

    return df, completion


# =====================================================================
# Configuration inventory
# =====================================================================

def configuration_inventory(df):

    inventory = (
        df[
            [
                "case_id",
                "parameter",
                "level",
                "parameter_value",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            ["parameter", "level", "case_id"]
        )
        .reset_index(drop=True)
    )

    inventory.to_csv(
        PROCESSED / "01_configuration_inventory.csv",
        index=False,
    )

    return inventory


# =====================================================================
# Function-level aggregation
# =====================================================================

def aggregate_function_level(df):

    agg = (
        df.groupby(
            [
                "case_id",
                "parameter",
                "level",
                "parameter_value",
                "function_id",
                "category",
            ],
            dropna=False,
            as_index=False,
        )
        .agg(
            median_corrected_error=(
                "corrected_error",
                "median",
            ),
            mean_corrected_error=(
                "corrected_error",
                "mean",
            ),
            min_corrected_error=(
                "corrected_error",
                "min",
            ),
            max_corrected_error=(
                "corrected_error",
                "max",
            ),
            q25_corrected_error=(
                "corrected_error",
                lambda x: x.quantile(0.25),
            ),
            q75_corrected_error=(
                "corrected_error",
                lambda x: x.quantile(0.75),
            ),
            median_wall_seconds=(
                "wall_seconds",
                "median",
            ),
            n_runs=(
                "run_id",
                "size",
            ),
        )
        .sort_values(
            ["case_id", "function_id"]
        )
        .reset_index(drop=True)
    )

    if not (agg["n_runs"] == 5).all():
        raise RuntimeError(
            "Function-level aggregation found a cell with n_runs != 5."
        )

    agg.to_csv(
        PROCESSED / "02_function_level_sensitivity_summary.csv",
        index=False,
    )

    return agg


# =====================================================================
# Descriptive statistics across functions
# =====================================================================

def descriptive_statistics(agg):

    desc = (
        agg.groupby(
            [
                "case_id",
                "parameter",
                "level",
                "parameter_value",
            ],
            dropna=False,
            as_index=False,
        )
        .agg(
            n_functions=(
                "function_id",
                "size",
            ),
            median_function_error=(
                "median_corrected_error",
                "median",
            ),
            mean_function_error=(
                "median_corrected_error",
                "mean",
            ),
            q25_function_error=(
                "median_corrected_error",
                lambda x: x.quantile(0.25),
            ),
            q75_function_error=(
                "median_corrected_error",
                lambda x: x.quantile(0.75),
            ),
            min_function_error=(
                "median_corrected_error",
                "min",
            ),
            max_function_error=(
                "median_corrected_error",
                "max",
            ),
        )
        .sort_values(
            ["parameter", "level", "case_id"]
        )
        .reset_index(drop=True)
    )

    if not (desc["n_functions"] == 30).all():
        raise RuntimeError(
            "Expected 30 function-level observations per configuration."
        )

    desc.to_csv(
        STATISTICS / "03_parameter_sensitivity_descriptive_statistics.csv",
        index=False,
    )

    return desc


# =====================================================================
# Paired matrix
# =====================================================================

def paired_matrix(agg):

    matrix = agg.pivot(
        index="function_id",
        columns="case_id",
        values="median_corrected_error",
    ).sort_index()

    if matrix.shape != (30, 17):
        raise RuntimeError(
            f"Expected paired matrix shape (30,17), found {matrix.shape}."
        )

    if matrix.isna().any().any():
        raise RuntimeError(
            "Missing values in paired function-level matrix."
        )

    matrix.to_csv(
        PROCESSED / "04_function_level_paired_matrix.csv"
    )

    return matrix


# =====================================================================
# Wilcoxon/Holm planned comparisons
# =====================================================================

def planned_comparisons(
    matrix,
    inventory,
):

    rng = np.random.default_rng(
        BOOTSTRAP_SEED
    )

    baseline = matrix["baseline"].to_numpy(
        dtype=float
    )

    perturbation_inventory = (
        inventory[
            inventory["case_id"] != "baseline"
        ]
        .copy()
        .sort_values(
            ["parameter", "level", "case_id"]
        )
        .reset_index(drop=True)
    )

    rows = []

    for _, meta in perturbation_inventory.iterrows():

        case_id = str(meta["case_id"])

        pert = matrix[case_id].to_numpy(
            dtype=float
        )

        diff = pert - baseline

        # Wilcoxon signed-rank.
        # method='auto' allows SciPy to handle ties/zeros appropriately.
        try:
            result = wilcoxon(
                pert,
                baseline,
                alternative="two-sided",
                zero_method="wilcox",
                method="auto",
            )

            statistic = float(result.statistic)
            p_raw = float(result.pvalue)

        except ValueError:
            # All paired differences exactly zero.
            statistic = 0.0
            p_raw = 1.0

        rbc = paired_rank_biserial(diff)

        median_diff = float(
            np.median(diff)
        )

        ci_lo, ci_hi = bootstrap_median_ci(
            diff,
            rng,
        )

        # W/T/L from perturbation perspective.
        better = pert < baseline
        worse = pert > baseline

        tied = np.isclose(
            pert,
            baseline,
            rtol=TIE_RTOL,
            atol=TIE_ATOL,
        )

        # Remove numerically tied entries from better/worse.
        better = better & (~tied)
        worse = worse & (~tied)

        wins = int(np.sum(better))
        ties = int(np.sum(tied))
        losses = int(np.sum(worse))

        rows.append(
            {
                "case_id": case_id,
                "parameter": meta["parameter"],
                "level": meta["level"],
                "parameter_value": meta["parameter_value"],
                "comparison":
                    f"{case_id} vs baseline",
                "n_functions": 30,
                "wilcoxon_statistic":
                    statistic,
                "p_raw":
                    p_raw,
                "rank_biserial_perturbation_minus_baseline":
                    rbc,
                "median_paired_difference_perturbation_minus_baseline":
                    median_diff,
                "bootstrap95_median_diff_low":
                    ci_lo,
                "bootstrap95_median_diff_high":
                    ci_hi,
                "wins_perturbation_better":
                    wins,
                "ties":
                    ties,
                "losses_perturbation_worse":
                    losses,
            }
        )

    out = pd.DataFrame(rows)

    if len(out) != 16:
        raise RuntimeError(
            f"Expected 16 planned comparisons; found {len(out)}."
        )

    out["p_holm"] = holm_adjust(
        out["p_raw"].to_numpy()
    )

    out["significant_holm_0.05"] = (
        out["p_holm"] < 0.05
    )

    # Put adjusted p near raw p.
    columns = [
        "case_id",
        "parameter",
        "level",
        "parameter_value",
        "comparison",
        "n_functions",
        "wilcoxon_statistic",
        "p_raw",
        "p_holm",
        "significant_holm_0.05",
        "rank_biserial_perturbation_minus_baseline",
        "median_paired_difference_perturbation_minus_baseline",
        "bootstrap95_median_diff_low",
        "bootstrap95_median_diff_high",
        "wins_perturbation_better",
        "ties",
        "losses_perturbation_worse",
    ]

    out = out[columns]

    out.to_csv(
        STATISTICS / "05_wilcoxon_holm_perturbations_vs_baseline.csv",
        index=False,
    )

    return out


# =====================================================================
# Function-wise paired differences
# =====================================================================

def function_wise_differences(
    matrix,
    inventory,
):

    baseline = matrix["baseline"]

    rows = []

    pert_inventory = inventory[
        inventory["case_id"] != "baseline"
    ].copy()

    for _, meta in pert_inventory.iterrows():

        case_id = str(meta["case_id"])

        diff = (
            matrix[case_id]
            - baseline
        )

        for fid, value in diff.items():

            rows.append(
                {
                    "case_id": case_id,
                    "parameter": meta["parameter"],
                    "level": meta["level"],
                    "parameter_value": meta["parameter_value"],
                    "function_id": int(fid),
                    "baseline_median_error":
                        float(baseline.loc[fid]),
                    "perturbation_median_error":
                        float(matrix.loc[fid, case_id]),
                    "paired_difference_perturbation_minus_baseline":
                        float(value),
                }
            )

    out = pd.DataFrame(rows).sort_values(
        ["parameter", "level", "function_id"]
    )

    if len(out) != 16 * 30:
        raise RuntimeError(
            f"Expected 480 paired differences; found {len(out)}."
        )

    out.to_csv(
        PROCESSED / "06_function_wise_paired_differences.csv",
        index=False,
    )

    return out


# =====================================================================
# Parameter-level characterization
# =====================================================================

def parameter_characterization(
    comparisons,
):

    rows = []

    for parameter, sub in comparisons.groupby(
        "parameter",
        sort=True,
    ):

        sub = sub.copy()

        sig = sub[
            sub["significant_holm_0.05"]
        ]

        rows.append(
            {
                "parameter": parameter,
                "n_perturbations_tested": int(len(sub)),
                "n_holm_significant": int(len(sig)),
                "any_holm_significant":
                    bool(len(sig) > 0),
                "min_raw_p":
                    float(sub["p_raw"].min()),
                "min_holm_p":
                    float(sub["p_holm"].min()),
                "max_abs_rank_biserial":
                    float(
                        sub[
                            "rank_biserial_perturbation_minus_baseline"
                        ].abs().max()
                    ),
            }
        )

    out = pd.DataFrame(rows)

    if len(out) != 8:
        raise RuntimeError(
            f"Expected 8 parameters; found {len(out)}."
        )

    out.to_csv(
        STATISTICS / "07_parameter_level_characterization.csv",
        index=False,
    )

    return out


# =====================================================================
# Machine-readable summary
# =====================================================================

def write_summary(
    comparisons,
    parameter_summary,
    raw_sha,
):

    significant = comparisons[
        comparisons["significant_holm_0.05"]
    ]

    summary = {
        "analysis": "DMGSO OFAT parameter sensitivity",
        "reviewer_comment": "Reviewer 5 Comment 6",
        "analysis_role":
            "characterization only; not parameter tuning",
        "core_modified": False,
        "baseline_retuned": False,
        "primary_outcome": PRIMARY_OUTCOME,
        "benchmark": "CEC2014 F1-F30",
        "dimension": EXPECTED_D,
        "fe_budget": EXPECTED_FE,
        "run_ids": EXPECTED_RUN_IDS,
        "n_expected_unique_runs": EXPECTED_RUNS,
        "n_configurations": 17,
        "n_perturbations": 16,
        "runs_per_function_configuration": 5,
        "function_aggregation": "median over five run_ids",
        "inferential_unit": "benchmark function",
        "n_inferential_units": 30,
        "planned_test":
            "two-sided paired Wilcoxon signed-rank",
        "multiplicity":
            "Holm correction across 16 perturbation-vs-baseline comparisons",
        "effect_size":
            "paired rank-biserial correlation",
        "paired_difference":
            "perturbation corrected_error minus baseline corrected_error",
        "bootstrap_ci":
            "percentile 95% CI for median paired difference",
        "bootstrap_reps": BOOTSTRAP_REPS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "lower_error_is_better": True,
        "n_holm_significant_comparisons":
            int(len(significant)),
        "holm_significant_case_ids":
            significant["case_id"].astype(str).tolist(),
        "n_parameters_with_any_holm_significant_perturbation":
            int(
                parameter_summary[
                    "any_holm_significant"
                ].sum()
            ),
        "raw_sha256": raw_sha,
        "interpretation_guardrail":
            (
                "The analysis characterizes local OFAT sensitivity "
                "around the frozen baseline settings. It does not "
                "establish global parameter optimality, parameter "
                "interactions, or justify retuning the baseline."
            ),
    }

    write_json(
        STATISTICS / "08_parameter_sensitivity_analysis_summary.json",
        summary,
    )


# =====================================================================
# Reproducibility manifest
# =====================================================================

def write_manifest():

    script = Path(__file__).resolve()

    output_files = sorted(
        list(PROCESSED.glob("*"))
        + list(STATISTICS.glob("*"))
        + [
            AUDIT
            / "parameter_sensitivity_analysis_input_audit.json"
        ]
    )

    hashes = {}

    for p in output_files:
        if p.exists() and p.is_file():
            hashes[
                str(p.relative_to(ROOT))
            ] = sha256_file(p)

    manifest = {
        "analysis_script": str(script),
        "analysis_script_sha256":
            sha256_file(script),
        "raw_input": str(RAW),
        "raw_input_sha256":
            sha256_file(RAW),
        "completion_record":
            str(COMPLETION),
        "completion_record_sha256":
            sha256_file(COMPLETION),
        "expected_core_sha256":
            EXPECTED_CORE_SHA,
        "expected_protocol_sha256":
            EXPECTED_PROTOCOL_SHA,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "outputs": hashes,
        "core_modified": False,
        "raw_modified": False,
        "analysis_role":
            "confirmatory OFAT sensitivity characterization",
    }

    write_json(
        AUDIT
        / "parameter_sensitivity_analysis_manifest.json",
        manifest,
    )


# =====================================================================
# Console report
# =====================================================================

def print_report(
    comparisons,
    parameter_summary,
):

    print()
    print("=" * 92)
    print("DMGSO PARAMETER-SENSITIVITY ANALYSIS")
    print("=" * 92)

    print(
        "Primary outcome: corrected_error"
    )
    print(
        "Inferential unit: CEC2014 benchmark function (n=30)"
    )
    print(
        "Design: frozen baseline + 16 OFAT perturbations; "
        "five run_ids per function/configuration"
    )
    print(
        "Multiplicity: Holm correction across 16 planned "
        "perturbation-vs-baseline comparisons"
    )

    print()
    print("PLANNED PERTURBATION-vs-BASELINE COMPARISONS")

    display_cols = [
        "case_id",
        "parameter",
        "level",
        "parameter_value",
        "p_raw",
        "p_holm",
        "rank_biserial_perturbation_minus_baseline",
        "median_paired_difference_perturbation_minus_baseline",
        "bootstrap95_median_diff_low",
        "bootstrap95_median_diff_high",
        "wins_perturbation_better",
        "ties",
        "losses_perturbation_worse",
    ]

    print(
        comparisons[display_cols].to_string(
            index=False,
            float_format=lambda x: f"{x:.6e}",
        )
    )

    print()
    print("PARAMETER-LEVEL CHARACTERIZATION")

    print(
        parameter_summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.6e}",
        )
    )

    print()
    n_sig = int(
        comparisons[
            "significant_holm_0.05"
        ].sum()
    )

    print(
        f"Holm-significant perturbations: "
        f"{n_sig}/16"
    )

    if n_sig:
        print("Significant case_ids:")
        for case_id in comparisons.loc[
            comparisons[
                "significant_holm_0.05"
            ],
            "case_id",
        ]:
            print(f"  - {case_id}")
    else:
        print(
            "No perturbation differs significantly from "
            "baseline after Holm correction."
        )

    print()
    print(
        "Interpretation guardrail: this is local OFAT "
        "sensitivity characterization around the frozen "
        "baseline. It is not parameter tuning and does not "
        "establish global parameter optimality or interactions."
    )

    print("=" * 92)


# =====================================================================
# Main
# =====================================================================

def main():

    ensure_dirs()

    print("[1/9] Auditing frozen full-run inputs...")
    df, completion = load_and_audit()
    print("      INPUT AUDIT PASS")

    print("[2/9] Writing configuration inventory...")
    inventory = configuration_inventory(df)

    print("[3/9] Aggregating five run_ids at function level...")
    agg = aggregate_function_level(df)

    print("[4/9] Computing descriptive statistics...")
    descriptive_statistics(agg)

    print("[5/9] Constructing paired function-level matrix...")
    matrix = paired_matrix(agg)

    print("[6/9] Running 16 planned Wilcoxon/Holm comparisons...")
    comparisons = planned_comparisons(
        matrix,
        inventory,
    )

    print("[7/9] Writing function-wise paired differences...")
    function_wise_differences(
        matrix,
        inventory,
    )

    print("[8/9] Characterizing parameter-level sensitivity...")
    parameter_summary = parameter_characterization(
        comparisons
    )

    print("[9/9] Writing summary and reproducibility manifest...")

    raw_sha = sha256_file(RAW)

    write_summary(
        comparisons,
        parameter_summary,
        raw_sha,
    )

    write_manifest()

    print_report(
        comparisons,
        parameter_summary,
    )

    print()
    print("PARAMETER-SENSITIVITY ANALYSIS PASS")
    print(
        f"Outputs written only below: {ROOT}"
    )


if __name__ == "__main__":
    main()
