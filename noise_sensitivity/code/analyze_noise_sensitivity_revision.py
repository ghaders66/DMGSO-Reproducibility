#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Statistical analysis of the frozen DMGSO observational-noise experiment.

Reviewer 5 noise-sensitivity analysis.

Scientific rules
----------------
- Read ONLY the frozen raw dataset.
- Verify its SHA256 before analysis.
- Never modify the frozen raw dataset.
- Primary outcome: true_final_error.
- Five deterministic run-indexed configurations are aggregated by the
  median within each (function, noise level).
- Benchmark function is the inferential unit (n = 12).
- Omnibus comparison: Friedman test across the four noise levels.
- Planned comparisons: each nonzero noise level versus eta = 0 control.
- Multiplicity correction: Holm across the three planned comparisons.
- Effect size: paired rank-biserial correlation.
- Additional reporting:
    * median paired difference (noisy - control)
    * bootstrap 95% CI for the median paired difference
    * Win/Tie/Loss counts relative to control
- No parameter tuning or post-hoc selection is performed.
- All generated outputs remain under Revision/noise_sensitivity/.
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
from scipy.stats import friedmanchisquare, rankdata, wilcoxon


# ---------------------------------------------------------------------
# Frozen experiment definition
# ---------------------------------------------------------------------

EXP_ROOT = Path(
    "/mnt/d/PHD/DMGSO_Information Scince/Revision/noise_sensitivity"
)

RAW_PATH = (
    EXP_ROOT
    / "raw"
    / "noise_sensitivity_cec2014_run_summary_FROZEN.csv"
)

PROTOCOL_PATH = (
    EXP_ROOT
    / "configs"
    / "noise_sensitivity_protocol.json"
)

PROCESSED_DIR = EXP_ROOT / "processed"
STATISTICS_DIR = EXP_ROOT / "statistics"
FIGURES_DIR = EXP_ROOT / "figures"
AUDIT_DIR = EXP_ROOT / "audit"

EXPECTED_RAW_SHA256 = (
    "26bf67a1e92cb32106a40fb7eca73c7ca8430a63907ca6e585c365b3b6c3193b"
)

EXPECTED_PROTOCOL_VERSION = "R5-C8-noise-v1.0"

EXPECTED_FUNCTIONS = [1, 2, 3, 4, 8, 12, 17, 20, 22, 23, 26, 30]
EXPECTED_RUN_IDS = [0, 1, 2, 3, 4]
EXPECTED_NOISE_LEVELS = [0.0, 1e-4, 1e-3, 1e-2]
EXPECTED_DIMENSION = 30
EXPECTED_BUDGET = 300_000
EXPECTED_ROWS = (
    len(EXPECTED_FUNCTIONS)
    * len(EXPECTED_RUN_IDS)
    * len(EXPECTED_NOISE_LEVELS)
)

PRIMARY_OUTCOME = "true_final_error"

# Fixed deterministic bootstrap seed for reproducibility.
BOOTSTRAP_SEED = 20261001
BOOTSTRAP_REPS = 100_000

# Numerical tolerance only for W/T/L equality classification.
# Exact equality is expected to be uncommon; this prevents machine-level
# floating-point differences from being labeled as substantive changes.
WTL_ATOL = 1e-12
WTL_RTOL = 1e-12


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def ensure_dirs() -> None:
    for p in (
        PROCESSED_DIR,
        STATISTICS_DIR,
        FIGURES_DIR,
        AUDIT_DIR,
    ):
        p.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj) -> None:
    path.write_text(
        json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def eta_label(eta: float) -> str:
    if eta == 0.0:
        return "eta_0"
    return f"eta_{eta:.0e}".replace("+", "")


# ---------------------------------------------------------------------
# Holm correction
# ---------------------------------------------------------------------

def holm_adjust(pvalues: list[float]) -> list[float]:
    """
    Holm step-down family-wise error correction.

    Returns adjusted p-values in original input order.
    """
    p = np.asarray(pvalues, dtype=float)
    m = len(p)

    order = np.argsort(p)
    sorted_p = p[order]

    adjusted_sorted = np.empty(m, dtype=float)

    running_max = 0.0
    for i, pv in enumerate(sorted_p):
        candidate = (m - i) * pv
        running_max = max(running_max, candidate)
        adjusted_sorted[i] = min(1.0, running_max)

    adjusted = np.empty(m, dtype=float)
    adjusted[order] = adjusted_sorted

    return adjusted.tolist()


# ---------------------------------------------------------------------
# Effect size
# ---------------------------------------------------------------------

def paired_rank_biserial(noisy: np.ndarray, control: np.ndarray) -> float:
    """
    Paired rank-biserial correlation based on noisy - control.

    Positive value:
        noisy condition tends to have larger true final error than control
        -> degradation under noise.

    Negative value:
        noisy condition tends to have smaller true final error than control.

    Zero differences are removed, matching the usual Wilcoxon convention.
    """
    d = np.asarray(noisy, dtype=float) - np.asarray(control, dtype=float)

    nz = d != 0.0
    d = d[nz]

    if d.size == 0:
        return 0.0

    ranks = rankdata(np.abs(d), method="average")

    r_plus = float(np.sum(ranks[d > 0]))
    r_minus = float(np.sum(ranks[d < 0]))

    denom = r_plus + r_minus
    if denom == 0.0:
        return 0.0

    return (r_plus - r_minus) / denom


# ---------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------

def bootstrap_median_difference_ci(
    differences: np.ndarray,
    seed: int,
    reps: int = BOOTSTRAP_REPS,
) -> tuple[float, float]:
    """
    Percentile bootstrap 95% CI for the median paired difference.

    Resampling unit = benchmark function.
    """
    d = np.asarray(differences, dtype=float)

    if d.ndim != 1 or d.size == 0:
        raise ValueError("Bootstrap differences must be a non-empty 1-D array.")

    rng = np.random.default_rng(seed)

    # n=12, so 100k x 12 is small and reproducible.
    idx = rng.integers(0, d.size, size=(reps, d.size))
    boot = np.median(d[idx], axis=1)

    low, high = np.quantile(boot, [0.025, 0.975])

    return float(low), float(high)


# ---------------------------------------------------------------------
# Integrity audit
# ---------------------------------------------------------------------

def audit_inputs() -> tuple[pd.DataFrame, dict]:
    if not RAW_PATH.exists():
        raise FileNotFoundError(RAW_PATH)

    if not PROTOCOL_PATH.exists():
        raise FileNotFoundError(PROTOCOL_PATH)

    raw_sha = sha256_file(RAW_PATH)

    if raw_sha != EXPECTED_RAW_SHA256:
        raise RuntimeError(
            "Frozen raw SHA256 mismatch.\n"
            f"Expected: {EXPECTED_RAW_SHA256}\n"
            f"Actual:   {raw_sha}\n"
            "Analysis aborted."
        )

    with PROTOCOL_PATH.open("r", encoding="utf-8") as f:
        protocol = json.load(f)

    if protocol.get("protocol_version") != EXPECTED_PROTOCOL_VERSION:
        raise RuntimeError(
            "Protocol-version mismatch: "
            f"{protocol.get('protocol_version')} != "
            f"{EXPECTED_PROTOCOL_VERSION}"
        )

    df = pd.read_csv(RAW_PATH)

    required = {
        "protocol_version",
        "function_id",
        "category",
        "dimension",
        "run_id",
        "noise_eta",
        "noise_seed",
        "noise_scale_Sf",
        "noise_sigma",
        "FE_budget",
        "fe_used",
        "best_noisy_f_reported",
        "true_final_f",
        "f_opt",
        "true_final_error",
        "n_accept",
        "n_reloc",
        "wall_time_sec",
        "status",
    }

    missing_columns = sorted(required - set(df.columns))
    if missing_columns:
        raise RuntimeError(
            f"Missing required raw columns: {missing_columns}"
        )

    problems: list[str] = []

    if len(df) != EXPECTED_ROWS:
        problems.append(
            f"row count {len(df)} != expected {EXPECTED_ROWS}"
        )

    if set(df["function_id"].astype(int)) != set(EXPECTED_FUNCTIONS):
        problems.append("function set mismatch")

    if set(df["run_id"].astype(int)) != set(EXPECTED_RUN_IDS):
        problems.append("run_id set mismatch")

    actual_eta = sorted(df["noise_eta"].astype(float).unique())
    expected_eta = sorted(EXPECTED_NOISE_LEVELS)

    if len(actual_eta) != len(expected_eta) or not np.allclose(
        actual_eta,
        expected_eta,
        rtol=0.0,
        atol=1e-15,
    ):
        problems.append(
            f"noise levels mismatch: {actual_eta}"
        )

    if set(df["dimension"].astype(int)) != {EXPECTED_DIMENSION}:
        problems.append("dimension mismatch")

    if set(df["FE_budget"].astype(int)) != {EXPECTED_BUDGET}:
        problems.append("FE budget mismatch")

    if not (df["status"].astype(str) == "ok").all():
        problems.append("non-ok status detected")

    if (df["fe_used"].astype(int) > df["FE_budget"].astype(int)).any():
        problems.append("FE budget violation detected")

    if not np.isfinite(df[PRIMARY_OUTCOME].astype(float)).all():
        problems.append("non-finite primary outcome detected")

    key_cols = [
        "function_id",
        "run_id",
        "noise_eta",
        "FE_budget",
    ]

    duplicate_count = int(df.duplicated(key_cols).sum())

    if duplicate_count != 0:
        problems.append(
            f"duplicate design keys detected: {duplicate_count}"
        )

    # Every function x eta cell must contain exactly five run_ids.
    cell_counts = (
        df.groupby(["function_id", "noise_eta"])
        .size()
        .reset_index(name="n")
    )

    bad_cells = cell_counts[cell_counts["n"] != len(EXPECTED_RUN_IDS)]

    if not bad_cells.empty:
        problems.append(
            "one or more function x noise cells do not contain five runs"
        )

    # Pairing check: same noise seed across eta for each function/run_id.
    seed_counts = (
        df.groupby(["function_id", "run_id"])["noise_seed"]
        .nunique()
    )

    if not (seed_counts == 1).all():
        problems.append(
            "noise-seed pairing mismatch across eta levels"
        )

    # Noise scale must remain fixed within each function.
    scale_counts = (
        df.groupby("function_id")["noise_scale_Sf"]
        .nunique()
    )

    if not (scale_counts == 1).all():
        problems.append(
            "noise scale is not fixed within one or more functions"
        )

    # sigma = eta * Sf.
    expected_sigma = (
        df["noise_eta"].astype(float)
        * df["noise_scale_Sf"].astype(float)
    )

    if not np.allclose(
        df["noise_sigma"].astype(float),
        expected_sigma,
        rtol=1e-12,
        atol=1e-12,
    ):
        problems.append("noise_sigma != noise_eta * noise_scale_Sf")

    # Protocol version must be constant.
    if set(df["protocol_version"].astype(str)) != {
        EXPECTED_PROTOCOL_VERSION
    }:
        problems.append("raw protocol version mismatch")

    audit = {
        "pass": len(problems) == 0,
        "problems": problems,
        "raw_path": str(RAW_PATH),
        "raw_sha256": raw_sha,
        "protocol_path": str(PROTOCOL_PATH),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "rows": int(len(df)),
        "expected_rows": EXPECTED_ROWS,
        "unique_design_keys": int(
            df[key_cols].drop_duplicates().shape[0]
        ),
        "duplicate_design_keys": duplicate_count,
        "functions": sorted(
            df["function_id"].astype(int).unique().tolist()
        ),
        "run_ids": sorted(
            df["run_id"].astype(int).unique().tolist()
        ),
        "noise_levels": sorted(
            df["noise_eta"].astype(float).unique().tolist()
        ),
        "dimension": sorted(
            df["dimension"].astype(int).unique().tolist()
        ),
        "FE_budget": sorted(
            df["FE_budget"].astype(int).unique().tolist()
        ),
        "primary_outcome": PRIMARY_OUTCOME,
    }

    write_json(
        AUDIT_DIR / "noise_analysis_input_audit.json",
        audit,
    )

    if problems:
        raise RuntimeError(
            "Input audit failed:\n- " + "\n- ".join(problems)
        )

    return df, protocol


# ---------------------------------------------------------------------
# Function-level aggregation
# ---------------------------------------------------------------------

def make_function_level_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Inferential unit = benchmark function.

    For each function and eta, aggregate the five deterministic run-indexed
    configurations using the median true final error.
    """
    grouped = (
        df.groupby(
            ["function_id", "category", "noise_eta"],
            as_index=False,
        )
        .agg(
            median_true_final_error=("true_final_error", "median"),
            mean_true_final_error=("true_final_error", "mean"),
            min_true_final_error=("true_final_error", "min"),
            max_true_final_error=("true_final_error", "max"),
            q25_true_final_error=(
                "true_final_error",
                lambda x: float(np.quantile(x, 0.25)),
            ),
            q75_true_final_error=(
                "true_final_error",
                lambda x: float(np.quantile(x, 0.75)),
            ),
            n_runs=("true_final_error", "size"),
            median_fe_used=("fe_used", "median"),
            median_n_accept=("n_accept", "median"),
            median_n_reloc=("n_reloc", "median"),
            median_wall_time_sec=("wall_time_sec", "median"),
        )
        .sort_values(["function_id", "noise_eta"])
        .reset_index(drop=True)
    )

    if not (grouped["n_runs"] == len(EXPECTED_RUN_IDS)).all():
        raise RuntimeError(
            "Function-level aggregation does not contain five runs per cell."
        )

    out = (
        PROCESSED_DIR
        / "01_function_level_noise_summary.csv"
    )
    grouped.to_csv(out, index=False)

    return grouped


# ---------------------------------------------------------------------
# Descriptive statistics
# ---------------------------------------------------------------------

def make_descriptive_summary(
    function_level: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for eta in EXPECTED_NOISE_LEVELS:
        x = (
            function_level.loc[
                np.isclose(
                    function_level["noise_eta"],
                    eta,
                    rtol=0.0,
                    atol=1e-15,
                ),
                "median_true_final_error",
            ]
            .astype(float)
            .to_numpy()
        )

        rows.append(
            {
                "noise_eta": eta,
                "n_functions": int(x.size),
                "median_function_error": float(np.median(x)),
                "mean_function_error": float(np.mean(x)),
                "q25_function_error": float(np.quantile(x, 0.25)),
                "q75_function_error": float(np.quantile(x, 0.75)),
                "min_function_error": float(np.min(x)),
                "max_function_error": float(np.max(x)),
            }
        )

    desc = pd.DataFrame(rows)

    desc.to_csv(
        STATISTICS_DIR / "02_noise_descriptive_statistics.csv",
        index=False,
    )

    return desc


# ---------------------------------------------------------------------
# Paired matrix
# ---------------------------------------------------------------------

def make_paired_matrix(
    function_level: pd.DataFrame,
) -> pd.DataFrame:

    pivot = function_level.pivot(
        index="function_id",
        columns="noise_eta",
        values="median_true_final_error",
    )

    pivot = pivot.reindex(
        index=EXPECTED_FUNCTIONS,
        columns=EXPECTED_NOISE_LEVELS,
    )

    if pivot.isna().any().any():
        raise RuntimeError(
            "Missing value in function-level paired matrix."
        )

    pivot.columns = [eta_label(float(c)) for c in pivot.columns]
    pivot = pivot.reset_index()

    pivot.to_csv(
        PROCESSED_DIR / "03_function_level_paired_matrix.csv",
        index=False,
    )

    return pivot


# ---------------------------------------------------------------------
# Friedman omnibus test
# ---------------------------------------------------------------------

def friedman_analysis(
    paired: pd.DataFrame,
) -> dict:

    arrays = [
        paired[eta_label(eta)].astype(float).to_numpy()
        for eta in EXPECTED_NOISE_LEVELS
    ]

    result = friedmanchisquare(*arrays)

    n = len(paired)
    k = len(arrays)

    # Kendall's W for Friedman repeated-measures design.
    kendall_w = float(result.statistic) / (n * (k - 1))

    out = {
        "test": "Friedman",
        "inferential_unit": "CEC2014 benchmark function",
        "n_functions": int(n),
        "k_conditions": int(k),
        "noise_levels": EXPECTED_NOISE_LEVELS,
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "kendall_W": float(kendall_w),
    }

    write_json(
        STATISTICS_DIR / "04_friedman_noise_test.json",
        out,
    )

    return out


# ---------------------------------------------------------------------
# Planned Wilcoxon comparisons
# ---------------------------------------------------------------------

def planned_comparisons(
    paired: pd.DataFrame,
) -> pd.DataFrame:

    control = paired[eta_label(0.0)].astype(float).to_numpy()

    rows = []

    noisy_levels = EXPECTED_NOISE_LEVELS[1:]

    for j, eta in enumerate(noisy_levels):
        noisy = paired[eta_label(eta)].astype(float).to_numpy()
        d = noisy - control

        # SciPy two-sided paired Wilcoxon.
        # zero_method='wilcox' removes exact zero differences.
        if np.all(d == 0.0):
            statistic = 0.0
            p_raw = 1.0
        else:
            wr = wilcoxon(
                noisy,
                control,
                zero_method="wilcox",
                correction=False,
                alternative="two-sided",
                method="auto",
            )
            statistic = float(wr.statistic)
            p_raw = float(wr.pvalue)

        rbc = paired_rank_biserial(noisy, control)

        med_diff = float(np.median(d))

        ci_low, ci_high = bootstrap_median_difference_ci(
            d,
            seed=BOOTSTRAP_SEED + j,
            reps=BOOTSTRAP_REPS,
        )

        close = np.isclose(
            noisy,
            control,
            rtol=WTL_RTOL,
            atol=WTL_ATOL,
        )

        # Lower error is better.
        wins = int(np.sum((noisy < control) & (~close)))
        losses = int(np.sum((noisy > control) & (~close)))
        ties = int(np.sum(close))

        rows.append(
            {
                "comparison": f"eta={eta:g} vs eta=0",
                "noise_eta": eta,
                "n_functions": int(len(d)),
                "wilcoxon_statistic": statistic,
                "p_raw": p_raw,
                "rank_biserial_noisy_minus_control": rbc,
                "median_paired_difference_noisy_minus_control": med_diff,
                "bootstrap95_median_diff_low": ci_low,
                "bootstrap95_median_diff_high": ci_high,
                "wins_noisy_better": wins,
                "ties": ties,
                "losses_noisy_worse": losses,
            }
        )

    adjusted = holm_adjust([r["p_raw"] for r in rows])

    for r, p_holm in zip(rows, adjusted):
        r["p_holm"] = float(p_holm)
        r["significant_holm_0.05"] = bool(p_holm < 0.05)

    result = pd.DataFrame(rows)

    result.to_csv(
        STATISTICS_DIR / "05_wilcoxon_holm_noise_vs_control.csv",
        index=False,
    )

    return result


# ---------------------------------------------------------------------
# Function-wise paired differences
# ---------------------------------------------------------------------

def function_wise_differences(
    paired: pd.DataFrame,
) -> pd.DataFrame:

    out = paired.copy()

    control = out[eta_label(0.0)].astype(float)

    for eta in EXPECTED_NOISE_LEVELS[1:]:
        col = eta_label(eta)
        out[f"diff_{col}_minus_control"] = (
            out[col].astype(float) - control
        )

        out[f"ratio_{col}_over_control"] = np.where(
            control != 0.0,
            out[col].astype(float) / control,
            np.nan,
        )

    out.to_csv(
        PROCESSED_DIR / "06_function_wise_noise_differences.csv",
        index=False,
    )

    return out


# ---------------------------------------------------------------------
# Secondary diagnostics
# ---------------------------------------------------------------------

def secondary_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Descriptive only.
    These are not the primary inferential outcomes.
    """
    result = (
        df.groupby("noise_eta", as_index=False)
        .agg(
            n_runs=("true_final_error", "size"),
            median_fe_used=("fe_used", "median"),
            min_fe_used=("fe_used", "min"),
            max_fe_used=("fe_used", "max"),
            median_n_accept=("n_accept", "median"),
            median_n_reloc=("n_reloc", "median"),
            median_wall_time_sec=("wall_time_sec", "median"),
            total_wall_time_sec=("wall_time_sec", "sum"),
        )
        .sort_values("noise_eta")
        .reset_index(drop=True)
    )

    result.to_csv(
        STATISTICS_DIR / "07_secondary_execution_diagnostics.csv",
        index=False,
    )

    return result


# ---------------------------------------------------------------------
# Machine-readable summary
# ---------------------------------------------------------------------

def build_summary(
    raw: pd.DataFrame,
    protocol: dict,
    desc: pd.DataFrame,
    friedman: dict,
    comparisons: pd.DataFrame,
) -> dict:

    summary = {
        "analysis_scope": (
            "Controlled observational-noise sensitivity of frozen DMGSO V4 "
            "on 12 representative CEC2014 functions at D=30."
        ),
        "primary_outcome": PRIMARY_OUTCOME,
        "aggregation": (
            "Median across five deterministic run-indexed configurations "
            "within each function and noise level."
        ),
        "inferential_unit": "benchmark function",
        "n_functions": len(EXPECTED_FUNCTIONS),
        "noise_levels": EXPECTED_NOISE_LEVELS,
        "raw_rows": int(len(raw)),
        "raw_sha256": EXPECTED_RAW_SHA256,
        "protocol_version": protocol.get("protocol_version"),
        "friedman": friedman,
        "descriptive": desc.to_dict(orient="records"),
        "planned_comparisons": comparisons.to_dict(orient="records"),
        "interpretation_guardrail": (
            "This experiment characterizes sensitivity to the specified "
            "additive Gaussian observational-noise model only. It does not "
            "establish universal robustness to arbitrary noise processes, "
            "additional constraints, expensive objectives, dynamic problems, "
            "or real engineering applications."
        ),
    }

    write_json(
        STATISTICS_DIR / "08_noise_analysis_summary.json",
        summary,
    )

    return summary


# ---------------------------------------------------------------------
# Output hashes / reproducibility manifest
# ---------------------------------------------------------------------

def make_manifest() -> dict:

    tracked_files = [
        PROCESSED_DIR / "01_function_level_noise_summary.csv",
        STATISTICS_DIR / "02_noise_descriptive_statistics.csv",
        PROCESSED_DIR / "03_function_level_paired_matrix.csv",
        STATISTICS_DIR / "04_friedman_noise_test.json",
        STATISTICS_DIR / "05_wilcoxon_holm_noise_vs_control.csv",
        PROCESSED_DIR / "06_function_wise_noise_differences.csv",
        STATISTICS_DIR / "07_secondary_execution_diagnostics.csv",
        STATISTICS_DIR / "08_noise_analysis_summary.json",
    ]

    outputs = {}

    for p in tracked_files:
        if not p.exists():
            raise FileNotFoundError(
                f"Expected analysis output missing: {p}"
            )
        outputs[str(p.relative_to(EXP_ROOT))] = sha256_file(p)

    script_path = Path(__file__).resolve()

    manifest = {
        "analysis_script": str(script_path),
        "analysis_script_sha256": sha256_file(script_path),
        "raw_frozen_path": str(RAW_PATH),
        "raw_frozen_sha256": sha256_file(RAW_PATH),
        "protocol_path": str(PROTOCOL_PATH),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_reps": BOOTSTRAP_REPS,
        "outputs": outputs,
    }

    write_json(
        AUDIT_DIR / "noise_analysis_manifest.json",
        manifest,
    )

    return manifest


# ---------------------------------------------------------------------
# Console report
# ---------------------------------------------------------------------

def print_report(
    desc: pd.DataFrame,
    friedman: dict,
    comparisons: pd.DataFrame,
) -> None:

    print()
    print("=" * 72)
    print("DMGSO NOISE-SENSITIVITY ANALYSIS")
    print("=" * 72)

    print(f"Frozen raw SHA256: {EXPECTED_RAW_SHA256}")
    print(f"Primary outcome:   {PRIMARY_OUTCOME}")
    print("Inferential unit:  benchmark function (n=12)")
    print()

    print("FUNCTION-LEVEL DESCRIPTIVE STATISTICS")
    print(
        desc.to_string(
            index=False,
            float_format=lambda x: f"{x:.6e}",
        )
    )

    print()
    print("FRIEDMAN TEST")
    print(
        f"chi2={friedman['statistic']:.6f}, "
        f"p={friedman['p_value']:.8g}, "
        f"Kendall_W={friedman['kendall_W']:.6f}"
    )

    print()
    print("PLANNED NOISE-vs-CONTROL COMPARISONS")
    cols = [
        "comparison",
        "wilcoxon_statistic",
        "p_raw",
        "p_holm",
        "rank_biserial_noisy_minus_control",
        "median_paired_difference_noisy_minus_control",
        "bootstrap95_median_diff_low",
        "bootstrap95_median_diff_high",
        "wins_noisy_better",
        "ties",
        "losses_noisy_worse",
    ]

    print(
        comparisons[cols].to_string(
            index=False,
            float_format=lambda x: f"{x:.6e}",
        )
    )

    print()
    print("Interpretation:")
    print(
        "Positive paired differences and positive rank-biserial values "
        "indicate larger true final error under noise (degradation)."
    )
    print(
        "Win/Tie/Loss is reported from the noisy condition's perspective; "
        "lower true final error is better."
    )
    print("=" * 72)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    ensure_dirs()

    print("[1/8] Auditing frozen inputs...")
    raw, protocol = audit_inputs()
    print("      INPUT AUDIT PASS")

    print("[2/8] Aggregating five run_ids at function level...")
    function_level = make_function_level_table(raw)

    print("[3/8] Computing descriptive statistics...")
    desc = make_descriptive_summary(function_level)

    print("[4/8] Constructing paired function-level matrix...")
    paired = make_paired_matrix(function_level)

    print("[5/8] Running Friedman omnibus test...")
    friedman = friedman_analysis(paired)

    print("[6/8] Running planned Wilcoxon/Holm comparisons...")
    comparisons = planned_comparisons(paired)

    print("[7/8] Writing paired differences and secondary diagnostics...")
    function_wise_differences(paired)
    secondary_diagnostics(raw)

    print("[8/8] Writing summary and reproducibility manifest...")
    build_summary(
        raw,
        protocol,
        desc,
        friedman,
        comparisons,
    )
    make_manifest()

    print_report(desc, friedman, comparisons)

    print()
    print("ANALYSIS PASS")
    print(
        f"Outputs written only below: {EXP_ROOT}"
    )


if __name__ == "__main__":
    main()
