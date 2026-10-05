#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Exploratory mechanism diagnostics for the frozen DMGSO noise experiment.

IMPORTANT
---------
- This script is strictly read-only with respect to the frozen experiment.
- It does NOT modify the frozen raw dataset.
- It does NOT modify the confirmatory statistical-analysis outputs.
- It writes ONLY below:
      noise_sensitivity/diagnostics/
- Results are exploratory/diagnostic and are NOT causal evidence.

Questions examined:
1. Does observational noise alter the number of accepted moves?
2. Does observational noise alter the number of relocations?
3. Are changes in these internal search outcomes associated with changes
   in true final error?
4. Which functions show the strongest simultaneous changes?
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
from scipy.stats import spearmanr


# ---------------------------------------------------------------------
# Paths and frozen definitions
# ---------------------------------------------------------------------

EXP_ROOT = Path(
    "/mnt/d/PHD/DMGSO_Information Scince/Revision/noise_sensitivity"
)

RAW_PATH = (
    EXP_ROOT
    / "raw"
    / "noise_sensitivity_cec2014_run_summary_FROZEN.csv"
)

DIAG_ROOT = EXP_ROOT / "diagnostics"
PROCESSED_DIR = DIAG_ROOT / "processed"
STATISTICS_DIR = DIAG_ROOT / "statistics"
AUDIT_DIR = DIAG_ROOT / "audit"

EXPECTED_RAW_SHA256 = (
    "26bf67a1e92cb32106a40fb7eca73c7ca8430a63907ca6e585c365b3b6c3193b"
)

EXPECTED_FUNCTIONS = [1, 2, 3, 4, 8, 12, 17, 20, 22, 23, 26, 30]
EXPECTED_RUN_IDS = [0, 1, 2, 3, 4]
EXPECTED_NOISE_LEVELS = [0.0, 1e-4, 1e-3, 1e-2]
EXPECTED_ROWS = 240

OUTCOMES = [
    "true_final_error",
    "n_accept",
    "n_reloc",
]


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def ensure_dirs():
    for p in (PROCESSED_DIR, STATISTICS_DIR, AUDIT_DIR):
        p.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj):
    path.write_text(
        json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def eta_label(eta: float) -> str:
    if eta == 0.0:
        return "eta_0"
    return f"eta_{eta:.0e}".replace("+", "")


# ---------------------------------------------------------------------
# Input audit
# ---------------------------------------------------------------------

def load_and_audit() -> pd.DataFrame:
    if not RAW_PATH.exists():
        raise FileNotFoundError(RAW_PATH)

    actual_sha = sha256_file(RAW_PATH)

    if actual_sha != EXPECTED_RAW_SHA256:
        raise RuntimeError(
            "Frozen raw SHA mismatch.\n"
            f"Expected: {EXPECTED_RAW_SHA256}\n"
            f"Actual:   {actual_sha}\n"
            "Diagnostic analysis aborted."
        )

    df = pd.read_csv(RAW_PATH)

    required = {
        "function_id",
        "run_id",
        "noise_eta",
        "true_final_error",
        "n_accept",
        "n_reloc",
        "status",
    }

    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing columns: {sorted(missing)}")

    problems = []

    if len(df) != EXPECTED_ROWS:
        problems.append(f"Expected 240 rows, found {len(df)}")

    if set(df["function_id"].astype(int)) != set(EXPECTED_FUNCTIONS):
        problems.append("Function set mismatch")

    if set(df["run_id"].astype(int)) != set(EXPECTED_RUN_IDS):
        problems.append("Run-ID set mismatch")

    actual_eta = sorted(df["noise_eta"].astype(float).unique())
    if not np.allclose(
        actual_eta,
        EXPECTED_NOISE_LEVELS,
        rtol=0.0,
        atol=1e-15,
    ):
        problems.append(f"Noise-level mismatch: {actual_eta}")

    if not (df["status"].astype(str) == "ok").all():
        problems.append("Non-ok status detected")

    if df.duplicated(
        ["function_id", "run_id", "noise_eta"]
    ).any():
        problems.append("Duplicate function/run/noise key detected")

    for col in OUTCOMES:
        if not np.isfinite(df[col].astype(float)).all():
            problems.append(f"Non-finite values in {col}")

    audit = {
        "pass": not problems,
        "problems": problems,
        "raw_path": str(RAW_PATH),
        "raw_sha256": actual_sha,
        "rows": int(len(df)),
        "analysis_type": (
            "exploratory mechanism diagnostic; not causal inference"
        ),
        "confirmatory_outputs_modified": False,
    }

    write_json(
        AUDIT_DIR / "diagnostic_input_audit.json",
        audit,
    )

    if problems:
        raise RuntimeError(
            "Diagnostic input audit failed:\n- "
            + "\n- ".join(problems)
        )

    return df


# ---------------------------------------------------------------------
# Function-level medians
# ---------------------------------------------------------------------

def function_level_summary(df: pd.DataFrame) -> pd.DataFrame:

    g = (
        df.groupby(
            ["function_id", "category", "noise_eta"],
            as_index=False,
        )
        .agg(
            median_true_error=("true_final_error", "median"),
            median_n_accept=("n_accept", "median"),
            median_n_reloc=("n_reloc", "median"),
            mean_n_accept=("n_accept", "mean"),
            mean_n_reloc=("n_reloc", "mean"),
            n_runs=("run_id", "size"),
        )
        .sort_values(["function_id", "noise_eta"])
        .reset_index(drop=True)
    )

    if not (g["n_runs"] == 5).all():
        raise RuntimeError(
            "Expected five runs per function/noise cell."
        )

    g.to_csv(
        PROCESSED_DIR / "01_function_level_mechanism_summary.csv",
        index=False,
    )

    return g


# ---------------------------------------------------------------------
# Paired changes relative to noiseless control
# ---------------------------------------------------------------------

def paired_changes(g: pd.DataFrame) -> pd.DataFrame:

    rows = []

    for fid in EXPECTED_FUNCTIONS:

        sub = g[g["function_id"] == fid].copy()

        control = sub[
            np.isclose(
                sub["noise_eta"],
                0.0,
                rtol=0.0,
                atol=1e-15,
            )
        ]

        if len(control) != 1:
            raise RuntimeError(
                f"Control row missing/duplicated for F{fid}"
            )

        c = control.iloc[0]

        for eta in EXPECTED_NOISE_LEVELS[1:]:

            noisy = sub[
                np.isclose(
                    sub["noise_eta"],
                    eta,
                    rtol=0.0,
                    atol=1e-15,
                )
            ]

            if len(noisy) != 1:
                raise RuntimeError(
                    f"Noisy row missing/duplicated for F{fid}, eta={eta}"
                )

            n = noisy.iloc[0]

            rows.append(
                {
                    "function_id": fid,
                    "category": n["category"],
                    "noise_eta": eta,

                    "control_true_error":
                        float(c["median_true_error"]),
                    "noisy_true_error":
                        float(n["median_true_error"]),
                    "delta_true_error":
                        float(
                            n["median_true_error"]
                            - c["median_true_error"]
                        ),

                    "control_n_accept":
                        float(c["median_n_accept"]),
                    "noisy_n_accept":
                        float(n["median_n_accept"]),
                    "delta_n_accept":
                        float(
                            n["median_n_accept"]
                            - c["median_n_accept"]
                        ),

                    "control_n_reloc":
                        float(c["median_n_reloc"]),
                    "noisy_n_reloc":
                        float(n["median_n_reloc"]),
                    "delta_n_reloc":
                        float(
                            n["median_n_reloc"]
                            - c["median_n_reloc"]
                        ),
                }
            )

    out = pd.DataFrame(rows)

    out.to_csv(
        PROCESSED_DIR / "02_paired_mechanism_changes.csv",
        index=False,
    )

    return out


# ---------------------------------------------------------------------
# Noise-level descriptive mechanism changes
# ---------------------------------------------------------------------

def level_summary(changes: pd.DataFrame) -> pd.DataFrame:

    rows = []

    for eta in EXPECTED_NOISE_LEVELS[1:]:

        s = changes[
            np.isclose(
                changes["noise_eta"],
                eta,
                rtol=0.0,
                atol=1e-15,
            )
        ]

        rows.append(
            {
                "noise_eta": eta,
                "n_functions": len(s),

                "median_delta_true_error":
                    float(np.median(s["delta_true_error"])),

                "median_delta_n_accept":
                    float(np.median(s["delta_n_accept"])),

                "q25_delta_n_accept":
                    float(np.quantile(s["delta_n_accept"], 0.25)),

                "q75_delta_n_accept":
                    float(np.quantile(s["delta_n_accept"], 0.75)),

                "functions_accept_increased":
                    int(np.sum(s["delta_n_accept"] > 0)),

                "functions_accept_unchanged":
                    int(np.sum(s["delta_n_accept"] == 0)),

                "functions_accept_decreased":
                    int(np.sum(s["delta_n_accept"] < 0)),

                "median_delta_n_reloc":
                    float(np.median(s["delta_n_reloc"])),

                "q25_delta_n_reloc":
                    float(np.quantile(s["delta_n_reloc"], 0.25)),

                "q75_delta_n_reloc":
                    float(np.quantile(s["delta_n_reloc"], 0.75)),

                "functions_reloc_increased":
                    int(np.sum(s["delta_n_reloc"] > 0)),

                "functions_reloc_unchanged":
                    int(np.sum(s["delta_n_reloc"] == 0)),

                "functions_reloc_decreased":
                    int(np.sum(s["delta_n_reloc"] < 0)),
            }
        )

    out = pd.DataFrame(rows)

    out.to_csv(
        STATISTICS_DIR / "03_mechanism_change_summary.csv",
        index=False,
    )

    return out


# ---------------------------------------------------------------------
# Exploratory associations
# ---------------------------------------------------------------------

def safe_spearman(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if np.all(x == x[0]) or np.all(y == y[0]):
        return np.nan, np.nan

    result = spearmanr(x, y)

    return float(result.statistic), float(result.pvalue)


def association_analysis(
    changes: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for eta in EXPECTED_NOISE_LEVELS[1:]:

        s = changes[
            np.isclose(
                changes["noise_eta"],
                eta,
                rtol=0.0,
                atol=1e-15,
            )
        ].copy()

        rho_a, p_a = safe_spearman(
            s["delta_n_accept"],
            s["delta_true_error"],
        )

        rho_r, p_r = safe_spearman(
            s["delta_n_reloc"],
            s["delta_true_error"],
        )

        rows.append(
            {
                "noise_eta": eta,
                "association":
                    "delta_n_accept vs delta_true_error",
                "n_functions": len(s),
                "spearman_rho": rho_a,
                "p_value_exploratory": p_a,
            }
        )

        rows.append(
            {
                "noise_eta": eta,
                "association":
                    "delta_n_reloc vs delta_true_error",
                "n_functions": len(s),
                "spearman_rho": rho_r,
                "p_value_exploratory": p_r,
            }
        )

    # Pooled exploratory view across all 36 function/noise contrasts.
    # This is descriptive only because each function contributes repeatedly.
    rho_a, p_a = safe_spearman(
        changes["delta_n_accept"],
        changes["delta_true_error"],
    )

    rho_r, p_r = safe_spearman(
        changes["delta_n_reloc"],
        changes["delta_true_error"],
    )

    rows.append(
        {
            "noise_eta": "pooled_descriptive",
            "association":
                "delta_n_accept vs delta_true_error",
            "n_functions": len(changes),
            "spearman_rho": rho_a,
            "p_value_exploratory": p_a,
        }
    )

    rows.append(
        {
            "noise_eta": "pooled_descriptive",
            "association":
                "delta_n_reloc vs delta_true_error",
            "n_functions": len(changes),
            "spearman_rho": rho_r,
            "p_value_exploratory": p_r,
        }
    )

    out = pd.DataFrame(rows)

    out.to_csv(
        STATISTICS_DIR / "04_exploratory_mechanism_associations.csv",
        index=False,
    )

    return out


# ---------------------------------------------------------------------
# Strongest-degradation cases
# ---------------------------------------------------------------------

def strongest_cases(changes: pd.DataFrame) -> pd.DataFrame:

    out = changes.sort_values(
        "delta_true_error",
        ascending=False,
    ).reset_index(drop=True)

    out.insert(0, "degradation_rank", np.arange(1, len(out) + 1))

    out.to_csv(
        PROCESSED_DIR / "05_ranked_degradation_cases.csv",
        index=False,
    )

    return out


# ---------------------------------------------------------------------
# Run-level paired diagnostic
# ---------------------------------------------------------------------

def run_level_changes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Exact paired contrasts at function x run_id level.

    Useful because the same deterministic run_id is used across eta levels.
    This remains exploratory and is not used to replace the confirmatory
    function-level inference.
    """

    control = (
        df[np.isclose(df["noise_eta"], 0.0)]
        [
            [
                "function_id",
                "run_id",
                "true_final_error",
                "n_accept",
                "n_reloc",
            ]
        ]
        .rename(
            columns={
                "true_final_error": "control_true_error",
                "n_accept": "control_n_accept",
                "n_reloc": "control_n_reloc",
            }
        )
    )

    noisy = df[~np.isclose(df["noise_eta"], 0.0)].copy()

    merged = noisy.merge(
        control,
        on=["function_id", "run_id"],
        how="left",
        validate="many_to_one",
    )

    merged["delta_true_error"] = (
        merged["true_final_error"] - merged["control_true_error"]
    )

    merged["delta_n_accept"] = (
        merged["n_accept"] - merged["control_n_accept"]
    )

    merged["delta_n_reloc"] = (
        merged["n_reloc"] - merged["control_n_reloc"]
    )

    cols = [
        "function_id",
        "category",
        "run_id",
        "noise_eta",
        "control_true_error",
        "true_final_error",
        "delta_true_error",
        "control_n_accept",
        "n_accept",
        "delta_n_accept",
        "control_n_reloc",
        "n_reloc",
        "delta_n_reloc",
    ]

    merged = merged[cols].sort_values(
        ["function_id", "run_id", "noise_eta"]
    )

    merged.to_csv(
        PROCESSED_DIR / "06_run_level_paired_diagnostics.csv",
        index=False,
    )

    return merged


# ---------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------

def write_manifest():
    files = sorted(
        list(PROCESSED_DIR.glob("*"))
        + list(STATISTICS_DIR.glob("*"))
    )

    outputs = {
        str(p.relative_to(DIAG_ROOT)): sha256_file(p)
        for p in files
        if p.is_file()
    }

    script = Path(__file__).resolve()

    manifest = {
        "analysis_script": str(script),
        "analysis_script_sha256": sha256_file(script),
        "raw_frozen_path": str(RAW_PATH),
        "raw_frozen_sha256": sha256_file(RAW_PATH),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "analysis_type": (
            "exploratory mechanism diagnostic"
        ),
        "causal_claims_permitted": False,
        "confirmatory_analysis_modified": False,
        "outputs": outputs,
    }

    write_json(
        AUDIT_DIR / "diagnostic_manifest.json",
        manifest,
    )


# ---------------------------------------------------------------------
# Console
# ---------------------------------------------------------------------

def print_report(
    levels: pd.DataFrame,
    associations: pd.DataFrame,
    ranked: pd.DataFrame,
):
    print()
    print("=" * 78)
    print("DMGSO NOISE MECHANISM DIAGNOSTICS")
    print("=" * 78)
    print(
        "Exploratory only — these associations do not establish causality."
    )
    print()

    print("MEDIAN CHANGES RELATIVE TO NOISELESS CONTROL")
    print(
        levels.to_string(
            index=False,
            float_format=lambda x: f"{x:.6e}",
        )
    )

    print()
    print("EXPLORATORY SPEARMAN ASSOCIATIONS")
    print(
        associations.to_string(
            index=False,
            float_format=lambda x: f"{x:.6e}",
        )
    )

    print()
    print("TOP 10 LARGEST TRUE-ERROR DEGRADATION CASES")
    cols = [
        "degradation_rank",
        "function_id",
        "noise_eta",
        "delta_true_error",
        "delta_n_accept",
        "delta_n_reloc",
    ]

    print(
        ranked[cols].head(10).to_string(
            index=False,
            float_format=lambda x: f"{x:.6e}",
        )
    )

    print()
    print(
        "Interpretation guardrail: changes in acceptance or relocation "
        "are diagnostic associations only and must not be interpreted "
        "as demonstrated causes of noise-induced degradation."
    )
    print("=" * 78)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    ensure_dirs()

    print("[1/7] Auditing frozen raw input...")
    df = load_and_audit()
    print("      DIAGNOSTIC INPUT AUDIT PASS")

    print("[2/7] Computing function-level mechanism summaries...")
    g = function_level_summary(df)

    print("[3/7] Computing paired changes relative to control...")
    changes = paired_changes(g)

    print("[4/7] Summarizing acceptance/relocation changes...")
    levels = level_summary(changes)

    print("[5/7] Computing exploratory associations...")
    associations = association_analysis(changes)

    print("[6/7] Ranking degradation cases and run-level contrasts...")
    ranked = strongest_cases(changes)
    run_level_changes(df)

    print("[7/7] Writing diagnostic manifest...")
    write_manifest()

    print_report(levels, associations, ranked)

    print()
    print("DIAGNOSTIC ANALYSIS PASS")
    print(f"Outputs written only below: {DIAG_ROOT}")


if __name__ == "__main__":
    main()
