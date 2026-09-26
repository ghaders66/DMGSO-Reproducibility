# -*- coding: utf-8 -*-
"""
Safe resume utility for the frozen CEC2014 external-comparison experiment.

Purpose
-------
Resume an interrupted run WITHOUT modifying the frozen experiment runner,
DMGSO core, wrapper, or baseline implementation.

Safety design
-------------
- Dry-run is the DEFAULT behavior.
- Existing completed records are never recomputed.
- Existing CSV is never deleted or overwritten.
- --execute is required before any new result is appended.
- A timestamped backup of the current CSV is created automatically before append.
- Existing schema, duplicates, status, FE budgets, finite values, and benchmark
  errors are validated before continuation.
- Missing combinations are appended in the same canonical order as the original
  frozen runner: function -> run_id -> algorithm.
- The script is restart-safe: if interrupted again, rerun it and already written
  combinations will be skipped.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Resolve official Revision paths from this helper's own location.
# Expected location:
# Revision/cec2014/code/resume_cec2014_external_revision.py
# ---------------------------------------------------------------------
CODE_DIR = Path(__file__).resolve().parent
CEC_DIR = CODE_DIR.parent
RAW_DIR = CEC_DIR / "raw"
BACKUP_DIR = RAW_DIR / "backups"

RUN_SUMMARY = RAW_DIR / "cec2014_run_summary_revision.csv"

sys.path.insert(0, str(CODE_DIR))

# Import the exact frozen implementation used by the original experiment.
from run_cec2014_all_algorithms_revision import (  # noqa: E402
    BASE_SEED,
    append_csv,
    run_dmgso,
    summarize_results,
    validate_result,
)
from wrapper_cec2014_revision import get_cec2014_problem  # noqa: E402
from baseline_algorithms_revision import run_baseline_algorithm  # noqa: E402


# ---------------------------------------------------------------------
# Frozen main-experiment protocol
# ---------------------------------------------------------------------
FUNCTIONS = list(range(1, 31))
DIMENSION = 30
RUNS = 30
BUDGET = 300_000
ALGORITHMS = ["DMGSO", "CMA-ES", "DE", "PSO"]

KEY_COLUMNS = ["algorithm", "function_id", "dimension", "run_id"]

REQUIRED_COLUMNS = [
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
]


def canonical_expected_keys() -> List[Tuple[str, int, int, int]]:
    """Return expected keys in the original frozen runner's execution order."""
    keys: List[Tuple[str, int, int, int]] = []
    for fid in FUNCTIONS:
        for run_id in range(RUNS):
            for alg in ALGORITHMS:
                keys.append((alg, fid, DIMENSION, run_id))
    return keys


def load_and_validate_existing(path: Path) -> pd.DataFrame:
    """Read and strictly validate the existing interrupted CSV."""
    if not path.exists():
        raise FileNotFoundError(
            f"Expected interrupted run CSV was not found:\n{path}\n"
            "Nothing has been changed."
        )

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise RuntimeError(
            f"Could not parse existing CSV safely:\n{path}\n{exc}\n"
            "Nothing has been changed."
        ) from exc

    missing_columns = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_columns:
        raise RuntimeError(
            "Existing CSV schema does not match the frozen runner.\n"
            f"Missing columns: {missing_columns}\n"
            "Nothing has been changed."
        )

    if df.empty:
        raise RuntimeError("Existing CSV is empty. Nothing has been changed.")

    # Normalize only for validation; do not rewrite existing data.
    alg = df["algorithm"].astype(str).str.upper()
    function_id = pd.to_numeric(df["function_id"], errors="coerce")
    dimension = pd.to_numeric(df["dimension"], errors="coerce")
    run_id = pd.to_numeric(df["run_id"], errors="coerce")
    fe_budget = pd.to_numeric(df["FE_budget_max"], errors="coerce")
    fe_used = pd.to_numeric(df["fe_used"], errors="coerce")
    best_f = pd.to_numeric(df["best_f"], errors="coerce")
    f_opt = pd.to_numeric(df["f_opt"], errors="coerce")
    error = pd.to_numeric(df["error"], errors="coerce")

    numeric_checks = {
        "function_id": function_id,
        "dimension": dimension,
        "run_id": run_id,
        "FE_budget_max": fe_budget,
        "fe_used": fe_used,
        "best_f": best_f,
        "f_opt": f_opt,
        "error": error,
    }
    bad_numeric = {
        name: int(series.isna().sum())
        for name, series in numeric_checks.items()
        if series.isna().any()
    }
    if bad_numeric:
        raise RuntimeError(
            f"Non-numeric/NaN values found in required fields: {bad_numeric}\n"
            "Nothing has been changed."
        )

    # Protocol checks.
    if set(alg.unique()) - set(ALGORITHMS):
        raise RuntimeError(
            f"Unexpected algorithms present: {sorted(set(alg.unique()) - set(ALGORITHMS))}"
        )
    if not set(function_id.astype(int).unique()).issubset(set(FUNCTIONS)):
        raise RuntimeError("Unexpected CEC2014 function_id found.")
    if set(dimension.astype(int).unique()) != {DIMENSION}:
        raise RuntimeError(
            f"Dimension mismatch. Expected only D={DIMENSION}; "
            f"found {sorted(set(dimension.astype(int).unique()))}."
        )
    if not set(run_id.astype(int).unique()).issubset(set(range(RUNS))):
        raise RuntimeError("Unexpected run_id found.")
    if set(fe_budget.astype(int).unique()) != {BUDGET}:
        raise RuntimeError(
            f"Budget mismatch. Expected only {BUDGET}; "
            f"found {sorted(set(fe_budget.astype(int).unique()))}."
        )

    if not (df["suite"].astype(str) == "CEC2014").all():
        raise RuntimeError("Unexpected suite value found.")
    if not (df["status"].astype(str).str.lower() == "ok").all():
        bad = df.loc[df["status"].astype(str).str.lower() != "ok", KEY_COLUMNS + ["status"]]
        raise RuntimeError(
            "Non-ok rows found in existing CSV:\n"
            + bad.to_string(index=False)
        )

    if (fe_used.astype(int) < 1).any():
        raise RuntimeError("Invalid FE count < 1 found.")
    if (fe_used.astype(int) > BUDGET).any():
        bad = df.loc[fe_used.astype(int) > BUDGET, KEY_COLUMNS + ["fe_used", "FE_budget_max"]]
        raise RuntimeError(
            "FE-budget violation found:\n" + bad.to_string(index=False)
        )

    if not np.isfinite(best_f.to_numpy(dtype=float)).all():
        raise RuntimeError("Non-finite best_f found.")
    if not np.isfinite(f_opt.to_numpy(dtype=float)).all():
        raise RuntimeError("Non-finite f_opt found.")
    if not np.isfinite(error.to_numpy(dtype=float)).all():
        raise RuntimeError("Non-finite error found.")

    # Existing stored error must be consistent with best_f - f_opt to numerical precision.
    recomputed_error = best_f.to_numpy(dtype=float) - f_opt.to_numpy(dtype=float)
    stored_error = error.to_numpy(dtype=float)
    if not np.allclose(stored_error, recomputed_error, rtol=1e-10, atol=1e-8):
        idx = np.where(~np.isclose(stored_error, recomputed_error, rtol=1e-10, atol=1e-8))[0][:10]
        raise RuntimeError(
            "Stored error is inconsistent with best_f - f_opt. "
            f"Example row indices: {idx.tolist()}"
        )

    if (recomputed_error < -1e-8).any():
        idx = np.where(recomputed_error < -1e-8)[0][:10]
        raise RuntimeError(
            "Materially negative benchmark error found. "
            f"Example row indices: {idx.tolist()}"
        )

    # Duplicate guard using the exact frozen-run identity.
    dup = pd.DataFrame(
        {
            "algorithm": alg,
            "function_id": function_id.astype(int),
            "dimension": dimension.astype(int),
            "run_id": run_id.astype(int),
        }
    ).duplicated(keep=False)

    if dup.any():
        duplicate_rows = df.loc[dup, KEY_COLUMNS].sort_values(KEY_COLUMNS)
        raise RuntimeError(
            "Duplicate algorithm/function/dimension/run records detected:\n"
            + duplicate_rows.to_string(index=False)
        )

    return df


def completed_key_set(df: pd.DataFrame) -> Set[Tuple[str, int, int, int]]:
    return {
        (
            str(row.algorithm).upper(),
            int(row.function_id),
            int(row.dimension),
            int(row.run_id),
        )
        for row in df[KEY_COLUMNS].itertuples(index=False)
    }


def print_audit(df: pd.DataFrame, missing: List[Tuple[str, int, int, int]]) -> None:
    expected_total = len(FUNCTIONS) * RUNS * len(ALGORITHMS)
    existing_total = len(df)

    print("=" * 72)
    print("CEC2014 SAFE-RESUME AUDIT")
    print("=" * 72)
    print(f"CSV:            {RUN_SUMMARY}")
    print(f"Existing rows:  {existing_total}")
    print(f"Expected final: {expected_total}")
    print(f"Missing rows:   {len(missing)}")
    print(f"Completion:     {100.0 * existing_total / expected_total:.2f}%")

    complete_functions = []
    for fid in FUNCTIONS:
        n = int((pd.to_numeric(df["function_id"]) == fid).sum())
        if n == RUNS * len(ALGORITHMS):
            complete_functions.append(fid)

    if complete_functions:
        print(
            "Fully complete functions: "
            + f"F{complete_functions[0]:02d}-F{complete_functions[-1]:02d}"
            if complete_functions == list(range(complete_functions[0], complete_functions[-1] + 1))
            else "Fully complete functions: " + ", ".join(f"F{x:02d}" for x in complete_functions)
        )
    else:
        print("Fully complete functions: none")

    if missing:
        first = missing[0]
        last = missing[-1]
        print(
            "First missing:  "
            f"F{first[1]:02d} / run_id={first[3]:02d} / {first[0]}"
        )
        print(
            "Last missing:   "
            f"F{last[1]:02d} / run_id={last[3]:02d} / {last[0]}"
        )
    else:
        print("First missing:  none")
        print("Dataset is already complete.")

    print("=" * 72)


def make_backup(path: Path) -> Path:
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = BACKUP_DIR / f"{path.stem}_pre_resume_{stamp}{path.suffix}"
    if backup.exists():
        raise FileExistsError(f"Backup path unexpectedly exists: {backup}")
    shutil.copy2(path, backup)
    return backup


def run_one_missing(
    alg_upper: str,
    fid: int,
    run_id: int,
) -> Dict:
    problem = get_cec2014_problem(fid, DIMENSION)
    f_opt = float(problem.optimum)
    baseline_seed = BASE_SEED + run_id

    if alg_upper == "DMGSO":
        result, wall = run_dmgso(
            f=problem.objective,
            lb=problem.lower_bound,
            ub=problem.upper_bound,
            dimension=DIMENSION,
            budget=BUDGET,
            run_id=run_id,
            trace=False,
            trace_every=10,
        )
        best_f = float(result.best_f)
        fe_used = int(result.fe_used)
        n_accept = int(result.n_accept)
        n_reloc = int(result.n_reloc)
        seed_value = ""
    else:
        baseline_result = run_baseline_algorithm(
            algorithm=alg_upper,
            f=problem.objective,
            lb=problem.lower_bound,
            ub=problem.upper_bound,
            budget=BUDGET,
            seed=baseline_seed,
        )
        best_f = float(baseline_result.best_f)
        fe_used = int(baseline_result.fe_used)
        wall = float(baseline_result.wall_time_sec)
        n_accept = np.nan
        n_reloc = np.nan
        seed_value = baseline_seed

    error = validate_result(
        algorithm=alg_upper,
        fid=fid,
        run_id=run_id,
        best_f=best_f,
        f_opt=f_opt,
        fe_used=fe_used,
        budget=BUDGET,
    )

    return {
        "suite": "CEC2014",
        "algorithm": alg_upper,
        "function_id": fid,
        "function_name": problem.name,
        "category": problem.category,
        "dimension": DIMENSION,
        "run_id": run_id,
        "seed": seed_value,
        "FE_budget_max": BUDGET,
        "fe_used": fe_used,
        "best_f": best_f,
        "f_opt": f_opt,
        "error": error,
        "n_accept": n_accept,
        "n_reloc": n_reloc,
        "wall_time_sec": wall,
        "status": "ok",
    }


def final_audit() -> None:
    df = load_and_validate_existing(RUN_SUMMARY)
    completed = completed_key_set(df)
    expected = canonical_expected_keys()
    missing = [key for key in expected if key not in completed]

    if len(df) != len(expected):
        raise RuntimeError(
            f"Final row count mismatch: {len(df)} != {len(expected)}"
        )
    if missing:
        raise RuntimeError(
            f"Final audit still finds {len(missing)} missing combinations. "
            f"First missing: {missing[0]}"
        )

    print()
    print("[PASS] FINAL RESUME AUDIT")
    print(f"[PASS] Unique complete rows: {len(df)} / {len(expected)}")
    print("[PASS] No duplicates")
    print("[PASS] All status=ok")
    print("[PASS] FE budgets valid")
    print("[PASS] Errors finite and non-negative within tolerance")

    summarize_results(RUN_SUMMARY)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Safely audit and resume the interrupted frozen CEC2014 "
            "external-comparison experiment."
        )
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help=(
            "Actually compute and append missing combinations. "
            "Without this flag the script performs a read-only dry-run."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = load_and_validate_existing(RUN_SUMMARY)
    completed = completed_key_set(df)
    expected = canonical_expected_keys()
    missing = [key for key in expected if key not in completed]

    # Strong set-integrity check: all existing rows must belong to expected protocol.
    unexpected = sorted(completed - set(expected))
    if unexpected:
        raise RuntimeError(
            "Existing CSV contains combinations outside the frozen protocol. "
            f"Examples: {unexpected[:10]}"
        )

    print_audit(df, missing)

    if not missing:
        print("[INFO] Nothing to resume.")
        final_audit()
        return

    if not args.execute:
        print("[DRY-RUN] No file was modified.")
        print("[DRY-RUN] If the audit above is correct, rerun with --execute.")
        return

    backup = make_backup(RUN_SUMMARY)
    print(f"[BACKUP] {backup}")
    print(f"[EXECUTE] Resuming {len(missing)} missing combinations...")

    # Re-read completed keys before every append session. If this script is ever
    # interrupted and restarted, the next invocation will naturally skip rows
    # already persisted by append_csv().
    completed_now = completed_key_set(load_and_validate_existing(RUN_SUMMARY))

    for alg_upper, fid, dimension, run_id in expected:
        key = (alg_upper, fid, dimension, run_id)
        if key in completed_now:
            continue

        row = run_one_missing(
            alg_upper=alg_upper,
            fid=fid,
            run_id=run_id,
        )

        # Last-moment duplicate guard against accidental external modification.
        current_df = pd.read_csv(RUN_SUMMARY, usecols=KEY_COLUMNS)
        current_keys = completed_key_set(current_df)
        if key in current_keys:
            raise RuntimeError(
                f"Refusing duplicate append; key appeared during resume: {key}"
            )

        append_csv(row, RUN_SUMMARY)
        completed_now.add(key)

        seed_text = (
            "deterministic"
            if alg_upper == "DMGSO"
            else str(BASE_SEED + run_id)
        )
        print(
            f"[OK] {alg_upper:7s} | "
            f"F{fid:02d} | D={DIMENSION} | "
            f"run_id={run_id:02d} | "
            f"seed={seed_text} | "
            f"error={float(row['error']):.6e} | "
            f"FE={int(row['fe_used'])}/{BUDGET}"
        )

    final_audit()
    print()
    print("CEC2014 interrupted experiment resumed successfully.")


if __name__ == "__main__":
    main()
