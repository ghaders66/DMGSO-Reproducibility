# -*- coding: utf-8 -*-
"""
CEC2014 revision runner for DMGSO, CMA-ES, DE, and PSO.

Revision protocol
-----------------
- CEC2014 dimension: configurable (main experiment D=30)
- Common maximum FE budget: configurable (main experiment 300,000)
- DMGSO: deterministic configurations indexed by run_id
- Stochastic baselines: explicit seed = 1000 + run_id
- Strict FE-budget validation
- Fail-fast validation for non-finite results and materially negative error
- Raw results preserved separately from processed summaries
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


# ============================================================
# Paths and local Revision imports
# ============================================================

CODE_DIR = Path(__file__).resolve().parent
CEC_DIR = CODE_DIR.parent
REVISION_ROOT = CEC_DIR.parent

sys.path.insert(0, str(CODE_DIR))

from dmgso_core_revision import DMGSOConfig, dmgso_optimize
from wrapper_cec2014_revision import get_cec2014_problem
from baseline_algorithms_revision import run_baseline_algorithm


RAW_DIR = CEC_DIR / "raw"
PROCESSED_DIR = CEC_DIR / "processed"
TRACE_DIR = RAW_DIR / "traces"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
TRACE_DIR.mkdir(parents=True, exist_ok=True)


ERROR_TOL = 1e-8
BASE_SEED = 1000


# ============================================================
# CSV utilities
# ============================================================

def append_csv(row: Dict, path: Path) -> None:
    exists = path.exists()

    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(row.keys()),
        )

        if not exists:
            writer.writeheader()

        writer.writerow(row)


def write_trace_csv(rows: List[dict], path: Path) -> None:
    if not rows:
        return

    keys = sorted(
        {key for row in rows for key in row.keys()}
    )

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=keys,
        )
        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# Validation
# ============================================================

def validate_result(
    algorithm: str,
    fid: int,
    run_id: int,
    best_f: float,
    f_opt: float,
    fe_used: int,
    budget: int,
) -> float:
    """
    Validate one completed optimization result.

    Returns
    -------
    error : float
        best_f - f_opt

    Raises
    ------
    RuntimeError
        If a non-finite result, FE-budget violation, or materially
        negative benchmark error is detected.
    """

    if not np.isfinite(best_f):
        raise RuntimeError(
            f"{algorithm} F{fid} run {run_id}: "
            f"non-finite best_f={best_f}"
        )

    if not np.isfinite(f_opt):
        raise RuntimeError(
            f"F{fid}: non-finite declared optimum={f_opt}"
        )

    if fe_used < 1:
        raise RuntimeError(
            f"{algorithm} F{fid} run {run_id}: "
            f"invalid FE count={fe_used}"
        )

    if fe_used > budget:
        raise RuntimeError(
            f"{algorithm} F{fid} run {run_id}: "
            f"FE budget exceeded ({fe_used} > {budget})"
        )

    error = float(best_f - f_opt)

    if error < -ERROR_TOL:
        raise RuntimeError(
            f"{algorithm} F{fid} run {run_id}: "
            f"best_f={best_f:.16e} is below "
            f"f_opt={f_opt:.16e}; error={error:.16e}. "
            "Benchmark pipeline stopped."
        )

    # Tiny negative round-off only.
    if -ERROR_TOL <= error < 0.0:
        error = 0.0

    return error


# ============================================================
# DMGSO
# ============================================================

def run_dmgso(
    f,
    lb: np.ndarray,
    ub: np.ndarray,
    dimension: int,
    budget: int,
    run_id: int,
    trace: bool,
    trace_every: int,
):
    cfg = DMGSOConfig(
        version="V4",
        D=dimension,
        FE_budget=budget,
        trace_enable=trace,
        trace_every=trace_every,
        refine_enable=False,
    )

    t0 = time.perf_counter()

    result = dmgso_optimize(
        f=f,
        lb=lb,
        ub=ub,
        cfg=cfg,
        run_id=run_id,
    )

    wall = time.perf_counter() - t0

    return result, float(wall)


# ============================================================
# Descriptive function-level summary
# ============================================================

def summarize_results(run_summary_path: Path) -> None:
    df = pd.read_csv(run_summary_path)

    expected_unique = [
        "algorithm",
        "function_id",
        "dimension",
        "run_id",
    ]

    duplicates = df.duplicated(
        subset=expected_unique,
        keep=False,
    )

    if duplicates.any():
        duplicate_rows = df.loc[
            duplicates,
            expected_unique,
        ]

        raise RuntimeError(
            "Duplicate algorithm/function/run records detected:\n"
            + duplicate_rows.to_string(index=False)
        )

    function_summary = (
        df.groupby(
            [
                "suite",
                "algorithm",
                "function_id",
                "category",
                "dimension",
            ],
            as_index=False,
        )
        .agg(
            configurations_or_runs=("run_id", "count"),
            error_mean=("error", "mean"),
            error_median=("error", "median"),
            error_std=("error", "std"),
            error_min=("error", "min"),
            error_max=("error", "max"),
            best_f_mean=("best_f", "mean"),
            best_f_median=("best_f", "median"),
            fe_used_mean=("fe_used", "mean"),
            fe_used_median=("fe_used", "median"),
            wall_time_mean=("wall_time_sec", "mean"),
            wall_time_median=("wall_time_sec", "median"),
        )
    )

    output_path = (
        PROCESSED_DIR
        / "cec2014_function_summary.csv"
    )

    function_summary.to_csv(
        output_path,
        index=False,
    )

    print(
        f"[OK] Function-level descriptive summary: "
        f"{output_path}"
    )


# ============================================================
# Main experiment
# ============================================================

def run_all_algorithms(
    functions: List[int],
    dimension: int,
    runs: int,
    budget: int,
    algorithms: List[str],
    trace: bool,
    trace_every: int,
    overwrite: bool,
) -> None:

    run_summary_path = (
        RAW_DIR
        / "cec2014_run_summary_revision.csv"
    )

    if run_summary_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists:\n{run_summary_path}\n"
            "Use --overwrite only when intentionally starting "
            "a new experiment."
        )

    if overwrite and run_summary_path.exists():
        run_summary_path.unlink()

    for fid in functions:

        problem = get_cec2014_problem(
            fid,
            dimension,
        )

        f_opt = float(problem.optimum)

        for run_id in range(runs):

            baseline_seed = BASE_SEED + run_id

            for alg in algorithms:

                alg_upper = alg.upper()

                if alg_upper == "DMGSO":

                    result, wall = run_dmgso(
                        f=problem.objective,
                        lb=problem.lower_bound,
                        ub=problem.upper_bound,
                        dimension=dimension,
                        budget=budget,
                        run_id=run_id,
                        trace=trace,
                        trace_every=trace_every,
                    )

                    best_f = float(result.best_f)
                    fe_used = int(result.fe_used)

                    n_accept = int(result.n_accept)
                    n_reloc = int(result.n_reloc)

                    seed_value = ""

                    if trace and result.trace_rows:
                        trace_path = (
                            TRACE_DIR
                            / (
                                f"cec2014_DMGSO_"
                                f"f{fid:02d}_D{dimension}_"
                                f"run{run_id:02d}_trace.csv"
                            )
                        )

                        write_trace_csv(
                            result.trace_rows,
                            trace_path,
                        )

                else:

                    if alg_upper not in {
                        "CMA-ES",
                        "DE",
                        "PSO",
                    }:
                        raise ValueError(
                            f"Unsupported algorithm: {alg}"
                        )

                    baseline_result = (
                        run_baseline_algorithm(
                            algorithm=alg_upper,
                            f=problem.objective,
                            lb=problem.lower_bound,
                            ub=problem.upper_bound,
                            budget=budget,
                            seed=baseline_seed,
                        )
                    )

                    best_f = float(
                        baseline_result.best_f
                    )

                    fe_used = int(
                        baseline_result.fe_used
                    )

                    wall = float(
                        baseline_result.wall_time_sec
                    )

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
                    budget=budget,
                )

                row = {
                    "suite": "CEC2014",
                    "algorithm": alg_upper,
                    "function_id": fid,
                    "function_name": problem.name,
                    "category": problem.category,
                    "dimension": dimension,
                    "run_id": run_id,
                    "seed": seed_value,
                    "FE_budget_max": budget,
                    "fe_used": fe_used,
                    "best_f": best_f,
                    "f_opt": f_opt,
                    "error": error,
                    "n_accept": n_accept,
                    "n_reloc": n_reloc,
                    "wall_time_sec": wall,
                    "status": "ok",
                }

                append_csv(
                    row,
                    run_summary_path,
                )

                seed_text = (
                    "deterministic"
                    if alg_upper == "DMGSO"
                    else str(baseline_seed)
                )

                print(
                    f"[OK] {alg_upper:7s} | "
                    f"F{fid:02d} | D={dimension} | "
                    f"run_id={run_id:02d} | "
                    f"seed={seed_text} | "
                    f"error={error:.6e} | "
                    f"FE={fe_used}/{budget}"
                )

    summarize_results(
        run_summary_path
    )

    print()
    print(
        "CEC2014 revision evaluation finished."
    )
    print(
        f"Raw run-level results: {run_summary_path}"
    )


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Run the controlled DMGSO revision "
            "experiment on CEC2014."
        )
    )

    parser.add_argument(
        "--functions",
        nargs="+",
        type=int,
        default=list(range(1, 31)),
    )

    parser.add_argument(
        "--dimension",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--budget",
        type=int,
        default=300000,
    )

    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=[
            "DMGSO",
            "CMA-ES",
            "DE",
            "PSO",
        ],
    )

    parser.add_argument(
        "--trace",
        action="store_true",
    )

    parser.add_argument(
        "--trace-every",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    run_all_algorithms(
        functions=args.functions,
        dimension=args.dimension,
        runs=args.runs,
        budget=args.budget,
        algorithms=args.algorithms,
        trace=args.trace,
        trace_every=args.trace_every,
        overwrite=args.overwrite,
    )
