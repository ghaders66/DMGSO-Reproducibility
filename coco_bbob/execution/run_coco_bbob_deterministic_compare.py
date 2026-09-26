# -*- coding: utf-8 -*-
"""
COCO/BBOB deterministic comparison runner.

Algorithms:
    DMGSO
    NELDER-MEAD
    POWELL
    COBYLA
    COMPASS

Recommended main paper setting:
    D = 2, 5, 10, 20, 40
    FE_budget = 1000 * D
    BBOB functions = 1--24

Quick test:
    python runners/run_coco_bbob_deterministic_compare.py --dims 2 --budget-mult 200 --max-functions 3 --runs 1 --overwrite

Main run:
    python runners/run_coco_bbob_deterministic_compare.py --dims 2 5 10 20 --budget-mult 1000 --max-functions 24 --runs 5 --overwrite
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Dict, List, Callable

import numpy as np
import pandas as pd
import re 

try:
    import cocoex
    print("[INFO] cocoex successfully loaded")
except ImportError as exc:
    raise ImportError(
        "cocoex is not installed. Install it with:\n"
        "    python -m pip install coco-experiment"
    ) from exc

try:
    from scipy.optimize import minimize
except ImportError as exc:
    raise ImportError(
        "scipy is required for Nelder-Mead, Powell, and COBYLA.\n"
        "Install it with:\n"
        "    python -m pip install scipy"
    ) from exc


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.dmgso_core import DMGSOConfig, dmgso_optimize


OUT_TABLES = ROOT / "outputs" / "summaries" / "tables"
OUT_TRACES = ROOT / "outputs" / "traces" / "csv"

OUT_TABLES.mkdir(parents=True, exist_ok=True)
OUT_TRACES.mkdir(parents=True, exist_ok=True)


class EvalCounter:
    def __init__(self, f: Callable[[np.ndarray], float], budget: int):
        self.f = f
        self.budget = int(budget)
        self.n_eval = 0
        self.best_f = float("inf")
        self.best_x = None

    def __call__(self, x: np.ndarray) -> float:
        if self.n_eval >= self.budget:
            return self.best_f

        x = np.asarray(x, dtype=float)
        val = float(self.f(x))

        self.n_eval += 1

        if np.isfinite(val) and val < self.best_f:
            self.best_f = val
            self.best_x = x.copy()

        return val


def append_csv(row: Dict, path: Path) -> None:
    exists = path.exists()

    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def write_trace_csv(rows: List[dict], path: Path) -> None:
    if not rows:
        return

    keys = sorted({k for row in rows for k in row.keys()})

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def get_problem_metadata(problem) -> Dict:
    """
    Extract BBOB metadata robustly from cocoex problem object.

    Typical problem.id:
        bbob_f001_i01_d02
    """

    dim = int(problem.dimension)

    pid = str(getattr(problem, "id", ""))

    fid = getattr(problem, "number", None)
    iid = getattr(problem, "instance", None)

    if fid is None:
        m = re.search(r"f(\d+)", pid)
        if m:
            fid = int(m.group(1))

    if iid is None:
        m = re.search(r"i(\d+)", pid)
        if m:
            iid = int(m.group(1))

    if fid is None:
        raise ValueError(f"Could not infer function_id from problem.id={pid}")

    if iid is None:
        iid = -1

    try:
        f_opt = float(problem.optimal_value)
    except Exception:
        f_opt = 0.0

    return {
        "function_id": int(fid),
        "instance_id": int(iid),
        "dimension": dim,
        "function_name": pid if pid else f"bbob_f{int(fid):03d}_d{dim:02d}",
        "f_opt": f_opt,
    }


def make_objective(problem):
    def objective(x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        val = float(problem(x))
        if not np.isfinite(val):
            return float("inf")
        return val

    return objective


def deterministic_x0(lb: np.ndarray, ub: np.ndarray, run_id: int) -> np.ndarray:
    """
    Deterministic initial point.

    run_id=0: center
    run_id>0: deterministic low-discrepancy-like sinusoidal perturbation
    """
    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)

    center = 0.5 * (lb + ub)

    if run_id == 0:
        return center

    D = len(lb)
    idx = np.arange(1, D + 1)
    phase = np.sin((run_id + 1) * idx * 1.61803398875)
    x = center + 0.35 * (ub - lb) * phase

    return np.clip(x, lb, ub)


def run_dmgso(problem, budget: int, run_id: int, trace: bool, trace_every: int):
    dim = int(problem.dimension)
    lb = np.asarray(problem.lower_bounds, dtype=float)
    ub = np.asarray(problem.upper_bounds, dtype=float)

    cfg = DMGSOConfig(
        version="V4",
        D=dim,
        FE_budget=budget,
        trace_enable=trace,
        trace_every=trace_every,
        refine_enable=False,
    )

    t0 = time.time()

    result = dmgso_optimize(
        f=make_objective(problem),
        lb=lb,
        ub=ub,
        cfg=cfg,
        run_id=run_id,
    )

    wall = time.time() - t0

    return {
        "best_f": float(result.best_f),
        "fe_used": int(result.fe_used),
        "n_accept": int(getattr(result, "n_accept", 0)),
        "n_reloc": int(getattr(result, "n_reloc", 0)),
        "wall_time_sec": wall,
        "trace_rows": getattr(result, "trace_rows", []),
    }


def run_scipy_method(problem, method: str, budget: int, run_id: int):
    dim = int(problem.dimension)
    lb = np.asarray(problem.lower_bounds, dtype=float)
    ub = np.asarray(problem.upper_bounds, dtype=float)

    counter = EvalCounter(make_objective(problem), budget)
    x0 = deterministic_x0(lb, ub, run_id)

    bounds = list(zip(lb, ub))

    options = {
        "maxfev": budget,
        "maxiter": budget,
        "disp": False,
    }

    if method.upper() == "POWELL":
        options = {
            "maxfev": budget,
            "maxiter": budget,
            "disp": False,
        }

        scipy_method = "Powell"

        kwargs = {
            "bounds": bounds,
            "options": options,
        }

    elif method.upper() == "NELDER-MEAD":
        options = {
            "maxfev": budget,
            "maxiter": budget,
            "disp": False,
            "adaptive": True,
        }

        scipy_method = "Nelder-Mead"

        kwargs = {
            "options": options,
        }

    elif method.upper() == "COBYLA":
        scipy_method = "COBYLA"

        constraints = []

        for i in range(dim):
            constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda x, i=i: x[i] - lb[i],
                }
            )
            constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda x, i=i: ub[i] - x[i],
                }
            )

        kwargs = {
            "constraints": constraints,
            "options": {
                "maxiter": budget,
                "disp": False,
                "rhobeg": 0.25 * float(np.mean(ub - lb)),
                "tol": 1e-12,
            },
        }

    else:
        raise ValueError(f"Unsupported scipy method: {method}")

    t0 = time.time()

    try:
        minimize(
            counter,
            x0,
            method=scipy_method,
            **kwargs,
        )
    except Exception:
        pass

    wall = time.time() - t0

    return {
        "best_f": float(counter.best_f),
        "fe_used": int(min(counter.n_eval, budget)),
        "n_accept": np.nan,
        "n_reloc": np.nan,
        "wall_time_sec": wall,
        "trace_rows": [],
    }


def run_compass(problem, budget: int, run_id: int):
    """
    Deterministic coordinate/compass search with bound handling.
    """
    dim = int(problem.dimension)
    lb = np.asarray(problem.lower_bounds, dtype=float)
    ub = np.asarray(problem.upper_bounds, dtype=float)

    f = make_objective(problem)

    x = deterministic_x0(lb, ub, run_id)
    best_f = float(f(x))
    fe = 1

    step = 0.25 * float(np.mean(ub - lb))
    step_min = 1e-12 * float(np.mean(ub - lb))

    n_accept = 0

    t0 = time.time()

    while fe < budget and step > step_min:
        improved = False

        for j in range(dim):
            for sign in (-1.0, 1.0):
                if fe >= budget:
                    break

                y = x.copy()
                y[j] += sign * step
                y = np.clip(y, lb, ub)

                fy = float(f(y))
                fe += 1

                if fy < best_f:
                    x = y
                    best_f = fy
                    improved = True
                    n_accept += 1
                    break

            if improved or fe >= budget:
                break

        if not improved:
            step *= 0.5

    wall = time.time() - t0

    return {
        "best_f": float(best_f),
        "fe_used": int(fe),
        "n_accept": int(n_accept),
        "n_reloc": 0,
        "wall_time_sec": wall,
        "trace_rows": [],
    }


def run_algorithm(problem, algorithm: str, budget: int, run_id: int, trace: bool, trace_every: int):
    alg = algorithm.upper()

    if alg == "DMGSO":
        return run_dmgso(problem, budget, run_id, trace, trace_every)

    if alg == "NELDER-MEAD":
        return run_scipy_method(problem, "NELDER-MEAD", budget, run_id)

    if alg == "POWELL":
        return run_scipy_method(problem, "POWELL", budget, run_id)

    if alg == "COBYLA":
        return run_scipy_method(problem, "COBYLA", budget, run_id)

    if alg == "COMPASS":
        return run_compass(problem, budget, run_id)

    raise ValueError(f"Unknown algorithm: {algorithm}")


def summarize_results(run_summary_path: Path) -> None:
    df = pd.read_csv(run_summary_path)

    function_summary = (
        df.groupby(
            ["suite", "algorithm", "function_id", "dimension"],
            dropna=False,
        )
        .agg(
            runs=("run_id", "count"),
            instances=("instance_id", "nunique"),
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
            n_accept_mean=("n_accept", "mean"),
            n_reloc_mean=("n_reloc", "mean"),
        )
        .reset_index()
    )

    dimension_summary = (
        df.groupby(
            ["suite", "algorithm", "dimension"],
            dropna=False,
        )
        .agg(
            runs=("run_id", "count"),
            functions=("function_id", "nunique"),
            instances=("instance_id", "nunique"),
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
            n_accept_mean=("n_accept", "mean"),
            n_reloc_mean=("n_reloc", "mean"),
        )
        .reset_index()
    )

    function_summary.to_csv(
        OUT_TABLES / "coco_bbob_deterministic_function_summary.csv",
        index=False,
    )

    dimension_summary.to_csv(
        OUT_TABLES / "coco_bbob_deterministic_dimension_summary.csv",
        index=False,
    )

    print("[OK] saved:", OUT_TABLES / "coco_bbob_deterministic_function_summary.csv")
    print("[OK] saved:", OUT_TABLES / "coco_bbob_deterministic_dimension_summary.csv")


def run_coco_bbob(
    dims: List[int],
    budget_mult: int,
    max_functions: int | None,
    runs: int,
    algorithms: List[str],
    trace: bool,
    trace_every: int,
    overwrite: bool,
) -> None:
    run_summary_path = OUT_TABLES / "coco_bbob_deterministic_run_summary.csv"

    if overwrite and run_summary_path.exists():
        run_summary_path.unlink()

    suite = cocoex.Suite("bbob", "", "")

    for problem in suite:
        meta = get_problem_metadata(problem)

        fid = int(meta["function_id"])
        dim = int(meta["dimension"])

        if dim not in dims:
            continue

        if max_functions is not None and fid > max_functions:
            continue

        budget = int(budget_mult * dim)

        for run_id in range(runs):
            for algorithm in algorithms:
                alg = algorithm.upper()

                objective_problem = suite.get_problem(problem.index)
                meta = get_problem_metadata(objective_problem)

                try:
                    result = run_algorithm(
                        problem=objective_problem,
                        algorithm=alg,
                        budget=budget,
                        run_id=run_id,
                        trace=trace,
                        trace_every=trace_every,
                    )

                    best_f = float(result["best_f"])
                    fe_used = int(result["fe_used"])
                    n_accept = result["n_accept"]
                    n_reloc = result["n_reloc"]
                    wall = float(result["wall_time_sec"])
                    status = "ok"

                    if trace and alg == "DMGSO" and result["trace_rows"]:
                        trace_path = (
                            OUT_TRACES
                            / f"trace_coco_bbob_DMGSO_f{fid:02d}_i{meta['instance_id']}_D{dim}_run{run_id}.csv"
                        )
                        write_trace_csv(result["trace_rows"], trace_path)

                except Exception as exc:
                    best_f = float("inf")
                    fe_used = budget
                    n_accept = np.nan
                    n_reloc = np.nan
                    wall = np.nan
                    status = f"failed: {type(exc).__name__}: {exc}"

                f_opt = float(meta["f_opt"])
                error = float(best_f - f_opt)

                if not np.isfinite(error):
                    error = float("inf")

                row = {
                    "suite": "COCO-BBOB",
                    "algorithm": alg,
                    "function_id": fid,
                    "instance_id": meta["instance_id"],
                    "function_name": meta["function_name"],
                    "dimension": dim,
                    "run_id": run_id,
                    "FE_budget": budget,
                    "budget_mult": budget_mult,
                    "fe_used": fe_used,
                    "best_f": best_f,
                    "f_opt": f_opt,
                    "error": error,
                    "n_accept": n_accept,
                    "n_reloc": n_reloc,
                    "wall_time_sec": wall,
                    "status": status,
                }

                append_csv(row, run_summary_path)

                print(
                    f"[{status}] {alg:11s} | F{fid:02d} | "
                    f"I{meta['instance_id']} | D={dim:2d} | "
                    f"run={run_id + 1}/{runs} | "
                    f"budget={budget} | error={error:.3e} | FE={fe_used}"
                )

    summarize_results(run_summary_path)

    print("\nCOCO/BBOB deterministic comparison finished.")
    print("Run summary:", run_summary_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run deterministic DFO comparison on COCO/BBOB."
    )

    parser.add_argument(
        "--dims",
        nargs="+",
        type=int,
        default=[2, 5, 10, 20],
        help="Standard BBOB dimensions. Recommended: 2 5 10 20.",
    )

    parser.add_argument(
        "--budget-mult",
        type=int,
        default=1000,
        help="FE_budget = budget_mult * D. Recommended main setting: 1000.",
    )

    parser.add_argument(
        "--max-functions",
        type=int,
        default=24,
        help="Use 24 for full BBOB function set.",
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Number of deterministic restarts / repeated initializations.",
    )

    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=["DMGSO", "NELDER-MEAD", "POWELL", "COBYLA", "COMPASS"],
    )

    parser.add_argument(
        "--trace",
        action="store_true",
        help="Save DMGSO traces only.",
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

    run_coco_bbob(
        dims=args.dims,
        budget_mult=args.budget_mult,
        max_functions=args.max_functions,
        runs=args.runs,
        algorithms=args.algorithms,
        trace=args.trace,
        trace_every=args.trace_every,
        overwrite=args.overwrite,
    )