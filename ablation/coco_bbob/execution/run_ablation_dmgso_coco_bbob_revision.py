#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audited DMGSO V0--V4 ablation on COCO/BBOB for the Information Sciences revision.

Architecture implemented by the imported audited core:
    V0: directional sensing only
    V1: V0 + adaptive step regulation
    V2: V1 + long-range sensing
    V3: V1 + memory guidance
    V4: full architecture = step + long-range + memory + relocation

Main protocol:
    BBOB functions: F1--F24
    dimensions: 2, 5, 10, 20, 40
    COCO instances: whatever is present in the local bbob suite
                    (the script records and audits them explicitly)
    deterministic configurations: run_id 0--4
    FE budget: 1000 * D
    same run_id is paired across V0--V4
    refine_enable=False

Critical safeguards:
    - imports the audited CEC2014 revision core, not the historical BBOB core
    - f_opt is NEVER assumed to be zero
    - f_opt is looked up from bbob_fopt_cocoex_2.8.2.csv by problem_id
    - missing/mismatched f_opt => fail fast
    - negative error beyond tolerance => fail fast
    - FE budget overrun => fail fast
    - output is separate from historical BBOB ablation output
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

try:
    import cocoex
except ImportError as exc:
    raise ImportError(
        "cocoex is not installed in the active environment. "
        "Activate .venv_revision_py311 and install coco-experiment if needed."
    ) from exc


# -------------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------------

REV = Path("/mnt/d/PHD/DMGSO_Information Scince/Revision")
CEC_CODE = REV / "cec2014" / "code"
BBOB_ROOT = REV / "coco_bbob"

if not CEC_CODE.exists():
    raise FileNotFoundError(f"Audited CEC code directory not found: {CEC_CODE}")

sys.path.insert(0, str(CEC_CODE))

from dmgso_core_revision import DMGSOConfig, dmgso_optimize  # noqa: E402

FOPT_CSV = BBOB_ROOT / "processed" / "bbob_fopt_cocoex_2.8.2.csv"

OUT_ROOT = BBOB_ROOT / "revision_audited_ablation"
OUT_RAW = OUT_ROOT / "raw"
OUT_PROCESSED = OUT_ROOT / "processed"
OUT_TRACES = OUT_ROOT / "traces"

for p in (OUT_ROOT, OUT_RAW, OUT_PROCESSED, OUT_TRACES):
    p.mkdir(parents=True, exist_ok=True)

RUN_SUMMARY = OUT_RAW / "ablation_coco_bbob_audited_run_summary.csv"

ERROR_TOL = 1e-8

VERSION_DESCRIPTIONS = {
    "V0": "Pure directional sensing",
    "V1": "V0 + adaptive step regulation",
    "V2": "V1 + long-range sensing",
    "V3": "V1 + memory guidance",
    "V4": "Full DMGSO combining long-range sensing, memory guidance, and relocation",
}

VERSIONS_DEFAULT = ["V0", "V1", "V2", "V3", "V4"]
DIMS_DEFAULT = [2, 5, 10, 20, 40]


# -------------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------------

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
    keys = sorted({k for r in rows for k in r.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def load_fopt_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Corrected BBOB f_opt table not found: {path}")

    df = pd.read_csv(path)

    required = {
        "suite", "cocoex_version", "function_id", "instance_id",
        "dimension", "problem_id", "f_opt"
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"f_opt table missing required columns: {missing}")

    df = df.copy()
    df["problem_id"] = df["problem_id"].astype(str)
    df["f_opt"] = pd.to_numeric(df["f_opt"], errors="coerce")

    if df["problem_id"].duplicated().any():
        dup = df.loc[df["problem_id"].duplicated(keep=False), "problem_id"].unique()
        raise ValueError(
            "Duplicate problem_id values in f_opt table; cannot perform unique lookup: "
            + ", ".join(map(str, dup[:10]))
        )

    if df["f_opt"].isna().any() or not np.isfinite(df["f_opt"]).all():
        raise ValueError("Non-finite f_opt values found in corrected BBOB optimum table.")

    if not (df["suite"].astype(str).str.lower() == "bbob").all():
        raise ValueError("Corrected optimum table contains non-bbob rows.")

    versions = sorted(df["cocoex_version"].astype(str).unique().tolist())
    print(f"[audit] f_opt table rows={len(df)}, cocoex_version(s)={versions}")
    return df.set_index("problem_id", drop=False)


def extract_problem_metadata(problem, fopt: pd.DataFrame) -> Dict:
    pid = str(getattr(problem, "id", "")).strip()
    if not pid:
        raise ValueError("COCO problem has empty problem.id")

    if pid not in fopt.index:
        raise KeyError(
            f"No corrected f_opt entry for problem_id={pid}. "
            "The run is stopped to avoid an invalid zero-optimum fallback."
        )

    row = fopt.loc[pid]
    if isinstance(row, pd.DataFrame):
        raise ValueError(f"Non-unique f_opt lookup for problem_id={pid}")

    dim = int(problem.dimension)
    fid = int(row["function_id"])
    iid = int(row["instance_id"])
    f_opt = float(row["f_opt"])

    # Cross-check metadata from the suite against the frozen optimum table.
    if int(row["dimension"]) != dim:
        raise ValueError(
            f"Dimension mismatch for {pid}: suite D={dim}, f_opt table D={row['dimension']}"
        )

    number = getattr(problem, "number", None)
    if number is not None and int(number) != fid:
        raise ValueError(
            f"Function mismatch for {pid}: suite F={number}, f_opt table F={fid}"
        )

    instance = getattr(problem, "instance", None)
    if instance is not None and int(instance) != iid:
        raise ValueError(
            f"Instance mismatch for {pid}: suite I={instance}, f_opt table I={iid}"
        )

    return {
        "function_id": fid,
        "instance_id": iid,
        "dimension": dim,
        "problem_id": pid,
        "function_name": pid,
        "f_opt": f_opt,
        "cocoex_version_fopt_table": str(row["cocoex_version"]),
    }


def make_objective(problem):
    def objective(x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        val = float(problem(x))
        if not np.isfinite(val):
            raise FloatingPointError(
                f"Non-finite objective returned for problem {problem.id}: {val}"
            )
        return val
    return objective


def validate_result(
    version: str,
    meta: Dict,
    run_id: int,
    best_f: float,
    f_opt: float,
    fe_used: int,
    budget: int,
) -> float:
    if not np.isfinite(best_f):
        raise RuntimeError(
            f"{version} {meta['problem_id']} run_id={run_id}: "
            f"non-finite best_f={best_f}"
        )

    if not np.isfinite(f_opt):
        raise RuntimeError(
            f"{meta['problem_id']}: non-finite corrected f_opt={f_opt}"
        )

    if fe_used < 1:
        raise RuntimeError(
            f"{version} {meta['problem_id']} run_id={run_id}: "
            f"invalid FE count={fe_used}"
        )

    if fe_used > budget:
        raise RuntimeError(
            f"{version} {meta['problem_id']} run_id={run_id}: "
            f"FE budget exceeded ({fe_used} > {budget})"
        )

    error = float(best_f - f_opt)

    if error < -ERROR_TOL:
        raise RuntimeError(
            f"{version} {meta['problem_id']} run_id={run_id}: "
            f"best_f={best_f:.16e} below corrected f_opt={f_opt:.16e}; "
            f"error={error:.16e}"
        )

    if -ERROR_TOL <= error < 0.0:
        error = 0.0

    return error


def run_single_version(
    problem,
    version: str,
    budget: int,
    run_id: int,
    trace: bool,
    trace_every: int,
):
    dim = int(problem.dimension)
    lb = np.asarray(problem.lower_bounds, dtype=float)
    ub = np.asarray(problem.upper_bounds, dtype=float)

    cfg = DMGSOConfig(
        version=version,
        D=dim,
        FE_budget=budget,
        trace_enable=trace,
        trace_every=trace_every,
        refine_enable=False,
    )

    t0 = time.perf_counter()
    result = dmgso_optimize(
        f=make_objective(problem),
        lb=lb,
        ub=ub,
        cfg=cfg,
        run_id=run_id,
    )
    wall = time.perf_counter() - t0
    return result, float(wall)


# -------------------------------------------------------------------------
# Post-run summaries
# -------------------------------------------------------------------------

def summarize_ablation(run_summary_path: Path) -> None:
    df = pd.read_csv(run_summary_path)

    expected_cols = {
        "suite", "version", "function_id", "instance_id", "dimension",
        "run_id", "FE_budget", "fe_used", "best_f", "f_opt", "error",
        "n_accept", "n_reloc", "wall_time_sec", "status"
    }
    missing = sorted(expected_cols - set(df.columns))
    if missing:
        raise ValueError(f"Run summary missing required columns: {missing}")

    if not (df["status"] == "ok").all():
        bad = df.loc[df["status"] != "ok"]
        raise RuntimeError(
            "Non-ok run(s) present in audited BBOB output:\n"
            + bad.head(20).to_string(index=False)
        )

    key = ["version", "function_id", "instance_id", "dimension", "run_id"]
    if df.duplicated(key).any():
        dup = df.loc[df.duplicated(key, keep=False), key]
        raise RuntimeError("Duplicate audited BBOB records:\n" + dup.to_string(index=False))

    # Function x dimension descriptive summary
    function_summary = (
        df.groupby(
            ["suite", "version", "function_id", "dimension"],
            as_index=False, dropna=False
        )
        .agg(
            deterministic_configurations=("run_id", "count"),
            instances=("instance_id", "nunique"),
            error_mean=("error", "mean"),
            error_median=("error", "median"),
            error_std=("error", "std"),
            error_min=("error", "min"),
            error_max=("error", "max"),
            fe_used_mean=("fe_used", "mean"),
            fe_used_median=("fe_used", "median"),
            n_accept_mean=("n_accept", "mean"),
            n_accept_median=("n_accept", "median"),
            n_reloc_mean=("n_reloc", "mean"),
            n_reloc_median=("n_reloc", "median"),
            wall_time_mean=("wall_time_sec", "mean"),
            wall_time_median=("wall_time_sec", "median"),
        )
    )

    dimension_summary = (
        df.groupby(["suite", "version", "dimension"], as_index=False, dropna=False)
        .agg(
            records=("run_id", "count"),
            functions=("function_id", "nunique"),
            instances=("instance_id", "nunique"),
            error_mean=("error", "mean"),
            error_median=("error", "median"),
            error_std=("error", "std"),
            fe_used_mean=("fe_used", "mean"),
            fe_used_median=("fe_used", "median"),
            n_accept_mean=("n_accept", "mean"),
            n_accept_median=("n_accept", "median"),
            n_reloc_mean=("n_reloc", "mean"),
            n_reloc_median=("n_reloc", "median"),
            wall_time_mean=("wall_time_sec", "mean"),
            wall_time_median=("wall_time_sec", "median"),
        )
    )

    version_summary = (
        df.groupby(["suite", "version"], as_index=False, dropna=False)
        .agg(
            records=("run_id", "count"),
            dimensions=("dimension", "nunique"),
            functions=("function_id", "nunique"),
            instances=("instance_id", "nunique"),
            error_mean=("error", "mean"),
            error_median=("error", "median"),
            error_std=("error", "std"),
            fe_used_mean=("fe_used", "mean"),
            fe_used_median=("fe_used", "median"),
            n_accept_mean=("n_accept", "mean"),
            n_accept_median=("n_accept", "median"),
            n_reloc_mean=("n_reloc", "mean"),
            n_reloc_median=("n_reloc", "median"),
            wall_time_mean=("wall_time_sec", "mean"),
            wall_time_median=("wall_time_sec", "median"),
        )
    )

    function_summary.to_csv(
        OUT_PROCESSED / "ablation_coco_bbob_audited_function_summary.csv",
        index=False,
    )
    dimension_summary.to_csv(
        OUT_PROCESSED / "ablation_coco_bbob_audited_dimension_summary.csv",
        index=False,
    )
    version_summary.to_csv(
        OUT_PROCESSED / "ablation_coco_bbob_audited_version_summary.csv",
        index=False,
    )

    print("[saved]", OUT_PROCESSED / "ablation_coco_bbob_audited_function_summary.csv")
    print("[saved]", OUT_PROCESSED / "ablation_coco_bbob_audited_dimension_summary.csv")
    print("[saved]", OUT_PROCESSED / "ablation_coco_bbob_audited_version_summary.csv")


# -------------------------------------------------------------------------
# Main experiment
# -------------------------------------------------------------------------

def run_ablation(
    versions: List[str],
    dims: List[int],
    budget_mult: int,
    max_functions: int,
    runs: int,
    trace: bool,
    trace_every: int,
    overwrite: bool,
) -> None:

    for version in versions:
        if version not in VERSION_DESCRIPTIONS:
            raise ValueError(
                f"Unknown version {version}. Allowed: {list(VERSION_DESCRIPTIONS)}"
            )

    fopt = load_fopt_table(FOPT_CSV)

    if overwrite and RUN_SUMMARY.exists():
        RUN_SUMMARY.unlink()

    existing = set()
    if RUN_SUMMARY.exists():
        old = pd.read_csv(RUN_SUMMARY)
        required_key = ["version", "problem_id", "run_id"]
        if not set(required_key).issubset(old.columns):
            raise ValueError(
                "Existing audited run file has incompatible schema. "
                "Use --overwrite after archiving it."
            )
        existing = {
            (str(r.version), str(r.problem_id), int(r.run_id))
            for r in old[required_key].itertuples(index=False)
        }
        print(f"[resume] existing completed rows: {len(existing)}")

    suite = cocoex.Suite("bbob", "", "")

    # Audit suite coverage before running.
    selected_meta = []
    for problem in suite:
        # Filter by dimension/function BEFORE corrected f_opt lookup.
        # The local f_opt table intentionally contains only the revision grid.
        dim_pre = int(problem.dimension)
        fid_pre = getattr(problem, "number", None)

        if dim_pre not in dims:
            continue
        if fid_pre is not None and int(fid_pre) > max_functions:
            continue

        meta = extract_problem_metadata(problem, fopt)

        if meta["function_id"] <= max_functions:
            selected_meta.append(meta)

    if not selected_meta:
        raise RuntimeError("No BBOB problems selected by the requested grid.")

    selected_df = pd.DataFrame(selected_meta)
    print(
        "[audit] selected grid:",
        f"functions={selected_df['function_id'].nunique()}, "
        f"instances={selected_df['instance_id'].nunique()}, "
        f"dimensions={sorted(selected_df['dimension'].unique().tolist())}, "
        f"problem_ids={len(selected_df)}"
    )

    # Re-create suite because it was iterated above.
    suite = cocoex.Suite("bbob", "", "")

    for problem in suite:
        # Filter the COCO suite before querying the corrected optimum table.
        dim_pre = int(problem.dimension)
        fid_pre = getattr(problem, "number", None)

        if dim_pre not in dims:
            continue
        if fid_pre is not None and int(fid_pre) > max_functions:
            continue

        meta = extract_problem_metadata(problem, fopt)
        fid = meta["function_id"]
        dim = meta["dimension"]

        if fid > max_functions:
            continue

        budget = int(budget_mult * dim)

        for run_id in range(runs):
            for version in versions:
                key = (version, meta["problem_id"], run_id)
                if key in existing:
                    print(f"[skip] {version} {meta['problem_id']} run_id={run_id}")
                    continue

                # Obtain a fresh problem object for each deterministic configuration.
                objective_problem = suite.get_problem(problem.index)
                meta_run = extract_problem_metadata(objective_problem, fopt)

                result, wall = run_single_version(
                    problem=objective_problem,
                    version=version,
                    budget=budget,
                    run_id=run_id,
                    trace=trace,
                    trace_every=trace_every,
                )

                best_f = float(result.best_f)
                fe_used = int(result.fe_used)
                n_accept = int(getattr(result, "n_accept", 0))
                n_reloc = int(getattr(result, "n_reloc", 0))
                f_opt = float(meta_run["f_opt"])

                error = validate_result(
                    version=version,
                    meta=meta_run,
                    run_id=run_id,
                    best_f=best_f,
                    f_opt=f_opt,
                    fe_used=fe_used,
                    budget=budget,
                )

                if trace and getattr(result, "trace_rows", None):
                    trace_path = (
                        OUT_TRACES /
                        f"trace_{version}_{meta_run['problem_id']}_run{run_id}.csv"
                    )
                    write_trace_csv(result.trace_rows, trace_path)

                row = {
                    "suite": "COCO-BBOB",
                    "version": version,
                    "version_description": VERSION_DESCRIPTIONS[version],
                    "function_id": meta_run["function_id"],
                    "instance_id": meta_run["instance_id"],
                    "function_name": meta_run["function_name"],
                    "problem_id": meta_run["problem_id"],
                    "dimension": meta_run["dimension"],
                    "run_id": run_id,
                    "configuration_type": "deterministic",
                    "FE_budget": budget,
                    "budget_mult": budget_mult,
                    "fe_used": fe_used,
                    "best_f": best_f,
                    "f_opt": f_opt,
                    "error": error,
                    "n_accept": n_accept,
                    "n_reloc": n_reloc,
                    "wall_time_sec": wall,
                    "status": "ok",
                    "cocoex_version_fopt_table": meta_run["cocoex_version_fopt_table"],
                }
                append_csv(row, RUN_SUMMARY)

                print(
                    f"[ok] {version:2s} | {meta_run['problem_id']} | "
                    f"run={run_id+1}/{runs} | budget={budget} | "
                    f"error={error:.3e} | FE={fe_used} | "
                    f"accept={n_accept} | reloc={n_reloc}"
                )

    summarize_ablation(RUN_SUMMARY)

    print("\nDMGSO audited COCO/BBOB ablation finished.")
    print("Raw run summary:", RUN_SUMMARY)
    print("Processed summaries:", OUT_PROCESSED)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run audited DMGSO V0--V4 ablation on COCO/BBOB."
    )
    p.add_argument("--versions", nargs="+", default=VERSIONS_DEFAULT)
    p.add_argument("--dims", nargs="+", type=int, default=DIMS_DEFAULT)
    p.add_argument("--budget-mult", type=int, default=1000)
    p.add_argument("--max-functions", type=int, default=24)
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--trace", action="store_true")
    p.add_argument("--trace-every", type=int, default=20)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_ablation(
        versions=args.versions,
        dims=args.dims,
        budget_mult=args.budget_mult,
        max_functions=args.max_functions,
        runs=args.runs,
        trace=args.trace,
        trace_every=args.trace_every,
        overwrite=args.overwrite,
    )
