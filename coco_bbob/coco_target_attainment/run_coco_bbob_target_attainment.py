#!/usr/bin/env python3
"""
Supplementary COCO/BBOB target-attainment experiment.

Purpose
-------
Generate native COCO/BBOB logging data for standard cocopp ECDF /
target-attainment reporting while reusing the exact optimizer implementations
from the frozen deterministic BBOB comparison runner.

This experiment supplements, but does not replace, the frozen 45,000-row
numerical comparison used in the manuscript.

Protocol
--------
Suite       : COCO/BBOB noiseless F1-F24
Dimensions  : 2, 5, 10, 20, 40
Instances   : 1,2,3,4,5,71,72,73,74,75,76,77,78,79,80
Runs        : run_id = 0,...,4
Algorithms  : DMGSO, NELDER-MEAD, POWELL, COBYLA, COMPASS
Max budget  : 1000 * D objective evaluations

Important
---------
- Optimizer implementations are imported from the frozen comparison runner.
- No optimizer implementation is redefined here.
- Native cocoex.Observer logging records the evaluation trajectories.
- Runtime from this supplementary experiment is NOT used for runtime claims.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import platform
import shutil
import sys
import time
from pathlib import Path

import cocoex
import numpy as np
import pandas as pd
import scipy


SCRIPT_PATH = Path(__file__).resolve()
COMMENT_DIR = SCRIPT_PATH.parent
COCO_BBOB_DIR = COMMENT_DIR.parent
REPRO_ROOT = COCO_BBOB_DIR.parent

SOURCE_RUNNER = (
    COCO_BBOB_DIR
    / "execution"
    / "run_coco_bbob_deterministic_compare.py"
)

OUT_TABLES = COMMENT_DIR / "tables"
OUT_MANIFEST = COMMENT_DIR / "manifest"
OUT_EXDATA = COMMENT_DIR / "exdata"

EXPECTED_SOURCE_SHA256 = (
    "64c793fa99fedb65ad85098c91023279be5f9247216b4823d00b10a10e78494a"
)

ALGORITHMS = [
    "DMGSO",
    "NELDER-MEAD",
    "POWELL",
    "COBYLA",
    "COMPASS",
]

FULL_DIMS = [2, 5, 10, 20, 40]

EXPECTED_INSTANCES = {
    1, 2, 3, 4, 5,
    71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
}

EXPECTED_FUNCTIONS = set(range(1, 25))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def import_source_runner():
    if not SOURCE_RUNNER.exists():
        raise FileNotFoundError(
            f"Frozen source runner not found: {SOURCE_RUNNER}"
        )

    actual_hash = sha256_file(SOURCE_RUNNER)

    if actual_hash != EXPECTED_SOURCE_SHA256:
        raise RuntimeError(
            "Frozen source runner SHA256 mismatch.\n"
            f"Expected: {EXPECTED_SOURCE_SHA256}\n"
            f"Actual:   {actual_hash}\n"
            "Refusing to run because optimizer provenance changed."
        )

    execution_dir = str(SOURCE_RUNNER.parent)
    if execution_dir not in sys.path:
        sys.path.insert(0, execution_dir)

    # The frozen comparison runner imports:
    #     from core.dmgso_core import ...
    #
    # In the reproducibility package, the corresponding frozen core file is
    # stored beside the runner as execution/dmgso_core.py.  Reconstruct that
    # historical import namespace without modifying either frozen source file.
    import types

    core_path = SOURCE_RUNNER.parent / "dmgso_core.py"
    if not core_path.exists():
        raise FileNotFoundError(
            f"Frozen DMGSO core not found: {core_path}"
        )

    core_pkg = types.ModuleType("core")
    core_pkg.__path__ = [str(SOURCE_RUNNER.parent)]
    sys.modules["core"] = core_pkg

    core_spec = importlib.util.spec_from_file_location(
        "core.dmgso_core",
        core_path,
    )
    if core_spec is None or core_spec.loader is None:
        raise ImportError(f"Cannot import frozen core: {core_path}")

    core_module = importlib.util.module_from_spec(core_spec)
    sys.modules["core.dmgso_core"] = core_module
    core_spec.loader.exec_module(core_module)
    core_pkg.dmgso_core = core_module

    spec = importlib.util.spec_from_file_location(
        "frozen_bbob_compare",
        SOURCE_RUNNER,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {SOURCE_RUNNER}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "run_algorithm"):
        raise AttributeError(
            "Frozen source runner does not expose run_algorithm()."
        )

    return module, actual_hash


def validate_suite_protocol(
    dims: list[int],
    function_ids: set[int],
    instance_ids: set[int],
) -> None:
    suite = cocoex.Suite("bbob", "", "")

    observed_functions = set()
    observed_instances_by_dim = {d: set() for d in dims}

    for problem in suite:
        fid = int(problem.id_function)
        iid = int(problem.id_instance)
        dim = int(problem.dimension)

        if dim not in dims:
            continue
        if fid not in function_ids:
            continue

        observed_functions.add(fid)

        # Validate only the instance subset requested for this experiment.
        # This permits reduced smoke tests (e.g. instance 1 only), while the
        # full protocol still validates the exact frozen 15-instance set.
        if iid in instance_ids:
            observed_instances_by_dim[dim].add(iid)

    if observed_functions != function_ids:
        raise RuntimeError(
            "COCO function-set validation failed.\n"
            f"Expected: {sorted(function_ids)}\n"
            f"Actual:   {sorted(observed_functions)}"
        )

    for dim in dims:
        actual = observed_instances_by_dim[dim]
        if actual != instance_ids:
            raise RuntimeError(
                f"COCO instance-set validation failed at D={dim}.\n"
                f"Expected: {sorted(instance_ids)}\n"
                f"Actual:   {sorted(actual)}"
            )


def get_problem_metadata(problem) -> dict:
    return {
        "function_id": int(problem.id_function),
        "instance_id": int(problem.id_instance),
        "dimension": int(problem.dimension),
        "problem_id": str(problem.id),
        "problem_name": str(problem.name),
    }


def append_csv(row: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()

    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def prepare_output(overwrite: bool) -> None:
    if overwrite:
        for path in [OUT_TABLES, OUT_MANIFEST, OUT_EXDATA]:
            if path.exists():
                shutil.rmtree(path)

    for path in [OUT_TABLES, OUT_MANIFEST, OUT_EXDATA]:
        path.mkdir(parents=True, exist_ok=True)


def observer_options(
    algorithm: str,
    result_folder: str,
    run_count: int,
    budget_mult: int,
) -> str:
    description = (
        f"COCO/BBOB Target-Attainment; "
        f"algorithm={algorithm}; "
        f"runs={run_count}; "
        f"budget={budget_mult}D"
    )

    return (
        f"result_folder: {result_folder} "
        f"algorithm_name: {algorithm} "
        f'algorithm_info: "{description}"'
    )


def run_experiment(
    source_module,
    dims: list[int],
    function_ids: set[int],
    instance_ids: set[int],
    runs: int,
    algorithms: list[str],
    budget_mult: int,
    overwrite: bool,
) -> Path:

    prepare_output(overwrite=overwrite)

    summary_path = OUT_TABLES / "coco_target_attainment_run_summary.csv"

    if summary_path.exists() and not overwrite:
        raise FileExistsError(
            f"{summary_path} already exists. "
            "Use --overwrite only when intentionally starting a fresh experiment."
        )

    validate_suite_protocol(
        dims=dims,
        function_ids=function_ids,
        instance_ids=instance_ids,
    )

    original_cwd = Path.cwd()

    # COCO interprets result_folder relative to ./exdata.
    # Therefore run from COMMENT_DIR so all native logger output remains
    # isolated inside coco_target_attainment/exdata/.
    try:
        import os
        os.chdir(COMMENT_DIR)

        for algorithm in algorithms:
            alg = algorithm.upper()

            if alg not in ALGORITHMS:
                raise ValueError(f"Unsupported algorithm: {algorithm}")

            result_folder = alg.replace("/", "_")

            observer = cocoex.Observer(
                "bbob",
                observer_options(
                    algorithm=alg,
                    result_folder=result_folder,
                    run_count=runs,
                    budget_mult=budget_mult,
                ),
            )

            suite = cocoex.Suite("bbob", "", "")

            selected = []

            for problem in suite:
                meta = get_problem_metadata(problem)

                if meta["dimension"] not in dims:
                    continue
                if meta["function_id"] not in function_ids:
                    continue
                if meta["instance_id"] not in instance_ids:
                    continue

                selected.append(
                    (
                        int(problem.index),
                        meta["function_id"],
                        meta["instance_id"],
                        meta["dimension"],
                    )
                )

            expected_problem_count = (
                len(function_ids) * len(instance_ids) * len(dims)
            )

            if len(selected) != expected_problem_count:
                raise RuntimeError(
                    f"Selected problem count mismatch for {alg}: "
                    f"{len(selected)} != {expected_problem_count}"
                )

            for problem_index, fid, iid, dim in selected:
                budget = int(budget_mult * dim)

                for run_id in range(runs):
                    objective_problem = suite.get_problem(problem_index)
                    objective_problem.observe_with(observer)

                    start = time.perf_counter()

                    try:
                        result = source_module.run_algorithm(
                            problem=objective_problem,
                            algorithm=alg,
                            budget=budget,
                            run_id=run_id,
                            trace=False,
                            trace_every=10,
                        )

                        elapsed = time.perf_counter() - start

                        best_f = float(result["best_f"])
                        fe_used = int(result["fe_used"])
                        observed_fe = int(objective_problem.evaluations)

                        if observed_fe != fe_used:
                            raise RuntimeError(
                                f"FE mismatch: optimizer={fe_used}, "
                                f"COCO observer={observed_fe}"
                            )

                        if fe_used > budget:
                            raise RuntimeError(
                                f"FE budget exceeded: {fe_used} > {budget}"
                            )

                        status = "ok"
                        error_message = ""

                    except Exception as exc:
                        elapsed = time.perf_counter() - start
                        best_f = float("nan")
                        fe_used = int(objective_problem.evaluations)
                        observed_fe = int(objective_problem.evaluations)
                        status = "failed"
                        error_message = (
                            f"{type(exc).__name__}: {exc}"
                        )

                    row = {
                        "suite": "bbob",
                        "algorithm": alg,
                        "function_id": fid,
                        "instance_id": iid,
                        "dimension": dim,
                        "run_id": run_id,
                        "FE_budget": budget,
                        "budget_mult": budget_mult,
                        "fe_used": fe_used,
                        "coco_observed_fe": observed_fe,
                        "best_f_optimizer": best_f,
                        "wall_time_sec_not_for_runtime_claims": elapsed,
                        "status": status,
                        "error_message": error_message,
                    }

                    append_csv(row, summary_path)

                    print(
                        f"[{status}] {alg:11s} | "
                        f"F{fid:02d} | I{iid:02d} | "
                        f"D={dim:2d} | run={run_id} | "
                        f"FE={fe_used}/{budget}",
                        flush=True,
                    )

                    objective_problem.free()

                    if status != "ok":
                        raise RuntimeError(
                            f"Experiment failed for {alg}, "
                            f"F{fid}, I{iid}, D={dim}, run={run_id}: "
                            f"{error_message}"
                        )

            observer = None

    finally:
        import os
        os.chdir(original_cwd)

    return summary_path


def audit_summary(
    summary_path: Path,
    dims: list[int],
    function_ids: set[int],
    instance_ids: set[int],
    runs: int,
    algorithms: list[str],
) -> dict:

    df = pd.read_csv(summary_path)

    expected_rows = (
        len(algorithms)
        * len(function_ids)
        * len(instance_ids)
        * len(dims)
        * runs
    )

    duplicate_cols = [
        "algorithm",
        "function_id",
        "instance_id",
        "dimension",
        "run_id",
    ]

    duplicate_count = int(
        df.duplicated(subset=duplicate_cols).sum()
    )

    failed_count = int((df["status"] != "ok").sum())

    fe_mismatch_count = int(
        (
            df["fe_used"].astype(int)
            != df["coco_observed_fe"].astype(int)
        ).sum()
    )

    budget_violation_count = int(
        (
            df["fe_used"].astype(int)
            > df["FE_budget"].astype(int)
        ).sum()
    )

    audit = {
        "expected_rows": expected_rows,
        "actual_rows": int(len(df)),
        "duplicate_keys": duplicate_count,
        "failed_rows": failed_count,
        "fe_mismatches": fe_mismatch_count,
        "budget_violations": budget_violation_count,
        "algorithms": sorted(df["algorithm"].unique().tolist()),
        "dimensions": sorted(
            int(x) for x in df["dimension"].unique()
        ),
        "functions": sorted(
            int(x) for x in df["function_id"].unique()
        ),
        "instances": sorted(
            int(x) for x in df["instance_id"].unique()
        ),
        "run_ids": sorted(
            int(x) for x in df["run_id"].unique()
        ),
    }

    if audit["actual_rows"] != expected_rows:
        raise RuntimeError(
            f"Row-count audit failed: "
            f"{audit['actual_rows']} != {expected_rows}"
        )

    if duplicate_count != 0:
        raise RuntimeError(
            f"Duplicate-key audit failed: {duplicate_count}"
        )

    if failed_count != 0:
        raise RuntimeError(
            f"Failed-run audit failed: {failed_count}"
        )

    if fe_mismatch_count != 0:
        raise RuntimeError(
            f"FE-consistency audit failed: {fe_mismatch_count}"
        )

    if budget_violation_count != 0:
        raise RuntimeError(
            f"Budget audit failed: {budget_violation_count}"
        )

    return audit


def package_version(name: str) -> str:
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return "unknown"


def write_manifest(
    source_hash: str,
    args,
    summary_path: Path,
    audit: dict,
) -> Path:

    manifest = {
        "experiment": "COCO/BBOB Target-Attainment Analysis",
        "purpose": (
            "Native COCO logging for standard cocopp "
            "target-attainment / ECDF reporting."
        ),
        "replaces_frozen_numerical_comparison": False,
        "runtime_results_used_for_runtime_claims": False,
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pandas": pd.__version__,
            "cocoex": getattr(cocoex, "__version__", "unknown"),
            "cocopp": package_version("cocopp"),
        },
        "source_runner": str(SOURCE_RUNNER),
        "source_runner_sha256": source_hash,
        "expected_source_runner_sha256": EXPECTED_SOURCE_SHA256,
        "protocol": {
            "suite": "bbob",
            "function_ids": sorted(args.function_ids),
            "dimensions": args.dims,
            "instance_ids": sorted(args.instance_ids),
            "run_ids": list(range(args.runs)),
            "runs_per_problem": args.runs,
            "algorithms": args.algorithms,
            "budget_rule": f"{args.budget_mult} * D",
            "budget_mult": args.budget_mult,
        },
        "summary_csv": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "audit": audit,
    }

    OUT_MANIFEST.mkdir(parents=True, exist_ok=True)

    path = OUT_MANIFEST / "coco_target_attainment_manifest.json"

    with path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate native COCO/BBOB data for standard target-attainment and ECDF reporting "
            "using the frozen optimizer implementations."
        )
    )

    parser.add_argument(
        "--dims",
        nargs="+",
        type=int,
        default=FULL_DIMS,
    )

    parser.add_argument(
        "--function-ids",
        nargs="+",
        type=int,
        default=sorted(EXPECTED_FUNCTIONS),
    )

    parser.add_argument(
        "--instance-ids",
        nargs="+",
        type=int,
        default=sorted(EXPECTED_INSTANCES),
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=ALGORITHMS,
    )

    parser.add_argument(
        "--budget-mult",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Smoke test: F1, instance 1, D=2, run_id=0, "
            "all five algorithms, budget=100*D."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    args.algorithms = [x.upper() for x in args.algorithms]
    args.function_ids = set(args.function_ids)
    args.instance_ids = set(args.instance_ids)

    if args.smoke:
        args.dims = [2]
        args.function_ids = {1}
        args.instance_ids = {1}
        args.runs = 1
        args.budget_mult = 100

    unknown_algorithms = sorted(
        set(args.algorithms) - set(ALGORITHMS)
    )
    if unknown_algorithms:
        raise ValueError(
            f"Unsupported algorithms: {unknown_algorithms}"
        )

    source_module, source_hash = import_source_runner()

    print("=" * 72)
    print("COCO/BBOB TARGET-ATTAINMENT EXPERIMENT")
    print("=" * 72)
    print("Python       :", sys.version.split()[0])
    print("NumPy        :", np.__version__)
    print("SciPy        :", scipy.__version__)
    print("pandas       :", pd.__version__)
    print("cocoex       :", getattr(cocoex, "__version__", "unknown"))
    print("cocopp       :", package_version("cocopp"))
    print("Source SHA256:", source_hash)
    print("Dimensions   :", args.dims)
    print("Functions    :", sorted(args.function_ids))
    print("Instances    :", sorted(args.instance_ids))
    print("Runs         :", args.runs)
    print("Algorithms   :", args.algorithms)
    print("Budget rule  :", f"{args.budget_mult}D")
    print("Smoke        :", args.smoke)
    print("=" * 72)

    summary_path = run_experiment(
        source_module=source_module,
        dims=args.dims,
        function_ids=args.function_ids,
        instance_ids=args.instance_ids,
        runs=args.runs,
        algorithms=args.algorithms,
        budget_mult=args.budget_mult,
        overwrite=args.overwrite,
    )

    audit = audit_summary(
        summary_path=summary_path,
        dims=args.dims,
        function_ids=args.function_ids,
        instance_ids=args.instance_ids,
        runs=args.runs,
        algorithms=args.algorithms,
    )

    manifest_path = write_manifest(
        source_hash=source_hash,
        args=args,
        summary_path=summary_path,
        audit=audit,
    )

    print()
    print("=" * 72)
    print("AUDIT PASS")
    print("=" * 72)
    for key, value in audit.items():
        print(f"{key}: {value}")

    print()
    print("Summary :", summary_path)
    print("Manifest:", manifest_path)
    print("Exdata  :", OUT_EXDATA)
    print("=" * 72)


if __name__ == "__main__":
    main()
