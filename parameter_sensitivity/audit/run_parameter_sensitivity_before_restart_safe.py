# -*- coding: utf-8 -*-
"""
DMGSO parameter-sensitivity runner.

Purpose
-------
Controlled one-factor-at-a-time (OFAT) sensitivity characterization
for Reviewer 5, Comment 6.

IMPORTANT
---------
- Uses the frozen Revision DMGSO core without modifying it.
- Uses the existing CEC2014 Revision wrapper.
- Reads the frozen experimental protocol from JSON.
- Verifies core and protocol SHA-256 integrity.
- Uses a fresh DMGSOConfig object for every optimization run.
- Changes exactly one designated scalar parameter in each perturbed case.
- Supports preflight, smoke test, determinism check, and resumable full runs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


# ============================================================
# Project paths
# ============================================================

THIS_FILE = Path(__file__).resolve()
CODE_DIR = THIS_FILE.parent
STUDY_DIR = CODE_DIR.parent
REVISION_ROOT = STUDY_DIR.parent

PROTOCOL_PATH = STUDY_DIR / "configs" / "sensitivity_protocol.json"
PROTOCOL_HASH_PATH = STUDY_DIR / "audit" / "sensitivity_protocol_sha256.txt"

RAW_DIR = STUDY_DIR / "raw"
LOG_DIR = STUDY_DIR / "logs"
AUDIT_DIR = STUDY_DIR / "audit"

RAW_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
AUDIT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Utilities
# ============================================================

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_recorded_hash(path: Path) -> str:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise RuntimeError(f"Empty SHA-256 record: {path}")
    return text.split()[0]


def append_csv(row: Dict[str, Any], path: Path) -> None:
    exists = path.exists()

    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(row.keys()),
        )
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def utc_timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


# ============================================================
# Protocol and integrity validation
# ============================================================

def load_and_validate_protocol() -> Tuple[Dict[str, Any], Path, str, str]:
    if not PROTOCOL_PATH.exists():
        raise RuntimeError(f"Protocol missing: {PROTOCOL_PATH}")

    if not PROTOCOL_HASH_PATH.exists():
        raise RuntimeError(
            f"Recorded protocol SHA-256 missing: {PROTOCOL_HASH_PATH}"
        )

    protocol = load_json(PROTOCOL_PATH)

    actual_protocol_hash = sha256_file(PROTOCOL_PATH)
    recorded_protocol_hash = read_recorded_hash(PROTOCOL_HASH_PATH)

    if actual_protocol_hash != recorded_protocol_hash:
        raise RuntimeError(
            "Protocol SHA-256 mismatch.\n"
            f"Recorded: {recorded_protocol_hash}\n"
            f"Actual:   {actual_protocol_hash}"
        )

    required_python = protocol["environment"]["python_required"]
    actual_python = platform.python_version()

    if actual_python != required_python:
        raise RuntimeError(
            "Python version mismatch.\n"
            f"Required: {required_python}\n"
            f"Actual:   {actual_python}"
        )

    core_path = REVISION_ROOT / protocol["core"]["canonical_path"]

    if not core_path.exists():
        raise RuntimeError(f"Frozen core missing: {core_path}")

    expected_core_hash = protocol["core"]["sha256"]
    actual_core_hash = sha256_file(core_path)

    if actual_core_hash != expected_core_hash:
        raise RuntimeError(
            "Frozen core SHA-256 mismatch.\n"
            f"Expected: {expected_core_hash}\n"
            f"Actual:   {actual_core_hash}"
        )

    return (
        protocol,
        core_path,
        actual_core_hash,
        actual_protocol_hash,
    )


# ============================================================
# Import frozen Revision implementation
# ============================================================

def import_revision_modules(core_path: Path):
    revision_code_dir = core_path.parent

    if str(revision_code_dir) not in sys.path:
        sys.path.insert(0, str(revision_code_dir))

    from dmgso_core_revision import DMGSOConfig, dmgso_optimize
    from wrapper_cec2014_revision import get_cec2014_problem

    return DMGSOConfig, dmgso_optimize, get_cec2014_problem


# ============================================================
# Baseline configuration validation
# ============================================================

BASELINE_KEYS = (
    "k_diag",
    "s0_frac",
    "delta0_frac",
    "Delta0_frac",
    "s_min_frac",
    "s_max_frac",
    "c_plus",
    "c_minus",
    "stag_trigger",
    "max_relocs",
    "mem_best_B",
    "mem_trap_B",
    "mem_pull",
    "mem_push",
    "gate_stag_scale",
    "eps",
)


def values_equal(a: Any, b: Any) -> bool:
    if isinstance(a, (float, np.floating)) or isinstance(
        b, (float, np.floating)
    ):
        return bool(np.isclose(float(a), float(b), rtol=0.0, atol=1e-15))
    return a == b


def validate_baseline_defaults(protocol, DMGSOConfig) -> None:
    baseline = protocol["baseline"]

    cfg = DMGSOConfig(
        version=protocol["algorithm"]["version"],
        D=protocol["benchmark"]["dimension"],
        FE_budget=protocol["benchmark"]["maximum_function_evaluations"],
        trace_enable=False,
        refine_enable=False,
    )

    actual = asdict(cfg)

    mismatches = []

    for key in BASELINE_KEYS:
        if key not in baseline:
            mismatches.append(f"{key}: missing from protocol baseline")
            continue

        if not values_equal(actual[key], baseline[key]):
            mismatches.append(
                f"{key}: core={actual[key]!r}, protocol={baseline[key]!r}"
            )

    if mismatches:
        raise RuntimeError(
            "Baseline configuration mismatch:\n  "
            + "\n  ".join(mismatches)
        )

    if actual["refine_enable"] is not False:
        raise RuntimeError("refine_enable must remain False.")


# ============================================================
# Build frozen OFAT cases
# ============================================================

def build_cases(protocol: Dict[str, Any]) -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = [
        {
            "case_id": "baseline",
            "parameter": "baseline",
            "level": "baseline",
            "value": None,
        }
    ]

    for parameter, spec in protocol["sensitivity_parameters"].items():
        perturbations = spec["perturbations"]

        if len(perturbations) != 2:
            raise RuntimeError(
                f"{parameter}: exactly two perturbations are required."
            )

        labels = spec.get("labels")

        for idx, value in enumerate(perturbations):
            if labels is not None:
                level = labels[idx]
            else:
                level = "lower" if idx == 0 else "upper"

            cases.append(
                {
                    "case_id": f"{parameter}__{level}",
                    "parameter": parameter,
                    "level": level,
                    "value": value,
                }
            )

    expected = protocol["design"]["total_configurations"]

    if len(cases) != expected:
        raise RuntimeError(
            f"Expected {expected} configurations, got {len(cases)}."
        )

    return cases


# ============================================================
# Fresh configuration + OFAT validation
# ============================================================

def make_config(
    protocol: Dict[str, Any],
    case: Dict[str, Any],
    DMGSOConfig,
):
    cfg = DMGSOConfig(
        version=protocol["algorithm"]["version"],
        D=protocol["benchmark"]["dimension"],
        FE_budget=protocol["benchmark"]["maximum_function_evaluations"],
        trace_enable=False,
        refine_enable=False,
    )

    baseline_cfg = asdict(cfg)

    if case["parameter"] != "baseline":
        parameter = case["parameter"]

        if parameter not in protocol["sensitivity_parameters"]:
            raise RuntimeError(f"Unauthorized sensitivity parameter: {parameter}")

        setattr(cfg, parameter, case["value"])

    effective_cfg = asdict(cfg)

    changed = [
        key
        for key in effective_cfg
        if not values_equal(effective_cfg[key], baseline_cfg[key])
    ]

    if case["parameter"] == "baseline":
        if changed:
            raise RuntimeError(
                f"Baseline unexpectedly changed fields: {changed}"
            )
    else:
        if changed != [case["parameter"]]:
            raise RuntimeError(
                f"OFAT violation in {case['case_id']}: changed={changed}"
            )

    return cfg, effective_cfg


# ============================================================
# Result extraction and validation
# ============================================================

def extract_result_fields(result) -> Tuple[float, int]:
    if isinstance(result, dict):
        best_candidates = ("best_f", "f_best", "best_value")
        fe_candidates = ("fe_used", "FE_used", "n_fe", "fes")

        best_f = None
        fe_used = None

        for key in best_candidates:
            if key in result:
                best_f = result[key]
                break

        for key in fe_candidates:
            if key in result:
                fe_used = result[key]
                break

        if best_f is None or fe_used is None:
            raise RuntimeError(
                "Could not identify best_f/fe_used in DMGSO result dictionary. "
                f"Available keys: {sorted(result.keys())}"
            )

        return float(best_f), int(fe_used)

    best_f = getattr(result, "best_f", None)
    fe_used = getattr(result, "fe_used", None)

    if best_f is None or fe_used is None:
        raise RuntimeError(
            "Could not identify best_f and fe_used in DMGSO result object."
        )

    return float(best_f), int(fe_used)


def corrected_error(
    best_f: float,
    f_opt: float,
    tolerance: float,
) -> float:
    if not np.isfinite(best_f):
        raise RuntimeError(f"Non-finite best_f: {best_f}")

    if not np.isfinite(f_opt):
        raise RuntimeError(f"Non-finite f_opt: {f_opt}")

    error = float(best_f - f_opt)

    if error < -tolerance:
        raise RuntimeError(
            f"Materially negative corrected error: {error:.16e}"
        )

    if -tolerance <= error < 0.0:
        error = 0.0

    return error


# ============================================================
# Resume support
# ============================================================

RESULT_PATH = RAW_DIR / "parameter_sensitivity_runs.csv"


def completed_keys(path: Path) -> set:
    if not path.exists():
        return set()

    import pandas as pd

    df = pd.read_csv(path)

    required = {"case_id", "function_id", "run_id"}

    if not required.issubset(df.columns):
        raise RuntimeError(
            f"Existing result file has incompatible schema: {path}"
        )

    if df.duplicated(
        subset=["case_id", "function_id", "run_id"]
    ).any():
        raise RuntimeError(
            "Duplicate completed-run keys detected in existing raw results."
        )

    return set(
        zip(
            df["case_id"].astype(str),
            df["function_id"].astype(int),
            df["run_id"].astype(int),
        )
    )


# ============================================================
# Single optimization run
# ============================================================

def execute_one(
    protocol,
    case,
    fid,
    run_id,
    DMGSOConfig,
    dmgso_optimize,
    get_cec2014_problem,
):
    D = int(protocol["benchmark"]["dimension"])
    budget = int(
        protocol["benchmark"]["maximum_function_evaluations"]
    )
    tolerance = float(
        protocol["benchmark"]["negative_error_tolerance"]
    )

    problem = get_cec2014_problem(
        function_id=fid,
        dimension=D,
    )

    cfg, effective_cfg = make_config(
        protocol,
        case,
        DMGSOConfig,
    )

    t0 = time.perf_counter()

    result = dmgso_optimize(
        f=problem.objective,
        lb=problem.lower_bound,
        ub=problem.upper_bound,
        cfg=cfg,
        run_id=run_id,
    )

    wall_seconds = time.perf_counter() - t0

    best_f, fe_used = extract_result_fields(result)

    if fe_used < 1 or fe_used > budget:
        raise RuntimeError(
            f"FE-budget violation: F{fid}, run_id={run_id}, "
            f"case={case['case_id']}, FE={fe_used}, budget={budget}"
        )

    error = corrected_error(
        best_f,
        float(problem.optimum),
        tolerance,
    )

    row: Dict[str, Any] = {
        "timestamp": utc_timestamp(),
        "case_id": case["case_id"],
        "parameter": case["parameter"],
        "level": case["level"],
        "parameter_value": (
            "" if case["value"] is None else case["value"]
        ),
        "function_id": int(fid),
        "category": problem.category,
        "dimension": D,
        "run_id": int(run_id),
        "fe_budget": budget,
        "fe_used": int(fe_used),
        "best_f": float(best_f),
        "f_opt": float(problem.optimum),
        "corrected_error": float(error),
        "wall_seconds": float(wall_seconds),
    }

    # Store the complete effective configuration with every run.
    for key, value in effective_cfg.items():
        row[f"cfg_{key}"] = value

    return row


# ============================================================
# Audit manifest
# ============================================================

def write_preflight_manifest(
    protocol,
    core_path,
    core_hash,
    protocol_hash,
    cases,
):
    manifest = {
        "timestamp": utc_timestamp(),
        "runner": str(THIS_FILE),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "protocol_path": str(PROTOCOL_PATH),
        "protocol_sha256": protocol_hash,
        "core_path": str(core_path),
        "core_sha256": core_hash,
        "benchmark": protocol["benchmark"],
        "number_of_cases": len(cases),
        "cases": cases,
    }

    path = AUDIT_DIR / "preflight_manifest.json"

    with path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return path


# ============================================================
# Preflight
# ============================================================

def preflight():
    protocol, core_path, core_hash, protocol_hash = (
        load_and_validate_protocol()
    )

    DMGSOConfig, dmgso_optimize, get_cec2014_problem = (
        import_revision_modules(core_path)
    )

    validate_baseline_defaults(protocol, DMGSOConfig)

    cases = build_cases(protocol)

    # Validate every case before any optimization is run.
    for case in cases:
        make_config(protocol, case, DMGSOConfig)

    # Validate all benchmark functions load correctly.
    D = int(protocol["benchmark"]["dimension"])

    for fid in protocol["benchmark"]["function_ids"]:
        p = get_cec2014_problem(fid, D)

        if p.dimension != D:
            raise RuntimeError(
                f"Dimension mismatch for CEC2014 F{fid}."
            )

        if not np.all(np.isfinite(p.lower_bound)):
            raise RuntimeError(f"Non-finite lower bound for F{fid}.")

        if not np.all(np.isfinite(p.upper_bound)):
            raise RuntimeError(f"Non-finite upper bound for F{fid}.")

        if not np.isfinite(p.optimum):
            raise RuntimeError(f"Non-finite optimum for F{fid}.")

    manifest = write_preflight_manifest(
        protocol,
        core_path,
        core_hash,
        protocol_hash,
        cases,
    )

    print("[OK] Protocol SHA-256 verified:", protocol_hash)
    print("[OK] Frozen core SHA-256 verified:", core_hash)
    print("[OK] Python version verified:", platform.python_version())
    print("[OK] Baseline configuration matches frozen core.")
    print("[OK] 17 OFAT configurations validated.")
    print("[OK] CEC2014 F1-F30 wrapper validation passed.")
    print("[OK] Preflight manifest:", manifest)
    print("\nPRE-FLIGHT PASS")

    return (
        protocol,
        core_path,
        core_hash,
        protocol_hash,
        cases,
        DMGSOConfig,
        dmgso_optimize,
        get_cec2014_problem,
    )


# ============================================================
# Smoke test
# ============================================================

def smoke_test():
    (
        protocol,
        core_path,
        core_hash_before,
        _,
        cases,
        DMGSOConfig,
        dmgso_optimize,
        get_cec2014_problem,
    ) = preflight()

    # Deliberately small smoke test:
    # F1, baseline, run_id=0, with a reduced FE budget.
    smoke_protocol = json.loads(json.dumps(protocol))
    smoke_protocol["benchmark"]["maximum_function_evaluations"] = 3000

    case = cases[0]

    row = execute_one(
        smoke_protocol,
        case,
        fid=1,
        run_id=0,
        DMGSOConfig=DMGSOConfig,
        dmgso_optimize=dmgso_optimize,
        get_cec2014_problem=get_cec2014_problem,
    )

    smoke_path = RAW_DIR / "smoke_test.csv"

    if smoke_path.exists():
        smoke_path.unlink()

    append_csv(row, smoke_path)

    core_hash_after = sha256_file(core_path)

    if core_hash_after != core_hash_before:
        raise RuntimeError(
            "Frozen core changed during smoke test."
        )

    print("\n[OK] Smoke test completed.")
    print("[OK] Core unchanged after smoke test.")
    print("[OK] Output:", smoke_path)
    print(
        f"[OK] F1 baseline run_id=0: "
        f"FE={row['fe_used']}, "
        f"error={row['corrected_error']:.16e}"
    )
    print("\nSMOKE TEST PASS")


# ============================================================
# Determinism check
# ============================================================

def determinism_check():
    (
        protocol,
        core_path,
        core_hash_before,
        _,
        cases,
        DMGSOConfig,
        dmgso_optimize,
        get_cec2014_problem,
    ) = preflight()

    check_protocol = json.loads(json.dumps(protocol))
    check_protocol["benchmark"]["maximum_function_evaluations"] = 3000

    case = cases[0]

    row1 = execute_one(
        check_protocol,
        case,
        fid=1,
        run_id=0,
        DMGSOConfig=DMGSOConfig,
        dmgso_optimize=dmgso_optimize,
        get_cec2014_problem=get_cec2014_problem,
    )

    row2 = execute_one(
        check_protocol,
        case,
        fid=1,
        run_id=0,
        DMGSOConfig=DMGSOConfig,
        dmgso_optimize=dmgso_optimize,
        get_cec2014_problem=get_cec2014_problem,
    )

    exact_fields = (
        "fe_used",
        "best_f",
        "f_opt",
        "corrected_error",
    )

    for field in exact_fields:
        if row1[field] != row2[field]:
            raise RuntimeError(
                f"Determinism failure for {field}: "
                f"{row1[field]!r} != {row2[field]!r}"
            )

    core_hash_after = sha256_file(core_path)

    if core_hash_after != core_hash_before:
        raise RuntimeError(
            "Frozen core changed during determinism check."
        )

    print("\n[OK] Repeated identical run produced identical result.")
    print("[OK] Core unchanged.")
    print(
        f"[OK] best_f={row1['best_f']:.16e}, "
        f"error={row1['corrected_error']:.16e}, "
        f"FE={row1['fe_used']}"
    )
    print("\nDETERMINISM CHECK PASS")


# ============================================================
# Full resumable experiment
# ============================================================

def full_run():
    (
        protocol,
        core_path,
        core_hash_before,
        protocol_hash,
        cases,
        DMGSOConfig,
        dmgso_optimize,
        get_cec2014_problem,
    ) = preflight()

    done = completed_keys(RESULT_PATH)

    function_ids = [
        int(x) for x in protocol["benchmark"]["function_ids"]
    ]
    run_ids = [
        int(x) for x in protocol["benchmark"]["run_ids"]
    ]

    expected_total = (
        len(cases) * len(function_ids) * len(run_ids)
    )

    print(
        f"\nFull sensitivity design: "
        f"{len(cases)} cases × "
        f"{len(function_ids)} functions × "
        f"{len(run_ids)} run_ids = "
        f"{expected_total} runs"
    )
    print(f"Already completed: {len(done)}")
    print(f"Raw output: {RESULT_PATH}\n")

    completed_now = 0

    for case in cases:
        for fid in function_ids:
            for run_id in run_ids:
                key = (
                    case["case_id"],
                    int(fid),
                    int(run_id),
                )

                if key in done:
                    continue

                row = execute_one(
                    protocol,
                    case,
                    fid,
                    run_id,
                    DMGSOConfig,
                    dmgso_optimize,
                    get_cec2014_problem,
                )

                append_csv(row, RESULT_PATH)

                done.add(key)
                completed_now += 1

                print(
                    f"[{len(done):4d}/{expected_total}] "
                    f"{case['case_id']:35s} "
                    f"F{fid:02d} "
                    f"run={run_id} "
                    f"FE={row['fe_used']:6d} "
                    f"err={row['corrected_error']:.6e} "
                    f"time={row['wall_seconds']:.2f}s",
                    flush=True,
                )

    if len(done) != expected_total:
        raise RuntimeError(
            f"Run-count mismatch: {len(done)} != {expected_total}"
        )

    core_hash_after = sha256_file(core_path)

    if core_hash_after != core_hash_before:
        raise RuntimeError(
            "Frozen core SHA-256 changed during full experiment."
        )

    completion = {
        "timestamp": utc_timestamp(),
        "status": "complete",
        "expected_runs": expected_total,
        "observed_unique_runs": len(done),
        "runs_completed_this_invocation": completed_now,
        "core_sha256_before": core_hash_before,
        "core_sha256_after": core_hash_after,
        "protocol_sha256": protocol_hash,
        "raw_result_path": str(RESULT_PATH),
    }

    completion_path = AUDIT_DIR / "full_run_completion.json"

    with completion_path.open("w", encoding="utf-8") as f:
        json.dump(completion, f, indent=2)

    print("\n[OK] All expected sensitivity runs are present.")
    print("[OK] Frozen core hash unchanged.")
    print("[OK] Completion record:", completion_path)
    print("\nFULL RUN COMPLETE")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="DMGSO frozen-core parameter sensitivity study"
    )

    mode = parser.add_mutually_exclusive_group(required=True)

    mode.add_argument(
        "--preflight",
        action="store_true",
        help="Validate protocol, environment, core, configs, and CEC wrapper.",
    )

    mode.add_argument(
        "--smoke",
        action="store_true",
        help="Run a small F1 baseline smoke test.",
    )

    mode.add_argument(
        "--determinism-check",
        action="store_true",
        help="Repeat an identical small run and verify identical outputs.",
    )

    mode.add_argument(
        "--full",
        action="store_true",
        help="Run/resume the complete frozen sensitivity experiment.",
    )

    args = parser.parse_args()

    if args.preflight:
        preflight()
    elif args.smoke:
        smoke_test()
    elif args.determinism_check:
        determinism_check()
    elif args.full:
        full_run()


if __name__ == "__main__":
    main()
