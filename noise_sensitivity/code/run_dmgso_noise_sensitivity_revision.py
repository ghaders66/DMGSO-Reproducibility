#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DMGSO noise-sensitivity experiment for Reviewer 5, Comment 8.

Scientific freeze
-----------------
- Frozen DMGSO V4 core is imported; it is never modified.
- Frozen CEC2014 wrapper is imported; benchmark definitions are never modified.
- D = 30, FE budget = 300,000 for the full experiment.
- Representative CEC2014 subset: F1,F2,F3,F4,F8,F12,F17,F20,F22,F23,F26,F30.
- Noise levels eta = {0, 1e-4, 1e-3, 1e-2}.
- Additive Gaussian observational noise:
      f_tilde(x) = f(x) + eta * S_f * z, z ~ N(0,1)
- S_f is fixed per function from the IQR of 128 deterministic Halton
  reference points in the benchmark domain. A numerical floor is used only
  if the IQR is degenerate.
- For a given (function_id, run_id), all nonzero noise levels use the same
  standard-normal random stream, scaled by eta. This creates paired noise
  realizations across levels.
- DMGSO run_id remains the deterministic search-configuration index.
- Final solution quality is assessed by reevaluating result.best_x on the
  original noiseless objective. This assessment evaluation is NOT supplied
  to DMGSO and is NOT counted in the optimization FE budget.
- No parameter tuning is performed.

All generated files are written ONLY below:
    Revision/noise_sensitivity/
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

# ---------------------------------------------------------------------
# Frozen protocol
# ---------------------------------------------------------------------
PROJECT_ROOT = Path("/mnt/d/PHD/DMGSO_Information Scince/Revision")
EXP_ROOT = PROJECT_ROOT / "noise_sensitivity"

DIRS = {
    "audit": EXP_ROOT / "audit",
    "code": EXP_ROOT / "code",
    "configs": EXP_ROOT / "configs",
    "figures": EXP_ROOT / "figures",
    "logs": EXP_ROOT / "logs",
    "processed": EXP_ROOT / "processed",
    "raw": EXP_ROOT / "raw",
    "statistics": EXP_ROOT / "statistics",
}

CORE_PATH = PROJECT_ROOT / "cec2014" / "code" / "dmgso_core_revision.py"
WRAPPER_PATH = PROJECT_ROOT / "cec2014" / "code" / "wrapper_cec2014_revision.py"

EXPECTED_CORE_SHA256 = "cadf35f0ae664619527707caf2deea4a163e79eb970cf273e53f5490f9cd9191"
EXPECTED_WRAPPER_SHA256 = "4b59f50b4d0a60034cc9dfc9dedf427a3f3eabb220583322e405bf6746ce2732"

FUNCTIONS = [1, 2, 3, 4, 8, 12, 17, 20, 22, 23, 26, 30]
DIMENSION = 30
FULL_BUDGET = 300_000
RUN_IDS = list(range(5))
NOISE_LEVELS = [0.0, 1e-4, 1e-3, 1e-2]
N_SCALE_POINTS = 128

PROTOCOL_VERSION = "R5-C8-noise-v1.0"

RAW_COLUMNS = [
    "protocol_version", "function_id", "category", "dimension", "run_id",
    "noise_eta", "noise_seed", "noise_scale_Sf", "noise_sigma",
    "FE_budget", "fe_used", "best_noisy_f_reported", "true_final_f",
    "f_opt", "true_final_error", "n_accept", "n_reloc",
    "wall_time_sec", "status"
]


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def ensure_dirs() -> None:
    for p in DIRS.values():
        p.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def append_csv_fsync(row: Dict, path: Path, columns: List[str]) -> None:
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        if not exists:
            w.writeheader()
        w.writerow(row)
        f.flush()
        os.fsync(f.fileno())


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def prime_numbers(n: int) -> List[int]:
    out = []
    x = 2
    while len(out) < n:
        isprime = True
        r = int(x ** 0.5)
        for p in range(2, r + 1):
            if x % p == 0:
                isprime = False
                break
        if isprime:
            out.append(x)
        x += 1
    return out


def radical_inverse(index: int, base: int) -> float:
    f = 1.0
    r = 0.0
    i = int(index)
    while i > 0:
        f /= base
        r += f * (i % base)
        i //= base
    return r


def halton_points(n: int, d: int) -> np.ndarray:
    bases = prime_numbers(d)
    pts = np.empty((n, d), dtype=float)
    # Start at index 1; no random scrambling.
    for i in range(n):
        idx = i + 1
        for j, b in enumerate(bases):
            pts[i, j] = radical_inverse(idx, b)
    return pts


def noise_seed(function_id: int, run_id: int) -> int:
    # Intentionally independent of eta so each paired noise level uses the
    # same z-sequence, merely rescaled by eta.
    return int(8_000_000 + 10_000 * function_id + run_id)


def protocol_dict() -> Dict:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "purpose": "Reviewer 5 Comment 8 controlled observational-noise sensitivity",
        "project_root": str(PROJECT_ROOT),
        "experiment_root": str(EXP_ROOT),
        "frozen_core": str(CORE_PATH),
        "frozen_wrapper": str(WRAPPER_PATH),
        "expected_core_sha256": EXPECTED_CORE_SHA256,
        "expected_wrapper_sha256": EXPECTED_WRAPPER_SHA256,
        "benchmark": "CEC2014",
        "functions": FUNCTIONS,
        "dimension": DIMENSION,
        "full_FE_budget": FULL_BUDGET,
        "run_ids": RUN_IDS,
        "noise_levels_eta": NOISE_LEVELS,
        "noise_model": "additive Gaussian observational noise",
        "formula": "f_tilde(x)=f(x)+eta*S_f*z, z~N(0,1)",
        "scale_definition": (
            f"IQR of noiseless objective values at {N_SCALE_POINTS} fixed "
            "unscrambled Halton reference points over each function domain; "
            "degenerate-IQR floor=max(1e-12,1e-12*max(1,median(abs(f))))"
        ),
        "pairing": (
            "For each function_id and run_id, all nonzero eta levels use the "
            "same standard-normal stream; run_id controls DMGSO deterministic "
            "configuration and noise_seed controls observational noise."
        ),
        "primary_outcome": (
            "true noiseless error at result.best_x; final noiseless assessment "
            "is not supplied to optimizer and not counted in optimization FE"
        ),
        "core_changed": False,
        "parameter_retuning": False,
        "scale_points": N_SCALE_POINTS,
        "planned_full_runs": len(FUNCTIONS) * len(RUN_IDS) * len(NOISE_LEVELS),
    }


def write_protocol() -> Tuple[Path, str]:
    ensure_dirs()
    p = DIRS["configs"] / "noise_sensitivity_protocol.json"
    payload = json.dumps(protocol_dict(), indent=2, sort_keys=True) + "\n"
    atomic_write_text(p, payload)
    return p, hashlib.sha256(payload.encode("utf-8")).hexdigest()


def verify_frozen_sources() -> Tuple[str, str]:
    if not CORE_PATH.exists():
        raise FileNotFoundError(CORE_PATH)
    if not WRAPPER_PATH.exists():
        raise FileNotFoundError(WRAPPER_PATH)
    c = sha256_file(CORE_PATH)
    w = sha256_file(WRAPPER_PATH)
    if c != EXPECTED_CORE_SHA256:
        raise RuntimeError(f"Frozen core SHA mismatch:\nexpected {EXPECTED_CORE_SHA256}\nactual   {c}")
    if w != EXPECTED_WRAPPER_SHA256:
        raise RuntimeError(f"CEC wrapper SHA mismatch:\nexpected {EXPECTED_WRAPPER_SHA256}\nactual   {w}")
    return c, w


def load_frozen_modules():
    core = load_module("dmgso_core_revision_noise_frozen", CORE_PATH)
    wrapper = load_module("wrapper_cec2014_revision_noise_frozen", WRAPPER_PATH)
    return core, wrapper


def compute_noise_scales(wrapper) -> Dict[int, float]:
    ensure_dirs()
    scale_path = DIRS["configs"] / "cec2014_noise_scales.csv"
    rows = []
    scales = {}
    unit = halton_points(N_SCALE_POINTS, DIMENSION)

    for fid in FUNCTIONS:
        problem = wrapper.get_cec2014_problem(function_id=fid, dimension=DIMENSION)
        lb = np.asarray(problem.lower_bound, dtype=float)
        ub = np.asarray(problem.upper_bound, dtype=float)
        X = lb + unit * (ub - lb)
        vals = np.asarray([float(problem.objective(x)) for x in X], dtype=float)
        if not np.all(np.isfinite(vals)):
            raise RuntimeError(f"Non-finite reference objective values for F{fid}")

        q25, q75 = np.quantile(vals, [0.25, 0.75])
        iqr = float(q75 - q25)
        med_abs = float(np.median(np.abs(vals)))
        floor = max(1e-12, 1e-12 * max(1.0, med_abs))
        Sf = max(iqr, floor)

        scales[fid] = Sf
        rows.append({
            "function_id": fid,
            "dimension": DIMENSION,
            "n_reference_points": N_SCALE_POINTS,
            "reference_design": "unscrambled_Halton",
            "q25": float(q25),
            "q75": float(q75),
            "IQR": iqr,
            "scale_floor": floor,
            "noise_scale_Sf": Sf,
        })

    cols = list(rows[0].keys())
    with scale_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    return scales


class NoisyObjective:
    def __init__(self, true_f, sigma: float, seed: int):
        self.true_f = true_f
        self.sigma = float(sigma)
        self.rng = np.random.default_rng(int(seed))
        self.calls = 0

    def __call__(self, x) -> float:
        true_val = float(self.true_f(np.asarray(x, dtype=float)))
        self.calls += 1
        if self.sigma == 0.0:
            return true_val
        return true_val + self.sigma * float(self.rng.standard_normal())


def run_one(core, wrapper, scales, fid: int, run_id: int, eta: float, budget: int) -> Dict:
    problem = wrapper.get_cec2014_problem(function_id=fid, dimension=DIMENSION)
    Sf = float(scales[fid])
    sigma = float(eta * Sf)
    seed = noise_seed(fid, run_id)

    noisy_f = NoisyObjective(problem.objective, sigma=sigma, seed=seed)

    cfg = core.DMGSOConfig(
        version="V4",
        D=DIMENSION,
        FE_budget=int(budget),
        trace_enable=False,
        refine_enable=False,
    )

    t0 = time.time()
    result = core.dmgso_optimize(
        f=noisy_f,
        lb=np.asarray(problem.lower_bound, dtype=float),
        ub=np.asarray(problem.upper_bound, dtype=float),
        cfg=cfg,
        run_id=int(run_id),
    )
    wall = time.time() - t0

    if noisy_f.calls != int(result.fe_used):
        raise RuntimeError(
            f"FE mismatch F{fid} run{run_id} eta={eta}: "
            f"wrapper_calls={noisy_f.calls}, core_fe={result.fe_used}"
        )
    if int(result.fe_used) > int(budget):
        raise RuntimeError("FE budget violation")

    # Assessment only; never fed back to DMGSO and not counted in optimization FE.
    true_final_f = float(problem.objective(np.asarray(result.best_x, dtype=float)))
    f_opt = float(problem.optimum)
    true_error = true_final_f - f_opt
    if not np.isfinite(true_error):
        raise RuntimeError("Non-finite true final error")

    return {
        "protocol_version": PROTOCOL_VERSION,
        "function_id": fid,
        "category": str(problem.category),
        "dimension": DIMENSION,
        "run_id": run_id,
        "noise_eta": eta,
        "noise_seed": seed,
        "noise_scale_Sf": Sf,
        "noise_sigma": sigma,
        "FE_budget": int(budget),
        "fe_used": int(result.fe_used),
        "best_noisy_f_reported": float(result.best_f),
        "true_final_f": true_final_f,
        "f_opt": f_opt,
        "true_final_error": true_error,
        "n_accept": int(getattr(result, "n_accept", 0)),
        "n_reloc": int(getattr(result, "n_reloc", 0)),
        "wall_time_sec": wall,
        "status": "ok",
    }


def read_completed_keys(path: Path) -> set:
    if not path.exists():
        return set()
    keys = set()
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("status") != "ok":
                continue
            keys.add((
                int(row["function_id"]),
                int(row["run_id"]),
                float(row["noise_eta"]),
                int(row["FE_budget"]),
            ))
    return keys


def audit_raw(path: Path, expected_budget: int, expect_full: bool) -> Dict:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    keys = []
    problems = []
    for i, r in enumerate(rows, start=2):
        try:
            key = (int(r["function_id"]), int(r["run_id"]), float(r["noise_eta"]), int(r["FE_budget"]))
            keys.append(key)
            if r["status"] != "ok":
                problems.append(f"line {i}: status={r['status']}")
            if int(r["fe_used"]) > int(r["FE_budget"]):
                problems.append(f"line {i}: FE violation")
            if not np.isfinite(float(r["true_final_error"])):
                problems.append(f"line {i}: non-finite true_final_error")
        except Exception as e:
            problems.append(f"line {i}: parse error {e}")

    duplicate_count = len(keys) - len(set(keys))
    if duplicate_count:
        problems.append(f"duplicate keys={duplicate_count}")

    expected = {
        (fid, rid, eta, expected_budget)
        for fid in FUNCTIONS for rid in RUN_IDS for eta in NOISE_LEVELS
    }
    actual = set(keys)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)

    if expect_full and missing:
        problems.append(f"missing full-design keys={len(missing)}")
    if expect_full and extra:
        problems.append(f"unexpected full-design keys={len(extra)}")

    report = {
        "path": str(path),
        "rows": len(rows),
        "unique_keys": len(actual),
        "duplicate_count": duplicate_count,
        "expected_full_rows": len(expected),
        "missing_full_keys": len(missing),
        "extra_full_keys": len(extra),
        "problems": problems,
        "pass": len(problems) == 0,
    }
    return report


def preflight() -> None:
    ensure_dirs()
    protocol_path, protocol_sha = write_protocol()
    core_sha, wrapper_sha = verify_frozen_sources()
    core, wrapper = load_frozen_modules()
    scales = compute_noise_scales(wrapper)

    # Validate baseline config values are the frozen defaults used by the paper.
    cfg = core.DMGSOConfig(D=DIMENSION, FE_budget=FULL_BUDGET)
    expected_defaults = {
        "k_diag": 8,
        "s0_frac": 0.02,
        "delta0_frac": 0.005,
        "Delta0_frac": 0.05,
        "c_plus": 1.05,
        "c_minus": 0.70,
        "stag_trigger": 60,
        "max_relocs": 20,
        "mem_best_B": 8,
        "mem_trap_B": 10,
        "mem_pull": 0.35,
        "mem_push": 0.25,
        "gate_stag_scale": 80,
    }
    for k, v in expected_defaults.items():
        actual = getattr(cfg, k)
        if actual != v:
            raise RuntimeError(f"Frozen default mismatch {k}: expected {v}, actual {actual}")

    # Scale reproducibility: compute twice.
    scales2 = compute_noise_scales(wrapper)
    for fid in FUNCTIONS:
        if scales[fid] != scales2[fid]:
            raise RuntimeError(f"Non-reproducible noise scale for F{fid}")

    # Noise-stream pairing check: same z stream scaled by eta.
    seed = noise_seed(FUNCTIONS[0], RUN_IDS[0])
    z1 = np.random.default_rng(seed).standard_normal(16)
    z2 = np.random.default_rng(seed).standard_normal(16)
    if not np.array_equal(z1, z2):
        raise RuntimeError("Noise RNG reproducibility failed")

    env = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "core_sha256": core_sha,
        "wrapper_sha256": wrapper_sha,
        "protocol_sha256": protocol_sha,
        "protocol_path": str(protocol_path),
        "runner_path": str(Path(__file__).resolve()),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
    }
    atomic_write_text(DIRS["audit"] / "environment_and_hashes.json",
                      json.dumps(env, indent=2, sort_keys=True) + "\n")

    print(f"[OK] Frozen core SHA256:    {core_sha}")
    print(f"[OK] Frozen wrapper SHA256: {wrapper_sha}")
    print(f"[OK] Protocol SHA256:       {protocol_sha}")
    print(f"[OK] Python:                {sys.version.split()[0]}")
    print(f"[OK] Noise scales:          {len(scales)} functions")
    print(f"[OK] Planned full design:   {len(FUNCTIONS)} × {len(NOISE_LEVELS)} × {len(RUN_IDS)} = "
          f"{len(FUNCTIONS)*len(NOISE_LEVELS)*len(RUN_IDS)} runs")
    print("PRE-FLIGHT PASS")


def smoke() -> None:
    preflight()
    core, wrapper = load_frozen_modules()
    scales = compute_noise_scales(wrapper)

    smoke_path = DIRS["audit"] / "smoke_test_results.csv"
    if smoke_path.exists():
        smoke_path.unlink()

    # Small execution test only: one control and one noisy case.
    for eta in (0.0, 1e-3):
        row = run_one(core, wrapper, scales, fid=1, run_id=0, eta=eta, budget=3000)
        append_csv_fsync(row, smoke_path, RAW_COLUMNS)
        print(f"[SMOKE OK] F01 run=0 eta={eta:g} FE={row['fe_used']} "
              f"true_error={row['true_final_error']:.6e}")

    report = audit_raw(smoke_path, expected_budget=3000, expect_full=False)
    atomic_write_text(DIRS["audit"] / "smoke_test_audit.json",
                      json.dumps(report, indent=2, sort_keys=True) + "\n")
    if not report["pass"]:
        raise RuntimeError(f"Smoke audit failed: {report['problems']}")
    print("SMOKE TEST PASS")


def full_run(overwrite: bool = False) -> None:
    preflight()
    core, wrapper = load_frozen_modules()
    scales = compute_noise_scales(wrapper)

    raw_path = DIRS["raw"] / "noise_sensitivity_cec2014_run_summary.csv"
    if overwrite and raw_path.exists():
        raw_path.unlink()

    completed = read_completed_keys(raw_path)
    total = len(FUNCTIONS) * len(RUN_IDS) * len(NOISE_LEVELS)
    print(f"Full noise design: {len(FUNCTIONS)} functions × {len(NOISE_LEVELS)} levels × "
          f"{len(RUN_IDS)} run_ids = {total} runs")
    print(f"Already completed: {len(completed)}")

    # Order control first, then increasing noise, for easy monitoring.
    for fid in FUNCTIONS:
        for run_id in RUN_IDS:
            for eta in NOISE_LEVELS:
                key = (fid, run_id, eta, FULL_BUDGET)
                if key in completed:
                    continue
                row = run_one(core, wrapper, scales, fid, run_id, eta, FULL_BUDGET)
                append_csv_fsync(row, raw_path, RAW_COLUMNS)
                completed.add(key)
                print(
                    f"[OK] F{fid:02d} run={run_id} eta={eta:g} "
                    f"sigma={row['noise_sigma']:.6e} FE={row['fe_used']} "
                    f"true_err={row['true_final_error']:.6e} "
                    f"time={row['wall_time_sec']:.2f}s",
                    flush=True,
                )

    report = audit_raw(raw_path, expected_budget=FULL_BUDGET, expect_full=True)
    atomic_write_text(DIRS["audit"] / "full_run_audit.json",
                      json.dumps(report, indent=2, sort_keys=True) + "\n")
    if not report["pass"]:
        raise RuntimeError(f"FULL RUN AUDIT FAILED: {report['problems']}")

    frozen = DIRS["raw"] / "noise_sensitivity_cec2014_run_summary_FROZEN.csv"
    frozen.write_bytes(raw_path.read_bytes())
    hashes = {
        "raw_sha256": sha256_file(raw_path),
        "frozen_sha256": sha256_file(frozen),
        "identical": raw_path.read_bytes() == frozen.read_bytes(),
    }
    atomic_write_text(DIRS["audit"] / "frozen_raw_hashes.json",
                      json.dumps(hashes, indent=2, sort_keys=True) + "\n")
    print(f"FULL RUN PASS: {report['rows']} rows, {report['unique_keys']} unique keys")
    print(f"Frozen raw SHA256: {hashes['frozen_sha256']}")


def audit_full() -> None:
    ensure_dirs()
    raw_path = DIRS["raw"] / "noise_sensitivity_cec2014_run_summary.csv"
    report = audit_raw(raw_path, expected_budget=FULL_BUDGET, expect_full=True)
    atomic_write_text(DIRS["audit"] / "full_run_audit.json",
                      json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2))
    if not report["pass"]:
        raise SystemExit(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["preflight", "smoke", "full", "audit"])
    ap.add_argument("--overwrite", action="store_true",
                    help="Only for an intentional fresh full run; otherwise resume safely.")
    args = ap.parse_args()

    ensure_dirs()

    if args.mode == "preflight":
        preflight()
    elif args.mode == "smoke":
        smoke()
    elif args.mode == "full":
        full_run(overwrite=args.overwrite)
    elif args.mode == "audit":
        audit_full()


if __name__ == "__main__":
    main()
