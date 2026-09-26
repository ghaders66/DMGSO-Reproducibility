"""
dmgso_core.py

Final core implementation of DMGSO
(Deterministic Memory-Guided Sensing Optimization).

This implementation is aligned with the mathematically grounded
DMGSO formulation used in the manuscript:

- deterministic Halton-based initialization,
- deterministic axis-aligned and diagonal direction set,
- local directional probing (vL),
- long-range directional probing (vG),
- adaptive memory-guided search (vM),
- deterministic multi-scale directional fusion,
- multiplicative step-size adaptation,
- deterministic low-discrepancy relocation,
- optional trace output for surface/contour trajectory visualization.

The algorithm is fully deterministic for fixed objective function,
bounds, configuration, and run_id.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np


# ============================================================
# Basic utilities
# ============================================================


def ensure_float64(x: np.ndarray) -> np.ndarray:
    """Return input as a NumPy float64 array."""
    return np.asarray(x, dtype=np.float64)


def safe_eval(f: Callable[[np.ndarray], float], x: np.ndarray) -> float:
    """Safely evaluate the objective function and map invalid values to inf."""
    try:
        val = float(f(x))
        if not np.isfinite(val):
            return np.inf
        return val
    except Exception:
        return np.inf


def reflect_bounds(x: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    """Reflect bound-violating components back into the feasible domain."""
    x = ensure_float64(x).copy()
    lb = ensure_float64(lb)
    ub = ensure_float64(ub)

    for j in range(x.size):
        if lb[j] == -np.inf or ub[j] == np.inf:
            continue

        if ub[j] <= lb[j]:
            x[j] = lb[j]
            continue

        while x[j] < lb[j] or x[j] > ub[j]:
            if x[j] < lb[j]:
                x[j] = lb[j] + (lb[j] - x[j])
            if x[j] > ub[j]:
                x[j] = ub[j] - (x[j] - ub[j])

        x[j] = min(max(x[j], lb[j]), ub[j])

    return x


# ============================================================
# Deterministic low-discrepancy initialization and relocation
# ============================================================


def _first_n_primes(n_primes: int) -> List[int]:
    if n_primes <= 0:
        return []

    primes: List[int] = []
    candidate = 2

    while len(primes) < n_primes:
        is_prime = True
        r = int(math.sqrt(candidate))
        for p in primes:
            if p > r:
                break
            if candidate % p == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(candidate)
        candidate += 1

    return primes


def _van_der_corput(index: int, base: int) -> float:
    vdc = 0.0
    denom = 1.0
    i = index
    while i > 0:
        i, remainder = divmod(i, base)
        denom *= base
        vdc += remainder / denom
    return vdc


def halton_vector(index: int, dim: int) -> np.ndarray:
    """Generate a deterministic Halton vector in [0, 1]^dim."""
    if dim <= 0:
        return np.array([], dtype=np.float64)

    bases = _first_n_primes(dim)
    idx = index + 1
    return np.array(
        [_van_der_corput(idx, bases[d]) for d in range(dim)],
        dtype=np.float64,
    )


def deterministic_init(run_id: int, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    """Deterministic Halton-based initialization within bounds."""
    h = halton_vector(run_id, lb.size)
    return lb + h * (ub - lb)


def deterministic_relocate(
    run_id: int,
    reloc_count: int,
    lb: np.ndarray,
    ub: np.ndarray,
) -> np.ndarray:
    """Deterministic low-discrepancy relocation within bounds."""
    idx = run_id * 100000 + reloc_count * 97
    h = halton_vector(idx, lb.size)
    return lb + h * (ub - lb)


# ============================================================
# Deterministic direction set
# ============================================================


def build_directions(D: int, k_diag: int = 8) -> np.ndarray:
    """
    Build deterministic probing directions.

    The set contains 2D axis-aligned directions and k_diag deterministic
    diagonal sign-pattern directions. All directions are normalized.
    """
    dirs: List[np.ndarray] = []
    I = np.eye(D, dtype=np.float64)

    for j in range(D):
        dirs.append(I[j].copy())
        dirs.append(-I[j].copy())

    if k_diag > 0:
        for t in range(k_diag):
            v = np.ones(D, dtype=np.float64)
            for j in range(D):
                if ((j + t) % 2) == 1:
                    v[j] = -1.0
            dirs.append(v)

    directions = np.vstack(dirs)
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return directions / norms


# ============================================================
# Memory structures
# ============================================================


@dataclass
class MemoryEntry:
    x: np.ndarray
    f: float


class TopKMemory:
    """Bounded favorable memory storing the best accepted points."""

    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.items: List[MemoryEntry] = []

    def add(self, x: np.ndarray, f: float) -> None:
        self.items.append(MemoryEntry(x=ensure_float64(x).copy(), f=float(f)))
        self.items.sort(key=lambda e: e.f)
        if len(self.items) > self.capacity:
            self.items = self.items[: self.capacity]

    def best(self) -> Optional[MemoryEntry]:
        return self.items[0] if self.items else None


class FIFOBuffer:
    """Bounded unfavorable memory storing rejected/stagnating states."""

    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.items: List[MemoryEntry] = []

    def add(self, x: np.ndarray, f: float) -> None:
        self.items.append(MemoryEntry(x=ensure_float64(x).copy(), f=float(f)))
        if len(self.items) > self.capacity:
            self.items.pop(0)


# ============================================================
# Configuration and output logs
# ============================================================


@dataclass
class DMGSOConfig:
    """Configuration of the DMGSO optimizer."""

    # Ablation variant: V0, V1, V2, V3, DMGSO, or DMGSO-Refine
    version: str = "DMGSO"

    # Problem and budget
    D: int = 30
    FE_budget: int = 300000

    # Direction set
    k_diag: int = 8

    # Scale-dependent step and probing radii
    s0_frac: float = 0.02
    delta0_frac: float = 0.005
    Delta0_frac: float = 0.05
    s_min_frac: float = 1e-6
    s_max_frac: float = 0.2

    # Multiplicative step-size adaptation
    c_plus: float = 1.05
    c_minus: float = 0.70

    # Relocation
    stag_trigger: int = 60
    max_relocs: int = 20

    # Memory
    mem_best_B: int = 8
    mem_trap_B: int = 10
    mem_pull: float = 0.35
    mem_push: float = 0.25

    # Directional fusion gate
    gate_stag_scale: float = 80.0

    # Numerical constants
    eps: float = 1e-12

    # Optional refinement mode; keep False for the main manuscript unless reported
    refine_enable: bool = False
    refine_trigger_accepts: int = 3
    refine_max_steps: int = 4
    refine_step_shrink: float = 0.5
    refine_use_memory_blend: bool = True

    # Trace for behavioral visualization
    trace_enable: bool = False
    trace_every: int = 1


@dataclass
class DMGSORunLog:
    best_f: float
    best_x: np.ndarray
    fe_used: int
    n_accept: int
    n_reloc: int
    wall_time_sec: float
    stagnation_count: Optional[int] = None
    trace_rows: List[dict] = field(default_factory=list)


# ============================================================
# Internal helper functions
# ============================================================


def _range_scale(lb: np.ndarray, ub: np.ndarray) -> float:
    """RMS scale of the finite search-space ranges."""
    r = ensure_float64(ub - lb)
    finite_mask = np.isfinite(r)
    if not np.any(finite_mask):
        return 1.0
    r = r[finite_mask]
    return float(np.sqrt(np.mean(r * r)))


def vector_from_improvements(
    f0: float,
    f_list: np.ndarray,
    dirs: np.ndarray,
    eps: float,
) -> np.ndarray:
    """Construct a normalized improvement-weighted directional vector."""
    f_list = np.asarray(f_list, dtype=np.float64)

    if not np.isfinite(f0):
        return np.zeros(dirs.shape[1], dtype=np.float64)

    improvements = f0 - f_list
    improvements = np.where(np.isfinite(improvements), improvements, -np.inf)

    weights = np.maximum(0.0, improvements)
    weight_sum = float(np.sum(weights))

    if weight_sum <= eps:
        return np.zeros(dirs.shape[1], dtype=np.float64)

    weights /= weight_sum
    return (weights[:, None] * dirs).sum(axis=0)


def memory_guidance_vector(
    x: np.ndarray,
    favorable_memory: TopKMemory,
    unfavorable_memory: FIFOBuffer,
    cfg: DMGSOConfig,
) -> np.ndarray:
    """Compute adaptive memory-guided search vector vM."""
    D = x.size
    vM = np.zeros(D, dtype=np.float64)

    best_entry = favorable_memory.best()
    if best_entry is not None:
        dv = best_entry.x - x
        norm = np.linalg.norm(dv) + cfg.eps
        vM += cfg.mem_pull * (dv / norm)

    if len(unfavorable_memory.items) > 0:
        repulsion = np.zeros(D, dtype=np.float64)
        for entry in unfavorable_memory.items:
            dv = x - entry.x
            norm_sq = float(np.dot(dv, dv)) + cfg.eps
            repulsion += dv / norm_sq
        repulsion_norm = np.linalg.norm(repulsion) + cfg.eps
        vM += cfg.mem_push * (repulsion / repulsion_norm)

    return vM


def directional_fusion_weights(
    stagnation: int,
    vL: np.ndarray,
    vG: np.ndarray,
    vM: np.ndarray,
    cfg: DMGSOConfig,
) -> Tuple[float, float, float]:
    """
    Deterministically compute fusion weights for local, global, and memory vectors.
    """
    nL = float(np.linalg.norm(vL))
    nG = float(np.linalg.norm(vG))
    nM = float(np.linalg.norm(vM))

    wL = 0.55
    wG = 0.30
    wM = 0.15

    gate = min(1.0, stagnation / max(1.0, cfg.gate_stag_scale))
    wL = max(0.15, wL - 0.25 * gate)
    wG = min(0.45, wG + 0.15 * gate)
    wM = min(0.40, wM + 0.10 * gate)

    signal_sum = nL + nG + nM + cfg.eps
    wL *= (nL + cfg.eps) / signal_sum
    wG *= (nG + cfg.eps) / signal_sum
    wM *= (nM + cfg.eps) / signal_sum

    total = wL + wG + wM + cfg.eps
    return wL / total, wG / total, wM / total


def _variant_flags(version: str) -> Tuple[bool, bool, bool, bool]:
    """Return flags for step adaptation, long-range probing, memory, and relocation."""
    v = version.upper()
    has_step = v in ("V1", "V2", "V3", "V4", "DMGSO", "DMGSO-REFINE")
    has_global = v in ("V2", "V4", "DMGSO", "DMGSO-REFINE")
    has_memory = v in ("V3", "V4", "DMGSO", "DMGSO-REFINE")
    has_relocation = v in ("V4", "DMGSO", "DMGSO-REFINE")
    return has_step, has_global, has_memory, has_relocation


def _make_trace_row(
    iteration: int,
    fe_used: int,
    x: np.ndarray,
    f_cur: float,
    best_f: float,
    accepted: int,
    relocated: int,
    step_size: float,
    delta: float,
    Delta: float,
    stagnation: int,
    signal_norm: float,
    wL: float,
    wG: float,
    wM: float,
) -> dict:
    row = {
        "iter": int(iteration),
        "fe_used": int(fe_used),
        "f": float(f_cur),
        "best_f_so_far": float(best_f),
        "accepted": int(accepted),
        "relocated": int(relocated),
        "step_size": float(step_size),
        "delta": float(delta),
        "Delta": float(Delta),
        "stagnation": int(stagnation),
        "signal_norm": float(signal_norm),
        "wL": float(wL),
        "wG": float(wG),
        "wM": float(wM),
    }

    for j in range(min(4, x.size)):
        row[f"x{j + 1}"] = float(x[j])

    return row


# ============================================================
# Core optimizer
# ============================================================


def dmgso_optimize(
    f: Callable[[np.ndarray], float],
    lb: np.ndarray,
    ub: np.ndarray,
    cfg: DMGSOConfig,
    run_id: int = 0,
) -> DMGSORunLog:
    """
    Run DMGSO on a bounded continuous black-box optimization problem.

    Parameters
    ----------
    f : callable
        Objective function accepting a NumPy vector and returning a scalar.
    lb, ub : np.ndarray
        Lower and upper bounds.
    cfg : DMGSOConfig
        Optimizer configuration.
    run_id : int
        Deterministic index controlling Halton initialization/relocation.

    Returns
    -------
    DMGSORunLog
        Best result, evaluation count, event counts, wall time, and trace rows.
    """
    t0 = time.time()

    lb = ensure_float64(lb)
    ub = ensure_float64(ub)
    D = lb.size

    dirs = build_directions(D, k_diag=cfg.k_diag)
    K = dirs.shape[0]

    scale = _range_scale(lb, ub)
    s = cfg.s0_frac * scale
    s_min = cfg.s_min_frac * scale
    s_max = cfg.s_max_frac * scale

    range_vec = ub - lb
    finite_range_vec = np.where(np.isfinite(range_vec), range_vec, 0.0)
    range_norm = float(np.linalg.norm(finite_range_vec) / np.sqrt(max(1, D)))
    range_norm = max(cfg.eps, range_norm)

    delta = min(cfg.delta0_frac * scale, 0.1 * range_norm)
    Delta = min(cfg.Delta0_frac * scale, 0.5 * range_norm)

    favorable_memory = TopKMemory(cfg.mem_best_B)
    unfavorable_memory = FIFOBuffer(cfg.mem_trap_B)

    x = deterministic_init(run_id, lb, ub)
    f_cur = safe_eval(f, x)
    fe = 1

    x_best = x.copy()
    f_best = f_cur
    favorable_memory.add(x, f_cur)

    stagnation = 0
    n_accept = 0
    n_reloc = 0
    reloc_count = 0
    refine_count = 0
    consecutive_accepts = 0

    has_step, has_global, has_memory, has_relocation = _variant_flags(cfg.version)
    refine_allowed = cfg.version.upper() == "DMGSO-REFINE" and cfg.refine_enable

    trace_rows: List[dict] = []
    iteration = 0

    last_signal_norm = 0.0
    last_wL, last_wG, last_wM = 1.0, 0.0, 0.0

    if cfg.trace_enable:
        trace_rows.append(
            _make_trace_row(
                iteration=iteration,
                fe_used=fe,
                x=x,
                f_cur=f_cur,
                best_f=f_best,
                accepted=0,
                relocated=0,
                step_size=s,
                delta=delta,
                Delta=Delta,
                stagnation=stagnation,
                signal_norm=last_signal_norm,
                wL=last_wL,
                wG=last_wG,
                wM=last_wM,
            )
        )

    while fe < cfg.FE_budget:
        # ----------------------------------------------------
        # Local directional probing -> vL
        # ----------------------------------------------------
        local_values: List[float] = []
        local_dirs: List[np.ndarray] = []

        for i in range(K):
            if fe >= cfg.FE_budget:
                break
            xi = reflect_bounds(x + delta * dirs[i], lb, ub)
            fi = safe_eval(f, xi)
            fe += 1
            local_values.append(fi)
            local_dirs.append(dirs[i])

        if not local_values:
            break

        vL = vector_from_improvements(
            f_cur,
            np.array(local_values, dtype=np.float64),
            np.array(local_dirs, dtype=np.float64),
            cfg.eps,
        )

        # ----------------------------------------------------
        # Long-range directional probing -> vG
        # ----------------------------------------------------
        vG = np.zeros(D, dtype=np.float64)
        if has_global and fe < cfg.FE_budget:
            global_values: List[float] = []
            global_dirs: List[np.ndarray] = []

            for i in range(K):
                if fe >= cfg.FE_budget:
                    break
                xi = reflect_bounds(x + Delta * dirs[i], lb, ub)
                fi = safe_eval(f, xi)
                fe += 1
                global_values.append(fi)
                global_dirs.append(dirs[i])

            if global_values:
                vG = vector_from_improvements(
                    f_cur,
                    np.array(global_values, dtype=np.float64),
                    np.array(global_dirs, dtype=np.float64),
                    cfg.eps,
                )

        # ----------------------------------------------------
        # Adaptive memory-guided search -> vM
        # ----------------------------------------------------
        vM = np.zeros(D, dtype=np.float64)
        if has_memory:
            vM = memory_guidance_vector(x, favorable_memory, unfavorable_memory, cfg)

        # ----------------------------------------------------
        # Multi-scale directional fusion
        # ----------------------------------------------------
        refine_active = (
            refine_allowed
            and consecutive_accepts >= cfg.refine_trigger_accepts
            and refine_count < cfg.refine_max_steps
        )

        if refine_active:
            if cfg.refine_use_memory_blend and has_memory:
                v_raw = 0.85 * vL + 0.15 * vM
                last_wL, last_wG, last_wM = 0.85, 0.0, 0.15
            else:
                v_raw = vL
                last_wL, last_wG, last_wM = 1.0, 0.0, 0.0
        else:
            if has_relocation:
                last_wL, last_wG, last_wM = directional_fusion_weights(
                    stagnation, vL, vG, vM, cfg
                )
                v_raw = last_wL * vL + last_wG * vG + last_wM * vM
            else:
                if has_global and not has_memory:
                    if np.linalg.norm(vG) > np.linalg.norm(vL) + cfg.eps:
                        v_raw = vG
                        last_wL, last_wG, last_wM = 0.0, 1.0, 0.0
                    else:
                        v_raw = vL
                        last_wL, last_wG, last_wM = 1.0, 0.0, 0.0
                elif has_memory and not has_global:
                    v_raw = vL + 0.5 * vM
                    last_wL, last_wG, last_wM = 2.0 / 3.0, 0.0, 1.0 / 3.0
                else:
                    v_raw = vL
                    last_wL, last_wG, last_wM = 1.0, 0.0, 0.0

        # ----------------------------------------------------
        # Normalize direction with deterministic fallback
        # ----------------------------------------------------
        signal_norm = float(np.linalg.norm(v_raw))
        last_signal_norm = signal_norm

        if signal_norm <= cfg.eps:
            # First deterministic fallback: direction toward box center.
            center = 0.5 * (lb + ub)
            v_raw = center - x
            signal_norm = float(np.linalg.norm(v_raw))

            # Second deterministic fallback: indexed direction from D.
            if signal_norm <= cfg.eps:
                v_raw = dirs[(stagnation * 7) % K].copy()
                signal_norm = float(np.linalg.norm(v_raw))

        v_hat = v_raw / (signal_norm + cfg.eps)

        if fe >= cfg.FE_budget:
            break

        # ----------------------------------------------------
        # Candidate generation and evaluation
        # ----------------------------------------------------
        s_eff = s
        if refine_active:
            s_eff = max(cfg.refine_step_shrink * s, s_min)

        x_new = reflect_bounds(x + s_eff * v_hat, lb, ub)
        f_new = safe_eval(f, x_new)
        fe += 1

        accepted_flag = 0
        relocated_flag = 0

        # ----------------------------------------------------
        # Acceptance and multiplicative step-size regulation
        # ----------------------------------------------------
        if f_new < f_cur:
            x = x_new
            f_cur = f_new
            n_accept += 1
            accepted_flag = 1
            stagnation = 0
            consecutive_accepts += 1

            favorable_memory.add(x, f_cur)
            if f_cur < f_best:
                f_best = f_cur
                x_best = x.copy()

            if has_step:
                if refine_active:
                    s = min(max(s_eff, s_min), s_max)
                else:
                    s = min(s * cfg.c_plus, s_max)

            if refine_active:
                refine_count += 1
            else:
                refine_count = 0
        else:
            stagnation += 1
            consecutive_accepts = 0
            refine_count = 0

            if has_memory:
                unfavorable_memory.add(x_new, f_new)

            if has_step:
                s = max(s * cfg.c_minus, s_min)

        # ----------------------------------------------------
        # Deterministic low-discrepancy relocation
        # ----------------------------------------------------
        if (
            has_relocation
            and stagnation >= cfg.stag_trigger
            and reloc_count < cfg.max_relocs
            and fe < cfg.FE_budget
        ):
            reloc_count += 1
            n_reloc += 1
            relocated_flag = 1

            x = deterministic_relocate(run_id, reloc_count, lb, ub)
            f_cur = safe_eval(f, x)
            fe += 1

            if f_cur < f_best:
                f_best = f_cur
                x_best = x.copy()
                favorable_memory.add(x, f_cur)

            s = cfg.s0_frac * scale
            stagnation = 0
            consecutive_accepts = 0
            refine_count = 0

        iteration += 1

        if cfg.trace_enable and (iteration % max(1, cfg.trace_every) == 0):
            trace_rows.append(
                _make_trace_row(
                    iteration=iteration,
                    fe_used=fe,
                    x=x,
                    f_cur=f_cur,
                    best_f=f_best,
                    accepted=accepted_flag,
                    relocated=relocated_flag,
                    step_size=s,
                    delta=delta,
                    Delta=Delta,
                    stagnation=stagnation,
                    signal_norm=last_signal_norm,
                    wL=last_wL,
                    wG=last_wG,
                    wM=last_wM,
                )
            )

    return DMGSORunLog(
        best_f=f_best,
        best_x=x_best,
        fe_used=fe,
        n_accept=n_accept,
        n_reloc=n_reloc,
        wall_time_sec=time.time() - t0,
        stagnation_count=stagnation,
        trace_rows=trace_rows,
    )


# ============================================================
# Backward-compatible aliases
# ============================================================

# These aliases allow existing runners using the previous MIDO-GS naming
# to keep working while the manuscript and new runners use DMGSO naming.
MIDOConfig = DMGSOConfig
MIDORunLog = DMGSORunLog
mido_optimize = dmgso_optimize
