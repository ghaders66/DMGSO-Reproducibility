# -*- coding: utf-8 -*-
"""
Baseline optimization algorithms for the DMGSO revision experiments.

Algorithms:
    - CMA-ES
    - Differential Evolution (DE)
    - Particle Swarm Optimization (PSO)

Revision protocol:
    - explicit reproducible seed for every stochastic run
    - strict FE-budget guard
    - actual FE consumption returned for every algorithm
    - historical algorithm parameters retained unless required for
      reproducibility or budget compliance

Author:
    Ghader Saadati
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.optimize import differential_evolution


# ============================================================
# Result container
# ============================================================

@dataclass
class BaselineResult:
    algorithm: str
    best_x: np.ndarray
    best_f: float
    fe_used: int
    wall_time_sec: float


# ============================================================
# CMA-ES
# ============================================================

def run_cmaes(
    f: Callable,
    lb: np.ndarray,
    ub: np.ndarray,
    budget: int,
    seed: int,
    sigma0: float = 0.3,
) -> BaselineResult:
    """
    CMA-ES wrapper.

    Historical settings retained:
        - midpoint initial mean
        - sigma0 = 0.3
        - package-internal stopping criteria

    Revision changes:
        - explicit seed
        - strict FE-budget guard
        - explicit FE accounting
    """

    try:
        import cma
    except ImportError as exc:
        raise ImportError(
            "CMA-ES package not found.\n"
            "Install with:\n"
            "    pip install cma"
        ) from exc

    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)

    if budget <= 0:
        raise ValueError("budget must be positive")

    x0 = lb + 0.5 * (ub - lb)

    opts = {
        "bounds": [lb.tolist(), ub.tolist()],
        "maxfevals": int(budget),
        "seed": int(seed),
        "verbose": -9,
    }

    t0 = time.perf_counter()

    es = cma.CMAEvolutionStrategy(
        x0.tolist(),
        sigma0,
        opts,
    )

    fe_used = 0
    best_f = np.inf
    best_x = None

    while not es.stop():
        xs = es.ask()
        generation_size = len(xs)

        # Do not start a generation that would exceed the FE budget.
        if fe_used + generation_size > budget:
            break

        vals = [
            float(f(np.asarray(x, dtype=float)))
            for x in xs
        ]

        fe_used += generation_size

        gen_best_idx = int(np.argmin(vals))
        gen_best_f = float(vals[gen_best_idx])

        if gen_best_f < best_f:
            best_f = gen_best_f
            best_x = np.asarray(
                xs[gen_best_idx],
                dtype=float,
            ).copy()

        es.tell(xs, vals)

    wall = time.perf_counter() - t0

    if best_x is None:
        raise RuntimeError(
            "CMA-ES completed without any objective evaluation."
        )

    if fe_used > budget:
        raise RuntimeError(
            f"CMA-ES exceeded FE budget: {fe_used} > {budget}"
        )

    return BaselineResult(
        algorithm="CMA-ES",
        best_x=best_x,
        best_f=float(best_f),
        fe_used=int(fe_used),
        wall_time_sec=float(wall),
    )


# ============================================================
# Differential Evolution
# ============================================================

def run_de(
    f: Callable,
    lb: np.ndarray,
    ub: np.ndarray,
    budget: int,
    seed: int,
) -> BaselineResult:
    """
    Differential Evolution wrapper.

    Historical settings retained:
        - strategy = best1bin
        - popsize = 15
        - tol = 0
        - mutation = (0.5, 1.0)
        - recombination = 0.9
        - polish = False
        - updating = deferred
        - workers = 1

    Revision changes:
        - explicit seed
        - maxiter corrected so total nfev cannot exceed budget
    """

    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)

    dim = len(lb)

    if budget <= 0:
        raise ValueError("budget must be positive")

    bounds = [
        (float(a), float(b))
        for a, b in zip(lb, ub)
    ]

    popsize = 15

    # For unconstrained continuous DE:
    # initial population evaluations = popsize * dimension.
    population_evals = popsize * dim

    if budget < population_evals:
        raise ValueError(
            f"DE budget ({budget}) is smaller than the initial "
            f"population cost ({population_evals})."
        )

    # scipy DE total evaluations without polishing:
    #
    #     nfev = (maxiter + 1) * popsize * dim
    #
    # Hence subtract one generation for the initial population.
    maxiter = (budget // population_evals) - 1

    t0 = time.perf_counter()

    result = differential_evolution(
        func=f,
        bounds=bounds,
        strategy="best1bin",
        maxiter=int(maxiter),
        popsize=popsize,
        tol=0.0,
        mutation=(0.5, 1.0),
        recombination=0.9,
        seed=int(seed),
        polish=False,
        init="latinhypercube",
        updating="deferred",
        workers=1,
    )

    wall = time.perf_counter() - t0

    fe_used = int(result.nfev)

    if fe_used > budget:
        raise RuntimeError(
            f"DE exceeded FE budget: {fe_used} > {budget}"
        )

    return BaselineResult(
        algorithm="DE",
        best_x=np.asarray(result.x, dtype=float),
        best_f=float(result.fun),
        fe_used=fe_used,
        wall_time_sec=float(wall),
    )


# ============================================================
# PSO
# ============================================================

def run_pso(
    f: Callable,
    lb: np.ndarray,
    ub: np.ndarray,
    budget: int,
    seed: int,
    swarm_size: int = 30,
) -> BaselineResult:
    """
    Simple self-contained PSO.

    Historical settings retained:
        - swarm size = 30
        - w = 0.7
        - c1 = 2.0
        - c2 = 2.0
        - zero initial velocity
        - bound handling by clipping

    Revision changes:
        - seed supplied externally
        - different stochastic realization can be used for each run
        - strict FE-budget guard
    """

    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)

    dim = len(lb)

    if budget < swarm_size:
        raise ValueError(
            f"PSO budget ({budget}) is smaller than swarm size "
            f"({swarm_size})."
        )

    rng = np.random.default_rng(int(seed))

    w = 0.7
    c1 = 2.0
    c2 = 2.0

    t0 = time.perf_counter()

    # Initial population
    X = rng.uniform(
        lb,
        ub,
        size=(swarm_size, dim),
    )

    V = np.zeros(
        (swarm_size, dim),
        dtype=float,
    )

    P = X.copy()

    P_f = np.array(
        [float(f(x)) for x in X],
        dtype=float,
    )

    fe_used = swarm_size

    g_idx = int(np.argmin(P_f))

    G = P[g_idx].copy()
    G_f = float(P_f[g_idx])

    # Each complete PSO iteration costs exactly swarm_size FEs.
    while fe_used + swarm_size <= budget:

        r1 = rng.random((swarm_size, dim))
        r2 = rng.random((swarm_size, dim))

        V = (
            w * V
            + c1 * r1 * (P - X)
            + c2 * r2 * (G - X)
        )

        X = X + V
        X = np.clip(X, lb, ub)

        F = np.array(
            [float(f(x)) for x in X],
            dtype=float,
        )

        fe_used += swarm_size

        improve = F < P_f

        P[improve] = X[improve]
        P_f[improve] = F[improve]

        idx = int(np.argmin(P_f))

        if P_f[idx] < G_f:
            G_f = float(P_f[idx])
            G = P[idx].copy()

    wall = time.perf_counter() - t0

    if fe_used > budget:
        raise RuntimeError(
            f"PSO exceeded FE budget: {fe_used} > {budget}"
        )

    return BaselineResult(
        algorithm="PSO",
        best_x=G,
        best_f=float(G_f),
        fe_used=int(fe_used),
        wall_time_sec=float(wall),
    )


# ============================================================
# Unified dispatcher
# ============================================================

def run_baseline_algorithm(
    algorithm: str,
    f: Callable,
    lb: np.ndarray,
    ub: np.ndarray,
    budget: int,
    seed: int,
) -> BaselineResult:
    """
    Unified baseline interface.

    Every stochastic baseline requires an explicit seed.
    """

    algorithm = algorithm.upper()

    if algorithm == "CMA-ES":
        return run_cmaes(
            f=f,
            lb=lb,
            ub=ub,
            budget=budget,
            seed=seed,
        )

    if algorithm == "DE":
        return run_de(
            f=f,
            lb=lb,
            ub=ub,
            budget=budget,
            seed=seed,
        )

    if algorithm == "PSO":
        return run_pso(
            f=f,
            lb=lb,
            ub=ub,
            budget=budget,
            seed=seed,
        )

    raise ValueError(
        f"Unknown baseline algorithm: {algorithm}"
    )
