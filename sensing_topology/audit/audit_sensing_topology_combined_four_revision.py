# -*- coding: utf-8 -*-
"""
Combined Four-Topology Frozen Audit for the DMGSO Sensing-Topology Revision
===========================================================================

Purpose
-------
Audit, without modifying any source file, the complete four-topology design used
to answer the reviewer request concerning sensing-direction number and topology.

The combined design consists of:
    HISTORICAL
    AXIS_ONLY
    FIXED8_UNIQUE
    DIM_ADAPTIVE

The HISTORICAL baseline is extracted read-only from the frozen COCO/BBOB DMGSO
comparison files. The three revision-only topologies are read from the frozen
sensing-topology experiment. The 14,400-row combined table is created in memory
only and is NOT written as a new raw CSV.

Frozen sources
--------------
1) Historical D=20 source:
   coco_bbob/backup/frozen_20260911/
   coco_bbob_deterministic_D2_D20_corrected_FROZEN.csv

2) Historical D=40 source:
   coco_bbob/backup/frozen_20260911/
   coco_bbob_deterministic_D40_corrected_FROZEN.csv

3) Revision topology source:
   sensing_topology/raw/topology_full_run_summary_revision.csv

Expected source SHA-256 values
------------------------------
Historical D2-D20:
68c2ae2b4776714d2a3fcd4202232db1d745e1d71fe6465cff6102c9ab14d229

Historical D40:
9e67d3bbe0f8d8c19f382235107abba7698f2eec59068eb5f51da60c95a3877d

Revision topology:
7f9f59bd2211ac86b51827d3ca6262aff339f1752e9ca92907497ff33a64e04c

Target design
-------------
Topologies   : 4
Functions    : BBOB F1-F24
Instances    : 15 (1-5 and 71-80)
Dimensions   : D=20, D=40
run_id       : 0-4
Budget       : 1000 x D
Rows/topology: 3,600
Total rows   : 14,400

Scientific interpretation
-------------------------
The audit verifies exact matching on:
    function_id, instance_id, dimension, run_id

It does not treat BBOB instances or run_id values as independent stochastic
replications. Statistical-unit decisions belong to the analysis script.

Output
------
sensing_topology/statistics/combined_four_topology_frozen_audit.txt

Existing output is never overwritten.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# 1. PATHS
# =============================================================================

CODE_DIR = Path(__file__).resolve().parent
TOPO_DIR = CODE_DIR.parent
REVISION = TOPO_DIR.parent

HIST_D20_FILE = (
    REVISION
    / "coco_bbob"
    / "backup"
    / "frozen_20260911"
    / "coco_bbob_deterministic_D2_D20_corrected_FROZEN.csv"
)

HIST_D40_FILE = (
    REVISION
    / "coco_bbob"
    / "backup"
    / "frozen_20260911"
    / "coco_bbob_deterministic_D40_corrected_FROZEN.csv"
)

REV_TOPO_FILE = (
    TOPO_DIR
    / "raw"
    / "topology_full_run_summary_revision.csv"
)

OUT_DIR = TOPO_DIR / "statistics"
REPORT_FILE = OUT_DIR / "combined_four_topology_frozen_audit.txt"

EXPECTED_HASH_D20 = (
    "68c2ae2b4776714d2a3fcd4202232db1d745e1d71fe6465cff6102c9ab14d229"
)
EXPECTED_HASH_D40 = (
    "9e67d3bbe0f8d8c19f382235107abba7698f2eec59068eb5f51da60c95a3877d"
)
EXPECTED_HASH_REV = (
    "7f9f59bd2211ac86b51827d3ca6262aff339f1752e9ca92907497ff33a64e04c"
)

TOPOLOGIES = [
    "HISTORICAL",
    "AXIS_ONLY",
    "FIXED8_UNIQUE",
    "DIM_ADAPTIVE",
]
FUNCTIONS = list(range(1, 25))
INSTANCES = [1, 2, 3, 4, 5] + list(range(71, 81))
DIMENSIONS = [20, 40]
RUN_IDS = list(range(5))
BUDGET_MULT = 1000

EXPECTED_PER_TOPOLOGY = 3600
EXPECTED_TOTAL = 14400

PAIR_KEY = ["function_id", "instance_id", "dimension", "run_id"]


# =============================================================================
# 2. HELPERS
# =============================================================================

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fail(message: str) -> None:
    raise RuntimeError(f"COMBINED TOPOLOGY AUDIT FAILED: {message}")


def verify_file(path: Path, expected_hash: str, label: str) -> str:
    if not path.exists():
        fail(f"{label} source not found: {path}")

    actual = sha256_file(path)

    if actual != expected_hash:
        fail(
            f"{label} SHA-256 mismatch.\n"
            f"Expected: {expected_hash}\n"
            f"Observed: {actual}"
        )

    return actual


# =============================================================================
# 3. LOAD HISTORICAL BASELINE
# =============================================================================

def load_historical() -> pd.DataFrame:
    d20 = pd.read_csv(HIST_D20_FILE)
    d40 = pd.read_csv(HIST_D40_FILE)

    required = {
        "algorithm",
        "function_id",
        "instance_id",
        "dimension",
        "run_id",
        "FE_budget",
        "budget_mult",
        "fe_used",
        "best_f",
        "f_opt_official",
        "error_corrected",
        "n_accept",
        "n_reloc",
        "wall_time_sec",
        "status",
    }

    for label, df in [("D2-D20 historical", d20), ("D40 historical", d40)]:
        missing = required - set(df.columns)
        if missing:
            fail(f"{label} missing columns: {sorted(missing)}")

    hist = pd.concat(
        [
            d20[
                (d20["algorithm"].astype(str) == "DMGSO")
                & (pd.to_numeric(d20["dimension"]).astype(int) == 20)
            ],
            d40[
                (d40["algorithm"].astype(str) == "DMGSO")
                & (pd.to_numeric(d40["dimension"]).astype(int) == 40)
            ],
        ],
        ignore_index=True,
    ).copy()

    hist["topology"] = "HISTORICAL"

    if len(hist) != EXPECTED_PER_TOPOLOGY:
        fail(
            f"Historical DMGSO subset has {len(hist)} rows; "
            f"expected {EXPECTED_PER_TOPOLOGY}."
        )

    return hist


# =============================================================================
# 4. LOAD REVISION TOPOLOGIES
# =============================================================================

def load_revision_topologies() -> pd.DataFrame:
    df = pd.read_csv(REV_TOPO_FILE)

    required = {
        "topology",
        "function_id",
        "instance_id",
        "dimension",
        "run_id",
        "FE_budget",
        "budget_mult",
        "fe_used",
        "best_f",
        "f_opt_official",
        "error_corrected",
        "n_accept",
        "n_reloc",
        "wall_time_sec",
        "status",
    }

    missing = required - set(df.columns)
    if missing:
        fail(f"Revision topology source missing columns: {sorted(missing)}")

    expected_revision_topologies = {
        "AXIS_ONLY",
        "FIXED8_UNIQUE",
        "DIM_ADAPTIVE",
    }

    observed = set(df["topology"].astype(str).unique())
    if observed != expected_revision_topologies:
        fail(
            "Unexpected revision topology set.\n"
            f"Expected: {sorted(expected_revision_topologies)}\n"
            f"Observed: {sorted(observed)}"
        )

    if len(df) != 10800:
        fail(f"Revision topology source has {len(df)} rows; expected 10800.")

    return df.copy()


# =============================================================================
# 5. COMBINED AUDIT
# =============================================================================

def main() -> None:
    if REPORT_FILE.exists():
        raise FileExistsError(
            f"Audit report already exists and will not be overwritten:\n{REPORT_FILE}"
        )

    hash_d20 = verify_file(HIST_D20_FILE, EXPECTED_HASH_D20, "Historical D2-D20")
    hash_d40 = verify_file(HIST_D40_FILE, EXPECTED_HASH_D40, "Historical D40")
    hash_rev = verify_file(REV_TOPO_FILE, EXPECTED_HASH_REV, "Revision topology")

    hist = load_historical()
    rev = load_revision_topologies()

    combined = pd.concat([hist, rev], ignore_index=True, sort=False)

    if len(combined) != EXPECTED_TOTAL:
        fail(f"Combined design has {len(combined)} rows; expected {EXPECTED_TOTAL}.")

    # Global design sets.
    if sorted(combined["topology"].astype(str).unique()) != sorted(TOPOLOGIES):
        fail("Combined topology set mismatch.")

    if sorted(pd.to_numeric(combined["function_id"]).astype(int).unique()) != FUNCTIONS:
        fail("Combined function set mismatch.")

    if sorted(pd.to_numeric(combined["instance_id"]).astype(int).unique()) != INSTANCES:
        fail("Combined instance set mismatch.")

    if sorted(pd.to_numeric(combined["dimension"]).astype(int).unique()) != DIMENSIONS:
        fail("Combined dimension set mismatch.")

    if sorted(pd.to_numeric(combined["run_id"]).astype(int).unique()) != RUN_IDS:
        fail("Combined run_id set mismatch.")

    if set(pd.to_numeric(combined["budget_mult"]).astype(int).unique()) != {BUDGET_MULT}:
        fail("Combined budget multiplier mismatch.")

    # Budget accounting.
    expected_budget = (
        pd.to_numeric(combined["dimension"]).astype(int) * BUDGET_MULT
    )
    actual_budget = pd.to_numeric(combined["FE_budget"]).astype(int)

    if not np.array_equal(expected_budget.to_numpy(), actual_budget.to_numpy()):
        fail("FE_budget is inconsistent with 1000 x D.")

    if (
        pd.to_numeric(combined["fe_used"])
        > pd.to_numeric(combined["FE_budget"])
    ).any():
        fail("FE-budget violation detected.")

    if not (
        combined["status"].astype(str).str.lower() == "ok"
    ).all():
        fail("Non-ok row detected.")

    # Per-topology completeness.
    counts_by_topology = combined.groupby("topology").size()

    for topology in TOPOLOGIES:
        n = int(counts_by_topology.get(topology, 0))
        if n != EXPECTED_PER_TOPOLOGY:
            fail(
                f"{topology} has {n} rows; expected {EXPECTED_PER_TOPOLOGY}."
            )

    # No duplicate within each topology.
    full_key = ["topology"] + PAIR_KEY
    duplicates = int(combined.duplicated(full_key).sum())
    if duplicates != 0:
        fail(f"Detected {duplicates} duplicate combined experiment keys.")

    # Exact four-way pairing.
    paired = combined.groupby(PAIR_KEY)["topology"].agg(
        count="count",
        nunique="nunique",
    )

    bad_pair_blocks = paired[
        (paired["count"] != 4) | (paired["nunique"] != 4)
    ]

    if not bad_pair_blocks.empty:
        fail(
            f"Detected {len(bad_pair_blocks)} incomplete/non-unique four-topology blocks."
        )

    expected_pair_blocks = (
        len(FUNCTIONS)
        * len(INSTANCES)
        * len(DIMENSIONS)
        * len(RUN_IDS)
    )  # 3600

    if len(paired) != expected_pair_blocks:
        fail(
            f"Observed {len(paired)} paired blocks; "
            f"expected {expected_pair_blocks}."
        )

    # Numerical error validity.
    err = pd.to_numeric(combined["error_corrected"], errors="coerce")
    if err.isna().any() or not np.isfinite(err.to_numpy(dtype=float)).all():
        fail("Non-finite corrected error detected.")

    if (err.to_numpy(dtype=float) < -1e-8).any():
        fail("Materially negative corrected error detected.")

    # Check historical corrected-error identity separately.
    hist_recomputed = (
        pd.to_numeric(hist["best_f"]).to_numpy(dtype=float)
        - pd.to_numeric(hist["f_opt_official"]).to_numpy(dtype=float)
    )
    hist_stored = pd.to_numeric(hist["error_corrected"]).to_numpy(dtype=float)

    hist_max_mismatch = float(
        np.max(np.abs(hist_recomputed - hist_stored))
    )

    if not np.allclose(
        hist_recomputed,
        hist_stored,
        rtol=1e-10,
        atol=1e-8,
    ):
        fail(
            "Historical corrected error is inconsistent with "
            "best_f - f_opt_official."
        )

    # Revision corrected-error identity.
    rev_recomputed = (
        pd.to_numeric(rev["best_f"]).to_numpy(dtype=float)
        - pd.to_numeric(rev["f_opt_official"]).to_numpy(dtype=float)
    )
    rev_stored = pd.to_numeric(rev["error_corrected"]).to_numpy(dtype=float)

    rev_max_mismatch = float(
        np.max(np.abs(rev_recomputed - rev_stored))
    )

    if not np.allclose(
        rev_recomputed,
        rev_stored,
        rtol=1e-10,
        atol=1e-8,
    ):
        fail(
            "Revision corrected error is inconsistent with "
            "best_f - f_opt_official."
        )

    # Verify matched official optima across all four topologies for every block.
    fopt_nunique = (
        combined.groupby(PAIR_KEY)["f_opt_official"].nunique(dropna=False)
    )
    inconsistent_fopt = int((fopt_nunique != 1).sum())

    if inconsistent_fopt != 0:
        fail(
            f"Official f* mismatch in {inconsistent_fopt} paired topology blocks."
        )

    lines = [
        "=" * 84,
        "DMGSO FOUR-TOPOLOGY COMBINED FROZEN AUDIT",
        "=" * 84,
        "",
        "Frozen source fingerprints:",
        f"  Historical D2-D20 : {hash_d20}",
        f"  Historical D40     : {hash_d40}",
        f"  Revision topology  : {hash_rev}",
        "",
        f"Historical DMGSO rows : {len(hist)} / {EXPECTED_PER_TOPOLOGY}",
        f"Revision topology rows: {len(rev)} / 10800",
        f"Combined in-memory rows: {len(combined)} / {EXPECTED_TOTAL}",
        "",
        "Rows by topology:",
        counts_by_topology.to_string(),
        "",
        f"Functions             : {len(FUNCTIONS)} (F1-F24)",
        f"Instances             : {len(INSTANCES)}",
        f"Dimensions            : {DIMENSIONS}",
        f"Run IDs               : {RUN_IDS}",
        f"Four-way paired blocks: {len(paired)} / {expected_pair_blocks}",
        f"Duplicate keys        : {duplicates}",
        f"Incomplete pair blocks: {len(bad_pair_blocks)}",
        "Non-ok rows           : 0",
        "FE-budget violations  : 0",
        "Non-finite errors     : 0",
        f"Historical error max mismatch: {hist_max_mismatch:.3e}",
        f"Revision error max mismatch  : {rev_max_mismatch:.3e}",
        f"Official f* mismatches       : {inconsistent_fopt}",
        "",
        "FINAL STATUS: COMBINED FOUR-TOPOLOGY AUDIT PASSED",
        "=" * 84,
    ]

    report = "\n".join(lines)
    print(report)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(report + "\n", encoding="utf-8")

    print(f"\n[OK] Audit report written to: {REPORT_FILE}")
    print("[OK] All frozen source files remained read-only.")
    print("[OK] No merged raw CSV was created.")


if __name__ == "__main__":
    main()
