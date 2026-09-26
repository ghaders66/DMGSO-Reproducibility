# -*- coding: utf-8 -*-
"""
Reviewer-Driven Combined Four-Topology Analysis for DMGSO
==========================================================

Purpose
-------
Analyze the complete sensing-topology experiment used in the Information
Sciences revision by combining, in memory only, the frozen historical DMGSO
baseline with the three revision-only controlled sensing topologies:

    HISTORICAL
    AXIS_ONLY
    FIXED8_UNIQUE
    DIM_ADAPTIVE

The analysis is intentionally reviewer-driven. It addresses questions about:
    - the number and geometry of deterministic sensing directions,
    - duplicated versus unique diagonal directions,
    - high-dimensional directional coverage,
    - and the performance-versus-function-evaluation cost trade-off.

No frozen raw source is modified and no merged raw CSV is created.

Frozen sources
--------------
Historical D=20 source:
    coco_bbob/backup/frozen_20260911/
    coco_bbob_deterministic_D2_D20_corrected_FROZEN.csv

Historical D=40 source:
    coco_bbob/backup/frozen_20260911/
    coco_bbob_deterministic_D40_corrected_FROZEN.csv

Revision topology source:
    sensing_topology/raw/topology_full_run_summary_revision.csv

Expected source SHA-256
-----------------------
Historical D2-D20:
68c2ae2b4776714d2a3fcd4202232db1d745e1d71fe6465cff6102c9ab14d229

Historical D40:
9e67d3bbe0f8d8c19f382235107abba7698f2eec59068eb5f51da60c95a3877d

Revision topology:
7f9f59bd2211ac86b51827d3ca6262aff339f1752e9ca92907497ff33a64e04c

Experimental design
-------------------
Functions            : BBOB F1-F24
Instances            : 15 official instances (1-5 and 71-80)
Dimensions           : D=20 and D=40
run_id               : 0,...,4
Budget               : 1000 x D
Rows per topology    : 3,600
Combined rows        : 14,400

Statistical unit
----------------
The inferential unit is the BBOB benchmark function (n=24), analyzed separately
for D=20 and D=40.

For each Function x Dimension x Topology cell, corrected error is summarized by
the median across:

    15 BBOB instances x 5 deterministic run_id configurations = 75 raw records.

Thus, neither the 15 BBOB instances nor the five deterministic configurations
are treated as independent statistical replications. This prevents
pseudo-replication and follows the reviewer requirement that benchmark
functions/problems serve as paired inferential units.

Omnibus analysis
----------------
For each dimension separately:
    - within-function ranks,
    - Friedman test across four topologies,
    - Kendall's W,
    - average and median topology ranks.

Pre-specified topology contrasts
--------------------------------
The pairwise tests are limited to contrasts that answer distinct reviewer
questions. All comparisons are interpreted as OLD -> NEW, where positive
rank-biserial correlation favors NEW.

1. HISTORICAL -> AXIS_ONLY
   Tests whether removing the historical stored diagonals changes performance.

2. HISTORICAL -> FIXED8_UNIQUE
   Tests the effect of replacing the historically duplicated diagonal set by
   eight genuinely unique diagonal patterns while preserving the same stored
   direction count and nominal normal-iteration FE cost.

3. AXIS_ONLY -> FIXED8_UNIQUE
   Tests whether adding eight unique deterministic diagonal directions improves
   performance relative to axis-only sensing.

4. FIXED8_UNIQUE -> DIM_ADAPTIVE
   Tests whether increasing unique diagonal coverage from a fixed eight
   directions to a dimension-scaled set produces additional benefit.

No exhaustive all-pairs post-hoc fishing is performed.

Multiplicity
------------
Primary Holm correction:
    four planned contrasts within each dimension.

Sensitivity Holm correction:
    all eight dimension-specific pairwise tests together.

The stricter global-eight correction is retained to show whether any conclusion
depends on the narrower dimension-specific multiplicity family.

Historical topology structure
-----------------------------
The historical implementation stores:

    K = 2D + 8 directions

but its parity-based diagonal construction collapses the eight stored diagonals
to two unique diagonal patterns. Therefore:

    stored directions      = 2D + 8
    unique directions      = 2D + 2
    duplicate directions   = 6
    normal iteration FE    = 2K + 1 = 4D + 17

FIXED8_UNIQUE intentionally keeps the same stored direction count and nominal
normal-iteration FE cost while replacing the duplicated diagonal structure with
eight unique deterministic dense patterns.

Interpretation guardrails
-------------------------
- A denser direction set is not assumed to be better.
- Statistical significance is interpreted together with rank direction,
  rank-biserial effect size, and win/tie/loss counts.
- Raw objective-error magnitudes are not pooled across BBOB functions for a
  global effect size because function scales differ strongly.
- Wall-clock time is descriptive only; it is not treated as pure algorithmic
  overhead because objective evaluation and platform effects contribute.
- The experiment supports conclusions about the implemented topology
  definitions, not universal causal claims about all possible direction sets.

Outputs
-------
sensing_topology/statistics/four_topology/

    00_four_topology_analysis_manifest.txt
    01_four_topology_function_level_medians.csv
    02_four_topology_average_ranks_friedman.csv
    03_four_topology_planned_pairwise.csv
    04_four_topology_cost_summary.csv
    05_four_topology_statistics_report.txt

Existing outputs are never silently overwritten.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, wilcoxon


# =============================================================================
# 1. PATHS AND FROZEN DESIGN
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

EXPECTED_HASH_D20 = (
    "68c2ae2b4776714d2a3fcd4202232db1d745e1d71fe6465cff6102c9ab14d229"
)
EXPECTED_HASH_D40 = (
    "9e67d3bbe0f8d8c19f382235107abba7698f2eec59068eb5f51da60c95a3877d"
)
EXPECTED_HASH_REV = (
    "7f9f59bd2211ac86b51827d3ca6262aff339f1752e9ca92907497ff33a64e04c"
)

OUT_DIR = TOPO_DIR / "statistics" / "four_topology"

OUT_MANIFEST = OUT_DIR / "00_four_topology_analysis_manifest.txt"
OUT_FUNCTION = OUT_DIR / "01_four_topology_function_level_medians.csv"
OUT_RANK = OUT_DIR / "02_four_topology_average_ranks_friedman.csv"
OUT_PAIR = OUT_DIR / "03_four_topology_planned_pairwise.csv"
OUT_COST = OUT_DIR / "04_four_topology_cost_summary.csv"
OUT_REPORT = OUT_DIR / "05_four_topology_statistics_report.txt"

OUTPUTS = [
    OUT_MANIFEST,
    OUT_FUNCTION,
    OUT_RANK,
    OUT_PAIR,
    OUT_COST,
    OUT_REPORT,
]

TOPOLOGIES = [
    "HISTORICAL",
    "AXIS_ONLY",
    "FIXED8_UNIQUE",
    "DIM_ADAPTIVE",
]

DIMENSIONS = [20, 40]
FUNCTIONS = list(range(1, 25))
INSTANCES = [1, 2, 3, 4, 5] + list(range(71, 81))
RUN_IDS = list(range(5))
BUDGET_MULT = 1000
EXPECTED_PER_TOPOLOGY = 3600
EXPECTED_TOTAL = 14400
ALPHA = 0.05

PAIR_KEY = ["function_id", "instance_id", "dimension", "run_id"]

PLANNED_CONTRASTS = [
    ("HISTORICAL", "AXIS_ONLY"),
    ("HISTORICAL", "FIXED8_UNIQUE"),
    ("AXIS_ONLY", "FIXED8_UNIQUE"),
    ("FIXED8_UNIQUE", "DIM_ADAPTIVE"),
]


# =============================================================================
# 2. SAFETY AND STATISTICAL HELPERS
# =============================================================================

def sha256_file(path: Path) -> str:
    """Return SHA-256 without modifying the file."""
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_hash(path: Path, expected: str, label: str) -> str:
    """Fail if a frozen source is missing or has changed."""
    if not path.exists():
        raise FileNotFoundError(f"{label} not found:\n{path}")

    actual = sha256_file(path)

    if actual != expected:
        raise RuntimeError(
            f"{label} SHA-256 mismatch.\n"
            f"Expected: {expected}\n"
            f"Observed: {actual}"
        )

    return actual


def refuse_overwrite() -> None:
    """
    Refuse to overwrite any existing derived statistical output.

    The output directory may exist, but no expected analysis artifact may be
    silently replaced.
    """
    existing = [p for p in OUTPUTS if p.exists()]

    if existing:
        raise FileExistsError(
            "Four-topology statistical output already exists. "
            "No files were overwritten:\n"
            + "\n".join(f"  {p}" for p in existing)
        )


def holm_adjust(p_values: Iterable[float]) -> np.ndarray:
    """Holm step-down family-wise error correction."""
    p = np.asarray(list(p_values), dtype=float)
    order = np.argsort(p)
    m = len(p)

    adjusted = np.empty(m, dtype=float)
    running_max = 0.0

    for i, idx in enumerate(order):
        candidate = (m - i) * p[idx]
        running_max = max(running_max, candidate)
        adjusted[idx] = min(running_max, 1.0)

    return adjusted


def rank_biserial_from_difference(difference: np.ndarray) -> float:
    """
    Paired rank-biserial correlation.

    difference = error_old - error_new

    Positive values favor NEW.
    Negative values favor OLD.
    """
    d = np.asarray(difference, dtype=float)

    # Match Wilcoxon zero_method='wilcox' by excluding exact zero differences.
    d = d[d != 0.0]

    if d.size == 0:
        return 0.0

    ranks = rankdata(np.abs(d), method="average")
    positive = float(ranks[d > 0].sum())
    negative = float(ranks[d < 0].sum())
    denominator = positive + negative

    if denominator == 0.0:
        return 0.0

    return (positive - negative) / denominator


# =============================================================================
# 3. LOAD AND VERIFY THE THREE FROZEN SOURCES
# =============================================================================

def load_combined() -> tuple[pd.DataFrame, dict[str, str]]:
    """
    Create the 14,400-row four-topology table in memory only.

    Returns
    -------
    combined : pandas.DataFrame
        Historical DMGSO + three revision-only topology datasets.
    hashes : dict
        Verified SHA-256 values for provenance reporting.
    """
    hashes = {
        "historical_D2_D20": verify_hash(
            HIST_D20_FILE,
            EXPECTED_HASH_D20,
            "Historical D2-D20 source",
        ),
        "historical_D40": verify_hash(
            HIST_D40_FILE,
            EXPECTED_HASH_D40,
            "Historical D40 source",
        ),
        "revision_topology": verify_hash(
            REV_TOPO_FILE,
            EXPECTED_HASH_REV,
            "Revision topology source",
        ),
    }

    d20 = pd.read_csv(HIST_D20_FILE)
    d40 = pd.read_csv(HIST_D40_FILE)
    rev = pd.read_csv(REV_TOPO_FILE)

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
        raise RuntimeError(
            f"Historical DMGSO subset has {len(hist)} rows; "
            f"expected {EXPECTED_PER_TOPOLOGY}."
        )

    if len(rev) != 3 * EXPECTED_PER_TOPOLOGY:
        raise RuntimeError(
            f"Revision topology source has {len(rev)} rows; expected 10800."
        )

    combined = pd.concat(
        [hist, rev],
        ignore_index=True,
        sort=False,
    )

    if len(combined) != EXPECTED_TOTAL:
        raise RuntimeError(
            f"Combined design has {len(combined)} rows; expected {EXPECTED_TOTAL}."
        )

    # Full pairing safety check repeated here even though the separate formal
    # audit has already passed.
    if combined.duplicated(["topology"] + PAIR_KEY).any():
        raise RuntimeError("Duplicate four-topology experiment key detected.")

    blocks = combined.groupby(PAIR_KEY)["topology"].agg(
        count="count",
        nunique="nunique",
    )

    if len(blocks) != 3600:
        raise RuntimeError(
            f"Expected 3600 four-way paired blocks, found {len(blocks)}."
        )

    if not ((blocks["count"] == 4) & (blocks["nunique"] == 4)).all():
        raise RuntimeError("Incomplete four-topology paired block detected.")

    if sorted(combined["topology"].astype(str).unique()) != sorted(TOPOLOGIES):
        raise RuntimeError("Unexpected combined topology set.")

    if sorted(pd.to_numeric(combined["dimension"]).astype(int).unique()) != DIMENSIONS:
        raise RuntimeError("Unexpected combined dimension set.")

    if sorted(pd.to_numeric(combined["function_id"]).astype(int).unique()) != FUNCTIONS:
        raise RuntimeError("Unexpected combined function set.")

    if sorted(pd.to_numeric(combined["instance_id"]).astype(int).unique()) != INSTANCES:
        raise RuntimeError("Unexpected combined instance set.")

    if sorted(pd.to_numeric(combined["run_id"]).astype(int).unique()) != RUN_IDS:
        raise RuntimeError("Unexpected combined run_id set.")

    if not (combined["status"].astype(str).str.lower() == "ok").all():
        raise RuntimeError("Non-ok combined row detected.")

    if (
        pd.to_numeric(combined["fe_used"])
        > pd.to_numeric(combined["FE_budget"])
    ).any():
        raise RuntimeError("Combined FE-budget violation detected.")

    return combined, hashes


# =============================================================================
# 4. FUNCTION-LEVEL AGGREGATION
# =============================================================================

def function_level(combined: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate 15 instances x 5 deterministic configurations to function level.

    The resulting 24 x 4 matrix at each dimension is the inferential dataset.
    """
    out = (
        combined.groupby(
            ["dimension", "function_id", "topology"],
            as_index=False,
        )
        .agg(
            n_raw_observations=("error_corrected", "count"),
            error_median=("error_corrected", "median"),
            error_mean=("error_corrected", "mean"),
            error_q25=("error_corrected", lambda x: x.quantile(0.25)),
            error_q75=("error_corrected", lambda x: x.quantile(0.75)),
            fe_used_median=("fe_used", "median"),
            wall_time_median=("wall_time_sec", "median"),
            n_accept_median=("n_accept", "median"),
            n_reloc_median=("n_reloc", "median"),
        )
    )

    expected_rows = len(DIMENSIONS) * len(FUNCTIONS) * len(TOPOLOGIES)

    if len(out) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} function-level rows, found {len(out)}."
        )

    if not (out["n_raw_observations"] == 75).all():
        raise RuntimeError(
            "Each Dimension x Function x Topology cell must contain "
            "15 x 5 = 75 raw observations."
        )

    out["rank_within_function"] = (
        out.groupby(["dimension", "function_id"])["error_median"]
        .rank(method="average", ascending=True)
    )

    return out


# =============================================================================
# 5. FRIEDMAN / RANK SUMMARY
# =============================================================================

def rank_analysis(function_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for D in DIMENSIONS:
        sub = function_df[function_df["dimension"] == D]

        pivot = sub.pivot(
            index="function_id",
            columns="topology",
            values="error_median",
        )[TOPOLOGIES]

        if pivot.shape != (24, 4):
            raise RuntimeError(
                f"Unexpected four-topology Friedman matrix at D={D}: "
                f"{pivot.shape}"
            )

        statistic, p_value = friedmanchisquare(
            *(pivot[topology].to_numpy(float) for topology in TOPOLOGIES)
        )

        n = pivot.shape[0]
        k = pivot.shape[1]
        kendall_w = float(statistic) / (n * (k - 1))

        summary = (
            sub.groupby("topology", as_index=False)
            .agg(
                average_rank=("rank_within_function", "mean"),
                median_rank=("rank_within_function", "median"),
                rank1_count=(
                    "rank_within_function",
                    lambda s: int(np.sum(np.isclose(s, 1.0))),
                ),
            )
        )

        for _, row in summary.iterrows():
            rows.append(
                {
                    "dimension": D,
                    "topology": row["topology"],
                    "average_rank": float(row["average_rank"]),
                    "median_rank": float(row["median_rank"]),
                    "rank1_count": int(row["rank1_count"]),
                    "friedman_statistic": float(statistic),
                    "friedman_p": float(p_value),
                    "kendall_W": kendall_w,
                    "friedman_significant_0.05": bool(p_value < ALPHA),
                }
            )

    return pd.DataFrame(rows)


# =============================================================================
# 6. PRE-SPECIFIED PAIRWISE CONTRASTS
# =============================================================================

def pairwise_analysis(function_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for D in DIMENSIONS:
        sub = function_df[function_df["dimension"] == D]

        pivot = sub.pivot(
            index="function_id",
            columns="topology",
            values="error_median",
        )[TOPOLOGIES]

        for old, new in PLANNED_CONTRASTS:
            old_values = pivot[old].to_numpy(float)
            new_values = pivot[new].to_numpy(float)

            improvement = old_values - new_values

            if np.all(improvement == 0.0):
                statistic = 0.0
                p_value = 1.0
            else:
                result = wilcoxon(
                    improvement,
                    zero_method="wilcox",
                    alternative="two-sided",
                    correction=False,
                    method="auto",
                )
                statistic = float(result.statistic)
                p_value = float(result.pvalue)

            tied = np.isclose(
                old_values,
                new_values,
                rtol=1e-9,
                atol=1e-12,
            )

            wins = int(((new_values < old_values) & (~tied)).sum())
            losses = int(((new_values > old_values) & (~tied)).sum())
            ties = int(tied.sum())

            rbc = rank_biserial_from_difference(improvement)

            direction = (
                "favors new topology"
                if rbc > 0
                else "favors old topology"
                if rbc < 0
                else "neutral"
            )

            rows.append(
                {
                    "dimension": D,
                    "comparison": f"{old} -> {new}",
                    "old_topology": old,
                    "new_topology": new,
                    "n_functions": len(improvement),
                    "wilcoxon_statistic": statistic,
                    "p_value_raw": p_value,
                    "rank_biserial_correlation": rbc,
                    "effect_direction": direction,
                    "wins_new_better": wins,
                    "ties": ties,
                    "losses_new_worse": losses,
                }
            )

    out = pd.DataFrame(rows)

    # Primary correction: four pre-specified contrasts inside each dimension.
    out["p_value_holm_within_dimension"] = np.nan

    for D in DIMENSIONS:
        mask = out["dimension"] == D

        out.loc[mask, "p_value_holm_within_dimension"] = holm_adjust(
            out.loc[mask, "p_value_raw"].to_numpy(float)
        )

    # Sensitivity correction: all eight dimension-specific tests together.
    out["p_value_holm_global_eight"] = holm_adjust(
        out["p_value_raw"].to_numpy(float)
    )

    out["significant_holm_within_dimension_0.05"] = (
        out["p_value_holm_within_dimension"] < ALPHA
    )

    out["significant_holm_global_eight_0.05"] = (
        out["p_value_holm_global_eight"] < ALPHA
    )

    return out


# =============================================================================
# 7. TOPOLOGY COST / STRUCTURE SUMMARY
# =============================================================================

def topology_structure(D: int, topology: str) -> tuple[int, int, int, int]:
    """
    Return:
        stored_directions,
        unique_directions,
        duplicate_directions,
        normal_iteration_fe
    """
    if topology == "HISTORICAL":
        stored = 2 * D + 8
        unique = 2 * D + 2
        duplicates = 6
        normal_fe = 2 * stored + 1
        return stored, unique, duplicates, normal_fe

    if topology == "AXIS_ONLY":
        stored = 2 * D
        unique = stored
        duplicates = 0
        normal_fe = 2 * stored + 1
        return stored, unique, duplicates, normal_fe

    if topology == "FIXED8_UNIQUE":
        stored = 2 * D + 8
        unique = stored
        duplicates = 0
        normal_fe = 2 * stored + 1
        return stored, unique, duplicates, normal_fe

    if topology == "DIM_ADAPTIVE":
        stored = 3 * D
        unique = stored
        duplicates = 0
        normal_fe = 2 * stored + 1
        return stored, unique, duplicates, normal_fe

    raise ValueError(f"Unknown topology: {topology}")


def cost_summary(combined: pd.DataFrame) -> pd.DataFrame:
    observed = (
        combined.groupby(["dimension", "topology"], as_index=False)
        .agg(
            raw_records=("error_corrected", "count"),
            FE_budget=("FE_budget", "first"),
            fe_used_median=("fe_used", "median"),
            fe_used_mean=("fe_used", "mean"),
            wall_time_median_sec=("wall_time_sec", "median"),
            wall_time_mean_sec=("wall_time_sec", "mean"),
            n_accept_median=("n_accept", "median"),
            n_reloc_median=("n_reloc", "median"),
        )
    )

    metadata_rows = []

    for D in DIMENSIONS:
        for topology in TOPOLOGIES:
            stored, unique, duplicates, normal_fe = topology_structure(
                D,
                topology,
            )

            metadata_rows.append(
                {
                    "dimension": D,
                    "topology": topology,
                    "stored_directions": stored,
                    "unique_directions": unique,
                    "duplicate_directions": duplicates,
                    "normal_iteration_fe": normal_fe,
                    "direction_uniqueness_fraction": unique / stored,
                }
            )

    metadata = pd.DataFrame(metadata_rows)

    out = observed.merge(
        metadata,
        on=["dimension", "topology"],
        how="left",
        validate="one_to_one",
    )

    # Relative nominal sensing cost uses AXIS_ONLY as a dimension-specific
    # structural reference only; it is not a performance normalization.
    axis_cost = (
        out[out["topology"] == "AXIS_ONLY"]
        .set_index("dimension")["normal_iteration_fe"]
        .to_dict()
    )

    out["normal_iteration_fe_vs_axis_ratio"] = [
        row.normal_iteration_fe / axis_cost[int(row.dimension)]
        for row in out.itertuples(index=False)
    ]

    return out


# =============================================================================
# 8. MANIFEST AND REPORT
# =============================================================================

def build_manifest(hashes: dict[str, str]) -> str:
    lines = [
        "DMGSO FOUR-TOPOLOGY STATISTICS — REVISION MANIFEST",
        "=" * 72,
        "",
        "Frozen inputs:",
        f"Historical D2-D20 SHA-256: {hashes['historical_D2_D20']}",
        f"Historical D40 SHA-256:     {hashes['historical_D40']}",
        f"Revision topology SHA-256:  {hashes['revision_topology']}",
        "",
        "Combined design:",
        "- Topologies: HISTORICAL, AXIS_ONLY, FIXED8_UNIQUE, DIM_ADAPTIVE",
        "- Functions: BBOB F1-F24",
        "- Instances: 15 official instances",
        "- Dimensions: D=20 and D=40",
        "- run_id: 0-4",
        "- Budget: 1000 x D",
        "- Rows per topology: 3600",
        "- Combined in-memory rows: 14400",
        "",
        "Inferential statistical unit:",
        "BBOB benchmark function (n=24), analyzed separately at each dimension.",
        "",
        "Function-level aggregation:",
        "Median corrected error across 15 instances x 5 deterministic configurations.",
        "",
        "Omnibus:",
        "Friedman test across four topologies and Kendall's W.",
        "",
        "Planned paired contrasts:",
        "HISTORICAL -> AXIS_ONLY",
        "HISTORICAL -> FIXED8_UNIQUE",
        "AXIS_ONLY -> FIXED8_UNIQUE",
        "FIXED8_UNIQUE -> DIM_ADAPTIVE",
        "",
        "Pairwise inference:",
        "Wilcoxon signed-rank test across 24 paired function-level medians.",
        "Rank-biserial correlation; positive values favor NEW topology.",
        "Win/tie/loss reported for direction.",
        "",
        "Multiplicity:",
        "Primary Holm correction: four planned contrasts within each dimension.",
        "Sensitivity Holm correction: all eight dimension-specific tests.",
        "",
        "Raw-data policy:",
        "All frozen inputs are read-only. No merged raw CSV is created.",
        "=" * 72,
    ]

    return "\n".join(lines)


def build_report(
    rank_df: pd.DataFrame,
    pair_df: pd.DataFrame,
    cost_df: pd.DataFrame,
) -> str:
    lines = [
        "=" * 84,
        "DMGSO SENSING TOPOLOGY — COMBINED FOUR-TOPOLOGY STATISTICAL ANALYSIS",
        "=" * 84,
        "",
        "Inferential unit: BBOB function (n=24), separately at D=20 and D=40.",
        "Function-level value: median corrected error across 75 matched raw records.",
        "",
    ]

    for D in DIMENSIONS:
        rank_sub = rank_df[rank_df["dimension"] == D]
        first = rank_sub.iloc[0]

        lines.append(f"D={D} — OMNIBUS TOPOLOGY COMPARISON")
        lines.append(
            f"Friedman chi2={first['friedman_statistic']:.6g}, "
            f"p={first['friedman_p']:.6g}, "
            f"Kendall W={first['kendall_W']:.4f}"
        )

        lines.append("Average ranks (lower is better):")

        for _, row in rank_sub.sort_values("average_rank").iterrows():
            lines.append(
                f"  {row['topology']}: "
                f"average rank={row['average_rank']:.4f}, "
                f"median rank={row['median_rank']:.4f}, "
                f"rank-1 count={int(row['rank1_count'])}"
            )

        lines.append("")
        lines.append(f"D={D} — PRE-SPECIFIED PAIRED CONTRASTS")

        pair_sub = pair_df[pair_df["dimension"] == D]

        for _, row in pair_sub.iterrows():
            lines.append(
                f"  {row['comparison']}: "
                f"W/T/L={int(row['wins_new_better'])}/"
                f"{int(row['ties'])}/"
                f"{int(row['losses_new_worse'])}, "
                f"r_rb={row['rank_biserial_correlation']:.4f}, "
                f"{row['effect_direction']}, "
                f"p_raw={row['p_value_raw']:.6g}, "
                f"p_Holm(dim)={row['p_value_holm_within_dimension']:.6g}, "
                f"p_Holm(global8)={row['p_value_holm_global_eight']:.6g}"
            )

        lines.append("")

    lines.append("TOPOLOGY STRUCTURE AND NOMINAL NORMAL-ITERATION COST")

    for _, row in cost_df.sort_values(["dimension", "topology"]).iterrows():
        lines.append(
            f"  D={int(row['dimension'])}, {row['topology']}: "
            f"stored={int(row['stored_directions'])}, "
            f"unique={int(row['unique_directions'])}, "
            f"duplicates={int(row['duplicate_directions'])}, "
            f"normal_iteration_FE={int(row['normal_iteration_fe'])}, "
            f"FE_vs_axis={row['normal_iteration_fe_vs_axis_ratio']:.3f}, "
            f"median_wall_sec={row['wall_time_median_sec']:.4f}"
        )

    lines.extend(
        [
            "",
            "INTERPRETATION GUARDRAILS",
            "- The omnibus Friedman test is considered before broad topology claims.",
            "- Pairwise tests are limited to four pre-specified reviewer-driven contrasts.",
            "- Statistical significance is interpreted jointly with direction,",
            "  rank-biserial effect size, and win/tie/loss counts.",
            "- HISTORICAL and FIXED8_UNIQUE have the same stored direction count and",
            "  nominal normal-iteration FE cost; they differ in diagonal uniqueness.",
            "- DIM_ADAPTIVE increases directional coverage and nominal sensing cost.",
            "- Wall time is descriptive and is not treated as pure algorithmic overhead.",
            "- BBOB instances and deterministic run_id values are not inferential",
            "  replications.",
            "",
            "All frozen raw inputs remained read-only. No merged raw CSV was created.",
            "=" * 84,
        ]
    )

    return "\n".join(lines)


# =============================================================================
# 9. MAIN
# =============================================================================

def main() -> None:
    refuse_overwrite()

    combined, hashes = load_combined()

    function_df = function_level(combined)
    rank_df = rank_analysis(function_df)
    pair_df = pairwise_analysis(function_df)
    cost_df = cost_summary(combined)

    manifest = build_manifest(hashes)
    report = build_report(rank_df, pair_df, cost_df)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    OUT_MANIFEST.write_text(manifest + "\n", encoding="utf-8")
    function_df.to_csv(OUT_FUNCTION, index=False)
    rank_df.to_csv(OUT_RANK, index=False)
    pair_df.to_csv(OUT_PAIR, index=False)
    cost_df.to_csv(OUT_COST, index=False)
    OUT_REPORT.write_text(report + "\n", encoding="utf-8")

    print(report)
    print()
    print("[OK] Four-topology analysis outputs written to:")
    for path in OUTPUTS:
        print(f"  {path}")
    print("[OK] Frozen raw inputs remained read-only.")
    print("[OK] No merged raw CSV was created.")


if __name__ == "__main__":
    main()
