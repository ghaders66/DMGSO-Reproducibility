#!/usr/bin/env python3
"""
Build the complete function-level numerical results table requested during
peer review.

IMPORTANT
---------
- No optimizer is rerun.
- No frozen input is modified.
- Statistics are computed directly from the provenance-verified corrected
  run-level COCO/BBOB datasets.
- One output row represents:
      dimension × function_id × algorithm
- Each row must contain:
      15 BBOB instances × 5 run_id configurations = 75 observations.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
COCO_DIR = SCRIPT_DIR.parent

FROZEN_DIR = COCO_DIR / "data" / "frozen"
OUTPUT_DIR = SCRIPT_DIR / "outputs"

D2_D20_PATH = (
    FROZEN_DIR
    / "coco_bbob_deterministic_D2_D20_corrected_FROZEN.csv"
)

D40_PATH = (
    FROZEN_DIR
    / "coco_bbob_deterministic_D40_corrected_FROZEN.csv"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "08_function_level_complete_numerical_results.csv"
)


# ---------------------------------------------------------------------
# Expected experimental structure
# ---------------------------------------------------------------------

EXPECTED_DIMENSIONS = {2, 5, 10, 20, 40}
EXPECTED_FUNCTIONS = set(range(1, 25))
EXPECTED_INSTANCES = {
    1, 2, 3, 4, 5,
    71, 72, 73, 74, 75,
    76, 77, 78, 79, 80,
}
EXPECTED_RUN_IDS = set(range(5))

EXPECTED_OBSERVATIONS_PER_ROW = (
    len(EXPECTED_INSTANCES) * len(EXPECTED_RUN_IDS)
)  # 75


# ---------------------------------------------------------------------
# Load frozen corrected observations
# ---------------------------------------------------------------------

frames = [
    pd.read_csv(D2_D20_PATH),
    pd.read_csv(D40_PATH),
]

df = pd.concat(frames, ignore_index=True)

required = {
    "algorithm",
    "function_id",
    "instance_id",
    "dimension",
    "run_id",
    "fe_used",
    "error_corrected",
}

missing = required.difference(df.columns)
if missing:
    raise RuntimeError(
        f"Frozen corrected data are missing required columns: "
        f"{sorted(missing)}"
    )

if len(df) != 45000:
    raise RuntimeError(
        f"Expected 45000 frozen corrected records, found {len(df)}."
    )

for col in [
    "function_id",
    "instance_id",
    "dimension",
    "run_id",
    "fe_used",
    "error_corrected",
]:
    df[col] = pd.to_numeric(df[col], errors="raise")

df["function_id"] = df["function_id"].astype(int)
df["instance_id"] = df["instance_id"].astype(int)
df["dimension"] = df["dimension"].astype(int)
df["run_id"] = df["run_id"].astype(int)

if set(df["dimension"].unique()) != EXPECTED_DIMENSIONS:
    raise RuntimeError("Unexpected or incomplete dimension set.")

if set(df["function_id"].unique()) != EXPECTED_FUNCTIONS:
    raise RuntimeError("Unexpected or incomplete function set.")

if set(df["instance_id"].unique()) != EXPECTED_INSTANCES:
    raise RuntimeError("Unexpected or incomplete instance set.")

if set(df["run_id"].unique()) != EXPECTED_RUN_IDS:
    raise RuntimeError("Unexpected or incomplete run_id set.")

if not np.isfinite(df["error_corrected"]).all():
    raise RuntimeError("Non-finite corrected errors detected.")

if not np.isfinite(df["fe_used"]).all():
    raise RuntimeError("Non-finite FE counts detected.")


# ---------------------------------------------------------------------
# Function-level descriptive aggregation
# ---------------------------------------------------------------------

group_cols = [
    "dimension",
    "function_id",
    "algorithm",
]

summary = (
    df.groupby(group_cols, sort=True)
      .agg(
          instances=("instance_id", "nunique"),
          configurations=("run_id", "nunique"),
          observations=("error_corrected", "size"),

          error_mean=("error_corrected", "mean"),
          error_median=("error_corrected", "median"),
          error_std=("error_corrected", "std"),
          error_q25=("error_corrected", lambda x: x.quantile(0.25)),
          error_q75=("error_corrected", lambda x: x.quantile(0.75)),
          error_min=("error_corrected", "min"),
          error_max=("error_corrected", "max"),

          fe_mean=("fe_used", "mean"),
          fe_median=("fe_used", "median"),
          fe_min=("fe_used", "min"),
          fe_max=("fe_used", "max"),
      )
      .reset_index()
)

summary["error_iqr"] = (
    summary["error_q75"] - summary["error_q25"]
)

# Standard COCO/BBOB budget used in this study: 1000 D.
summary["fe_budget"] = 1000 * summary["dimension"]

# Put reviewer-requested fields in a transparent order.
column_order = [
    "dimension",
    "function_id",
    "algorithm",
    "instances",
    "configurations",
    "observations",
    "error_mean",
    "error_median",
    "error_std",
    "error_q25",
    "error_q75",
    "error_iqr",
    "error_min",
    "error_max",
    "fe_budget",
    "fe_mean",
    "fe_median",
    "fe_min",
    "fe_max",
]

summary = summary[column_order]


# ---------------------------------------------------------------------
# Structural audit
# ---------------------------------------------------------------------

if not (summary["instances"] == 15).all():
    bad = summary.loc[summary["instances"] != 15]
    raise RuntimeError(
        "At least one function-level row does not contain 15 instances:\n"
        f"{bad.to_string(index=False)}"
    )

if not (summary["configurations"] == 5).all():
    bad = summary.loc[summary["configurations"] != 5]
    raise RuntimeError(
        "At least one function-level row does not contain 5 run_id "
        f"configurations:\n{bad.to_string(index=False)}"
    )

if not (
    summary["observations"] == EXPECTED_OBSERVATIONS_PER_ROW
).all():
    bad = summary.loc[
        summary["observations"] != EXPECTED_OBSERVATIONS_PER_ROW
    ]
    raise RuntimeError(
        "At least one function-level row does not contain 75 observations:\n"
        f"{bad.to_string(index=False)}"
    )

expected_rows = (
    len(EXPECTED_DIMENSIONS)
    * len(EXPECTED_FUNCTIONS)
    * df["algorithm"].nunique()
)

if len(summary) != expected_rows:
    raise RuntimeError(
        f"Expected {expected_rows} summary rows, found {len(summary)}."
    )


# ---------------------------------------------------------------------
# Write output
# ---------------------------------------------------------------------

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

summary.to_csv(
    OUTPUT_PATH,
    index=False,
    float_format="%.17g",
)

print("[PASS] Frozen corrected run-level inputs loaded.")
print(f"[PASS] Input records: {len(df)}")
print(
    "[PASS] Function-level structure: "
    "15 instances × 5 run_id configurations = 75 observations per row."
)
print(f"[PASS] Algorithms: {sorted(df['algorithm'].unique())}")
print(f"[PASS] Dimensions: {sorted(df['dimension'].unique())}")
print(f"[PASS] Functions: {df['function_id'].nunique()}")
print(f"[PASS] Output rows: {len(summary)}")
print(f"[PASS] Output written to:\n{OUTPUT_PATH}")
