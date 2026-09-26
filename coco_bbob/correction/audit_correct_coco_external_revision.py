from pathlib import Path
import sys
import numpy as np
import pandas as pd

# ============================================================
# Reviewer-driven COCO/BBOB audit + f* correction
# Historical raw files are READ-ONLY. No optimizer is rerun.
# No historical file is overwritten or appended.
# ============================================================

RAW_D20 = Path("/mnt/d/PHD/MIDO-GS/DMGSO_Project/KSB_DMGSO/outputs/summaries/tables/COCO_20/coco_bbob_deterministic_run_summary_D_20.csv")
RAW_D40 = Path("/mnt/d/PHD/MIDO-GS/DMGSO_Project/KSB_DMGSO/outputs/summaries/tables/COCO_40/coco_bbob_deterministic_run_summary_D40.csv")

# Official COCO/BBOB f* table previously extracted with cocoex 2.8.2
FOPT_FILE = Path("/mnt/d/PHD/DMGSO_Information Scince/Revision/coco_bbob/processed/bbob_fopt_cocoex_2.8.2.csv")

OUT_DIR = Path("/mnt/d/PHD/DMGSO_Information Scince/Revision/coco_bbob/statistics")
OUT_D20 = OUT_DIR / "coco_bbob_deterministic_D2_D20_corrected.csv"
OUT_D40 = OUT_DIR / "coco_bbob_deterministic_D40_corrected.csv"
AUDIT_LOG = OUT_DIR / "coco_bbob_external_audit_and_fopt_correction.txt"

KEY = ["algorithm", "function_id", "instance_id", "dimension", "run_id"]
PROBLEM_KEY = ["function_id", "instance_id", "dimension"]
ERROR_TOL = 1e-8
EXPECTED_ALGORITHMS = {"DMGSO", "NELDER-MEAD", "POWELL", "COBYLA", "COMPASS"}
EXPECTED_FUNCTIONS = set(range(1, 25))
EXPECTED_INSTANCES = {1, 2, 3, 4, 5, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80}
EXPECTED_RUNS = set(range(5))


def fail(msg: str):
    raise RuntimeError(msg)


def require_columns(df: pd.DataFrame, cols, label):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        fail(f"{label}: missing required columns: {missing}")


def audit_raw(df: pd.DataFrame, label: str, expected_dims, expected_rows: int):
    require_columns(
        df,
        KEY + ["FE_budget", "fe_used", "best_f", "status"],
        label,
    )

    if len(df) != expected_rows:
        fail(f"{label}: expected {expected_rows} rows, found {len(df)}")

    if set(df["dimension"].astype(int).unique()) != set(expected_dims):
        fail(f"{label}: unexpected dimensions {sorted(df['dimension'].unique())}")

    if set(df["algorithm"].astype(str).unique()) != EXPECTED_ALGORITHMS:
        fail(f"{label}: unexpected algorithm set {sorted(df['algorithm'].unique())}")

    if set(df["function_id"].astype(int).unique()) != EXPECTED_FUNCTIONS:
        fail(f"{label}: incomplete/unexpected function IDs")

    if set(df["instance_id"].astype(int).unique()) != EXPECTED_INSTANCES:
        fail(f"{label}: incomplete/unexpected instance IDs")

    if set(df["run_id"].astype(int).unique()) != EXPECTED_RUNS:
        fail(f"{label}: incomplete/unexpected run IDs")

    dup = int(df.duplicated(KEY).sum())
    if dup:
        fail(f"{label}: {dup} duplicate experiment keys found")

    best_f = pd.to_numeric(df["best_f"], errors="coerce")
    if (~np.isfinite(best_f.to_numpy())).any():
        fail(f"{label}: non-finite best_f detected")

    fe_used = pd.to_numeric(df["fe_used"], errors="coerce")
    fe_budget = pd.to_numeric(df["FE_budget"], errors="coerce")
    if (~np.isfinite(fe_used.to_numpy())).any() or (~np.isfinite(fe_budget.to_numpy())).any():
        fail(f"{label}: non-finite FE fields detected")
    if (fe_used > fe_budget).any():
        fail(f"{label}: FE budget violation detected")

    if not (df["status"].astype(str).str.lower() == "ok").all():
        fail(f"{label}: non-ok status detected")

    # Completeness at each dimension: 24 f x 15 instances x 5 runs x 5 algorithms = 9000
    by_dim = df.groupby("dimension", observed=True).size().to_dict()
    for d in expected_dims:
        if int(by_dim.get(d, 0)) != 9000:
            fail(f"{label}: D={d} expected 9000 rows, found {by_dim.get(d, 0)}")

    return {
        "rows": len(df),
        "duplicates": dup,
        "fe_violations": int((fe_used > fe_budget).sum()),
        "nonfinite_best_f": int((~np.isfinite(best_f.to_numpy())).sum()),
    }


def normalize_fopt_table(fopt: pd.DataFrame) -> pd.DataFrame:
    # Accept common naming variants without altering source file.
    rename = {}
    aliases = {
        "function": "function_id",
        "func_id": "function_id",
        "instance": "instance_id",
        "inst_id": "instance_id",
        "dim": "dimension",
        "D": "dimension",
        "fopt": "f_opt_official",
        "Fopt": "f_opt_official",
        "f_opt": "f_opt_official",
        "optimal_value": "f_opt_official",
    }
    for old, new in aliases.items():
        if old in fopt.columns and new not in fopt.columns:
            rename[old] = new
    fopt = fopt.rename(columns=rename).copy()

    require_columns(fopt, PROBLEM_KEY, "FOPT")

    if "f_opt_official" not in fopt.columns:
        # If there is exactly one plausible numeric value column beyond keys, use it.
        candidates = [c for c in fopt.columns if c not in PROBLEM_KEY]
        numeric_candidates = [c for c in candidates if pd.api.types.is_numeric_dtype(fopt[c])]
        if len(numeric_candidates) == 1:
            fopt = fopt.rename(columns={numeric_candidates[0]: "f_opt_official"})
        else:
            fail(
                "FOPT: cannot identify official optimum column. "
                f"Columns are: {list(fopt.columns)}"
            )

    fopt = fopt[PROBLEM_KEY + ["f_opt_official"]].copy()
    for c in PROBLEM_KEY:
        fopt[c] = pd.to_numeric(fopt[c], errors="raise").astype(int)
    fopt["f_opt_official"] = pd.to_numeric(fopt["f_opt_official"], errors="raise")

    if fopt.duplicated(PROBLEM_KEY).any():
        fail("FOPT: duplicate problem keys detected")
    if (~np.isfinite(fopt["f_opt_official"].to_numpy())).any():
        fail("FOPT: non-finite official optimum detected")

    return fopt


def correct_errors(df: pd.DataFrame, fopt: pd.DataFrame, label: str) -> pd.DataFrame:
    out = df.merge(
        fopt,
        on=PROBLEM_KEY,
        how="left",
        validate="many_to_one",
        indicator=True,
    )

    missing = int((out["_merge"] != "both").sum())
    if missing:
        fail(f"{label}: official f* missing for {missing} rows")
    out = out.drop(columns="_merge")

    out["error_corrected"] = pd.to_numeric(out["best_f"], errors="raise") - out["f_opt_official"]

    materially_negative = out["error_corrected"] < -ERROR_TOL
    if materially_negative.any():
        bad = out.loc[materially_negative, KEY + ["best_f", "f_opt_official", "error_corrected"]].head(10)
        fail(
            f"{label}: {int(materially_negative.sum())} materially negative corrected errors detected.\n"
            + bad.to_string(index=False)
        )

    # IMPORTANT: no clipping/flooring is performed.
    return out


def main():
    # Fail before doing any work if source files are unavailable.
    for p, label in [(RAW_D20, "RAW_D20"), (RAW_D40, "RAW_D40"), (FOPT_FILE, "FOPT_FILE")]:
        if not p.is_file():
            fail(f"{label} not found: {p}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Never overwrite revision outputs silently.
    existing = [p for p in [OUT_D20, OUT_D40, AUDIT_LOG] if p.exists()]
    if existing:
        fail(
            "Refusing to overwrite existing revision outputs:\n  "
            + "\n  ".join(str(p) for p in existing)
            + "\nRename/archive them first if a new revision is intentionally required."
        )

    d20 = pd.read_csv(RAW_D20)
    d40 = pd.read_csv(RAW_D40)
    fopt_raw = pd.read_csv(FOPT_FILE)

    a20 = audit_raw(d20, "D2-D20", {2, 5, 10, 20}, 36000)
    a40 = audit_raw(d40, "D40", {40}, 9000)

    # Cross-file key overlap must be zero because dimensions are complementary.
    overlap = pd.merge(d20[KEY], d40[KEY], on=KEY, how="inner")
    if len(overlap):
        fail(f"Cross-file overlap detected: {len(overlap)} experiment keys")

    if len(d20) + len(d40) != 45000:
        fail(f"Combined expected 45000 rows, found {len(d20) + len(d40)}")

    fopt = normalize_fopt_table(fopt_raw)

    # Ensure f* coverage includes every problem represented in these two raw files.
    represented = pd.concat([d20[PROBLEM_KEY], d40[PROBLEM_KEY]], ignore_index=True).drop_duplicates()
    coverage = represented.merge(fopt, on=PROBLEM_KEY, how="left", indicator=True)
    missing_problems = coverage.loc[coverage["_merge"] != "both", PROBLEM_KEY]
    if len(missing_problems):
        fail(f"FOPT coverage missing for {len(missing_problems)} represented BBOB problems")

    c20 = correct_errors(d20, fopt, "D2-D20")
    c40 = correct_errors(d40, fopt, "D40")

    # Preserve all historical columns unchanged; append only corrected fields.
    # Historical f_opt/error columns remain present for provenance but must not be used downstream.
    c20.to_csv(OUT_D20, index=False)
    c40.to_csv(OUT_D40, index=False)

    neg20 = int((c20["error_corrected"] < -ERROR_TOL).sum())
    neg40 = int((c40["error_corrected"] < -ERROR_TOL).sum())

    lines = [
        "COCO/BBOB EXTERNAL COMPARISON — REVIEWER-DRIVEN AUDIT & f* CORRECTION",
        "====================================================================",
        "",
        "Historical raw files: READ-ONLY (not overwritten, not appended)",
        f"RAW D2-D20: {RAW_D20}",
        f"RAW D40:    {RAW_D40}",
        f"Official f*: {FOPT_FILE}",
        "",
        "AUDIT RESULTS",
        f"D2-D20 rows: {a20['rows']} (expected 36000)",
        f"D40 rows:    {a40['rows']} (expected 9000)",
        f"Combined:    {a20['rows'] + a40['rows']} (expected 45000)",
        f"Duplicates D2-D20: {a20['duplicates']}",
        f"Duplicates D40:    {a40['duplicates']}",
        f"FE violations D2-D20: {a20['fe_violations']}",
        f"FE violations D40:    {a40['fe_violations']}",
        f"Non-finite best_f D2-D20: {a20['nonfinite_best_f']}",
        f"Non-finite best_f D40:    {a40['nonfinite_best_f']}",
        f"Materially negative corrected errors D2-D20 (< -{ERROR_TOL:g}): {neg20}",
        f"Materially negative corrected errors D40 (< -{ERROR_TOL:g}): {neg40}",
        "",
        "CORRECTION POLICY",
        "error_corrected = best_f - f_opt_official",
        "No clipping/flooring applied.",
        "Historical f_opt and error columns retained only for provenance.",
        "",
        "OUTPUTS",
        str(OUT_D20),
        str(OUT_D40),
        "",
        "FINAL STATUS: AUDIT PASSED",
    ]
    AUDIT_LOG.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
