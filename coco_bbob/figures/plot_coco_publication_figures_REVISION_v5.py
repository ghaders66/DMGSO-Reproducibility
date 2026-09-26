# -*- coding: utf-8 -*-
"""
REVISION plotter preserving the submitted COCO/BBOB figure style.
Uses frozen corrected errors and function-instance problem aggregation.

Input:
    outputs/summaries/tables/coco_bbob_deterministic_run_summary.csv

Notes:
    The current version assumes that all COCO/BBOB dimensions, including
    D = 2, 5, 10, 20, and 40, are already stored in one unified run-summary
    CSV file. No separate COCO_20 or COCO_40 input files are required.

Outputs:
    outputs/figures/coco_publication/

All figures:
    PNG, 300 dpi, Times New Roman, fixed algorithm colors.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]

TABLE_DIR = ROOT / "outputs" / "summaries" / "tables"
OUT_DIR = ROOT / "outputs" / "figures" / "coco_publication"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Frozen corrected inputs used for the revision analysis.
# The script is intentionally colocated with the frozen CSV files, so this
# works directly in WSL and remains portable if the folder is moved.
FROZEN_DIR = Path(__file__).resolve().parent
RUN_D2_D20 = FROZEN_DIR / "coco_bbob_deterministic_D2_D20_corrected_FROZEN.csv"
RUN_D40 = FROZEN_DIR / "coco_bbob_deterministic_D40_corrected_FROZEN.csv"
RUN_FILES = [RUN_D2_D20, RUN_D40]

# Keep revision figures separate from submitted outputs.
# frozen_20260911 -> backup -> coco_bbob
COCO_ROOT = FROZEN_DIR.parents[1]
OUT_DIR = COCO_ROOT / "statistics" / "external_comparison" / "figures_revision_submitted_style"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Fixed academic style
# ------------------------------------------------------------
def configure_times_new_roman():
    """Register Times New Roman explicitly to avoid Matplotlib font fallback.

    Works with native Windows Python and WSL when the Windows font directory is
    mounted. If Times New Roman is unavailable, a warning is printed and
    Liberation Serif is used as a metric-compatible fallback.
    """
    candidates = [
        Path(r"C:/Windows/Fonts/times.ttf"),
        Path(r"C:/Windows/Fonts/timesbd.ttf"),
        Path(r"C:/Windows/Fonts/timesi.ttf"),
        Path(r"C:/Windows/Fonts/timesbi.ttf"),
        Path("/mnt/c/Windows/Fonts/times.ttf"),
        Path("/mnt/c/Windows/Fonts/timesbd.ttf"),
        Path("/mnt/c/Windows/Fonts/timesi.ttf"),
        Path("/mnt/c/Windows/Fonts/timesbi.ttf"),
    ]
    found = []
    for fp in candidates:
        if fp.exists():
            try:
                font_manager.fontManager.addfont(str(fp))
                found.append(fp)
            except Exception as exc:
                print(f"[WARN] Could not register font {fp}: {exc}")
    if found:
        family = "Times New Roman"
        print(f"[OK] Times New Roman registered from: {found[0]}")
    else:
        family = "Liberation Serif"
        print("[WARN] Times New Roman font file not found; using Liberation Serif fallback.")
    return family


FONT_FAMILY = configure_times_new_roman()


plt.rcParams.update({
    "font.family": FONT_FAMILY,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 12,
    "axes.linewidth": 0.8,
    "grid.linewidth": 0.4,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.facecolor": "white",
})


ALG_COLORS = {
    "DMGSO": "blue",
    "POWELL": "red",
    "COMPASS": "green",
    "COBYLA": "orange",
    "NELDER-MEAD": "purple",
}

ALG_ORDER = [
    "DMGSO",
    "POWELL",
    "COMPASS",
    "COBYLA",
    "NELDER-MEAD",
]


def savefig(fig, filename: str):
    out = OUT_DIR / filename
    fig.savefig(
        out,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
        pad_inches=0.05,
    )
    plt.close(fig)
    print(f"[OK] saved: {out}")


def _read_if_exists(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    print(f"[WARN] file not found: {path}")
    return pd.DataFrame()


def load_run_data() -> pd.DataFrame:
    """Load and validate the two frozen corrected COCO/BBOB datasets."""
    frames = []
    for path in RUN_FILES:
        if not path.exists():
            raise FileNotFoundError(f"Frozen corrected input not found:\n{path}")
        print(f"[OK] Loading frozen corrected input: {path}")
        frames.append(pd.read_csv(path))

    df = pd.concat(frames, ignore_index=True)
    required_cols = [
        "suite", "algorithm", "function_id", "instance_id", "dimension",
        "run_id", "error_corrected", "fe_used", "wall_time_sec", "status"
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required corrected-data columns: {missing}")

    df = df[df["status"].astype(str).str.lower() == "ok"].copy()
    df["algorithm"] = df["algorithm"].astype(str).str.strip().replace({
        "Nelder-Mead": "NELDER-MEAD", "Nelder–Mead": "NELDER-MEAD",
        "Powell": "POWELL", "Compass": "COMPASS", "Cobyla": "COBYLA"
    })
    df = df.drop_duplicates(
        subset=["suite", "algorithm", "function_id", "instance_id", "dimension", "run_id"],
        keep="last",
    )
    for col in ["function_id", "instance_id", "dimension", "run_id",
                "error_corrected", "fe_used", "wall_time_sec", "n_accept", "n_reloc"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["function_id", "instance_id", "dimension", "run_id",
                           "algorithm", "error_corrected"])
    df["error"] = np.maximum(df["error_corrected"].astype(float), 0.0)
    df["log_error"] = np.log10(np.maximum(df["error"], 1e-12))

    print(f"[OK] Loaded corrected rows: {len(df):,}")
    print(f"[OK] Dimensions: {sorted(df['dimension'].astype(int).unique())}")
    print(f"[OK] Algorithms: {[a for a in ALG_ORDER if a in set(df['algorithm'])]}")
    return df


def build_problem_level_data(run_df: pd.DataFrame) -> pd.DataFrame:
    """Median across five run_id observations for each function-instance problem."""
    agg = {"error": ("error", "median")}
    if "fe_used" in run_df.columns:
        agg["fe_used"] = ("fe_used", "median")
    if "wall_time_sec" in run_df.columns:
        agg["wall_time_sec"] = ("wall_time_sec", "median")
    if "n_accept" in run_df.columns:
        agg["n_accept"] = ("n_accept", "mean")
    if "n_reloc" in run_df.columns:
        agg["n_reloc"] = ("n_reloc", "mean")
    return (run_df.groupby(["suite", "algorithm", "function_id", "instance_id", "dimension"],
                           as_index=False).agg(**agg))


def load_dimension_data(run_df: pd.DataFrame) -> pd.DataFrame:
    """Dimension summaries after the frozen five-observation problem aggregation."""
    problem_df = build_problem_level_data(run_df)
    agg = {
        "problems": ("error", "count"),
        "functions": ("function_id", "nunique"),
        "instances": ("instance_id", "nunique"),
        "error_mean": ("error", "mean"),
        "error_median": ("error", "median"),
        "error_std": ("error", "std"),
        "error_min": ("error", "min"),
        "error_max": ("error", "max"),
    }
    if "fe_used" in problem_df.columns:
        agg["fe_used_mean"] = ("fe_used", "mean")
        agg["fe_used_median"] = ("fe_used", "median")
    if "wall_time_sec" in problem_df.columns:
        agg["wall_time_mean"] = ("wall_time_sec", "mean")
        agg["wall_time_median"] = ("wall_time_sec", "median")
    if "n_accept" in problem_df.columns:
        agg["n_accept_mean"] = ("n_accept", "mean")
    if "n_reloc" in problem_df.columns:
        agg["n_reloc_mean"] = ("n_reloc", "mean")
    return (problem_df.groupby(["suite", "algorithm", "dimension"], as_index=False).agg(**agg))


# ------------------------------------------------------------
# Figure 1: Scalability curve
# ------------------------------------------------------------
def plot_scalability_curve(dim_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8.0, 5.2))

    for alg in ALG_ORDER:
        g = dim_df[dim_df["algorithm"] == alg].sort_values("dimension")

        if g.empty:
            continue

        ax.plot(
            g["dimension"],
            g["error_median"],
            marker="o",
            linewidth=2.0,
            markersize=4.5,
            color=ALG_COLORS.get(alg, "gray"),
            label=alg,
        )

    ax.set_yscale("log")
    ax.set_xlabel("Problem dimension")
    ax.set_ylabel("Median final error")
    ax.set_title("Scalability of deterministic methods on COCO/BBOB")
    ax.set_xticks(sorted(dim_df["dimension"].unique()))
    ax.grid(True, which="both", alpha=0.42)
    ax.legend(frameon=True, ncol=2)

    savefig(fig, "fig_coco_scalability_median_error.png")


# ------------------------------------------------------------
# Figure 2: FE efficiency by dimension
# ------------------------------------------------------------
def plot_fe_efficiency(dim_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8.0, 5.2))

    for alg in ALG_ORDER:
        g = dim_df[dim_df["algorithm"] == alg].sort_values("dimension")

        if g.empty:
            continue

        ax.plot(
            g["dimension"],
            g["fe_used_median"],
            marker="s",
            linewidth=2.0,
            markersize=4.5,
            color=ALG_COLORS.get(alg, "gray"),
            label=alg,
        )

    ax.set_xlabel("Problem dimension")
    ax.set_ylabel("Median function evaluations")
    ax.set_title("Evaluation-budget usage across COCO/BBOB dimensions")
    ax.set_xticks(sorted(dim_df["dimension"].unique()))
    ax.grid(True, alpha=0.42)
    ax.legend(frameon=True, ncol=2)

    savefig(fig, "fig_coco_fe_efficiency.png")


# ------------------------------------------------------------
# Figure 3: Performance profile
# ------------------------------------------------------------
def plot_performance_profile(run_df: pd.DataFrame):
    summary = build_problem_level_data(run_df)

    pivot = summary.pivot_table(
        index=["function_id", "instance_id", "dimension"],
        columns="algorithm",
        values="error",
    ).dropna()

    # Visualization-only floor prevents undefined ratios when corrected error is zero.
    pivot_for_ratio = pivot.clip(lower=1e-12)
    best = pivot_for_ratio.min(axis=1)
    ratios = pivot_for_ratio.div(best, axis=0)
    tau = np.logspace(0, 6, 400)

    fig, ax = plt.subplots(figsize=(8.0, 5.2))

    for alg in ALG_ORDER:
        if alg not in ratios.columns:
            continue

        values = ratios[alg].replace([np.inf, -np.inf], np.nan).dropna().values
        y = [(values <= t).mean() for t in tau]

        ax.plot(
            tau,
            y,
            linewidth=2.1,
            color=ALG_COLORS.get(alg, "gray"),
            label=alg,
        )

    ax.set_xscale("log")
    ax.set_xlabel(r"Performance ratio $\tau$")
    ax.set_ylabel("Proportion of problems")
    ax.set_title("Dolan--Moré performance profile on COCO/BBOB")
    ax.grid(True, which="both", alpha=0.42)
    ax.legend(frameon=True)

    savefig(fig, "fig_coco_performance_profile.png")


# ------------------------------------------------------------
# Figure 4: Rank heatmap
# ------------------------------------------------------------
def plot_rank_heatmap(run_df: pd.DataFrame):
    problem_df = build_problem_level_data(run_df)
    summary = (
        problem_df
        .groupby(["function_id", "dimension", "algorithm"], as_index=False)
        .agg(error_median=("error", "median"))
    )

    summary["rank"] = (
        summary
        .groupby(["function_id", "dimension"])["error_median"]
        .rank(method="min", ascending=True)
    )

    summary["problem"] = (
        "F" + summary["function_id"].astype(int).astype(str).str.zfill(2)
        + "-D" + summary["dimension"].astype(int).astype(str)
    )

    pivot = summary.pivot(
        index="problem",
        columns="algorithm",
        values="rank",
    )

    cols = [a for a in ALG_ORDER if a in pivot.columns]
    pivot = pivot[cols]

    def sort_key(idx):
        f = int(idx.split("-D")[0].replace("F", ""))
        d = int(idx.split("-D")[1])
        return d, f

    pivot = pivot.loc[sorted(pivot.index, key=sort_key)]

    fig, ax = plt.subplots(figsize=(7.8, 12.0))
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis_r")

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=35, ha="right")

    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=6)

    ax.set_xlabel("Algorithm")
    ax.set_ylabel("BBOB function-dimension pair")
    ax.set_title("Function-wise ranking across COCO/BBOB dimensions")

    cbar = fig.colorbar(im, ax=ax, shrink=0.82)
    cbar.set_label("Rank (1 = best)")

    savefig(fig, "fig_coco_rank_heatmap.png")


# ------------------------------------------------------------
# Figure 5: Robustness boxplot
# ------------------------------------------------------------
def plot_robustness_boxplot(run_df: pd.DataFrame):
    """Operational robustness: within-problem dispersion across the five observations.

    For each algorithm × dimension × function × instance block, compute the IQR of
    log10(corrected error) across run_id observations. A small numerical floor is
    used only to make exact-zero corrected errors finite on the log scale. Lower
    IQR therefore indicates greater within-problem stability.
    """
    algorithms = [a for a in ALG_ORDER if a in run_df["algorithm"].unique()]
    eps = 1e-12

    work = run_df.copy()
    work["log_error_robust"] = np.log10(np.maximum(work["error"].to_numpy(dtype=float), eps))

    keys = ["algorithm", "dimension", "function_id", "instance_id"]
    counts = work.groupby(keys, observed=True).size()
    if not (counts == 5).all():
        bad = counts[counts != 5]
        raise ValueError(
            "Robustness audit failed: expected exactly five observations per "
            f"algorithm-problem block; found {len(bad)} nonconforming blocks."
        )

    dispersion = (
        work.groupby(keys, observed=True)["log_error_robust"]
        .quantile([0.25, 0.75])
        .unstack()
        .reset_index()
    )
    dispersion["iqr_log_error"] = dispersion[0.75] - dispersion[0.25]

    # 5 algorithms × 5 dimensions × 24 functions × 15 instances
    # = 9,000 algorithm-problem blocks; each block contains exactly 5 observations.
    # Derive audit sizes directly from the frozen data to avoid reliance on
    # undeclared global constants.
    n_algorithms = run_df["algorithm"].nunique()
    n_dimensions = run_df["dimension"].nunique()
    n_functions = run_df["function_id"].nunique()
    n_instances = run_df["instance_id"].nunique()

    expected_per_algorithm = n_dimensions * n_functions * n_instances
    expected_blocks = n_algorithms * expected_per_algorithm
    if len(dispersion) != expected_blocks:
        raise ValueError(
            f"Robustness audit failed: expected {expected_blocks:,} algorithm-problem "
            f"blocks, found {len(dispersion):,}."
        )

    counts_by_algorithm = dispersion.groupby("algorithm", observed=True).size()
    bad_counts = counts_by_algorithm[counts_by_algorithm != expected_per_algorithm]
    if not bad_counts.empty:
        raise ValueError(
            "Robustness audit failed: expected "
            f"{expected_per_algorithm:,} problem blocks per algorithm; got "
            f"{counts_by_algorithm.to_dict()}."
        )

    print(
        f"[OK] Robustness audit: {len(dispersion):,} algorithm-problem blocks "
        f"({expected_per_algorithm:,} per algorithm), each based on five observations."
    )

    data = [
        dispersion.loc[dispersion["algorithm"] == alg, "iqr_log_error"].dropna().values
        for alg in algorithms
    ]

    fig, ax = plt.subplots(figsize=(8.0, 5.2))

    bp = ax.boxplot(
        data,
        tick_labels=algorithms,
        patch_artist=True,
        showfliers=False,
        widths=0.65,
        medianprops={"color": "black", "linewidth": 1.8},
        boxprops={"linewidth": 1.1},
        whiskerprops={"linewidth": 1.0},
        capprops={"linewidth": 1.0},
    )

    for patch, alg in zip(bp["boxes"], algorithms):
        patch.set_facecolor(ALG_COLORS.get(alg, "gray"))
        patch.set_alpha(0.68)

    ax.set_ylabel(r"Within-problem IQR of $\log_{10}$ final error")
    ax.set_xlabel("Algorithm")
    ax.set_title("Within-problem stability across COCO/BBOB observations")
    ax.grid(True, axis="y", alpha=0.35)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=ALG_COLORS.get(alg, "gray"), alpha=0.68)
        for alg in algorithms
    ]
    ax.legend(handles, algorithms, frameon=True, loc="upper right")

    summary = (
        dispersion.groupby("algorithm", observed=True)["iqr_log_error"]
        .agg(["count", "median", "mean", "max"])
        .reindex(algorithms)
    )
    print("[OK] Robustness metric: within-problem IQR of log10 corrected error across five observations.")
    print("[OK] Exact-zero corrected errors use a 1e-12 floor only for this log-scale dispersion metric.")
    print(summary.to_string(float_format=lambda x: f"{x:.6g}"))

    savefig(fig, "fig_coco_robustness_within_problem_iqr.png")

def plot_dmgso_internal_behavior(dim_df: pd.DataFrame):
    d = dim_df[dim_df["algorithm"] == "DMGSO"].sort_values("dimension")

    if d.empty:
        print("[WARN] No DMGSO rows found.")
        return

    fig, ax1 = plt.subplots(figsize=(8.0, 5.2))

    ax1.plot(
        d["dimension"],
        d["n_accept_mean"],
        marker="o",
        linewidth=2.0,
        markersize=4.5,
        color=ALG_COLORS["DMGSO"],
        label="Accepted updates",
    )

    ax1.set_xlabel("Problem dimension")
    ax1.set_ylabel("Mean accepted updates")
    ax1.grid(True, alpha=0.42)

    ax2 = ax1.twinx()

    ax2.plot(
        d["dimension"],
        d["n_reloc_mean"],
        marker="X",
        linewidth=2.0,
        markersize=5.5,
        color="#111111",
        label="Relocation events",
    )

    ax2.set_ylabel("Mean relocation events")

    ax1.set_title("Internal search activity of DMGSO on COCO/BBOB")
    ax1.set_xticks(sorted(d["dimension"].unique()))

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        frameon=True,
        loc="upper left",
    )

    savefig(fig, "fig_coco_dmgso_internal_behavior.png")


# ------------------------------------------------------------
# Figure 7: Runtime-accuracy trade-off
# ------------------------------------------------------------
def plot_runtime_accuracy_tradeoff(dim_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8.0, 5.2))

    for alg in ALG_ORDER:
        g = dim_df[dim_df["algorithm"] == alg].sort_values("dimension")

        if g.empty:
            continue

        ax.scatter(
            g["wall_time_median"],
            g["error_median"],
            s=70,
            color=ALG_COLORS.get(alg, "gray"),
            alpha=0.78,
            label=alg,
        )

        for _, row in g.iterrows():
            ax.text(
                row["wall_time_median"],
                row["error_median"],
                f"D{int(row['dimension'])}",
                fontsize=7,
                ha="left",
                va="bottom",
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Median wall time [s]")
    ax.set_ylabel("Median final error")
    ax.set_title("Runtime--accuracy trade-off on COCO/BBOB")
    ax.grid(True, which="both", alpha=0.38)
    ax.legend(frameon=True, ncol=2)

    savefig(fig, "fig_coco_runtime_accuracy_tradeoff.png")


# ------------------------------------------------------------
# Figure 8: Dimension-wise grouped bar
# ------------------------------------------------------------
def plot_dimension_grouped_bar(dim_df: pd.DataFrame):
    dims = sorted(dim_df["dimension"].unique())
    algorithms = [a for a in ALG_ORDER if a in dim_df["algorithm"].unique()]

    x = np.arange(len(dims))
    width = 0.82 / len(algorithms)

    fig, ax = plt.subplots(figsize=(8.8, 5.2))

    for i, alg in enumerate(algorithms):
        vals = []

        for dim in dims:
            v = dim_df[
                (dim_df["dimension"] == dim)
                & (dim_df["algorithm"] == alg)
            ]["error_median"]

            vals.append(float(v.iloc[0]) if len(v) else np.nan)

        vals = np.log10(np.maximum(vals, 1e-300))

        ax.bar(
            x + (i - len(algorithms) / 2) * width + width / 2,
            vals,
            width,
            color=ALG_COLORS.get(alg, "gray"),
            alpha=0.82,
            label=alg,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([f"D={d}" for d in dims])
    ax.set_ylabel(r"$\log_{10}$ median error")
    ax.set_xlabel("Problem dimension")
    ax.set_title("Dimension-wise median error comparison on COCO/BBOB")
    ax.grid(True, axis="y", alpha=0.38)
    ax.legend(frameon=True, ncol=2)

    savefig(fig, "fig_coco_dimension_grouped_bar.png")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    run_df = load_run_data()
    problem_df = build_problem_level_data(run_df)
    expected = 5 * 24 * 15 * 5
    if len(problem_df) != expected:
        raise RuntimeError(f"Expected {expected} algorithm-problem records after aggregation; found {len(problem_df)}")
    counts = problem_df.groupby(["dimension", "algorithm"]).size()
    if not (counts == 360).all():
        raise RuntimeError("Each dimension-algorithm block must contain exactly 360 function-instance problems.")
    print("[OK] Problem-level audit: 360 paired function-instance problems per dimension/algorithm.")
    dim_df = load_dimension_data(run_df)

    plot_scalability_curve(dim_df)
    plot_fe_efficiency(dim_df)
    plot_performance_profile(run_df)
    plot_rank_heatmap(run_df)
    plot_robustness_boxplot(run_df)
    plot_dmgso_internal_behavior(dim_df)
    plot_runtime_accuracy_tradeoff(dim_df)
    plot_dimension_grouped_bar(dim_df)

    print("\n[OK] All COCO/BBOB publication figures generated.")
    print(f"Output folder: {OUT_DIR}")


if __name__ == "__main__":
    main()