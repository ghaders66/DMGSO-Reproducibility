# -*- coding: utf-8 -*-
"""
Academic publication-grade visualizations for CEC2014 benchmark results.

Input (revision):
    D:\\PHD\\DMGSO_Information Scince\\Revision\\cec2014\\raw\\cec2014_run_summary_revision.csv

Outputs (revision):
    D:\\PHD\\DMGSO_Information Scince\\Revision\\cec2014\\statistics\\external_comparison\\figures

Generated figures:
    - Rank heatmap
    - Log-error heatmap
    - Performance profile
    - Category heatmap
    - Category grouped bar plot
    - Category radar rank chart
    - Robustness boxplot
    - DMGSO search activity plot

Output format:
    PNG only, 300 dpi
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager


# ============================================================
# Paths — CEC2014 Revision
# ============================================================

REVISION_ROOT = Path(
    "/mnt/d/PHD/DMGSO_Information Scince/Revision/cec2014"
)

INPUT = (
    REVISION_ROOT
    / "raw"
    / "cec2014_run_summary_revision.csv"
)

OUT_DIR = (
    REVISION_ROOT
    / "statistics"
    / "external_comparison"
    / "figures"
)

OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Times New Roman registration for WSL
# ============================================================

WINDOWS_TNR_FONTS = [
    Path("/mnt/c/Windows/Fonts/times.ttf"),
    Path("/mnt/c/Windows/Fonts/timesbd.ttf"),
    Path("/mnt/c/Windows/Fonts/timesi.ttf"),
    Path("/mnt/c/Windows/Fonts/timesbi.ttf"),
]

_found_tnr_files = []
for font_path in WINDOWS_TNR_FONTS:
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        _found_tnr_files.append(str(font_path))

if not _found_tnr_files:
    raise RuntimeError(
        "Times New Roman font files were not found under "
        "/mnt/c/Windows/Fonts/. Figures were not generated."
    )

try:
    TNR_PATH = font_manager.findfont(
        "Times New Roman",
        fallback_to_default=False,
    )
except ValueError as exc:
    raise RuntimeError(
        "Times New Roman files were found but Matplotlib could not "
        "register the family. Figures were not generated."
    ) from exc

print(f"[OK] Times New Roman registered: {TNR_PATH}")


# ============================================================
# Publication-quality plotting configuration
# ============================================================

plt.rcParams.update(
    {
        "font.family": "Times New Roman",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 13,
        "axes.linewidth": 1.0,
        "grid.linewidth": 0.5,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.facecolor": "white",
        "lines.antialiased": True,
        "patch.antialiased": True,
        "figure.autolayout": False,
    }
)


# ============================================================
# Algorithm color palette
# ============================================================

ALGORITHM_COLORS = {
    "DMGSO": "blue",
    "CMA-ES": "red",
    "DE": "green",
    "PSO": "orange",
    "COBYLA": "purple",
    "NELDER-MEAD": "brown",
    "NELDER–MEAD": "brown",
    "POWELL": "pink",
    "COMPASS": "gray",
}


# ============================================================
# Utility functions
# ============================================================

def get_algorithm_color(algorithm: str):
    return ALGORITHM_COLORS.get(str(algorithm).upper(), None)


def savefig(fig, filename: str) -> None:
    """
    Save figure as high-resolution PNG only.
    """

    out = OUT_DIR / filename

    fig.savefig(
        out,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
        pad_inches=0.08,
    )

    plt.close(fig)

    print(f"[OK] saved PNG: {out}")


def safe_log10(values):
    values = np.asarray(values, dtype=float)
    values = np.maximum(values, 1e-300)
    return np.log10(values)


def load_data() -> pd.DataFrame:
    """
    Load and preprocess CEC2014 run summary data.
    """

    if not INPUT.exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT}")

    df = pd.read_csv(INPUT)

    required_cols = ["function_id", "algorithm", "error"]

    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    numeric_cols = [
        "function_id",
        "error",
        "fe_used",
        "wall_time_sec",
        "n_accept",
        "n_reloc",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["function_id", "algorithm", "error"])

    df["function_id"] = df["function_id"].astype(int)
    df["algorithm"] = df["algorithm"].astype(str)

    df["error"] = np.maximum(df["error"].astype(float), 1e-300)
    df["log_error"] = np.log10(df["error"])

    if "category" not in df.columns:
        df["category"] = df["function_id"].apply(assign_cec2014_category)

    # --------------------------------------------------------
    # Revision-data integrity checks (do not alter plotting data)
    # --------------------------------------------------------
    expected_algorithms = {"DMGSO", "CMA-ES", "DE", "PSO"}
    found_algorithms = set(df["algorithm"].unique())

    missing_algorithms = expected_algorithms - found_algorithms
    if missing_algorithms:
        raise ValueError(
            f"Revision dataset is missing expected algorithms: "
            f"{sorted(missing_algorithms)}"
        )

    unexpected_algorithms = found_algorithms - expected_algorithms
    if unexpected_algorithms:
        print(
            "[WARN] Additional algorithms found and will also be plotted: "
            f"{sorted(unexpected_algorithms)}"
        )

    function_ids = sorted(df["function_id"].unique())
    if function_ids != list(range(1, 31)):
        print(
            "[WARN] Function IDs are not exactly F1--F30. "
            f"Found: {function_ids}"
        )

    if "dimension" in df.columns:
        dims = sorted(pd.to_numeric(df["dimension"], errors="coerce").dropna().unique())
        print(f"[INFO] Dimensions found: {dims}")

    if "fe_used" in df.columns:
        print(
            "[INFO] FE usage range: "
            f"{df['fe_used'].min():.0f} -- {df['fe_used'].max():.0f}"
        )

    counts = (
        df.groupby(["function_id", "algorithm"])
        .size()
        .rename("n")
    )
    print(
        "[INFO] Rows per function/algorithm: "
        f"min={int(counts.min())}, max={int(counts.max())}"
    )
    print(f"[INFO] Total valid rows loaded: {len(df)}")
    print(f"[INFO] Algorithms: {ordered_algorithms(df)}")

    return df


def assign_cec2014_category(function_id: int) -> str:
    """
    Assign CEC2014 function category.
    """

    if 1 <= function_id <= 3:
        return "Unimodal"

    if 4 <= function_id <= 16:
        return "Multimodal"

    if 17 <= function_id <= 22:
        return "Hybrid"

    if 23 <= function_id <= 30:
        return "Composition"

    return "Unknown"


def ordered_algorithms(df: pd.DataFrame) -> list[str]:
    """
    Return algorithms in preferred publication order.
    """

    preferred_order = [
        "DMGSO",
        "CMA-ES",
        "DE",
        "PSO",
        "COBYLA",
        "NELDER-MEAD",
        "NELDER–MEAD",
        "POWELL",
        "COMPASS",
    ]

    available = list(df["algorithm"].dropna().unique())

    ordered = [a for a in preferred_order if a in available]
    rest = sorted([a for a in available if a not in ordered])

    return ordered + rest


# ============================================================
# Figure 1: Function-wise rank heatmap
# ============================================================

def plot_rank_heatmap(df: pd.DataFrame) -> None:

    summary = (
        df.groupby(["function_id", "algorithm"], as_index=False)
        .agg(error_median=("error", "median"))
    )

    summary["rank"] = (
        summary.groupby("function_id")["error_median"]
        .rank(method="min", ascending=True)
    )

    algorithms = ordered_algorithms(summary)

    pivot = (
        summary.pivot(
            index="function_id",
            columns="algorithm",
            values="rank",
        )
        .reindex(columns=algorithms)
        .sort_index()
    )

    fig, ax = plt.subplots(figsize=(8.8, 9.2))

    im = ax.imshow(
        pivot.values,
        aspect="auto",
        cmap="viridis_r",
        interpolation="nearest",
    )

    ax.set_xticks(np.arange(len(pivot.columns)))

    ax.set_xticklabels(
        pivot.columns,
        rotation=35,
        ha="right",
    )

    ax.set_yticks(np.arange(len(pivot.index)))

    ax.set_yticklabels(
        [f"F{int(i)}" for i in pivot.index]
    )

    ax.set_xlabel("Algorithm")
    ax.set_ylabel("CEC2014 function")
    ax.set_title("Function-wise algorithm ranking on CEC2014")

    cbar = fig.colorbar(im, ax=ax, shrink=0.82)
    cbar.set_label("Rank (1 = best)")

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]

            if np.isfinite(val):
                ax.text(
                    j,
                    i,
                    f"{int(val)}",
                    ha="center",
                    va="center",
                    fontsize=10,
                )

    savefig(fig, "fig_cec2014_rank_heatmap.png")


# ============================================================
# Figure 2: Function-wise log-error heatmap
# ============================================================

def plot_error_heatmap(df: pd.DataFrame) -> None:

    summary = (
        df.groupby(["function_id", "algorithm"], as_index=False)
        .agg(error_median=("error", "median"))
    )

    algorithms = ordered_algorithms(summary)

    pivot = (
        summary.pivot(
            index="function_id",
            columns="algorithm",
            values="error_median",
        )
        .reindex(columns=algorithms)
        .sort_index()
    )

    Z = safe_log10(pivot.values)

    fig, ax = plt.subplots(figsize=(8.8, 9.2))

    im = ax.imshow(
        Z,
        aspect="auto",
        cmap="viridis",
        interpolation="nearest",
    )

    ax.set_xticks(np.arange(len(pivot.columns)))

    ax.set_xticklabels(
        pivot.columns,
        rotation=35,
        ha="right",
    )

    ax.set_yticks(np.arange(len(pivot.index)))

    ax.set_yticklabels(
        [f"F{int(i)}" for i in pivot.index]
    )

    ax.set_xlabel("Algorithm")
    ax.set_ylabel("CEC2014 function")
    ax.set_title(r"Median final error on CEC2014 ($\log_{10}$ scale)")

    cbar = fig.colorbar(im, ax=ax, shrink=0.82)
    cbar.set_label(r"$\log_{10}$ median error")

    savefig(fig, "fig_cec2014_log_error_heatmap.png")


# ============================================================
# Figure 3: Dolan-Moré performance profile
# ============================================================

def plot_performance_profile(df: pd.DataFrame) -> None:

    summary = (
        df.groupby(["function_id", "algorithm"], as_index=False)
        .agg(error_median=("error", "median"))
    )

    algorithms = ordered_algorithms(summary)

    pivot = (
        summary.pivot(
            index="function_id",
            columns="algorithm",
            values="error_median",
        )
        .reindex(columns=algorithms)
        .dropna()
    )

    best = pivot.min(axis=1)
    ratios = pivot.div(best, axis=0)

    tau = np.logspace(0, 6, 400)

    fig, ax = plt.subplots(figsize=(9.0, 5.5))

    for alg in ratios.columns:
        values = (
            ratios[alg]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .values
        )

        y = [(values <= t).mean() for t in tau]

        ax.plot(
            tau,
            y,
            linewidth=2.2,
            label=alg,
            color=get_algorithm_color(alg),
        )

    ax.set_xscale("log")
    ax.set_xlabel(r"Performance ratio $\tau$")
    ax.set_ylabel("Proportion of functions")
    ax.set_title("Dolan--Moré performance profile on CEC2014")

    ax.grid(
        True,
        which="both",
        linestyle="--",
        alpha=0.40,
    )

    ax.legend(frameon=True)

    savefig(fig, "fig_cec2014_performance_profile.png")


# ============================================================
# Figure 4: Category-level log-error heatmap
# ============================================================

def plot_category_heatmap(df: pd.DataFrame) -> None:

    summary = (
        df.groupby(["category", "algorithm"], as_index=False)
        .agg(error_median=("error", "median"))
    )

    categories = [
        "Unimodal",
        "Multimodal",
        "Hybrid",
        "Composition",
    ]

    algorithms = ordered_algorithms(summary)

    pivot = (
        summary.pivot(
            index="category",
            columns="algorithm",
            values="error_median",
        )
        .reindex(index=categories)
        .reindex(columns=algorithms)
    )

    Z = safe_log10(pivot.values)

    fig, ax = plt.subplots(figsize=(8.8, 5.2))

    im = ax.imshow(
        Z,
        aspect="auto",
        cmap="viridis_r",
        interpolation="nearest",
    )

    ax.set_xticks(np.arange(len(pivot.columns)))

    ax.set_xticklabels(
        pivot.columns,
        rotation=35,
        ha="right",
    )

    ax.set_yticks(np.arange(len(pivot.index)))

    ax.set_yticklabels(pivot.index)

    ax.set_xlabel("Algorithm")
    ax.set_ylabel("Function category")
    ax.set_title(r"Category-level median error ($\log_{10}$ scale)")

    cbar = fig.colorbar(im, ax=ax, shrink=0.82)
    cbar.set_label(r"$\log_{10}$ median error")

    savefig(fig, "fig_cec2014_category_heatmap.png")


# ============================================================
# Figure 5: Category-wise grouped bar plot
# ============================================================

def plot_category_grouped_bar(df: pd.DataFrame) -> None:

    summary = (
        df.groupby(["category", "algorithm"], as_index=False)
        .agg(error_median=("error", "median"))
    )

    categories = [
        "Unimodal",
        "Multimodal",
        "Hybrid",
        "Composition",
    ]

    algorithms = ordered_algorithms(summary)

    x = np.arange(len(categories))
    width = 0.80 / max(len(algorithms), 1)

    fig, ax = plt.subplots(figsize=(9.2, 5.5))

    for i, alg in enumerate(algorithms):

        vals = []

        for cat in categories:
            v = summary[
                (summary["category"] == cat)
                & (summary["algorithm"] == alg)
            ]["error_median"]

            vals.append(float(v.iloc[0]) if len(v) else np.nan)

        vals = safe_log10(vals)

        ax.bar(
            x + (i - len(algorithms) / 2) * width + width / 2,
            vals,
            width,
            label=alg,
            color=get_algorithm_color(alg),
            edgecolor="black",
            linewidth=0.5,
            alpha=0.85,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(categories)

    ax.set_ylabel(r"$\log_{10}$ median error")
    ax.set_xlabel("CEC2014 function category")
    ax.set_title("Category-wise performance comparison on CEC2014")

    ax.grid(
        True,
        axis="y",
        linestyle="--",
        alpha=0.40,
    )

    ax.legend(frameon=True, ncol=2)

    savefig(fig, "fig_cec2014_category_grouped_bar.png")


# ============================================================
# Figure 6: Category-wise radar rank chart
# ============================================================

def plot_category_radar_chart(df: pd.DataFrame) -> None:

    summary = (
        df.groupby(["category", "algorithm"], as_index=False)
        .agg(error_median=("error", "median"))
    )

    categories = [
        "Unimodal",
        "Multimodal",
        "Hybrid",
        "Composition",
    ]

    algorithms = ordered_algorithms(summary)

    summary["rank"] = (
        summary.groupby("category")["error_median"]
        .rank(method="min", ascending=True)
    )

    angles = np.linspace(
        0,
        2 * np.pi,
        len(categories),
        endpoint=False,
    )

    angles = np.concatenate([angles, [angles[0]]])

    fig = plt.figure(figsize=(8.0, 7.0))
    ax = fig.add_subplot(111, polar=True)

    for alg in algorithms:

        vals = []

        for cat in categories:
            v = summary[
                (summary["category"] == cat)
                & (summary["algorithm"] == alg)
            ]["rank"]

            vals.append(float(v.iloc[0]) if len(v) else np.nan)

        vals = np.asarray(vals, dtype=float)
        vals = np.concatenate([vals, [vals[0]]])

        ax.plot(
            angles,
            vals,
            linewidth=2.0,
            marker="o",
            markersize=6,
            label=alg,
            color=get_algorithm_color(alg),
        )

        ax.fill(
            angles,
            vals,
            color=get_algorithm_color(alg),
            alpha=0.08,
        )

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(
    categories,
    fontsize=14,
)

    max_rank = max(4, len(algorithms))

    ax.set_ylim(max_rank + 0.2, 0.8)

    ax.set_yticks(list(range(1, max_rank + 1)))
    ax.set_yticklabels(
    [str(i) for i in range(1, max_rank + 1)],
    fontsize=16,
)

    ax.set_title(
        "Category-wise algorithm rank profile on CEC2014\n"
        "(lower rank is better)",
        pad=20,
    )

    ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.30, 1.12),
        frameon=True,
    )

    savefig(fig, "fig_cec2014_category_radar_rank.png")


# ============================================================
# Figure 7: Robustness boxplot
# ============================================================

def plot_robustness_boxplot(df: pd.DataFrame) -> None:

    algorithms = ordered_algorithms(df)

    data = [
        df[df["algorithm"] == alg]["log_error"]
        .dropna()
        .values
        for alg in algorithms
    ]

    fig, ax = plt.subplots(figsize=(9.0, 5.6))

    bp = ax.boxplot(
        data,
        tick_labels=algorithms,
        patch_artist=True,
        showfliers=False,
        widths=0.65,
        medianprops={
            "color": "black",
            "linewidth": 1.8,
        },
        boxprops={
            "linewidth": 1.1,
        },
        whiskerprops={
            "linewidth": 1.0,
        },
        capprops={
            "linewidth": 1.0,
        },
    )

    for patch, alg in zip(bp["boxes"], algorithms):
        patch.set_facecolor(get_algorithm_color(alg) or "#808080")
        patch.set_alpha(0.68)

    ax.set_ylabel(r"$\log_{10}$ final error")
    ax.set_xlabel("Algorithm")
    ax.set_title("Robustness of final errors across CEC2014 runs")

    ax.grid(
        True,
        axis="y",
        linestyle="--",
        alpha=0.35,
    )

    handles = [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            color=get_algorithm_color(alg) or "#808080",
            alpha=0.68,
        )
        for alg in algorithms
    ]

    ax.legend(
        handles,
        algorithms,
        frameon=True,
        loc="upper left",
    )

    savefig(fig, "fig_cec2014_robustness_boxplot.png")


# ============================================================
# Figure 8: DMGSO internal search activity
# ============================================================

def plot_dmgso_search_activity(df: pd.DataFrame) -> None:

    d = df[df["algorithm"].str.upper() == "DMGSO"].copy()

    if d.empty:
        print("[WARN] No DMGSO rows found; search activity skipped.")
        return

    required = ["n_accept", "n_reloc"]

    missing = [c for c in required if c not in d.columns]

    if missing:
        print(
            f"[WARN] Missing DMGSO search-activity columns {missing}; "
            "search activity skipped."
        )
        return

    summary = (
        d.groupby(["function_id", "category"], as_index=False)
        .agg(
            n_accept_mean=("n_accept", "mean"),
            n_reloc_mean=("n_reloc", "mean"),
            error_median=("error", "median"),
        )
        .sort_values("function_id")
    )

    fig, ax1 = plt.subplots(figsize=(9.4, 5.6))

    x = summary["function_id"].values

    ax1.plot(
        x,
        summary["n_accept_mean"].values,
        marker="o",
        linewidth=2.0,
        markersize=4.5,
        label="Accepted updates",
        color=ALGORITHM_COLORS["DMGSO"],
    )

    ax1.set_xlabel("CEC2014 function")
    ax1.set_ylabel("Mean accepted updates")

    ax1.grid(
        True,
        linestyle="--",
        alpha=0.40,
    )

    ax2 = ax1.twinx()

    ax2.scatter(
        x,
        summary["n_reloc_mean"].values,
        marker="X",
        s=60,
        label="Relocation events",
        color="red",
    )

    ax2.set_ylabel("Mean relocation events")

    ax1.set_title("DMGSO internal search activity across CEC2014 functions")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        frameon=True,
        loc="upper right",
    )

    ax1.set_xticks(x)

    ax1.set_xticklabels(
        [f"F{int(i)}" for i in x],
        rotation=90,
    )

    savefig(fig, "fig_cec2014_dmgso_search_activity.png")


# ============================================================
# Main execution
# ============================================================

def main() -> None:

    df = load_data()

    plot_rank_heatmap(df)
    plot_error_heatmap(df)
    plot_performance_profile(df)
    plot_category_heatmap(df)
    plot_category_grouped_bar(df)
    plot_category_radar_chart(df)
    plot_robustness_boxplot(df)
    plot_dmgso_search_activity(df)

    print("\n[OK] All academic CEC2014 figures generated.")
    print(f"Output folder: {OUT_DIR}")


if __name__ == "__main__":
    main()