# -*- coding: utf-8 -*-
"""
Publication-grade visualizations for DMGSO COCO/BBOB ablation study.

Important:
The dimension-based plots use categorical x-positions so that the x-axis
is displayed exactly as: 2, 5, 10, 20, 40.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager


# ============================================================
# Paths
# ============================================================

REVISION_ROOT = Path("/mnt/d/PHD/DMGSO_Information Scince/Revision/coco_bbob/revision_audited_ablation")
RAW_FILE = REVISION_ROOT / "raw" / "ablation_coco_bbob_audited_run_summary.csv"
SUMMARY_DIR = REVISION_ROOT / "processed"
OUT_DIR = REVISION_ROOT / "figures_submitted_style_revision"
OUT_DIR.mkdir(parents=True, exist_ok=True)

VERSION_FILE = SUMMARY_DIR / "ablation_coco_bbob_audited_version_summary.csv"
DIMENSION_FILE = SUMMARY_DIR / "ablation_coco_bbob_audited_dimension_summary.csv"
FUNCTION_FILE = SUMMARY_DIR / "ablation_coco_bbob_audited_function_summary.csv"


# ============================================================
# Times New Roman registration (WSL -> Windows fonts)
# ============================================================

WINDOWS_FONT_DIR = Path("/mnt/c/Windows/Fonts")
TNR_FILES = [
    WINDOWS_FONT_DIR / "times.ttf",    # Regular
    WINDOWS_FONT_DIR / "timesbd.ttf",  # Bold
    WINDOWS_FONT_DIR / "timesi.ttf",   # Italic
    WINDOWS_FONT_DIR / "timesbi.ttf",  # Bold Italic
]

registered_fonts = []
for font_path in TNR_FILES:
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        registered_fonts.append(font_path.name)

if not registered_fonts:
    raise FileNotFoundError(
        "Times New Roman font files were not found in /mnt/c/Windows/Fonts/. "
        "Expected times.ttf, timesbd.ttf, timesi.ttf and timesbi.ttf."
    )

print(
    "[FONT] Times New Roman registered from Windows fonts: "
    + ", ".join(registered_fonts)
)


# ============================================================
# Global plotting style
# ============================================================

plt.rcParams.update(
    {
        "font.family": "Times New Roman",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "figure.titlesize": 12,
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.4,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.facecolor": "white",
    }
)


# ============================================================
# Constants
# ============================================================

VERSION_ORDER = ["V0", "V1", "V2", "V3", "V4"]

VERSION_LABELS = {
    "V0": "V0\nDirectional",
    "V1": "V1\n+ Step",
    "V2": "V2\n+ Long-range",
    "V3": "V3\n+ Memory",
    "V4": "V4\nFull",
}

VERSION_COLORS = {
    "V0": "gray",
    "V1": "blue",
    "V2": "orange",
    "V3": "green",
    "V4": "red",
}

DIMS = [2, 5, 10, 20, 40]
DIM_LABELS = ["2", "5", "10", "20", "40"]


# ============================================================
# Utility functions
# ============================================================

def savefig(fig: plt.Figure, filename: str) -> None:
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


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

    df = pd.read_csv(path)
    df = df.replace([np.inf, -np.inf], np.nan)

    return df


def ordered_versions(df: pd.DataFrame) -> list[str]:
    available = list(df["version"].dropna().unique())
    ordered = [v for v in VERSION_ORDER if v in available]
    rest = [v for v in available if v not in ordered]

    return ordered + rest


def safe_log10(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = np.maximum(values, 1e-300)

    return np.log10(values)


def normalize_for_radar(
    values: np.ndarray,
    lower_is_better: bool = True,
) -> np.ndarray:

    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)

    out = np.full_like(values, np.nan, dtype=float)

    if finite.sum() == 0:
        return out

    v = values[finite]
    vmin = np.min(v)
    vmax = np.max(v)

    if np.isclose(vmin, vmax):
        out[finite] = 1.0
        return out

    if lower_is_better:
        out[finite] = (vmax - v) / (vmax - vmin)
    else:
        out[finite] = (v - vmin) / (vmax - vmin)

    return out


def prepare_dimension_axis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert real dimensions into categorical x-positions.

    This prevents Matplotlib from displaying the x-axis in logarithmic
    or power-of-two notation such as 2^1, 2^2, 2^3.
    """

    dim_to_x = {dim: i for i, dim in enumerate(DIMS)}

    df = df[df["dimension"].isin(DIMS)].copy()
    df["x_pos"] = df["dimension"].map(dim_to_x)

    return df


def apply_dimension_ticks(ax: plt.Axes) -> None:
    x_pos = np.arange(len(DIMS))

    ax.set_xticks(x_pos)
    ax.set_xticklabels(DIM_LABELS)
    ax.set_xlim(-0.25, len(DIMS) - 0.75)


# ============================================================
# Figure 1: Version-level error bar plot
# ============================================================

def plot_version_error_bar(version_df: pd.DataFrame) -> None:
    df = version_df.copy()

    versions = ordered_versions(df)
    df = df.set_index("version").reindex(versions)

    x = np.arange(len(versions))
    width = 0.36

    median_vals = safe_log10(df["error_median"].to_numpy())
    mean_vals = safe_log10(df["error_mean"].to_numpy())

    fig, ax = plt.subplots(figsize=(8.4, 5.0))

    ax.bar(
        x - width / 2,
        median_vals,
        width,
        label="Median error",
        color="blue",
        alpha=0.82,
    )

    ax.bar(
        x + width / 2,
        mean_vals,
        width,
        label="Mean error",
        color="orange",
        alpha=0.82,
    )

    ax.set_xticks(x)
    ax.set_xticklabels([VERSION_LABELS.get(v, v) for v in versions])

    ax.set_xlabel("DMGSO ablation variant")
    ax.set_ylabel(r"$\log_{10}$ optimization error")
    ax.set_title("Version-level ablation performance on COCO/BBOB")

    ax.grid(True, axis="y", alpha=0.35)
    ax.legend(frameon=True)

    savefig(fig, "fig_ablation_coco_version_error_bar.png")


# ============================================================
# Figure 2: Version-level search dynamics
# ============================================================

def plot_version_search_dynamics(version_df: pd.DataFrame) -> None:
    """
    Plot search dynamics across DMGSO ablation variants.

    Left axis:
        Mean accepted updates

    Right axis:
        Mean relocation events
    """

    # ------------------------------------------------------------
    # Prepare data
    # ------------------------------------------------------------
    df = version_df.copy()

    versions = ordered_versions(df)

    df = df.set_index("version").reindex(versions)

    x = np.arange(len(versions))

    accept_vals = df["n_accept_mean"].fillna(0).to_numpy(dtype=float)

    if "n_reloc_mean" in df.columns:
        reloc_vals = df["n_reloc_mean"].fillna(0).to_numpy(dtype=float)
    else:
        reloc_vals = np.zeros(len(versions), dtype=float)

    # ------------------------------------------------------------
    # Create figure
    # ------------------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(9.0, 5.5))

    # ------------------------------------------------------------
    # Color mapping for each version
    # ------------------------------------------------------------
    bar_colors = [
        VERSION_COLORS.get(v, "gray")
        for v in versions
    ]

    # ------------------------------------------------------------
    # Bar plot: accepted updates
    # ------------------------------------------------------------
    bars = ax1.bar(
        x,
        accept_vals,
        width=0.60,
        color=bar_colors,
        alpha=0.82,
        edgecolor="black",
        linewidth=0.7,
    )

    # ------------------------------------------------------------
    # Left axis settings
    # ------------------------------------------------------------
    ax1.set_xticks(x)

    ax1.set_xticklabels(
        [VERSION_LABELS.get(v, v) for v in versions]
    )

    ax1.set_xlabel("DMGSO ablation variant")
    ax1.set_ylabel("Mean accepted updates")

    ax1.grid(
        True,
        axis="y",
        linestyle="--",
        alpha=0.35,
    )

    # ------------------------------------------------------------
    # Right axis: relocation events
    # ------------------------------------------------------------
    ax2 = ax1.twinx()

    reloc_line = ax2.plot(
        x,
        reloc_vals,
        color="black",
        marker="X",
        linestyle="--",
        linewidth=2.2,
        markersize=8,
        label="Relocation events",
    )

    ax2.set_ylabel("Mean relocation events")

    # ------------------------------------------------------------
    # Figure title
    # ------------------------------------------------------------
    ax1.set_title(
        "Search dynamics across DMGSO ablation variants"
    )

    # ------------------------------------------------------------
    # Custom legend
    # ------------------------------------------------------------
    version_handles = [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            color=VERSION_COLORS.get(v, "gray"),
            alpha=0.82,
        )
        for v in versions
    ]

    version_labels = [
        VERSION_LABELS.get(v, v).replace("\n", " ")
        for v in versions
    ]

    # Add relocation line to legend
    version_handles.append(reloc_line[0])
    version_labels.append("Relocation events")

    ax1.legend(
        version_handles,
        version_labels,
        loc="upper left",
        frameon=True,
        ncol=2,
    )

    # ------------------------------------------------------------
    # Save figure
    # ------------------------------------------------------------
    savefig(
        fig,
        "fig_ablation_coco_version_search_dynamics.png",
    )

# ============================================================
# Figure 3: Dimension-level scalability error
# ============================================================

def plot_dimension_scalability_error(dimension_df: pd.DataFrame) -> None:
    df = dimension_df.copy()
    df = prepare_dimension_axis(df)

    versions = ordered_versions(df)

    fig, ax = plt.subplots(figsize=(8.6, 5.2))

    for version in versions:
        g = df[df["version"] == version].sort_values("dimension")

        ax.plot(
            g["x_pos"],
            g["error_median"],
            marker="o",
            linewidth=2.0,
            markersize=5,
            label=version,
            color=VERSION_COLORS.get(version, None),
        )

    apply_dimension_ticks(ax)

    ax.set_xlabel("Problem dimension")
    ax.set_ylabel("Median optimization error")
    ax.set_title("Scalability of ablation variants across COCO/BBOB dimensions")

    ax.grid(True, which="major", alpha=0.35)
    ax.legend(frameon=True, ncol=2)

    savefig(fig, "fig_ablation_coco_dimension_scalability_error.png")


# ============================================================
# Figure 4: Dimension-level search dynamics
# ============================================================

def plot_dimension_search_dynamics(dimension_df: pd.DataFrame) -> None:
    df = dimension_df.copy()
    df = prepare_dimension_axis(df)

    versions = ordered_versions(df)

    fig, ax1 = plt.subplots(figsize=(8.8, 5.2))

    for version in versions:
        g = df[df["version"] == version].sort_values("dimension")

        ax1.plot(
            g["x_pos"],
            g["n_accept_mean"],
            marker="o",
            linewidth=1.9,
            markersize=5,
            label=f"{version} accepted",
            color=VERSION_COLORS.get(version, None),
        )

    apply_dimension_ticks(ax1)

    ax1.set_xlabel("Problem dimension")
    ax1.set_ylabel("Mean accepted updates")
    ax1.grid(True, which="major", alpha=0.35)

    ax2 = ax1.twinx()

    v4 = df[df["version"] == "V4"].sort_values("dimension")

    if not v4.empty and "n_reloc_mean" in v4.columns:
        ax2.plot(
            v4["x_pos"],
            v4["n_reloc_mean"],
            marker="X",
            linewidth=2.2,
            markersize=7,
            linestyle="--",
            color="black",
            label="V4 relocation",
        )

    apply_dimension_ticks(ax2)

    ax2.set_ylabel("Mean relocation events")

    ax1.set_title("Internal search dynamics across dimensions")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        frameon=True,
        loc="upper left",
        ncol=2,
    )

    savefig(fig, "fig_ablation_coco_dimension_search_dynamics.png")


# ============================================================
# Figure 5: Function-wise rank heatmap
# ============================================================

def plot_function_rank_heatmap(function_df: pd.DataFrame) -> None:
    df = function_df.copy()

    summary = (
        df.groupby(["function_id", "version"], as_index=False)
        .agg(error_median=("error_median", "median"))
    )

    summary["rank"] = (
        summary.groupby("function_id")["error_median"]
        .rank(method="min", ascending=True)
    )

    versions = ordered_versions(summary)

    pivot = (
        summary.pivot(
            index="function_id",
            columns="version",
            values="rank",
        )
        .reindex(columns=versions)
        .sort_index()
    )

    fig, ax = plt.subplots(figsize=(7.8, 8.4))

    im = ax.imshow(
        pivot.to_numpy(dtype=float),
        aspect="auto",
        cmap="viridis_r",
    )

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(
        [VERSION_LABELS.get(v, v).replace("\n", " ") for v in pivot.columns],
        rotation=35,
        ha="right",
    )

    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([f"F{int(i)}" for i in pivot.index])

    ax.set_xlabel("DMGSO ablation variant")
    ax.set_ylabel("BBOB function")
    ax.set_title("Function-wise rank heatmap of ablation variants")

    cbar = fig.colorbar(im, ax=ax, shrink=0.82)
    cbar.set_label("Rank: 1 = best")

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
                    fontsize=7,
                )

    savefig(fig, "fig_ablation_coco_function_rank_heatmap.png")


# ============================================================
# Figure 6: Function-wise log-error heatmap
# ============================================================

def plot_function_log_error_heatmap(function_df: pd.DataFrame) -> None:
    df = function_df.copy()

    summary = (
        df.groupby(["function_id", "version"], as_index=False)
        .agg(error_median=("error_median", "median"))
    )

    versions = ordered_versions(summary)

    pivot = (
        summary.pivot(
            index="function_id",
            columns="version",
            values="error_median",
        )
        .reindex(columns=versions)
        .sort_index()
    )

    z = safe_log10(pivot.to_numpy(dtype=float))

    fig, ax = plt.subplots(figsize=(7.8, 8.4))

    im = ax.imshow(
        z,
        aspect="auto",
        cmap="viridis",
    )

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(
        [VERSION_LABELS.get(v, v).replace("\n", " ") for v in pivot.columns],
        rotation=35,
        ha="right",
    )

    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([f"F{int(i)}" for i in pivot.index])

    ax.set_xlabel("DMGSO ablation variant")
    ax.set_ylabel("BBOB function")
    ax.set_title(r"Function-wise median error ($\log_{10}$ scale)")

    cbar = fig.colorbar(im, ax=ax, shrink=0.82)
    cbar.set_label(r"$\log_{10}$ median error")

    savefig(fig, "fig_ablation_coco_function_log_error_heatmap.png")


# ============================================================
# Figure 7: Robustness boxplot
# ============================================================

def plot_robustness_boxplot(function_df: pd.DataFrame) -> None:
    df = function_df.copy()

    versions = ordered_versions(df)
    data = []

    for version in versions:
        vals = (
            df[df["version"] == version]["error_median"]
            .dropna()
            .to_numpy(dtype=float)
        )

        vals = safe_log10(vals)
        data.append(vals)

    fig, ax = plt.subplots(figsize=(8.2, 5.2))

    bp = ax.boxplot(
        data,
        tick_labels=versions,
        patch_artist=True,
        showfliers=False,
        widths=0.65,
        medianprops={"color": "black", "linewidth": 1.7},
        boxprops={"linewidth": 1.0},
        whiskerprops={"linewidth": 1.0},
        capprops={"linewidth": 1.0},
    )

    for patch, version in zip(bp["boxes"], versions):
        patch.set_facecolor(VERSION_COLORS.get(version, "#808080"))
        patch.set_alpha(0.68)

    ax.set_xlabel("DMGSO ablation variant")
    ax.set_ylabel(r"$\log_{10}$ median error")
    ax.set_title("Robustness of ablation variants across BBOB functions")

    ax.grid(True, axis="y", alpha=0.35)

    handles = [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            color=VERSION_COLORS.get(version, "#808080"),
            alpha=0.68,
        )
        for version in versions
    ]

    ax.legend(handles, versions, frameon=True, loc="upper right")

    savefig(fig, "fig_ablation_coco_robustness_boxplot.png")


# ============================================================
# Figure 8: Radar profile
# ============================================================

def plot_ablation_radar(version_df: pd.DataFrame) -> None:
    df = version_df.copy()

    versions = ordered_versions(df)
    df = df.set_index("version").reindex(versions)

    metrics = [
        ("Error", "error_median", True),
        ("Mean error", "error_mean", True),
        ("Stability", "error_std", True),
        ("Accepted", "n_accept_mean", False),
        ("Efficiency", "fe_used_median", True),
    ]

    values = []

    for _, col, lower_is_better in metrics:
        vals = df[col].to_numpy(dtype=float)

        score = normalize_for_radar(
            vals,
            lower_is_better=lower_is_better,
        )

        values.append(score)

    scores = np.vstack(values).T

    labels = [metric[0] for metric in metrics]

    angles = np.linspace(
        0,
        2 * np.pi,
        len(labels),
        endpoint=False,
    )

    angles = np.concatenate([angles, [angles[0]]])

    fig = plt.figure(figsize=(7.2, 6.4))
    ax = fig.add_subplot(111, polar=True)

    for i, version in enumerate(versions):
        vals = scores[i, :]
        vals = np.concatenate([vals, [vals[0]]])

        ax.plot(
            angles,
            vals,
            linewidth=2.0,
            marker="o",
            markersize=4,
            color=VERSION_COLORS.get(version, None),
            label=version,
        )

        ax.fill(
            angles,
            vals,
            color=VERSION_COLORS.get(version, None),
            alpha=0.08,
        )

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)

    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.25, 0.50, 0.75, 1.00])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"])

    ax.set_title(
        "Normalized behavioral profile of DMGSO ablation variants\n"
        "(higher score is better)",
        pad=18,
    )

    ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.28, 1.12),
        frameon=True,
    )

    savefig(fig, "fig_ablation_coco_radar_profile.png")


# ============================================================
# Audited-revision provenance check
# ============================================================

def audit_raw_dataset() -> None:
    """Fail fast unless the exact audited 45,000-run design is present."""
    raw = load_csv(RAW_FILE)
    keys = ["version", "function_id", "instance_id", "dimension", "run_id"]
    required = set(keys + ["status", "error"])
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise ValueError(f"Audited raw dataset missing required columns: {missing}")

    checks = {
        "rows": len(raw) == 45000,
        "versions": sorted(raw["version"].unique().tolist()) == VERSION_ORDER,
        "dimensions": sorted(raw["dimension"].unique().tolist()) == DIMS,
        "functions": raw["function_id"].nunique() == 24,
        "instances": raw["instance_id"].nunique() == 15,
        "runs": raw["run_id"].nunique() == 5,
        "duplicates": raw.duplicated(keys).sum() == 0,
        "status": raw["status"].eq("ok").all(),
        "finite_error": np.isfinite(pd.to_numeric(raw["error"], errors="coerce")).all(),
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise ValueError(f"Audited raw dataset failed checks: {failed}")

    print("[AUDIT] PASS: 45,000 rows; V0-V4; 24 functions; "
          "15 instances; 5 runs; no duplicates; all OK.")

# ============================================================
# Main execution
# ============================================================

def main() -> None:
    audit_raw_dataset()

    version_df = load_csv(VERSION_FILE)
    dimension_df = load_csv(DIMENSION_FILE)
    function_df = load_csv(FUNCTION_FILE)

    plot_version_error_bar(version_df)
    plot_version_search_dynamics(version_df)
    plot_dimension_scalability_error(dimension_df)
    plot_dimension_search_dynamics(dimension_df)
    plot_function_rank_heatmap(function_df)
    plot_function_log_error_heatmap(function_df)
    plot_robustness_boxplot(function_df)
    plot_ablation_radar(version_df)

    print("\n[OK] All 8 submitted-style COCO/BBOB figures reconstructed from audited revision data.")
    print(f"Output folder: {OUT_DIR}")


if __name__ == "__main__":
    main()