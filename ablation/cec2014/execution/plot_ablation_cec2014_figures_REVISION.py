# -*- coding: utf-8 -*-
"""
Publication-grade visualizations for DMGSO/MIDO-GS CEC2014 ablation study.

Input CSV columns:
    suite,function,D,version,run,FE_budget,best_f,error,
    fe_used,n_accept,n_reloc,wall_time_sec

Outputs:
    outputs/ablation/cec2014/figures/
    outputs/ablation/cec2014/summaries/
"""

from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Times New Roman registration for Windows/WSL
# ============================================================
from matplotlib import font_manager as fm

_TNR_FILES = [
    "/mnt/c/Windows/Fonts/times.ttf",
    "/mnt/c/Windows/Fonts/timesbd.ttf",
    "/mnt/c/Windows/Fonts/timesi.ttf",
    "/mnt/c/Windows/Fonts/timesbi.ttf",
]

_registered_tnr = []

for _font_path in _TNR_FILES:
    if Path(_font_path).exists():
        try:
            fm.fontManager.addfont(_font_path)
            _registered_tnr.append(Path(_font_path).name)
        except Exception as _exc:
            print(f"[FONT WARNING] {_font_path}: {_exc}")

if not _registered_tnr:
    raise RuntimeError(
        "Times New Roman font files were not found under "
        "/mnt/c/Windows/Fonts."
    )

print(
    "[FONT] Times New Roman registered from Windows fonts: "
    + ", ".join(_registered_tnr)
)
# ============================================================



ROOT = Path(__file__).resolve().parent

DEFAULT_INPUT = ROOT / "ablation_cec2014_run_summary_revision.csv"

OUT_ROOT = ROOT / "figures_cec2014_revision"
OUT_FIG = OUT_ROOT / "figures"
OUT_SUM = OUT_ROOT / "summaries"

OUT_FIG.mkdir(parents=True, exist_ok=True)
OUT_SUM.mkdir(parents=True, exist_ok=True)


plt.rcParams.update(
    {
        "font.family": "Times New Roman",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 12,
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.4,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.facecolor": "white",
    }
)


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


def cec2014_category(fid: int) -> str:
    if 1 <= fid <= 3:
        return "Unimodal"
    if 4 <= fid <= 16:
        return "Multimodal"
    if 17 <= fid <= 22:
        return "Hybrid"
    if 23 <= fid <= 30:
        return "Composition"
    return "Unknown"


def savefig(fig, filename: str) -> None:
    out = OUT_FIG / filename
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


def safe_log10(values):
    values = np.asarray(values, dtype=float)
    values = np.maximum(values, 1e-300)
    return np.log10(values)


def ordered_versions(df: pd.DataFrame) -> list[str]:
    available = list(df["version"].dropna().unique())
    ordered = [v for v in VERSION_ORDER if v in available]
    rest = [v for v in available if v not in ordered]
    return ordered + rest


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Compatibility mapping: audited revision schema -> submitted plotting schema.
    # Numerical values and plotting settings are unchanged.
    df = df.rename(
        columns={
            "dimension": "D",
            "run_id": "run",
            "FE_budget_max": "FE_budget",
        }
    )

    # Legacy submitted plotting code expects labels F1, F2, ..., F30.
    # Preserve the audited integer function_id and create only the
    # compatibility display/processing column required by the old code.
    df["function"] = "F" + df["function_id"].astype(int).astype(str)

    required = [
        "suite",
        "function",
        "D",
        "version",
        "run",
        "FE_budget",
        "best_f",
        "error",
        "fe_used",
        "n_accept",
        "n_reloc",
        "wall_time_sec",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    numeric_cols = [
        "D",
        "run",
        "FE_budget",
        "best_f",
        "error",
        "fe_used",
        "n_accept",
        "n_reloc",
        "wall_time_sec",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["error"])

    df["function_id"] = (
        df["function"]
        .astype(str)
        .str.extract(r"F(\d+)", expand=False)
        .astype(int)
    )

    df["category"] = df["function_id"].apply(cec2014_category)
    df["error_pos"] = np.maximum(df["error"].astype(float), 1e-300)
    df["log_error"] = np.log10(df["error_pos"])

    return df


def build_summaries(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    version_summary = (
        df.groupby(["suite", "version", "D"], as_index=False)
        .agg(
            runs=("run", "count"),
            functions=("function_id", "nunique"),
            error_mean=("error_pos", "mean"),
            error_median=("error_pos", "median"),
            error_std=("error_pos", "std"),
            error_min=("error_pos", "min"),
            error_max=("error_pos", "max"),
            fe_used_mean=("fe_used", "mean"),
            fe_used_median=("fe_used", "median"),
            n_accept_mean=("n_accept", "mean"),
            n_accept_median=("n_accept", "median"),
            n_reloc_mean=("n_reloc", "mean"),
            n_reloc_median=("n_reloc", "median"),
            wall_time_mean=("wall_time_sec", "mean"),
            wall_time_median=("wall_time_sec", "median"),
        )
    )

    function_summary = (
        df.groupby(["suite", "version", "function_id", "function", "category", "D"], as_index=False)
        .agg(
            runs=("run", "count"),
            error_mean=("error_pos", "mean"),
            error_median=("error_pos", "median"),
            error_std=("error_pos", "std"),
            error_min=("error_pos", "min"),
            error_max=("error_pos", "max"),
            fe_used_mean=("fe_used", "mean"),
            fe_used_median=("fe_used", "median"),
            n_accept_mean=("n_accept", "mean"),
            n_accept_median=("n_accept", "median"),
            n_reloc_mean=("n_reloc", "mean"),
            n_reloc_median=("n_reloc", "median"),
            wall_time_mean=("wall_time_sec", "mean"),
            wall_time_median=("wall_time_sec", "median"),
        )
    )

    category_summary = (
        df.groupby(["suite", "version", "category", "D"], as_index=False)
        .agg(
            runs=("run", "count"),
            functions=("function_id", "nunique"),
            error_mean=("error_pos", "mean"),
            error_median=("error_pos", "median"),
            error_std=("error_pos", "std"),
            n_accept_mean=("n_accept", "mean"),
            n_reloc_mean=("n_reloc", "mean"),
            wall_time_mean=("wall_time_sec", "mean"),
        )
    )

    version_summary.to_csv(OUT_SUM / "ablation_cec2014_version_summary.csv", index=False)
    function_summary.to_csv(OUT_SUM / "ablation_cec2014_function_summary.csv", index=False)
    category_summary.to_csv(OUT_SUM / "ablation_cec2014_category_summary.csv", index=False)

    print(f"[OK] saved: {OUT_SUM / 'ablation_cec2014_version_summary.csv'}")
    print(f"[OK] saved: {OUT_SUM / 'ablation_cec2014_function_summary.csv'}")
    print(f"[OK] saved: {OUT_SUM / 'ablation_cec2014_category_summary.csv'}")

    return version_summary, function_summary, category_summary


def plot_version_error_bar(version_summary: pd.DataFrame) -> None:
    df = version_summary.copy()
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
    ax.set_ylabel(r"$\log_{10}$ optimization error")
    ax.set_xlabel("Ablation variant")
    ax.set_title("Version-level ablation performance on CEC2014")
    ax.grid(True, axis="y", alpha=0.35)
    ax.legend(frameon=True)

    savefig(fig, "fig_ablation_cec2014_version_error_bar.png")


def plot_version_search_dynamics(version_summary: pd.DataFrame) -> None:
    df = version_summary.copy()
    versions = ordered_versions(df)

    df = df.set_index("version").reindex(versions)

    x = np.arange(len(versions))

    accept_vals = df["n_accept_mean"].fillna(0).to_numpy(dtype=float)
    reloc_vals = df["n_reloc_mean"].fillna(0).to_numpy(dtype=float)

    fig, ax1 = plt.subplots(figsize=(8.4, 5.0))

    # Preserve the submitted-paper color semantics for V0--V4.
    variant_colors = {
        "V0": "gray",
        "V1": "blue",
        "V2": "orange",
        "V3": "green",
        "V4": "red",
    }

    variant_legend_labels = {
        "V0": "V0 Directional",
        "V1": "V1 + Step",
        "V2": "V2 + Long-range",
        "V3": "V3 + Memory",
        "V4": "V4 Full",
    }

    bars = ax1.bar(
        x,
        accept_vals,
        width=0.60,
        color=[variant_colors[v] for v in versions],
        alpha=0.80,
    )

    ax1.set_xticks(x)
    ax1.set_xticklabels([VERSION_LABELS.get(v, v) for v in versions])
    ax1.set_xlabel("Ablation variant")
    ax1.set_ylabel("Mean accepted updates")
    ax1.grid(True, axis="y", alpha=0.35)

    ax2 = ax1.twinx()

    relocation_line, = ax2.plot(
        x,
        reloc_vals,
        color="black",
        linestyle="--",
        marker="X",
        linewidth=2.2,
        markersize=8,
        label="Relocation events",
    )

    ax2.set_ylabel("Mean relocation events")
    ax1.set_title("Search dynamics across DMGSO ablation variants")

    # Variant-specific legend, consistent with the submitted figure.
    ax1.legend(
        list(bars) + [relocation_line],
        [variant_legend_labels[v] for v in versions] + ["Relocation events"],
        loc="upper left",
        frameon=True,
    )

    savefig(fig, "fig_ablation_cec2014_version_search_dynamics.png")


def plot_function_rank_heatmap(function_summary: pd.DataFrame) -> None:
    df = function_summary.copy()

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

    ax.set_xlabel("Ablation variant")
    ax.set_ylabel("CEC2014 function")
    ax.set_title("Function-wise rank heatmap of ablation variants")

    cbar = fig.colorbar(im, ax=ax, shrink=0.82)
    cbar.set_label("Rank (1 = best)")

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{int(val)}", ha="center", va="center", fontsize=7)

    savefig(fig, "fig_ablation_cec2014_function_rank_heatmap.png")


def plot_function_log_error_heatmap(function_summary: pd.DataFrame) -> None:
    df = function_summary.copy()

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

    Z = safe_log10(pivot.to_numpy(dtype=float))

    fig, ax = plt.subplots(figsize=(7.8, 8.4))

    im = ax.imshow(
        Z,
        aspect="auto",
        cmap="magma",
    )

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(
        [VERSION_LABELS.get(v, v).replace("\n", " ") for v in pivot.columns],
        rotation=35,
        ha="right",
    )

    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([f"F{int(i)}" for i in pivot.index])

    ax.set_xlabel("Ablation variant")
    ax.set_ylabel("CEC2014 function")
    ax.set_title(r"Function-wise median error ($\log_{10}$ scale)")

    cbar = fig.colorbar(im, ax=ax, shrink=0.82)
    cbar.set_label(r"$\log_{10}$ median error")

    savefig(fig, "fig_ablation_cec2014_function_log_error_heatmap.png")


def plot_category_grouped_bar(category_summary: pd.DataFrame) -> None:
    df = category_summary.copy()

    categories = ["Unimodal", "Multimodal", "Hybrid", "Composition"]
    versions = ordered_versions(df)

    x = np.arange(len(categories))
    width = 0.8 / len(versions)

    fig, ax = plt.subplots(figsize=(8.8, 5.0))

    for i, version in enumerate(versions):
        vals = []

        for cat in categories:
            v = df[
                (df["category"] == cat)
                & (df["version"] == version)
            ]["error_median"]

            vals.append(float(v.iloc[0]) if len(v) else np.nan)

        vals = safe_log10(vals)

        ax.bar(
            x + (i - len(versions) / 2) * width + width / 2,
            vals,
            width,
            label=version,
            color=VERSION_COLORS.get(version, None),
            alpha=0.82,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel(r"$\log_{10}$ median error")
    ax.set_xlabel("CEC2014 function category")
    ax.set_title("Category-wise ablation performance on CEC2014")
    ax.grid(True, axis="y", alpha=0.35)
    ax.legend(frameon=True, ncol=2)

    savefig(fig, "fig_ablation_cec2014_category_grouped_bar.png")


def plot_category_rank_heatmap(category_summary: pd.DataFrame) -> None:
    df = category_summary.copy()

    categories = ["Unimodal", "Multimodal", "Hybrid", "Composition"]

    summary = (
        df.groupby(["category", "version"], as_index=False)
        .agg(error_median=("error_median", "median"))
    )

    summary["rank"] = (
        summary.groupby("category")["error_median"]
        .rank(method="min", ascending=True)
    )

    versions = ordered_versions(summary)

    pivot = (
        summary.pivot(
            index="category",
            columns="version",
            values="rank",
        )
        .reindex(index=categories)
        .reindex(columns=versions)
    )

    fig, ax = plt.subplots(figsize=(7.6, 4.6))

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
    ax.set_yticklabels(pivot.index)

    ax.set_xlabel("Ablation variant")
    ax.set_ylabel("CEC2014 category")
    ax.set_title("Category-wise ranking of ablation variants")

    cbar = fig.colorbar(im, ax=ax, shrink=0.82)
    cbar.set_label("Rank (1 = best)")

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{int(val)}", ha="center", va="center", fontsize=8)

    savefig(fig, "fig_ablation_cec2014_category_rank_heatmap.png")


def plot_robustness_boxplot(df: pd.DataFrame) -> None:
    versions = ordered_versions(df)

    data = []

    for version in versions:
        vals = (
            df[df["version"] == version]["error_pos"]
            .dropna()
            .to_numpy(dtype=float)
        )

        data.append(safe_log10(vals))

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

    ax.set_xlabel("Ablation variant")
    ax.set_ylabel(r"$\log_{10}$ final error")
    ax.set_title("Robustness of ablation variants across CEC2014 runs")
    ax.grid(True, axis="y", alpha=0.35)

    handles = [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            color=VERSION_COLORS.get(v, "#808080"),
            alpha=0.68,
        )
        for v in versions
    ]

    ax.legend(handles, versions, frameon=True, loc="upper right")

    savefig(fig, "fig_ablation_cec2014_robustness_boxplot.png")


def plot_version_evolution(version_summary: pd.DataFrame) -> None:
    df = version_summary.copy()
    versions = ordered_versions(df)

    df = df.set_index("version").reindex(versions)

    x = np.arange(len(versions))

    fig, ax1 = plt.subplots(figsize=(8.6, 5.2))

    ax1.plot(
        x,
        safe_log10(df["error_median"].to_numpy()),
        marker="o",
        linewidth=2.2,
        label="Median error",
        color="blue",
    )

    ax1.plot(
        x,
        safe_log10(df["error_mean"].to_numpy()),
        marker="s",
        linewidth=2.2,
        label="Mean error",
        color="orange",
    )

    ax1.set_ylabel(r"$\log_{10}$ optimization error")
    ax1.set_xlabel("Ablation variant")
    ax1.set_xticks(x)
    ax1.set_xticklabels([VERSION_LABELS.get(v, v) for v in versions])
    ax1.grid(True, axis="y", alpha=0.35)

    ax2 = ax1.twinx()

    ax2.plot(
        x,
        df["n_reloc_mean"].fillna(0).to_numpy(dtype=float),
        marker="X",
        linestyle="--",
        linewidth=2.0,
        color="black",
        label="Relocation events",
    )

    ax2.set_ylabel("Mean relocation events")

    ax1.set_title("Evolution of optimization error and relocation across variants")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        frameon=True,
        loc="upper right",
    )

    savefig(fig, "fig_ablation_cec2014_version_evolution.png")


def main(input_csv: Path) -> None:
    df = load_data(input_csv)

    version_summary, function_summary, category_summary = build_summaries(df)

    plot_version_error_bar(version_summary)
    plot_version_search_dynamics(version_summary)
    plot_function_rank_heatmap(function_summary)
    plot_function_log_error_heatmap(function_summary)
    plot_category_grouped_bar(category_summary)
    plot_category_rank_heatmap(category_summary)
    plot_robustness_boxplot(df)
    plot_version_evolution(version_summary)

    print("\n[OK] All CEC2014 ablation figures generated.")
    print(f"Output folder: {OUT_FIG}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot publication-grade CEC2014 ablation figures."
    )

    parser.add_argument(
        "--input",
        type=str,
        default=str(DEFAULT_INPUT),
        help="Path to CEC2014 ablation run summary CSV.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(Path(args.input))