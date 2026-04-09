from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch, Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clustering.feature_extractor import PROFILE_FEATURE_COLUMNS, extract_building_profile_features
from src.config import FEATURES_BDG2_PATH


FIG_DIR = ROOT / "figures" / "paper"
BDG2_META_PATH = ROOT / "data" / "bdg2" / "filtered_building_meta.csv"
GEPIII_META_PATH = ROOT / "data" / "gepiii" / "building_metadata.csv"
GEPIII_FILTERED_IDS_PATH = ROOT / "data" / "gepiii" / "filtered_meter_ids.csv"

FIG1_PATH = FIG_DIR / "paper_fig01_site_distribution.png"
FIG2_PATH = FIG_DIR / "paper_fig02_building_type_distribution.png"
FIG3_PATH = FIG_DIR / "paper_fig03_exp1_main_and_baselines.png"
FIG4A_PATH = FIG_DIR / "paper_fig04a_per_building_cv_rmse.png"
FIG4C_PATH = FIG_DIR / "paper_fig04c_bdg2_headline_dashboard.png"
FIG4_PATH = FIG_DIR / "paper_fig04_lstm_bsplit_by_type.png"
FIG4B_PATH = FIG_DIR / "paper_fig04b_lstm_bsplit_low_support.png"
FIG5_PATH = FIG_DIR / "paper_fig05_ssplit_site_panels.png"
FIG6_PATH = FIG_DIR / "paper_fig06_kmeans_and_profiles.png"
FIG7_PATH = FIG_DIR / "paper_fig07_cold_start_performance.png"
FIG8_PATH = FIG_DIR / "paper_fig08_annual_mean_residual_by_type.png"
FIG9_PATH = FIG_DIR / "paper_fig09_eui_vs_cps_scatter.png"
FIG10_PATH = FIG_DIR / "paper_fig10_model_sensitivity.png"
FIG11_PATH = FIG_DIR / "paper_fig11_case_studies.png"
FIG12_PATH = FIG_DIR / "paper_fig12_cross_dataset_relative_ordering.png"
FIG14_PATH = FIG_DIR / "paper_fig14_heew_ranking_shift.png"
FIG15_PATH = FIG_DIR / "paper_fig15_lag_ablation.png"
FIG16_PATH = FIG_DIR / "paper_fig16_information_budget.png"
FIG18_PATH = FIG_DIR / "paper_fig18_parity_shift_transport.png"
FIGS1_PATH = FIG_DIR / "paper_figS1_heew_load_quartiles.png"

PALETTE = {
    "lgbm": "#3b6c8f",
    "lgbm_lag": "#5f8f3b",
    "lstm": "#b44b5c",
    "patchtst": "#2f7d6d",
    "lag baseline": "#8f7a3b",
    "single-building": "#7a5c99",
}

SAVE_DPI = 320
BASE_FONT_SIZE = 10.6
SMALL_FONT_SIZE = 9.3
PANEL_TITLE_SIZE = 11.6
ANNOTATION_FONT_SIZE = 9.0

plt.rcParams.update(
    {
        "font.family": "STIXGeneral",
        "mathtext.fontset": "stix",
        "font.size": BASE_FONT_SIZE,
        "axes.labelsize": BASE_FONT_SIZE,
        "axes.titlesize": PANEL_TITLE_SIZE,
        "axes.titleweight": "regular",
        "xtick.labelsize": SMALL_FONT_SIZE,
        "ytick.labelsize": SMALL_FONT_SIZE,
        "legend.fontsize": SMALL_FONT_SIZE,
        "axes.unicode_minus": False,
        "figure.dpi": 160,
        "savefig.dpi": SAVE_DPI,
    }
)


def _ensure_dir() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def _format_split_name(value: str) -> str:
    mapping = {
        "t_split": "T-split",
        "b_split": "B-split",
        "s_split": "S-split",
    }
    return mapping.get(value, value)


def _save_figure(fig: plt.Figure, path: Path, *, tight: bool = False) -> None:
    save_kwargs = {"dpi": SAVE_DPI, "facecolor": "white"}
    if tight:
        save_kwargs["bbox_inches"] = "tight"
        save_kwargs["pad_inches"] = 0.02
    fig.savefig(path, **save_kwargs)


def _load_bdg2_feature_frame() -> pd.DataFrame:
    frame = pd.read_parquet(
        FEATURES_BDG2_PATH,
        columns=["building_id", "site_id", "building_type", "timestamp", "meter_reading"],
    )
    frame["building_id"] = frame["building_id"].astype(str)
    frame["site_id"] = frame["site_id"].astype(str)
    frame["building_type"] = frame["building_type"].astype(str)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    return frame.sort_values(["building_id", "timestamp"]).reset_index(drop=True)


def _cv_rmse(frame: pd.DataFrame) -> float:
    y_true = frame["y_true"].to_numpy(dtype=np.float64)
    y_pred = frame["y_pred"].to_numpy(dtype=np.float64)
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    mean_y = float(np.mean(y_true))
    return rmse / mean_y if mean_y else np.nan


def build_figure_1() -> Path:
    meta = pd.read_csv(BDG2_META_PATH)
    eligible = set(pd.read_csv(ROOT / "tables" / "eligible_sites_for_loso.csv")["site_id"].astype(str))
    counts = (
        meta.groupby("site_id", sort=True)["building_id"]
        .nunique()
        .rename("building_count")
        .reset_index()
        .sort_values(["building_count", "site_id"], ascending=[False, True])
        .reset_index(drop=True)
    )
    counts["eligible"] = counts["site_id"].isin(eligible)

    fig, ax = plt.subplots(figsize=(10.8, 6.8))
    y = np.arange(len(counts), dtype=np.float64)
    bars = ax.barh(
        y,
        counts["building_count"],
        height=0.72,
        color=np.where(counts["eligible"], "#3b6c8f", "#c8d6e5"),
        edgecolor="#4a4a4a",
        linewidth=0.8,
    )
    for bar, eligible in zip(bars, counts["eligible"], strict=False):
        if not bool(eligible):
            bar.set_hatch("//")
    for ypos, value in zip(y, counts["building_count"], strict=False):
        ax.text(value + 3, ypos, f"{int(value)}", ha="left", va="center", fontsize=ANNOTATION_FONT_SIZE)

    ax.set_yticks(y, counts["site_id"])
    ax.invert_yaxis()
    ax.set_xlabel("Filtered building count")
    ax.set_ylabel("Site")
    ax.legend(
        handles=[
            Patch(facecolor="#3b6c8f", edgecolor="#4a4a4a", label="Eligible for S-split"),
            Patch(facecolor="#c8d6e5", edgecolor="#4a4a4a", hatch="//", label="Retained but not eligible"),
        ],
        frameon=False,
        loc="lower right",
    )
    ax.grid(axis="x", alpha=0.2, linestyle="--")
    fig.tight_layout()
    _save_figure(fig, FIG1_PATH)
    plt.close(fig)
    return FIG1_PATH


def build_figure_2() -> Path:
    bdg2 = pd.read_csv(BDG2_META_PATH)
    gepiii_meta = pd.read_csv(GEPIII_META_PATH)
    gepiii_ids = pd.read_csv(GEPIII_FILTERED_IDS_PATH)[["building_id"]]
    gepiii = gepiii_meta.merge(gepiii_ids, on="building_id", how="inner")
    gepiii["building_type"] = gepiii["primary_use"].astype(str)

    bdg2_counts = bdg2.groupby("building_type")["building_id"].nunique().rename("BDG2")
    gepiii_counts = gepiii.groupby("building_type")["building_id"].nunique().rename("GEPIII")
    counts = pd.concat([bdg2_counts, gepiii_counts], axis=1).fillna(0).astype(int)

    support = pd.read_csv(ROOT / "tables" / "exp1_type_sample_counts.csv")
    support = support[support["split_name"] == "b_split"][["building_type", "n_buildings"]]
    low_support = {
        row["building_type"]
        for _, row in support.iterrows()
        if int(row["n_buildings"]) < 20
    }

    counts = counts.assign(max_count=counts.max(axis=1)).sort_values("max_count", ascending=False).drop(columns="max_count")
    labels = [f"{name}*" if name in low_support else name for name in counts.index]
    y = np.arange(len(counts), dtype=np.float64)
    width = 0.36
    fig_height = max(7.0, 0.36 * len(counts) + 1.8)

    fig, ax = plt.subplots(figsize=(11.8, fig_height))
    ax.barh(y - width / 2, counts["BDG2"], height=width, color="#5b8fb9", label="BDG2")
    ax.barh(y + width / 2, counts["GEPIII"], height=width, color="#c97b63", label="GEPIII")

    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Filtered building count")
    ax.legend(frameon=False)
    ax.grid(axis="x", alpha=0.2, linestyle="--")
    fig.tight_layout()
    _save_figure(fig, FIG2_PATH)
    plt.close(fig)
    return FIG2_PATH


def build_figure_3() -> Path:
    main = pd.read_csv(ROOT / "tables" / "exp1_metrics.csv")
    baseline = pd.read_csv(ROOT / "tables" / "exp1_baseline_metrics.csv")
    repeated = pd.read_csv(ROOT / "tables" / "repeated_main_metrics_bdg2_summary.csv")
    history_budget = pd.read_csv(ROOT / "tables" / "exp1_history_budget_bsplit.csv")
    main_models = [
        model
        for model in ["lgbm", "lgbm_lag", "lstm", "patchtst"]
        if f"{model}_cv_rmse" in main.columns
    ]

    rows: list[dict[str, object]] = []
    for _, row in main.iterrows():
        split_name = str(row["split_name"])
        for model in main_models:
            rows.append(
                {
                    "split_name": split_name,
                    "label": model,
                    "cv_rmse": float(row[f"{model}_cv_rmse"]),
                    "err_low": np.nan,
                    "err_high": np.nan,
                }
            )
    for _, row in repeated.iterrows():
        split_name = str(row["split_name"])
        if split_name not in {"t_split", "b_split"}:
            continue
        mask = [
            record["split_name"] == split_name and record["label"] == row["model"]
            for record in rows
        ]
        idx = int(np.flatnonzero(mask)[0])
        rows[idx]["err_low"] = max(0.0, rows[idx]["cv_rmse"] - float(row["cv_rmse_ci95_low"]))
        rows[idx]["err_high"] = max(0.0, float(row["cv_rmse_ci95_high"]) - rows[idx]["cv_rmse"])

    label_map = {"naive": "lag baseline", "single_building": "single-building"}
    for _, row in baseline.iterrows():
        rows.append(
            {
                "split_name": str(row["split_name"]),
                "label": label_map[str(row["model"])],
                "cv_rmse": float(row["cv_rmse"]),
                "err_low": np.nan,
                "err_high": np.nan,
            }
        )

    data = pd.DataFrame(rows)
    split_order = ["t_split", "b_split", "s_split"]
    label_order = main_models + ["lag baseline", "single-building"]
    x = np.arange(len(split_order), dtype=np.float64)
    width = 0.12 if len(label_order) >= 6 else 0.15

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14.5, 6.2),
        gridspec_kw={"width_ratios": [1.45, 1.0]},
    )
    ax = axes[0]
    for idx, label in enumerate(label_order):
        subset = data[data["label"] == label].set_index("split_name").reindex(split_order)
        offsets = x + (idx - (len(label_order) - 1) / 2) * width
        values = subset["cv_rmse"].to_numpy(dtype=np.float64)
        mask = np.isfinite(values)
        if not mask.any():
            continue
        bars = ax.bar(
            offsets[mask],
            values[mask],
            width=width,
            color=PALETTE[label],
            label=label,
            alpha=0.92,
        )
        err_low = subset["err_low"].to_numpy(dtype=np.float64)[mask]
        err_high = subset["err_high"].to_numpy(dtype=np.float64)[mask]
        if np.isfinite(err_low).any() or np.isfinite(err_high).any():
            ax.errorbar(
                offsets[mask],
                values[mask],
                yerr=np.vstack([np.nan_to_num(err_low), np.nan_to_num(err_high)]),
                fmt="none",
                ecolor="#333333",
                elinewidth=1.1,
                capsize=4,
                zorder=3,
            )
        for bar, value in zip(bars, values[mask], strict=False):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value * 1.04,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=ANNOTATION_FONT_SIZE,
                rotation=90,
            )

    ax.set_xticks(x, [_format_split_name(name) for name in split_order])
    ax.set_ylabel("Row-wise CV(RMSE)")
    ax.set_yscale("log")
    ax.set_ylim(0.25, max(2.2, float(data["cv_rmse"].max()) * 1.12))
    ax.grid(axis="y", alpha=0.2, linestyle="--")
    ax.legend(ncol=3, frameon=False)

    focus_labels = [label for label in ["lgbm_lag", "lstm", "patchtst"] if label in label_order]
    if focus_labels:
        axins = inset_axes(ax, width="47%", height="50%", loc="upper left", borderpad=1.0)
        inset_x = np.arange(len(split_order), dtype=np.float64)
        inset_width = 0.22
        for idx, label in enumerate(focus_labels):
            subset = data[data["label"] == label].set_index("split_name").reindex(split_order)
            offsets = inset_x + (idx - (len(focus_labels) - 1) / 2) * inset_width
            values = subset["cv_rmse"].to_numpy(dtype=np.float64)
            axins.bar(
                offsets,
                values,
                width=inset_width,
                color=PALETTE[label],
                alpha=0.92,
            )
        focus_values = data[data["label"].isin(focus_labels)]["cv_rmse"].to_numpy(dtype=np.float64)
        axins.set_xticks(inset_x, ["T", "B", "S"])
        axins.set_ylim(float(focus_values.min()) * 0.94, float(focus_values.max()) * 1.08)
        axins.set_title("Parity-focus inset", fontsize=SMALL_FONT_SIZE)
        axins.grid(axis="y", alpha=0.18, linestyle="--")
        ax.indicate_inset_zoom(axins, edgecolor="#666666", alpha=0.9)

    ax = axes[1]
    history_order = [0, 1, 3, 7, 14]
    label_map = {
        "lgbm": "LightGBM (realised start = N)",
        "lstm": "LSTM (realised start = N + 7d)",
        "patchtst": "PatchTST (realised start = N + 7d)",
    }
    for model_name in ["lgbm", "lstm", "patchtst"]:
        subset = (
            history_budget[history_budget["model"] == model_name]
            .set_index("nominal_history_days")
            .reindex(history_order)
            .reset_index()
        )
        x_values = subset["nominal_history_days"].to_numpy(dtype=float)
        y_values = subset["cv_rmse"].to_numpy(dtype=float)
        ax.plot(
            x_values,
            y_values,
            marker="o",
            linewidth=2.0,
            markersize=5.5,
            color=PALETTE[model_name],
            label=label_map[model_name],
        )
        for x_value, y_value in zip(x_values, y_values, strict=False):
            ax.text(
                x_value,
                y_value + 0.004,
                f"{y_value:.3f}",
                ha="center",
                va="bottom",
                fontsize=ANNOTATION_FONT_SIZE,
            )

    ax.set_xticks(history_order)
    ax.set_xlabel("Nominal History Budget (days)")
    ax.set_ylabel("B-split CV(RMSE)")
    ax.grid(axis="both", alpha=0.2, linestyle="--")
    ax.legend(frameon=False, loc="upper left")

    fig.tight_layout()
    _save_figure(fig, FIG3_PATH)
    plt.close(fig)
    return FIG3_PATH


def build_figure_4() -> Path:
    pred = pd.read_csv(ROOT / "results" / "exp1_predictions" / "lstm_b_split.csv")
    pred["sq_error"] = (pred["y_pred"] - pred["y_true"]) ** 2
    grouped = (
        pred.groupby(["building_id", "building_type"], sort=True)
        .agg(mean_true=("y_true", "mean"), mse=("sq_error", "mean"))
        .reset_index()
    )
    grouped["cv_rmse"] = np.sqrt(grouped["mse"]) / grouped["mean_true"].replace({0.0: np.nan})
    grouped = grouped.drop(columns=["mean_true", "mse"])
    support = grouped.groupby("building_type")["building_id"].nunique().rename("n_buildings")
    grouped = grouped.merge(support, on="building_type", how="left")
    grouped["support_group"] = np.where(grouped["n_buildings"] >= 20, "high", "low")

    overall_median = float(grouped["cv_rmse"].median())
    high_support = grouped[grouped["support_group"] == "high"].copy()
    low_support = grouped[grouped["support_group"] == "low"].copy()

    high_order = (
        high_support.groupby("building_type")["cv_rmse"]
        .median()
        .sort_values()
        .index
        .tolist()
    )
    low_order = (
        low_support.groupby("building_type")["cv_rmse"]
        .median()
        .sort_values()
        .index
        .tolist()
    )

    fig, ax = plt.subplots(figsize=(10.5, 6.8))
    high_labels = [f"{name} (n={int(support[name])})" for name in high_order]
    high_data = [high_support.loc[high_support["building_type"] == name, "cv_rmse"].to_numpy() for name in high_order]
    box = ax.boxplot(
        high_data,
        vert=False,
        tick_labels=high_labels,
        patch_artist=True,
        showfliers=False,
    )
    for patch in box["boxes"]:
        patch.set(facecolor="#dbe7f1", edgecolor="#456882", linewidth=1.1)
    for median in box["medians"]:
        median.set(color="#b44b5c", linewidth=1.5)

    rng = np.random.default_rng(42)
    for idx, name in enumerate(high_order, start=1):
        values = high_support.loc[high_support["building_type"] == name, "cv_rmse"].to_numpy()
        jitter = rng.uniform(-0.12, 0.12, size=len(values))
        ax.scatter(values, np.full(len(values), idx) + jitter, s=14, alpha=0.28, color="#3b6c8f", edgecolors="none")
    ax.axvline(overall_median, linestyle="--", color="#444444", linewidth=1.1, label="Overall median")
    ax.set_xlabel("Per-building CV(RMSE)")
    ax.set_title("LSTM B-split by type: high-support types", fontsize=PANEL_TITLE_SIZE)
    ax.grid(axis="x", alpha=0.2, linestyle="--")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    _save_figure(fig, FIG4_PATH)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    low_labels = [f"{name}* (n={int(support[name])})" for name in low_order]
    for idx, name in enumerate(low_order, start=1):
        values = low_support.loc[low_support["building_type"] == name, "cv_rmse"].to_numpy()
        if len(values) > 1:
            ax.hlines(idx, values.min(), values.max(), color="#a0a0a0", linewidth=1.0, alpha=0.9)
        jitter = rng.uniform(-0.10, 0.10, size=len(values))
        ax.scatter(values, np.full(len(values), idx) + jitter, s=28, alpha=0.85, color="#8f6c3b", edgecolors="white", linewidths=0.5)
    ax.axvline(overall_median, linestyle="--", color="#444444", linewidth=1.1)
    ax.set_yticks(np.arange(1, len(low_order) + 1), low_labels)
    ax.set_xlabel("Per-building CV(RMSE)")
    ax.set_title("LSTM B-split by type: low-support types", fontsize=PANEL_TITLE_SIZE)
    ax.grid(axis="x", alpha=0.2, linestyle="--")

    fig.tight_layout()
    _save_figure(fig, FIG4B_PATH)
    plt.close(fig)
    return FIG4_PATH


def build_figure_4a() -> Path:
    metrics = pd.read_csv(ROOT / "tables" / "exp1_per_building_metrics.csv")
    summary = pd.read_csv(ROOT / "tables" / "exp1_per_building_summary.csv")
    split_order = ["t_split", "b_split", "s_split"]
    model_order = [model for model in ["lgbm", "lgbm_lag", "lstm", "patchtst"] if model in metrics["model"].astype(str).unique()]
    model_labels = {
        "lgbm": "LightGBM",
        "lgbm_lag": "LightGBM + lag",
        "lstm": "LSTM",
        "patchtst": "PatchTST",
    }

    cv_values = metrics["cv_rmse"].replace([np.inf, -np.inf], np.nan).dropna()
    if cv_values.empty:
        raise RuntimeError("Per-building CV(RMSE) table is empty.")
    y_min = max(float(cv_values.min()) * 0.85, 1e-3)
    y_max = float(cv_values.max()) * 1.15

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.6), sharey=True)
    rng = np.random.default_rng(42)

    for ax, split_name in zip(axes, split_order, strict=False):
        subset = metrics[metrics["split_name"] == split_name].copy()
        positions = np.arange(1, len(model_order) + 1)
        data = [
            subset.loc[subset["model"] == model, "cv_rmse"].to_numpy(dtype=np.float64)
            for model in model_order
        ]
        box = ax.boxplot(
            data,
            positions=positions,
            widths=0.58,
            patch_artist=True,
            showfliers=False,
            showmeans=True,
            meanprops={
                "marker": "D",
                "markerfacecolor": "white",
                "markeredgecolor": "#333333",
                "markersize": 4.8,
            },
        )
        for patch, model in zip(box["boxes"], model_order, strict=False):
            patch.set(facecolor=PALETTE[model], alpha=0.28, edgecolor=PALETTE[model], linewidth=1.1)
        for whisker in box["whiskers"]:
            whisker.set(color="#6b6b6b", linewidth=0.9)
        for cap in box["caps"]:
            cap.set(color="#6b6b6b", linewidth=0.9)
        for median in box["medians"]:
            median.set(color="#202020", linewidth=1.4)

        for pos, model in zip(positions, model_order, strict=False):
            values = subset.loc[subset["model"] == model, "cv_rmse"].to_numpy(dtype=np.float64)
            jitter = rng.uniform(-0.16, 0.16, size=len(values))
            ax.scatter(
                np.full(len(values), pos, dtype=np.float64) + jitter,
                values,
                s=5,
                alpha=0.10,
                color=PALETTE[model],
                edgecolors="none",
                rasterized=True,
            )
            share_best = summary.loc[
                (summary["split_name"] == split_name) & (summary["model"] == model),
                "share_best_buildings",
            ].iloc[0]
            ax.text(
                pos,
                0.98,
                f"best {share_best:.0%}",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=ANNOTATION_FONT_SIZE,
                color="#333333",
            )

        n_buildings = int(subset["building_id"].nunique())
        ax.set_title(f"{_format_split_name(split_name)} (n={n_buildings})", fontsize=PANEL_TITLE_SIZE)
        ax.set_xticks(positions, [model_labels[model] for model in model_order])
        ax.set_yscale("log")
        ax.set_ylim(y_min, y_max)
        ax.grid(axis="y", alpha=0.2, linestyle="--")
        ax.tick_params(axis="x", length=0)

    axes[0].set_ylabel("Per-building CV(RMSE)")
    fig.tight_layout()
    _save_figure(fig, FIG4A_PATH)
    plt.close(fig)
    return FIG4A_PATH


def build_figure_4c() -> Path:
    metrics = pd.read_csv(ROOT / "tables" / "exp1_metrics.csv").set_index("split_name")
    per_building = pd.read_csv(ROOT / "tables" / "exp1_per_building_summary.csv")
    per_building = per_building.set_index(["split_name", "model"])

    split_order = ["t_split", "b_split", "s_split"]
    model_order = ["lgbm", "lgbm_lag", "lstm", "patchtst"]
    model_labels = {
        "lgbm": "LightGBM",
        "lgbm_lag": "LightGBM+lag",
        "lstm": "LSTM",
        "patchtst": "PatchTST",
    }
    metric_labels = ["Pooled\n$\\downarrow$", "Median\n$\\downarrow$", "Share-best\n$\\uparrow$"]
    metric_columns = {
        "pooled": {
            "lgbm": "lgbm_cv_rmse",
            "lgbm_lag": "lgbm_lag_cv_rmse",
            "lstm": "lstm_cv_rmse",
            "patchtst": "patchtst_cv_rmse",
        },
        "median": "median_cv_rmse",
        "share": "share_best_buildings",
    }
    cmap = LinearSegmentedColormap.from_list(
        "headline_dashboard",
        ["#f6f1e7", "#d9e5d1", "#7ba05b", "#4d7d35"],
    )

    fig, axes = plt.subplots(1, len(split_order), figsize=(12.2, 4.8), sharey=True)
    heatmaps = []

    for ax, split_name in zip(axes, split_order, strict=False):
        value_frame = pd.DataFrame(index=model_order, columns=["pooled", "median", "share"], dtype=np.float64)
        for model_name in model_order:
            value_frame.loc[model_name, "pooled"] = float(metrics.loc[split_name, metric_columns["pooled"][model_name]])
            value_frame.loc[model_name, "median"] = float(per_building.loc[(split_name, model_name), metric_columns["median"]])
            value_frame.loc[model_name, "share"] = float(per_building.loc[(split_name, model_name), metric_columns["share"]])

        score_frame = pd.DataFrame(index=model_order, columns=["pooled", "median", "share"], dtype=np.float64)
        for metric_name in ["pooled", "median"]:
            values = value_frame[metric_name].to_numpy(dtype=np.float64)
            value_range = float(values.max() - values.min())
            if value_range <= 0:
                score_frame[metric_name] = 1.0
            else:
                score_frame[metric_name] = (values.max() - values) / value_range
        share_values = value_frame["share"].to_numpy(dtype=np.float64)
        share_range = float(share_values.max() - share_values.min())
        if share_range <= 0:
            score_frame["share"] = 1.0
        else:
            score_frame["share"] = (share_values - share_values.min()) / share_range

        image = ax.imshow(score_frame.to_numpy(dtype=np.float64), cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")
        heatmaps.append(image)

        for x_pos in np.arange(-0.5, 3.5, 1.0):
            ax.axvline(x_pos, color="white", linewidth=1.2, alpha=0.95)
        for y_pos in np.arange(-0.5, len(model_order) + 0.5, 1.0):
            ax.axhline(y_pos, color="white", linewidth=1.2, alpha=0.95)

        parity_row = model_order.index("lgbm_lag")
        ax.add_patch(
            Rectangle(
                (-0.5, parity_row - 0.5),
                3.0,
                1.0,
                fill=False,
                edgecolor=PALETTE["lgbm_lag"],
                linewidth=1.8,
            )
        )

        for row_index, model_name in enumerate(model_order):
            for col_index, metric_name in enumerate(["pooled", "median", "share"]):
                value = float(value_frame.iloc[row_index, col_index])
                score = float(score_frame.iloc[row_index, col_index])
                if metric_name == "share":
                    label = f"{100.0 * value:.1f}%"
                else:
                    label = f"{value:.3f}"
                text_color = "white" if score >= 0.60 else "#2a2a2a"
                font_weight = "bold" if score >= 0.92 else "regular"
                ax.text(
                    col_index,
                    row_index,
                    label,
                    ha="center",
                    va="center",
                    fontsize=ANNOTATION_FONT_SIZE,
                    color=text_color,
                    fontweight=font_weight,
                )

        ax.set_title(_format_split_name(split_name), fontsize=PANEL_TITLE_SIZE)
        ax.set_xticks(np.arange(3, dtype=np.float64), metric_labels)
        ax.tick_params(axis="x", length=0)
        if ax is axes[0]:
            ax.set_yticks(np.arange(len(model_order), dtype=np.float64), [model_labels[name] for name in model_order])
        else:
            ax.set_yticks(np.arange(len(model_order), dtype=np.float64), [""] * len(model_order))
        ax.tick_params(axis="y", length=0)

    fig.subplots_adjust(left=0.08, right=0.88, bottom=0.16, top=0.88, wspace=0.10)
    cax = fig.add_axes([0.905, 0.19, 0.015, 0.60])
    cbar = fig.colorbar(heatmaps[-1], cax=cax)
    cbar.set_label("Relative within-metric performance")
    _save_figure(fig, FIG4C_PATH)
    plt.close(fig)
    return FIG4C_PATH


def build_figure_5() -> Path:
    metrics = pd.read_csv(ROOT / "tables" / "exp1_site_metrics.csv")
    site_counts = pd.read_csv(ROOT / "tables" / "eligible_sites_for_loso.csv")
    site_counts["site_id"] = site_counts["site_id"].astype(str)
    counts_map = site_counts.set_index("site_id")["building_count"].to_dict()

    lstm_order = (
        metrics[metrics["model"] == "lstm"]
        .sort_values("cv_rmse")
        .loc[:, "site_id"]
        .astype(str)
        .tolist()
    )
    ordered_sites = [site for site in lstm_order if site in counts_map]
    ordered_metrics = metrics.copy()
    ordered_metrics["site_id"] = ordered_metrics["site_id"].astype(str)
    ordered_metrics = ordered_metrics[ordered_metrics["site_id"].isin(ordered_sites)].copy()
    ordered_metrics["site_id"] = pd.Categorical(ordered_metrics["site_id"], categories=ordered_sites, ordered=True)

    model_order = [model for model in ["lgbm", "lgbm_lag", "lstm", "patchtst"] if model in metrics["model"].astype(str).unique()]
    fig_width = 4.3 * len(model_order) + 1.8
    fig, axes = plt.subplots(1, len(model_order), figsize=(fig_width, 8.2), sharey=True)
    if len(model_order) == 1:
        axes = [axes]
    global_xmax = float(ordered_metrics["cv_rmse"].max()) * 1.08
    for ax, model in zip(axes, model_order, strict=False):
        subset = ordered_metrics[ordered_metrics["model"] == model].sort_values("site_id")
        sizes = subset["site_id"].map(counts_map).to_numpy(dtype=np.float64)
        scaled_sizes = 26 + 2.0 * sizes
        y = np.arange(len(subset), dtype=np.float64)
        ax.hlines(y, 0.0, subset["cv_rmse"].to_numpy(dtype=np.float64), color="#cfd8df", linewidth=1.0, alpha=0.9)
        ax.scatter(
            subset["cv_rmse"],
            y,
            s=scaled_sizes,
            color=PALETTE[model],
            alpha=0.88,
            edgecolors="white",
            linewidths=0.7,
            zorder=3,
        )
        weights = subset["site_id"].map(counts_map).to_numpy(dtype=np.float64)
        weighted_mean = float(np.average(subset["cv_rmse"], weights=weights))
        ax.axvline(weighted_mean, linestyle="--", color="#555555", linewidth=1.0)
        ax.set_xlabel("CV(RMSE)")
        title_map = {
            "lgbm": "LightGBM",
            "lgbm_lag": "LightGBM + lag",
            "lstm": "LSTM",
            "patchtst": "PatchTST",
        }
        ax.set_title(title_map[model], fontsize=PANEL_TITLE_SIZE)
        ax.grid(axis="x", alpha=0.2, linestyle="--")
        ax.set_xlim(0.0, global_xmax)
        ax.set_ylim(-0.6, len(subset) - 0.4)
        ax.invert_yaxis()
    axes[0].set_yticks(np.arange(len(ordered_sites)), ordered_sites)
    axes[0].set_ylabel("Site (ordered by LSTM CV(RMSE))")
    fig.tight_layout()
    _save_figure(fig, FIG5_PATH)
    plt.close(fig)
    return FIG5_PATH


def build_figure_6() -> Path:
    k_metrics = pd.read_csv(ROOT / "tables" / "kmeans_k_metrics.csv")
    labels = pd.read_csv(ROOT / "data" / "clustering" / "train_building_cluster_labels.csv")

    feature_frame = _load_bdg2_feature_frame()
    train_buildings = set(labels["building_id"].astype(str))
    train_frame = feature_frame[feature_frame["building_id"].isin(train_buildings)].reset_index(drop=True)
    profiles = extract_building_profile_features(train_frame)
    profiles = profiles.merge(labels, on="building_id", how="inner")

    profile_matrix = profiles[PROFILE_FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    max_values = np.maximum(profile_matrix.max(axis=1, keepdims=True), 1e-6)
    normalized = profile_matrix / max_values
    norm_df = pd.DataFrame(normalized, columns=PROFILE_FEATURE_COLUMNS)
    norm_df["cluster_label"] = profiles["cluster_label"].to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=(15.2, 6.2))

    ax = axes[0]
    ax.plot(k_metrics["k"], k_metrics["silhouette_score"], marker="o", color="#3b6c8f", linewidth=2, label="Silhouette")
    ax.set_xlabel("K")
    ax.set_ylabel("Silhouette score", color="#3b6c8f")
    ax.tick_params(axis="y", labelcolor="#3b6c8f")
    ax.set_xticks(k_metrics["k"])
    ax.grid(axis="x", alpha=0.15)
    ax2 = ax.twinx()
    ax2.plot(k_metrics["k"], k_metrics["calinski_harabasz_score"], marker="s", color="#b44b5c", linewidth=2, label="Calinski-Harabasz")
    ax2.set_ylabel("Calinski-Harabasz score", color="#b44b5c")
    ax2.tick_params(axis="y", labelcolor="#b44b5c")
    selected_k = int(k_metrics.loc[k_metrics["selected"], "k"].iloc[0])
    ax.axvline(selected_k, linestyle="--", color="#444444", linewidth=1.0)
    ax.set_title("Cluster selection metrics", fontsize=PANEL_TITLE_SIZE)

    hours = np.arange(24, dtype=np.float64)
    colors = {0: "#3b6c8f", 1: "#b44b5c", 2: "#2f7d6d"}
    ax = axes[1]
    for cluster_label, cluster_frame in norm_df.groupby("cluster_label", sort=True):
        weekday = cluster_frame[[f"workday_hour_{hour:02d}" for hour in range(24)]].to_numpy(dtype=np.float64)
        weekend = cluster_frame[[f"weekend_hour_{hour:02d}" for hour in range(24)]].to_numpy(dtype=np.float64)
        for values, linestyle, label_suffix in [
            (weekday, "-", "weekday"),
            (weekend, "--", "weekend"),
        ]:
            mean = values.mean(axis=0)
            q25 = np.quantile(values, 0.25, axis=0)
            q75 = np.quantile(values, 0.75, axis=0)
            ax.plot(hours, mean, linestyle=linestyle, linewidth=2, color=colors[int(cluster_label)], label=f"Cluster {int(cluster_label)} {label_suffix}")
            ax.fill_between(hours, q25, q75, color=colors[int(cluster_label)], alpha=0.10)

    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Normalised median load")
    ax.set_title("Cluster profile shapes", fontsize=PANEL_TITLE_SIZE)
    ax.set_xticks(range(0, 24, 3))
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    ax.grid(alpha=0.2, linestyle="--")

    fig.tight_layout(rect=[0, 0.03, 1, 1])
    _save_figure(fig, FIG6_PATH)
    plt.close(fig)
    return FIG6_PATH


def build_figure_7() -> Path:
    metrics = pd.read_csv(ROOT / "tables" / "exp2_cold_start_metrics.csv")
    model_order = ["lgbm", "lgbm_lag", "lstm", "patchtst"]
    title_map = {
        "lgbm": "LightGBM",
        "lgbm_lag": "LightGBM + lag",
        "lstm": "LSTM",
        "patchtst": "PatchTST",
    }
    all_values = metrics["cv_rmse"].to_numpy(dtype=np.float64)
    value_range = float(np.ptp(all_values))
    y_min = max(0.0, float(all_values.min()) - 0.05 * value_range)
    y_max = float(all_values.max()) + 0.08 * value_range
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.2), sharey=True)

    for ax, model in zip(axes.flat, model_order, strict=False):
        subset = metrics[metrics["model"] == model].copy()
        history_days = [3, 7, 14]
        baseline = []
        grouped = []
        for day in history_days:
            baseline.append(float(subset.loc[subset["strategy"] == f"all_mix_after_{day}d", "cv_rmse"].iloc[0]))
            grouped.append(float(subset.loc[subset["strategy"] == f"cold_start_cluster_group_{day}d", "cv_rmse"].iloc[0]))
        ax.plot(history_days, baseline, marker="o", linestyle="--", color="#555555", label="all_mix_after_Nd")
        ax.plot(history_days, grouped, marker="o", linestyle="-", color=PALETTE[model], label="cold_start_cluster_group_Nd")
        for x_value, y_value in zip(history_days, baseline, strict=False):
            ax.text(x_value, y_value + 0.004, f"{y_value:.3f}", ha="center", va="bottom", fontsize=ANNOTATION_FONT_SIZE)
        for x_value, y_value in zip(history_days, grouped, strict=False):
            ax.text(x_value, y_value - 0.010, f"{y_value:.3f}", ha="center", va="top", fontsize=ANNOTATION_FONT_SIZE)
        ax.set_xticks(history_days)
        ax.set_xlabel("Early-history window (days)")
        ax.set_ylabel("CV(RMSE)")
        ax.set_title(title_map[model], fontsize=PANEL_TITLE_SIZE)
        ax.grid(alpha=0.2, linestyle="--")
        ax.set_ylim(y_min, y_max)
        if model == "lgbm":
            ax.legend(frameon=False)

    fig.tight_layout()
    _save_figure(fig, FIG7_PATH)
    plt.close(fig)
    return FIG7_PATH


def build_figure_8() -> Path:
    ranking = pd.read_csv(ROOT / "results" / "exp3_eui_vs_cps_ranking.csv")
    support = ranking.groupby("building_type")["building_id"].size().rename("n_buildings")
    summary = (
        ranking.groupby("building_type")["annual_mean_residual"]
        .agg(median="median", q1=lambda s: s.quantile(0.25), q3=lambda s: s.quantile(0.75), min="min", max="max")
        .merge(support, left_index=True, right_index=True, how="left")
        .reset_index()
    )
    summary["support_group"] = np.where(summary["n_buildings"] >= 20, "high", "low")

    high_order = (
        summary.loc[summary["support_group"] == "high"]
        .sort_values("median")["building_type"]
        .tolist()
    )
    low_order = (
        summary.loc[summary["support_group"] == "low"]
        .sort_values("median")["building_type"]
        .tolist()
    )
    low_non_utility = [name for name in low_order if name != "Utility"]

    displayed_values = ranking.loc[ranking["building_type"].isin(high_order + low_non_utility), "annual_mean_residual"].to_numpy(dtype=np.float64)
    common_min = float(np.floor(displayed_values.min())) - 1.0
    common_max = float(np.ceil(displayed_values.max())) + 1.0

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(15, 7.8),
        sharey=False,
        gridspec_kw={"width_ratios": [1.45, 1.0]},
    )

    ax = axes[0]
    high_labels = [
        f"{name} (n={int(summary.loc[summary['building_type'] == name, 'n_buildings'].iloc[0])})"
        for name in high_order
    ]
    high_data = [
        ranking.loc[ranking["building_type"] == name, "annual_mean_residual"].to_numpy(dtype=np.float64)
        for name in high_order
    ]
    box = ax.boxplot(
        high_data,
        vert=False,
        tick_labels=high_labels,
        patch_artist=True,
        showfliers=False,
    )
    for patch in box["boxes"]:
        patch.set(facecolor="#e8f1ea", edgecolor="#52796f", linewidth=1.1)
    for median in box["medians"]:
        median.set(color="#b44b5c", linewidth=1.6)
    ax.axvline(0.0, linestyle="--", color="#444444", linewidth=1.1)
    ax.set_xlabel("Annual mean residual (observed - predicted, kWh/h)")
    ax.set_title("High-support types (n >= 20)")
    ax.set_xlim(common_min, common_max)
    ax.grid(axis="x", alpha=0.2, linestyle="--")

    ax = axes[1]
    low_labels = [
        f"{name}* (n={int(summary.loc[summary['building_type'] == name, 'n_buildings'].iloc[0])})"
        for name in low_non_utility
    ]
    y = np.arange(1, len(low_non_utility) + 1, dtype=np.float64)
    for idx, name in enumerate(low_non_utility, start=1):
        values = ranking.loc[ranking["building_type"] == name, "annual_mean_residual"].to_numpy(dtype=np.float64)
        if len(values) > 1:
            ax.hlines(idx, values.min(), values.max(), color="#9b9b9b", linewidth=1.0, alpha=0.9)
        ax.scatter(values, np.full(len(values), idx), s=34, color="#8f6c3b", alpha=0.88, edgecolors="white", linewidths=0.5, zorder=3)
        median_value = float(np.median(values))
        ax.scatter([median_value], [idx], s=52, marker="D", color="#b44b5c", edgecolors="white", linewidths=0.5, zorder=4)
    ax.axvline(0.0, linestyle="--", color="#444444", linewidth=1.1)
    ax.set_yticks(y, low_labels)
    ax.set_xlabel("Annual mean residual (observed - predicted, kWh/h)")
    ax.set_title("Low-support types excluding Utility")
    ax.set_xlim(common_min, common_max)
    ax.grid(axis="x", alpha=0.2, linestyle="--")

    fig.tight_layout()
    _save_figure(fig, FIG8_PATH)
    plt.close(fig)
    return FIG8_PATH


def build_figure_9() -> Path:
    ranking = pd.read_csv(ROOT / "results" / "exp3_eui_vs_cps_ranking.csv")
    ranking = ranking.dropna(subset=["eui_percentile", "cps_percentile"]).copy()
    support = ranking.groupby("building_type")["building_id"].size().sort_values(ascending=False)
    major_types = support[support >= 20].index.tolist()
    minor = ranking[~ranking["building_type"].isin(major_types)].copy()
    palette = {
        "Education": "#4c78a8",
        "Office": "#e45756",
        "Entertainment/public assembly": "#72b7b2",
        "Public services": "#54a24b",
        "Lodging/residential": "#b279a2",
    }
    case_frame = pd.read_csv(ROOT / "tables" / "exp3_case_study_buildings.csv")

    fig, ax = plt.subplots(figsize=(9.4, 8))
    if not minor.empty:
        ax.scatter(
            minor["eui_percentile"],
            minor["cps_percentile"],
            s=20,
            alpha=0.28,
            color="#b8c2cc",
            edgecolors="none",
            label="Low-support types (pooled)",
        )
    for building_type in major_types:
        subset = ranking[ranking["building_type"] == building_type]
        ax.scatter(
            subset["eui_percentile"],
            subset["cps_percentile"],
            s=30,
            alpha=0.82,
            color=palette.get(building_type, "#4c78a8"),
            edgecolors="white",
            linewidths=0.4,
            label=building_type,
        )
    for _, row in case_frame.iterrows():
        ax.scatter(
            [row["eui_percentile"]],
            [row["cps_percentile"]],
            s=70,
            marker="D",
            color="#b44b5c" if row["case_type"] == "A" else "#2f6c8f",
            edgecolors="white",
            linewidths=0.6,
            zorder=4,
        )
    ax.plot([0, 1], [0, 1], linestyle="--", color="#444444", linewidth=1.0)
    ax.axvline(0.2, linestyle=":", color="#9a9a9a", linewidth=0.9)
    ax.axvline(0.8, linestyle=":", color="#9a9a9a", linewidth=0.9)
    ax.axhline(0.2, linestyle=":", color="#9a9a9a", linewidth=0.9)
    ax.axhline(0.8, linestyle=":", color="#9a9a9a", linewidth=0.9)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("EUI percentile within building type")
    ax.set_ylabel("CPS percentile within building type")
    ax.grid(alpha=0.2, linestyle="--")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    fig.tight_layout()
    _save_figure(fig, FIG9_PATH, tight=True)
    plt.close(fig)
    return FIG9_PATH


def build_figure_10() -> Path:
    seed = pd.read_csv(ROOT / "tables" / "exp3_seed_sensitivity.csv")
    summary = pd.read_csv(ROOT / "tables" / "exp3_seed_sensitivity_summary.csv")
    order = ["lgbm", "lstm", "patchtst"]
    colors = {"lgbm": "#3b6c8f", "lstm": "#b44b5c", "patchtst": "#2f7d6d"}
    x = np.arange(len(order), dtype=np.float64)

    fig, ax = plt.subplots(figsize=(8.8, 5.3))
    for idx, model in enumerate(order):
        subset = seed[seed["model_name"] == model]
        jitter = np.linspace(-0.08, 0.08, num=len(subset)) if len(subset) > 1 else np.array([0.0])
        ax.scatter(np.full(len(subset), x[idx]) + jitter, subset["overall_spearman_rho"], s=48, color=colors[model], alpha=0.8, edgecolors="white", linewidths=0.6)
        summary_row = summary[summary["model_name"] == model].iloc[0]
        mean = float(summary_row["overall_spearman_rho_mean"])
        low = float(summary_row["overall_spearman_rho_ci95_low"])
        high = float(summary_row["overall_spearman_rho_ci95_high"])
        ax.errorbar(x[idx], mean, yerr=np.array([[mean - low], [high - mean]]), fmt="o", color="black", ecolor="black", elinewidth=1.3, capsize=5, markersize=6)

    ax.axhline(0.0, linestyle="--", color="#555555", linewidth=1.0)
    ax.set_xticks(x, ["LightGBM", "LSTM", "PatchTST"])
    ax.set_ylabel("Spearman rho between EUI rank and CPS rank")
    ax.grid(axis="y", alpha=0.2, linestyle="--")
    fig.tight_layout()
    _save_figure(fig, FIG10_PATH)
    plt.close(fig)
    return FIG10_PATH


def build_figure_11() -> Path:
    case_frame = pd.read_csv(ROOT / "tables" / "exp3_case_study_buildings.csv")
    pred = pd.read_csv(ROOT / "results" / "exp1_predictions" / "lstm_b_split.csv", parse_dates=["timestamp"])
    pred["building_id"] = pred["building_id"].astype(str)

    selected_ids = [
        "Lamb_warehouse_Allan",
        "Bear_science_Alison",
        "Panther_lodging_Edison",
    ]
    ordered = case_frame.set_index("building_id").reindex(selected_ids).dropna(how="all").reset_index()
    if len(ordered) < 3:
        ordered = case_frame.head(3).copy()

    fig, axes = plt.subplots(1, len(ordered), figsize=(5.3 * len(ordered), 4.9), sharex=False)
    if len(ordered) == 1:
        axes = [axes]
    else:
        axes = np.ravel(axes)

    for ax, (_, row) in zip(axes, ordered.iterrows(), strict=False):
        building_id = str(row["building_id"])
        subset = pred[pred["building_id"] == building_id].copy()
        subset["date"] = subset["timestamp"].dt.floor("D")
        daily = (
            subset.groupby("date", sort=True)
            .agg(actual=("y_true", "mean"), predicted=("y_pred", "mean"))
            .reset_index()
        )
        daily["residual"] = daily["actual"] - daily["predicted"]

        daily["actual_smooth"] = daily["actual"].rolling(window=21, center=True, min_periods=1).mean()
        daily["predicted_smooth"] = daily["predicted"].rolling(window=21, center=True, min_periods=1).mean()
        daily["residual_smooth"] = daily["residual"].rolling(window=21, center=True, min_periods=1).mean()

        ax.plot(daily["date"], daily["actual"], color="#6c757d", linewidth=0.6, alpha=0.28)
        ax.plot(daily["date"], daily["predicted"], color="#2f6c8f", linewidth=0.55, linestyle="--", alpha=0.22)
        ax.plot(daily["date"], daily["actual_smooth"], color="#5a6672", linewidth=1.6, label="Observed (21d mean)")
        ax.plot(daily["date"], daily["predicted_smooth"], color="#2f6c8f", linewidth=1.4, linestyle="--", label="Predicted (21d mean)")
        ax.set_title(f"{row['case_type']}: {building_id}", fontsize=PANEL_TITLE_SIZE)
        ax.set_ylabel("Daily mean load")
        ax.grid(alpha=0.18, linestyle="--")
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.tick_params(axis="x", rotation=25)

        ax2 = ax.twinx()
        positive = np.clip(daily["residual_smooth"].to_numpy(dtype=np.float64), a_min=0.0, a_max=None)
        negative = np.clip(daily["residual_smooth"].to_numpy(dtype=np.float64), a_min=None, a_max=0.0)
        ax2.fill_between(daily["date"], 0.0, positive, color="#c65d5d", alpha=0.16)
        ax2.fill_between(daily["date"], 0.0, negative, color="#5d84c6", alpha=0.16)
        ax2.axhline(0.0, color="#777777", linestyle=":", linewidth=0.9)
        ax2.set_yticks([])

        info = (
            f"{row['building_type']}\n"
            f"Area={row['floor_area']:.0f} sqm\n"
            f"EUI pct={row['eui_percentile']:.2f}\n"
            f"CPS pct={row['cps_percentile']:.2f}"
        )
        ax.text(
            0.02,
            0.96,
            info,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=ANNOTATION_FONT_SIZE,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.9},
        )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save_figure(fig, FIG11_PATH)
    plt.close(fig)
    return FIG11_PATH


def build_figure_12() -> Path:
    comparison = pd.read_csv(ROOT / "tables" / "exp4_cross_dataset_comparison.csv")
    panel_specs = [
        ("BDG2", "t_split", "BDG2 T-split"),
        ("BDG2", "b_split", "BDG2 B-split"),
        ("GEPIII", "t_split", "GEPIII T-split"),
        ("GEPIII", "b_split", "GEPIII B-split"),
    ]
    model_order = [model for model in ["lgbm", "lgbm_lag", "lstm", "patchtst"] if f"t_split_{model}_cv_rmse" in comparison.columns]
    model_labels = {
        "lgbm": "LightGBM",
        "lgbm_lag": "LightGBM + lag",
        "lstm": "LSTM",
        "patchtst": "PatchTST",
    }
    model_colors = [PALETTE[model] for model in model_order]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.ravel()

    for ax, (dataset, split_name, title) in zip(axes, panel_specs, strict=False):
        row = comparison[comparison["dataset"] == dataset].iloc[0]
        values = np.array([float(row[f"{split_name}_{model}_cv_rmse"]) for model in model_order], dtype=np.float64)
        normalized = values / values.min()
        labels = [model_labels[model] for model in model_order]
        bars = ax.bar(labels, normalized, color=model_colors)
        for bar, absolute in zip(bars, values, strict=False):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{absolute:.3f}", ha="center", va="bottom", fontsize=ANNOTATION_FONT_SIZE)
        ax.axhline(1.0, linestyle="--", color="#555555", linewidth=1.0)
        ax.set_ylim(0.95, float(normalized.max()) * 1.10)
        ax.set_ylabel("Relative CV(RMSE), best = 1.0")
        ax.set_title(title, fontsize=PANEL_TITLE_SIZE)
        ax.grid(axis="y", alpha=0.2, linestyle="--")
        ax.tick_params(axis="x", rotation=10)

    fig.tight_layout()
    _save_figure(fig, FIG12_PATH)
    plt.close(fig)
    return FIG12_PATH


def build_figure_14() -> Path:
    metrics = pd.read_csv(ROOT / "tables" / "heew_replication_metrics.csv")
    canonical = metrics[metrics["split_seed"].isna()].copy()
    if canonical.empty:
        raise FileNotFoundError("HEEW canonical metrics are required to render Figure 14.")

    pooled = canonical[["split_name", "model", "cv_rmse"]].copy()
    pooled["rank"] = pooled.groupby("split_name")["cv_rmse"].rank(method="min", ascending=True)
    per_building = canonical[["split_name", "model", "median_per_building_cv_rmse"]].copy()
    per_building["rank"] = per_building.groupby("split_name")["median_per_building_cv_rmse"].rank(method="min", ascending=True)
    model_labels = {
        "lgbm": "LightGBM",
        "lgbm_lag": "LightGBM+lag",
        "lstm": "LSTM",
        "patchtst": "PatchTST",
    }

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.6), sharey=True)
    panel_specs = [
        (axes[0], pooled, "cv_rmse", "Pooled row-wise CV(RMSE) rank"),
        (axes[1], per_building, "median_per_building_cv_rmse", "Median per-building CV(RMSE) rank"),
    ]
    for ax, frame, value_col, title in panel_specs:
        for model_name in ["lgbm", "lgbm_lag", "lstm", "patchtst"]:
            subset = frame[frame["model"] == model_name].set_index("split_name")
            x = np.array([0, 1], dtype=np.float64)
            y = np.array([subset.loc["t_split", "rank"], subset.loc["b_split", "rank"]], dtype=np.float64)
            values = np.array([subset.loc["t_split", value_col], subset.loc["b_split", value_col]], dtype=np.float64)
            ax.plot(x, y, color=PALETTE[model_name], marker="o", linewidth=2.0, markersize=6.0)
            ax.text(
                x[0] - 0.03,
                y[0],
                f"{model_labels[model_name]}\n{values[0]:.3f}",
                ha="right",
                va="center",
                fontsize=ANNOTATION_FONT_SIZE,
                color=PALETTE[model_name],
            )
            ax.text(
                x[1] + 0.03,
                y[1],
                f"{model_labels[model_name]}\n{values[1]:.3f}",
                ha="left",
                va="center",
                fontsize=ANNOTATION_FONT_SIZE,
                color=PALETTE[model_name],
            )
        ax.set_xticks([0, 1], ["T-split", "B-split"])
        ax.set_xlim(-0.35, 1.35)
        ax.set_title(title, fontsize=PANEL_TITLE_SIZE)
        ax.grid(axis="y", alpha=0.2, linestyle="--")
        ax.invert_yaxis()
    axes[0].set_ylabel("Rank (1 = best)")
    axes[0].set_yticks([1, 2, 3, 4])
    fig.tight_layout()
    _save_figure(fig, FIG14_PATH)
    plt.close(fig)
    return FIG14_PATH


def build_figure_15() -> Path:
    summary = pd.read_csv(ROOT / "tables" / "lag_ablation_summary_main.csv")
    order = ["C0", "A1", "A2", "A3", "B1", "B2", "B3", "C1", "D2", "D3", "D1", "D4"]
    offsets = np.linspace(-0.16, 0.16, num=4)
    offset_map = {
        "C0": 0.0,
        "A1": offsets[0],
        "A2": offsets[1],
        "A3": offsets[2],
        "B1": offsets[0],
        "B2": 0.0,
        "B3": offsets[3],
        "C1": 0.0,
        "D2": offsets[0],
        "D3": offsets[2],
        "D1": 0.0,
        "D4": 0.0,
    }
    stage_colors = {"stage1": "#4c78a8", "stage2": "#f28e2b"}
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), sharex=True)

    for ax, split_name in zip(axes, ["t_split", "b_split"], strict=False):
        subset = summary[summary["split_name"] == split_name].copy()
        subset["config_name"] = pd.Categorical(subset["config_name"], categories=order, ordered=True)
        subset = subset.sort_values(["lag_count", "config_name"])
        value_span = float(subset["pooled_cv_rmse_mean"].max() - subset["pooled_cv_rmse_mean"].min())
        text_dy_base = 0.018 * value_span if value_span > 0 else 0.01

        for stage_name in ["stage1", "stage2"]:
            stage_subset = subset[subset["stage"] == stage_name]
            x_values = stage_subset["lag_count"].to_numpy(dtype=np.float64) + stage_subset["config_name"].map(offset_map).to_numpy(dtype=np.float64)
            y_values = stage_subset["pooled_cv_rmse_mean"].to_numpy(dtype=np.float64)
            ax.scatter(
                x_values,
                y_values,
                s=48,
                color=stage_colors[stage_name],
                edgecolors="white",
                linewidths=0.6,
                alpha=0.92,
                label="Stage 1 configs" if stage_name == "stage1" else "Stage 2 configs",
                zorder=3,
            )
            for x_value, y_value, cfg in zip(x_values, y_values, stage_subset["config_name"], strict=False):
                dy = text_dy_base if float(offset_map[str(cfg)]) <= 0 else -text_dy_base
                ax.text(
                    x_value,
                    y_value + dy,
                    str(cfg),
                    fontsize=ANNOTATION_FONT_SIZE,
                    ha="center",
                    va="center",
                    color="#303030",
                    clip_on=False,
                )

        best_by_count = (
            subset.groupby("lag_count", as_index=False)["pooled_cv_rmse_mean"]
            .min()
            .sort_values("lag_count")
        )
        ax.plot(
            best_by_count["lag_count"].to_numpy(dtype=np.float64),
            best_by_count["pooled_cv_rmse_mean"].to_numpy(dtype=np.float64),
            color="#444444",
            linewidth=1.8,
            marker="o",
            markersize=4.2,
            label="Best pooled value at each lag count",
            zorder=2,
        )

        c1 = subset[subset["config_name"] == "C1"].iloc[0]
        ax.scatter(
            [float(c1["lag_count"])],
            [float(c1["pooled_cv_rmse_mean"])],
            s=90,
            marker="D",
            color="#b44b5c",
            edgecolors="white",
            linewidths=0.7,
            label="Canonical C1 triplet",
            zorder=4,
        )
        ax.axvline(3.0, linestyle="--", color="#777777", linewidth=1.0)
        ax.set_xticks([0, 1, 2, 3, 4, 5, 7])
        ax.set_xlabel("Lag-count complexity")
        ax.set_ylabel("Pooled CV(RMSE)")
        ax.set_title(_format_split_name(split_name), fontsize=PANEL_TITLE_SIZE)
        ax.grid(axis="y", alpha=0.2, linestyle="--")

    handles, labels = axes[0].get_legend_handles_labels()
    dedup: dict[str, object] = {}
    for handle, label in zip(handles, labels, strict=False):
        if label not in dedup:
            dedup[label] = handle
    fig.legend(dedup.values(), dedup.keys(), loc="upper center", ncol=4, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    _save_figure(fig, FIG15_PATH)
    plt.close(fig)
    return FIG15_PATH


def build_figure_16() -> Path:
    summary_path = ROOT / "tables" / "information_budget_summary_main_combined.csv"
    if not summary_path.exists():
        raise FileNotFoundError("Final information-budget summary is required to render Figure 16.")
    summary = pd.read_csv(summary_path)

    spec_order = [
        "lgbm_no_history",
        "lgbm_sparse",
        "lgbm_dense",
        "lstm_ctx24",
        "lstm_ctx72",
        "lstm_ctx168",
        "patchtst_ctx24",
        "patchtst_ctx72",
        "patchtst_ctx168",
    ]
    x_positions = {
        "lgbm_no_history": 0.0,
        "lgbm_sparse": 1.0,
        "lgbm_dense": 2.0,
        "lstm_ctx24": 4.0,
        "lstm_ctx72": 5.0,
        "lstm_ctx168": 6.0,
        "patchtst_ctx24": 8.0,
        "patchtst_ctx72": 9.0,
        "patchtst_ctx168": 10.0,
    }
    family_groups = {
        "tabular": ["lgbm_no_history", "lgbm_sparse", "lgbm_dense"],
        "lstm": ["lstm_ctx24", "lstm_ctx72", "lstm_ctx168"],
        "patchtst": ["patchtst_ctx24", "patchtst_ctx72", "patchtst_ctx168"],
    }
    family_colors = {
        "tabular": PALETTE["lgbm_lag"],
        "lstm": PALETTE["lstm"],
        "patchtst": PALETTE["patchtst"],
    }
    family_labels = {
        "tabular": "Tabular history budget",
        "lstm": "LSTM context budget",
        "patchtst": "PatchTST context budget",
    }
    label_map = {
        "lgbm_no_history": "No\nhistory",
        "lgbm_sparse": "Sparse\nlags",
        "lgbm_dense": "Dense\nsummary",
        "lstm_ctx24": "24h",
        "lstm_ctx72": "72h",
        "lstm_ctx168": "168h",
        "patchtst_ctx24": "24h",
        "patchtst_ctx72": "72h",
        "patchtst_ctx168": "168h",
    }
    metric_map = {
        "pooled_cv_rmse_mean": "Pooled CV(RMSE)",
        "median_per_building_cv_rmse_mean": "Median per-building CV(RMSE)",
    }
    err_map = {
        "pooled_cv_rmse_mean": "pooled_cv_rmse_sd",
        "median_per_building_cv_rmse_mean": "median_per_building_cv_rmse_sd",
    }

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 6.8), sharex=True)
    split_order = ["t_split", "b_split"]
    spec_ticks = [x_positions[spec_name] for spec_name in spec_order]
    spec_ticklabels = [label_map[spec_name] for spec_name in spec_order]

    for row_index, (metric_col, y_label) in enumerate(metric_map.items()):
        for col_index, split_name in enumerate(split_order):
            ax = axes[row_index, col_index]
            split_frame = summary[summary["split_name"] == split_name].copy()
            split_frame["spec_name"] = pd.Categorical(split_frame["spec_name"], categories=spec_order, ordered=True)
            split_frame = split_frame.sort_values("spec_name")

            for family_name, specs in family_groups.items():
                family_frame = split_frame[split_frame["spec_name"].isin(specs)].copy()
                x_values = np.array([x_positions[spec_name] for spec_name in family_frame["spec_name"]], dtype=np.float64)
                y_values = family_frame[metric_col].to_numpy(dtype=np.float64)
                y_err = family_frame[err_map[metric_col]].to_numpy(dtype=np.float64)
                ax.plot(
                    x_values,
                    y_values,
                    color=family_colors[family_name],
                    linewidth=2.0,
                    marker="o",
                    markersize=5.0,
                    label=family_labels[family_name],
                    zorder=3,
                )
                ax.errorbar(
                    x_values,
                    y_values,
                    yerr=y_err,
                    fmt="none",
                    ecolor=family_colors[family_name],
                    elinewidth=1.0,
                    capsize=2.5,
                    alpha=0.85,
                    zorder=2,
                )

            ax.axvspan(-0.45, 2.45, color="#f2f5f8", alpha=0.65, zorder=0)
            ax.axvspan(3.55, 6.45, color="#f8f3f2", alpha=0.65, zorder=0)
            ax.axvspan(7.55, 10.45, color="#eef6f3", alpha=0.65, zorder=0)
            ax.axvline(3.0, color="#999999", linestyle="--", linewidth=0.8)
            ax.axvline(7.0, color="#999999", linestyle="--", linewidth=0.8)
            ax.grid(axis="y", alpha=0.2, linestyle="--")
            ax.set_title(
                f"{_format_split_name(split_name)}: {y_label}" if row_index == 0 else y_label,
                fontsize=PANEL_TITLE_SIZE,
            )
            if col_index == 0:
                ax.set_ylabel(y_label)
            ax.set_xticks(spec_ticks, spec_ticklabels)
            ax.tick_params(axis="x", labelrotation=0)

    axes[0, 0].text(1.0, axes[0, 0].get_ylim()[1] * 0.985, "Tabular", ha="center", va="top", fontsize=SMALL_FONT_SIZE)
    axes[0, 0].text(5.0, axes[0, 0].get_ylim()[1] * 0.985, "LSTM", ha="center", va="top", fontsize=SMALL_FONT_SIZE)
    axes[0, 0].text(9.0, axes[0, 0].get_ylim()[1] * 0.985, "PatchTST", ha="center", va="top", fontsize=SMALL_FONT_SIZE)
    axes[0, 1].text(1.0, axes[0, 1].get_ylim()[1] * 0.985, "Tabular", ha="center", va="top", fontsize=SMALL_FONT_SIZE)
    axes[0, 1].text(5.0, axes[0, 1].get_ylim()[1] * 0.985, "LSTM", ha="center", va="top", fontsize=SMALL_FONT_SIZE)
    axes[0, 1].text(9.0, axes[0, 1].get_ylim()[1] * 0.985, "PatchTST", ha="center", va="top", fontsize=SMALL_FONT_SIZE)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save_figure(fig, FIG16_PATH)
    plt.close(fig)
    return FIG16_PATH


def build_figure_18() -> Path:
    bdg2_metrics = pd.read_csv(ROOT / "tables" / "exp1_metrics.csv").set_index("split_name")
    bdg2_per_building = pd.read_csv(ROOT / "tables" / "exp1_per_building_summary.csv").set_index(["split_name", "model"])

    gepiii_metrics = pd.read_csv(ROOT / "tables" / "exp4_cross_dataset_comparison.csv")
    gepiii_metrics = gepiii_metrics[gepiii_metrics["dataset"] == "GEPIII"].iloc[0]
    gepiii_per_building = pd.read_csv(ROOT / "tables" / "exp4_per_building_summary.csv").set_index(["split_name", "model"])

    heew_metrics = pd.read_csv(ROOT / "tables" / "heew_replication_metrics.csv")

    rows: list[dict[str, object]] = []
    for split_name in ["t_split", "b_split", "s_split"]:
        rows.append(
            {
                "dataset": "BDG2",
                "label": f"BDG2 {_format_split_name(split_name)}",
                "pooled_baseline": float(bdg2_metrics.loc[split_name, "lgbm_cv_rmse"]),
                "pooled_parity": float(bdg2_metrics.loc[split_name, "lgbm_lag_cv_rmse"]),
                "median_baseline": float(bdg2_per_building.loc[(split_name, "lgbm"), "median_cv_rmse"]),
                "median_parity": float(bdg2_per_building.loc[(split_name, "lgbm_lag"), "median_cv_rmse"]),
            }
        )

    for split_name in ["t_split", "b_split"]:
        rows.append(
            {
                "dataset": "GEPIII",
                "label": f"GEPIII {_format_split_name(split_name)}",
                "pooled_baseline": float(gepiii_metrics[f"{split_name}_lgbm_cv_rmse"]),
                "pooled_parity": float(gepiii_metrics[f"{split_name}_lgbm_lag_cv_rmse"]),
                "median_baseline": float(gepiii_per_building.loc[(split_name, "lgbm"), "median_cv_rmse"]),
                "median_parity": float(gepiii_per_building.loc[(split_name, "lgbm_lag"), "median_cv_rmse"]),
            }
        )

    for split_name in ["t_split", "b_split"]:
        subset = heew_metrics[(heew_metrics["split_name"] == split_name) & (heew_metrics["split_seed"].isna())]
        baseline = subset[subset["model"] == "lgbm"].iloc[0]
        parity = subset[subset["model"] == "lgbm_lag"].iloc[0]
        rows.append(
            {
                "dataset": "HEEW",
                "label": f"HEEW {_format_split_name(split_name)}",
                "pooled_baseline": float(baseline["cv_rmse"]),
                "pooled_parity": float(parity["cv_rmse"]),
                "median_baseline": float(baseline["median_per_building_cv_rmse"]),
                "median_parity": float(parity["median_per_building_cv_rmse"]),
            }
        )

    frame = pd.DataFrame(rows)
    frame["y"] = np.arange(len(frame) - 1, -1, -1, dtype=np.float64)
    group_fill = {"BDG2": "#eef4fa", "GEPIII": "#f8efe8", "HEEW": "#edf7f1"}
    panel_specs = [
        ("pooled", "Pooled CV(RMSE) $\\downarrow$"),
        ("median", "Median per-building CV(RMSE) $\\downarrow$"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.6), sharey=True)

    for ax, (metric_name, title) in zip(axes, panel_specs, strict=False):
        baseline_col = f"{metric_name}_baseline"
        parity_col = f"{metric_name}_parity"
        all_values = np.concatenate(
            [
                frame[baseline_col].to_numpy(dtype=np.float64),
                frame[parity_col].to_numpy(dtype=np.float64),
            ]
        )
        value_span = float(all_values.max() - all_values.min())
        x_pad = 0.08 * value_span if value_span > 0 else 0.05
        x_min = max(0.0, float(all_values.min()) - 1.4 * x_pad)
        x_max = float(all_values.max()) + 2.2 * x_pad

        for dataset_name in ["BDG2", "GEPIII", "HEEW"]:
            subset = frame[frame["dataset"] == dataset_name]
            ax.axhspan(
                float(subset["y"].min()) - 0.5,
                float(subset["y"].max()) + 0.5,
                color=group_fill[dataset_name],
                alpha=0.65,
                zorder=0,
            )

        for row in frame.itertuples(index=False):
            baseline_value = float(getattr(row, baseline_col))
            parity_value = float(getattr(row, parity_col))
            improvement = 100.0 * (baseline_value - parity_value) / baseline_value if baseline_value else 0.0
            y_value = float(row.y)

            ax.annotate(
                "",
                xy=(parity_value, y_value),
                xytext=(baseline_value, y_value),
                arrowprops=dict(arrowstyle="-|>", color="#777777", linewidth=1.7, shrinkA=2, shrinkB=2),
                zorder=2,
            )
            ax.scatter(
                [baseline_value],
                [y_value],
                s=56,
                color=PALETTE["lgbm"],
                edgecolors="white",
                linewidths=0.7,
                zorder=3,
            )
            ax.scatter(
                [parity_value],
                [y_value],
                s=68,
                marker="D",
                color=PALETTE["lgbm_lag"],
                edgecolors="white",
                linewidths=0.7,
                zorder=4,
            )
            ax.text(
                baseline_value + 0.55 * x_pad,
                y_value,
                f"{baseline_value:.3f}",
                ha="left",
                va="center",
                fontsize=ANNOTATION_FONT_SIZE,
                color=PALETTE["lgbm"],
            )
            ax.text(
                parity_value - 0.45 * x_pad,
                y_value,
                f"{parity_value:.3f}",
                ha="right",
                va="center",
                fontsize=ANNOTATION_FONT_SIZE,
                color=PALETTE["lgbm_lag"],
            )
            ax.text(
                parity_value + 0.52 * (baseline_value - parity_value),
                y_value + 0.24,
                f"-{improvement:.0f}%",
                ha="center",
                va="bottom",
                fontsize=ANNOTATION_FONT_SIZE,
                color="#555555",
            )

        ax.set_title(title, fontsize=PANEL_TITLE_SIZE)
        ax.set_xlim(x_min, x_max)
        ax.set_xlabel("CV(RMSE)")
        ax.grid(axis="x", alpha=0.18, linestyle="--")
        ax.set_ylim(-0.7, float(frame["y"].max()) + 0.7)

    axes[0].set_yticks(frame["y"].to_numpy(dtype=np.float64), frame["label"].tolist())
    axes[1].set_yticks(frame["y"].to_numpy(dtype=np.float64), [""] * len(frame))
    axes[0].set_ylabel("Dataset / protocol")

    legend_handles = [
        Patch(facecolor=PALETTE["lgbm"], edgecolor="none", label="LightGBM baseline"),
        Patch(facecolor=PALETTE["lgbm_lag"], edgecolor="none", label="LightGBM+lag parity check"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save_figure(fig, FIG18_PATH)
    plt.close(fig)
    return FIG18_PATH


def build_supplementary_figure_1() -> Path:
    metrics = pd.read_csv(ROOT / "tables" / "heew_replication_per_building_metrics.csv")
    if metrics.empty:
        raise FileNotFoundError("HEEW per-building metrics are required to render Supplementary Figure S1.")

    base_load = (
        metrics.loc[(metrics["split_name"] == "t_split") & (metrics["split_seed"].isna()), ["building_id", "mean_load"]]
        .drop_duplicates()
        .sort_values("mean_load")
        .reset_index(drop=True)
    )
    quartile_order = ["Q1", "Q2", "Q3", "Q4"]
    base_load["load_quartile"] = pd.qcut(
        base_load["mean_load"].rank(method="first"),
        q=4,
        labels=quartile_order,
    )
    metrics = metrics.merge(base_load[["building_id", "load_quartile"]], on="building_id", how="left")
    metrics["load_quartile"] = pd.Categorical(metrics["load_quartile"], categories=quartile_order, ordered=True)

    panel_specs = [
        (
            "Canonical T-split",
            metrics[(metrics["split_name"] == "t_split") & (metrics["split_seed"].isna())].copy(),
        ),
        (
            "B-split five-seed aggregate",
            metrics[(metrics["split_name"] == "b_split") & (metrics["split_seed"].notna())].copy(),
        ),
    ]
    model_order = ["lgbm", "lgbm_lag", "lstm", "patchtst"]
    model_labels = {
        "lgbm": "LightGBM",
        "lgbm_lag": "LightGBM+lag",
        "lstm": "LSTM",
        "patchtst": "PatchTST",
    }
    offsets = {
        "lgbm": -0.18,
        "lgbm_lag": -0.06,
        "lstm": 0.06,
        "patchtst": 0.18,
    }

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), sharey=True)
    quartile_positions = np.arange(len(quartile_order), dtype=np.float64)

    for ax, (title, frame) in zip(axes, panel_specs, strict=False):
        summary = (
            frame.groupby(["load_quartile", "model"], observed=False)["cv_rmse"]
            .median()
            .reset_index(name="median_cv_rmse")
        )
        for model_name in model_order:
            model_frame = frame[frame["model"] == model_name]
            for quartile_index, quartile in enumerate(quartile_order):
                values = model_frame.loc[model_frame["load_quartile"] == quartile, "cv_rmse"].to_numpy(dtype=np.float64)
                if values.size == 0:
                    continue
                x_values = np.full(values.shape, quartile_positions[quartile_index] + offsets[model_name], dtype=np.float64)
                ax.scatter(
                    x_values,
                    values,
                    s=18,
                    alpha=0.22,
                    color=PALETTE[model_name],
                    edgecolors="none",
                )
            model_summary = (
                summary[summary["model"] == model_name]
                .set_index("load_quartile")
                .reindex(quartile_order)["median_cv_rmse"]
                .to_numpy(dtype=np.float64)
            )
            ax.plot(
                quartile_positions + offsets[model_name],
                model_summary,
                marker="o",
                linewidth=2.0,
                markersize=5.0,
                color=PALETTE[model_name],
                label=model_labels[model_name],
            )
        ax.set_xticks(quartile_positions, quartile_order)
        ax.set_xlabel("Mean-load quartile")
        ax.set_title(title, fontsize=PANEL_TITLE_SIZE)
        ax.grid(axis="y", alpha=0.2, linestyle="--")
        ax.set_yscale("log")

    axes[0].set_ylabel("Per-building CV(RMSE)")
    axes[0].legend(frameon=False, loc="upper left")
    fig.tight_layout()
    _save_figure(fig, FIGS1_PATH)
    plt.close(fig)
    return FIGS1_PATH


def main() -> None:
    _ensure_dir()
    outputs = {
        "figure_1": build_figure_1(),
        "figure_2": build_figure_2(),
        "figure_3": build_figure_3(),
        "figure_4a": build_figure_4a(),
        "figure_4c": build_figure_4c(),
        "figure_4": build_figure_4(),
        "figure_4b": FIG4B_PATH,
        "figure_5": build_figure_5(),
        "figure_6": build_figure_6(),
        "figure_7": build_figure_7(),
        "figure_8": build_figure_8(),
        "figure_9": build_figure_9(),
        "figure_10": build_figure_10(),
        "figure_11": build_figure_11(),
        "figure_12": build_figure_12(),
        "figure_14": build_figure_14(),
        "figure_15": build_figure_15(),
        "figure_16": build_figure_16(),
        "figure_18": build_figure_18(),
        "supp_figure_1": build_supplementary_figure_1(),
    }
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
