from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import FIGURES_DIR, EXP1_PREDICTIONS_DIR, TABLES_DIR, ensure_phase2_dirs
from src.data_splitting import load_pickle


EXP1_METRICS_PATH = TABLES_DIR / "exp1_metrics.csv"
EXP1_METRICS_BY_TYPE_PATH = TABLES_DIR / "exp1_metrics_by_type.csv"
EXP1_TYPE_SAMPLE_COUNTS_PATH = TABLES_DIR / "exp1_type_sample_counts.csv"
EXP1_SITE_METRICS_PATH = TABLES_DIR / "exp1_site_metrics.csv"
EXP1_BOXPLOT_PATH = FIGURES_DIR / "exp1_boxplot_by_type.png"
EXP1_SITE_DIST_PATH = FIGURES_DIR / "exp1_site_error_dist.png"
EXP1_OVERVIEW_CVRMSE_PATH = FIGURES_DIR / "exp1_overall_cv_rmse.png"
EXP1_OVERVIEW_PANEL_PATH = FIGURES_DIR / "exp1_overall_metrics_panel.png"
EXP1_TYPE_HEATMAP_PATH = FIGURES_DIR / "exp1_type_cv_rmse_heatmap.png"
EXP1_SITE_RANKED_PATH = FIGURES_DIR / "exp1_site_ranked_cv_rmse.png"

SPLIT_ORDER = ["t_split", "b_split", "s_split"]
MODEL_ORDER = ["lgbm", "lgbm_lag", "lstm", "patchtst"]
MODEL_COLORS = {
    "lgbm": "#2f6690",
    "lgbm_lag": "#5f8f3b",
    "lstm": "#81b29a",
    "patchtst": "#d1495b",
}


def summarize_metrics(frame: pd.DataFrame, group_cols: list[str] | None = None) -> pd.DataFrame:
    group_cols = group_cols or []
    grouped = frame.groupby(group_cols) if group_cols else [((), frame)]
    rows: list[dict[str, object]] = []
    for key, group in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        mae = float(np.mean(np.abs(group["y_pred"] - group["y_true"])))
        rmse = float(np.sqrt(np.mean((group["y_pred"] - group["y_true"]) ** 2)))
        mean_y = float(np.mean(group["y_true"]))
        cv_rmse = float(rmse / mean_y) if mean_y != 0 else np.nan
        row = {
            "mae": mae,
            "rmse": rmse,
            "cv_rmse": cv_rmse,
            "n_rows": int(len(group)),
        }
        for idx, column in enumerate(group_cols):
            row[column] = key[idx]
        rows.append(row)
    return pd.DataFrame(rows)


def load_prediction_frames(predictions_dir: str | Path = EXP1_PREDICTIONS_DIR) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(Path(predictions_dir).glob("*.csv")):
        frame = pd.read_csv(path, parse_dates=["timestamp"])
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No prediction CSV files found in {predictions_dir}")
    return pd.concat(frames, ignore_index=True)


def compute_exp1_outputs(
    predictions_dir: str | Path = EXP1_PREDICTIONS_DIR,
    s_split_path: str | Path | None = None,
) -> dict[str, Path]:
    ensure_phase2_dirs()
    prediction_frame = load_prediction_frames(predictions_dir)

    overall = summarize_metrics(prediction_frame, ["split_name", "model"])
    by_type = summarize_metrics(prediction_frame, ["split_name", "model", "building_type"])
    type_support = (
        prediction_frame.groupby(["split_name", "building_type"])
        .agg(
            n_buildings=("building_id", "nunique"),
            n_eval_rows=("building_id", "size"),
        )
        .reset_index()
    )
    type_support["enough_buildings_for_main_conclusion"] = type_support["n_buildings"] >= 20
    by_type = by_type.merge(
        type_support,
        on=["split_name", "building_type"],
        how="left",
    )
    site_metrics = summarize_metrics(
        prediction_frame[prediction_frame["split_name"] == "s_split"],
        ["split_name", "model", "fold_id", "site_id"],
    )

    if s_split_path is not None and not site_metrics.empty:
        s_artifact = load_pickle(s_split_path)
        site_weights = {
            fold["test_site"]: fold["site_building_count"]
            for fold in s_artifact["folds"]
        }
        weighted_rows: list[dict[str, object]] = []
        for model, group in site_metrics.groupby("model"):
            weights = group["site_id"].map(site_weights).to_numpy(dtype=np.float64)
            weights = weights / weights.sum()
            weighted_rows.append(
                {
                    "split_name": "s_split",
                    "model": model,
                    "mae": float(np.average(group["mae"], weights=weights)),
                    "rmse": float(np.average(group["rmse"], weights=weights)),
                    "cv_rmse": float(np.average(group["cv_rmse"], weights=weights)),
                    "cv_rmse_site_std": float(group["cv_rmse"].std(ddof=0)),
                    "n_rows": int(group["n_rows"].sum()),
                }
            )

        s_weighted = pd.DataFrame(weighted_rows)
        overall = pd.concat(
            [
                overall[overall["split_name"] != "s_split"],
                s_weighted,
            ],
            ignore_index=True,
        )

    metrics_table = (
        overall.pivot(index="split_name", columns="model", values=["mae", "rmse", "cv_rmse"])
        .sort_index(axis=1)
    )
    metrics_table.columns = [f"{model}_{metric}" for metric, model in metrics_table.columns]
    metrics_table = metrics_table.reset_index()

    metrics_table.to_csv(EXP1_METRICS_PATH, index=False)
    by_type.to_csv(EXP1_METRICS_BY_TYPE_PATH, index=False)
    type_support.to_csv(EXP1_TYPE_SAMPLE_COUNTS_PATH, index=False)
    site_metrics.to_csv(EXP1_SITE_METRICS_PATH, index=False)

    _plot_boxplot_by_type(prediction_frame)
    _plot_site_error_dist(site_metrics)
    _plot_overall_cv_rmse(overall)
    _plot_overall_metric_panel(overall)
    _plot_type_heatmap(by_type)
    _plot_site_ranked_cv_rmse(site_metrics)

    return {
        "exp1_metrics": EXP1_METRICS_PATH,
        "exp1_metrics_by_type": EXP1_METRICS_BY_TYPE_PATH,
        "exp1_type_sample_counts": EXP1_TYPE_SAMPLE_COUNTS_PATH,
        "exp1_site_metrics": EXP1_SITE_METRICS_PATH,
        "exp1_boxplot_by_type": EXP1_BOXPLOT_PATH,
        "exp1_site_error_dist": EXP1_SITE_DIST_PATH,
        "exp1_overall_cv_rmse": EXP1_OVERVIEW_CVRMSE_PATH,
        "exp1_overall_metrics_panel": EXP1_OVERVIEW_PANEL_PATH,
        "exp1_type_cv_rmse_heatmap": EXP1_TYPE_HEATMAP_PATH,
        "exp1_site_ranked_cv_rmse": EXP1_SITE_RANKED_PATH,
    }


def _ordered_present(values: pd.Series, order: list[str]) -> list[str]:
    present = set(values.astype(str).unique().tolist())
    ordered = [value for value in order if value in present]
    extras = sorted(present - set(order))
    return ordered + extras


def _plot_overall_cv_rmse(overall: pd.DataFrame) -> None:
    splits = _ordered_present(overall["split_name"], SPLIT_ORDER)
    models = _ordered_present(overall["model"], MODEL_ORDER)
    x = np.arange(len(splits), dtype=np.float64)
    width = 0.22 if models else 0.8

    plt.figure(figsize=(10, 6))
    for idx, model in enumerate(models):
        subset = (
            overall[overall["model"] == model]
            .set_index("split_name")
            .reindex(splits)
        )
        offsets = x + (idx - (len(models) - 1) / 2) * width
        bars = plt.bar(
            offsets,
            subset["cv_rmse"].to_numpy(dtype=np.float64),
            width=width,
            label=model,
            color=MODEL_COLORS.get(model, "#666666"),
        )
        for bar, value in zip(bars, subset["cv_rmse"].to_numpy(dtype=np.float64), strict=False):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                value,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
            )

    plt.xticks(x, splits)
    plt.ylabel("CV(RMSE)")
    plt.title("Overall CV(RMSE) by Split and Model")
    plt.legend()
    plt.tight_layout()
    plt.savefig(EXP1_OVERVIEW_CVRMSE_PATH, dpi=200)
    plt.close()


def _plot_overall_metric_panel(overall: pd.DataFrame) -> None:
    splits = _ordered_present(overall["split_name"], SPLIT_ORDER)
    models = _ordered_present(overall["model"], MODEL_ORDER)
    metrics = [
        ("mae", "MAE", True),
        ("rmse", "RMSE", True),
        ("cv_rmse", "CV(RMSE)", False),
    ]
    x = np.arange(len(splits), dtype=np.float64)
    width = 0.22 if models else 0.8

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (metric_col, title, use_log) in zip(axes, metrics, strict=False):
        for idx, model in enumerate(models):
            subset = (
                overall[overall["model"] == model]
                .set_index("split_name")
                .reindex(splits)
            )
            offsets = x + (idx - (len(models) - 1) / 2) * width
            ax.bar(
                offsets,
                subset[metric_col].to_numpy(dtype=np.float64),
                width=width,
                label=model,
                color=MODEL_COLORS.get(model, "#666666"),
            )
        ax.set_xticks(x, splits)
        ax.set_title(title)
        if use_log:
            ax.set_yscale("log")
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Metric Value")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(EXP1_OVERVIEW_PANEL_PATH, dpi=200)
    plt.close(fig)


def _plot_type_heatmap(by_type: pd.DataFrame) -> None:
    if by_type.empty:
        return

    split_order = _ordered_present(by_type["split_name"], SPLIT_ORDER)
    model_order = _ordered_present(by_type["model"], MODEL_ORDER)
    type_order = (
        by_type.groupby("building_type")["n_rows"]
        .sum()
        .sort_values(ascending=False)
        .index
        .tolist()
    )

    fig, axes = plt.subplots(
        1,
        len(split_order),
        figsize=(5.5 * len(split_order), 8),
        sharey=True,
        constrained_layout=True,
    )
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    heatmaps: list[object] = []
    for ax, split_name in zip(axes, split_order, strict=False):
        pivot = (
            by_type[by_type["split_name"] == split_name]
            .pivot(index="building_type", columns="model", values="cv_rmse")
            .reindex(index=type_order, columns=model_order)
        )
        image = ax.imshow(pivot.to_numpy(dtype=np.float64), aspect="auto", cmap="YlOrRd")
        heatmaps.append(image)
        ax.set_title(split_name)
        ax.set_xticks(np.arange(len(model_order)), model_order)
        ax.set_yticks(np.arange(len(type_order)), type_order)
        ax.tick_params(axis="x", rotation=45)

    fig.colorbar(heatmaps[-1], ax=axes.ravel().tolist(), shrink=0.9, label="CV(RMSE)")
    fig.suptitle("CV(RMSE) Heatmap by Building Type, Split, and Model", y=0.98)
    fig.savefig(EXP1_TYPE_HEATMAP_PATH, dpi=200)
    plt.close(fig)


def _plot_site_ranked_cv_rmse(site_metrics: pd.DataFrame) -> None:
    if site_metrics.empty:
        return

    pivot = (
        site_metrics.pivot(index="site_id", columns="model", values="cv_rmse")
        .reindex(columns=_ordered_present(site_metrics["model"], MODEL_ORDER))
    )
    pivot["__mean__"] = pivot.mean(axis=1)
    pivot = pivot.sort_values("__mean__", ascending=False).drop(columns="__mean__")

    sites = pivot.index.tolist()
    models = pivot.columns.tolist()
    y = np.arange(len(sites), dtype=np.float64)
    height = 0.22 if models else 0.8

    plt.figure(figsize=(12, 8))
    for idx, model in enumerate(models):
        offsets = y + (idx - (len(models) - 1) / 2) * height
        plt.barh(
            offsets,
            pivot[model].to_numpy(dtype=np.float64),
            height=height,
            label=model,
            color=MODEL_COLORS.get(model, "#666666"),
        )

    plt.yticks(y, sites)
    plt.gca().invert_yaxis()
    plt.xlabel("CV(RMSE)")
    plt.ylabel("Site")
    plt.title("Ranked S-split Site Difficulty by Model")
    plt.legend()
    plt.tight_layout()
    plt.savefig(EXP1_SITE_RANKED_PATH, dpi=200)
    plt.close()


def _plot_boxplot_by_type(prediction_frame: pd.DataFrame) -> None:
    subset = prediction_frame.copy()
    subset["abs_error"] = np.abs(subset["y_pred"] - subset["y_true"])
    top_types = (
        subset["building_type"].value_counts().head(8).index.tolist()
    )
    subset = subset[subset["building_type"].isin(top_types)].copy()

    fig, axes = plt.subplots(1, max(subset["model"].nunique(), 1), figsize=(18, 6), sharey=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    for ax, (model, group) in zip(axes, subset.groupby("model", sort=True), strict=False):
        data = [group.loc[group["building_type"] == bt, "abs_error"].sample(
            n=min(3000, (group["building_type"] == bt).sum()),
            random_state=42,
        ).to_numpy() for bt in top_types]
        ax.boxplot(data, labels=top_types, showfliers=False)
        ax.set_title(model)
        ax.tick_params(axis="x", rotation=45)
        ax.set_ylabel("Absolute Error")
    plt.tight_layout()
    plt.savefig(EXP1_BOXPLOT_PATH, dpi=200)
    plt.close()


def _plot_site_error_dist(site_metrics: pd.DataFrame) -> None:
    if site_metrics.empty:
        return
    pivot = (
        site_metrics.pivot(index="site_id", columns="model", values="cv_rmse")
        .sort_index()
    )
    pivot.plot(kind="bar", figsize=(12, 6))
    plt.ylabel("CV(RMSE)")
    plt.title("S-split Site Error Distribution")
    plt.tight_layout()
    plt.savefig(EXP1_SITE_DIST_PATH, dpi=200)
    plt.close()
