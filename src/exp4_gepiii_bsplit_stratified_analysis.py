from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clustering.feature_extractor import (
    extract_building_profile_features,
    fit_profile_scaler,
    transform_profile_features,
)
from src.config import FEATURES_GEPIII_PATH, FIGURES_DIR, TABLES_DIR, ensure_phase2_dirs


PREDICTION_PATHS = {
    "lgbm": ROOT / "results" / "exp4_predictions" / "lgbm_b_split.csv",
    "lgbm_lag": ROOT / "results" / "exp4_predictions" / "lgbm_lag_b_split.csv",
    "lstm": ROOT / "results" / "exp4_predictions" / "lstm_b_split.csv",
    "patchtst": ROOT / "results" / "exp4_predictions" / "patchtst_b_split.csv",
}

TYPE_SUMMARY_PATH = TABLES_DIR / "exp4_gepiii_bsplit_type_stratified.csv"
SITE_SUMMARY_PATH = TABLES_DIR / "exp4_gepiii_bsplit_site_stratified.csv"
DIFFICULTY_SUMMARY_PATH = TABLES_DIR / "exp4_gepiii_bsplit_difficulty_stratified.csv"
CLUSTER_SUMMARY_PATH = TABLES_DIR / "exp4_gepiii_bsplit_cluster_stratified.csv"
CLUSTER_K_PATH = TABLES_DIR / "exp4_gepiii_bsplit_cluster_k_metrics.csv"
DRIVER_BUILDINGS_PATH = TABLES_DIR / "exp4_gepiii_bsplit_driver_buildings.csv"
OVERALL_PER_BUILDING_PATH = TABLES_DIR / "exp4_gepiii_bsplit_overall_per_building.csv"
FIGURE_PATH = FIGURES_DIR / "paper" / "paper_fig13_gepiii_bsplit_stratified.png"

FOCUS_MODELS = ["lgbm_lag", "lstm", "patchtst"]
PALETTE = {
    "lgbm_lag": "#2c7fb8",
    "lstm": "#d95f0e",
    "patchtst": "#238b45",
    "lgbm": "#6b6b6b",
}


def _load_predictions() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for model_name, path in PREDICTION_PATHS.items():
        frame = pd.read_csv(path, parse_dates=["timestamp"])
        frame["model"] = model_name
        frame["building_id"] = frame["building_id"].astype(str)
        frame["site_id"] = frame["site_id"].astype(str)
        frame["building_type"] = frame["building_type"].astype(str)
        frames.append(
            frame[
                [
                    "building_id",
                    "site_id",
                    "building_type",
                    "timestamp",
                    "model",
                    "y_true",
                    "y_pred",
                ]
            ]
        )
    merged = pd.concat(frames, ignore_index=True)
    merged["sqerr"] = (merged["y_true"] - merged["y_pred"]) ** 2
    merged["abserr"] = (merged["y_true"] - merged["y_pred"]).abs()
    return merged


def _compute_per_building_metrics(prediction_frame: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        prediction_frame.groupby(["model", "building_id", "site_id", "building_type"], sort=True)
        .agg(
            n_rows=("y_true", "size"),
            mean_true=("y_true", "mean"),
            mae=("abserr", "mean"),
            rmse=("sqerr", lambda s: float(np.sqrt(np.mean(s)))),
        )
        .reset_index()
    )
    grouped = grouped[grouped["mean_true"] >= 1.0].copy()
    grouped["cv_rmse"] = grouped["rmse"] / grouped["mean_true"]
    return grouped


def _summarize_stratum(
    prediction_frame: pd.DataFrame,
    per_building_frame: pd.DataFrame,
    stratum_col: str,
    *,
    ordered_levels: list[str] | None = None,
) -> pd.DataFrame:
    rowwise = (
        prediction_frame.groupby([stratum_col, "model"], sort=True, observed=False)
        .agg(
            n_rows=("y_true", "size"),
            n_buildings=("building_id", "nunique"),
            mean_true=("y_true", "mean"),
            mae=("abserr", "mean"),
            rmse=("sqerr", lambda s: float(np.sqrt(np.mean(s)))),
        )
        .reset_index()
    )
    rowwise["rowwise_cv_rmse"] = rowwise["rmse"] / rowwise["mean_true"].where(rowwise["mean_true"] >= 1.0)
    totals = (
        prediction_frame.groupby("model", sort=True, observed=False)
        .agg(
            total_rows=("y_true", "size"),
            total_true=("y_true", "sum"),
        )
        .reset_index()
    )
    rowwise = rowwise.merge(totals, on="model", how="left")
    rowwise["row_share"] = rowwise["n_rows"] / rowwise["total_rows"]
    load_share = (
        prediction_frame.groupby([stratum_col, "model"], sort=True, observed=False)["y_true"]
        .sum()
        .rename("stratum_true")
        .reset_index()
    )
    rowwise = rowwise.merge(load_share, on=[stratum_col, "model"], how="left")
    rowwise["load_share"] = rowwise["stratum_true"] / rowwise["total_true"]
    rowwise = rowwise.drop(columns=["total_rows", "total_true", "stratum_true"])

    building_summary = (
        per_building_frame.groupby([stratum_col, "model"], sort=True, observed=False)
        .agg(
            per_building_n=("building_id", "nunique"),
            per_building_mean_cv_rmse=("cv_rmse", "mean"),
            per_building_median_cv_rmse=("cv_rmse", "median"),
        )
        .reset_index()
    )

    wide = per_building_frame.pivot(index=["building_id", stratum_col], columns="model", values="cv_rmse").reset_index()
    wide = wide.dropna(subset=FOCUS_MODELS, how="all").copy()
    wide["best_model"] = wide[FOCUS_MODELS + ["lgbm"]].idxmin(axis=1)
    best_share = (
        wide.groupby([stratum_col, "best_model"], sort=True, observed=False)["building_id"]
        .size()
        .rename("n_best")
        .reset_index()
    )
    totals = (
        wide.groupby(stratum_col, sort=True, observed=False)["building_id"]
        .size()
        .rename("n_buildings_for_best_share")
        .reset_index()
    )
    best_share = best_share.merge(totals, on=stratum_col, how="left")
    best_share["share_best_buildings"] = best_share["n_best"] / best_share["n_buildings_for_best_share"]
    best_share = best_share.rename(columns={"best_model": "model"})

    summary = rowwise.merge(building_summary, on=[stratum_col, "model"], how="left")
    summary = summary.merge(
        best_share[[stratum_col, "model", "share_best_buildings"]],
        on=[stratum_col, "model"],
        how="left",
    )

    if ordered_levels is not None:
        summary[stratum_col] = pd.Categorical(summary[stratum_col], categories=ordered_levels, ordered=True)
        summary = summary.sort_values([stratum_col, "model"]).reset_index(drop=True)

    return summary


def _compute_difficulty_labels(per_building_frame: pd.DataFrame) -> pd.DataFrame:
    wide = per_building_frame.pivot(
        index=["building_id", "site_id", "building_type"],
        columns="model",
        values="cv_rmse",
    ).reset_index()
    wide["difficulty_proxy"] = wide[["lgbm", "lgbm_lag", "lstm", "patchtst"]].median(axis=1)
    wide = wide.replace([np.inf, -np.inf], np.nan).dropna(subset=["difficulty_proxy"]).copy()
    wide["difficulty_q"] = pd.qcut(
        wide["difficulty_proxy"],
        4,
        labels=["Q1 easiest", "Q2", "Q3", "Q4 hardest"],
    )
    return wide[["building_id", "difficulty_proxy", "difficulty_q"]]


def _compute_cluster_assignments() -> tuple[pd.DataFrame, pd.DataFrame]:
    full_frame = pd.read_parquet(FEATURES_GEPIII_PATH)
    full_frame["timestamp"] = pd.to_datetime(full_frame["timestamp"])
    full_frame["building_id"] = full_frame["building_id"].astype(str)
    full_frame = full_frame.sort_values(["building_id", "timestamp"]).reset_index(drop=True)
    test_buildings = (
        pd.read_csv(PREDICTION_PATHS["lgbm_lag"], usecols=["building_id"])["building_id"]
        .astype(str)
        .unique()
        .tolist()
    )
    test_building_set = set(test_buildings)
    test_frame = full_frame[full_frame["building_id"].isin(test_building_set)].reset_index(drop=True)
    train_frame = full_frame[~full_frame["building_id"].isin(test_building_set)].reset_index(drop=True)

    train_features = extract_building_profile_features(train_frame)
    scaled_train, scaler = fit_profile_scaler(train_features)

    metric_rows: list[dict[str, float | int]] = []
    for k in [3, 4, 5, 6]:
        model = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = model.fit_predict(scaled_train)
        metric_rows.append(
            {
                "k": int(k),
                "silhouette_score": float(silhouette_score(scaled_train, labels)),
            }
        )

    metrics = pd.DataFrame(metric_rows).sort_values("k").reset_index(drop=True)
    selected_k = int(metrics.sort_values(["silhouette_score", "k"], ascending=[False, True]).iloc[0]["k"])

    model = KMeans(n_clusters=selected_k, random_state=42, n_init=20)
    model.fit(scaled_train)

    test_features = extract_building_profile_features(test_frame)
    scaled_test = transform_profile_features(test_features, scaler)
    test_labels = model.predict(scaled_test)
    assignments = pd.DataFrame(
        {
            "building_id": test_features["building_id"].astype(str),
            "cluster_label": test_labels.astype(int),
        }
    )
    return assignments, metrics


def _compute_driver_buildings(
    prediction_frame: pd.DataFrame,
    per_building_frame: pd.DataFrame,
    difficulty_labels: pd.DataFrame,
    cluster_assignments: pd.DataFrame,
) -> pd.DataFrame:
    building_load = (
        prediction_frame.groupby("building_id", sort=True)["y_true"]
        .mean()
        .rename("mean_true")
        .reset_index()
    )
    pivot = per_building_frame.pivot(index="building_id", columns="model", values="cv_rmse").reset_index()
    mse_pieces: list[pd.DataFrame] = []
    for model_name in PREDICTION_PATHS:
        subset = prediction_frame.loc[prediction_frame["model"] == model_name]
        piece = (
            subset.groupby("building_id", sort=True)["sqerr"]
            .mean()
            .rename(f"mse_{model_name}")
            .reset_index()
        )
        mse_pieces.append(piece)

    merged = pivot.merge(building_load, on="building_id", how="left")
    meta = per_building_frame[["building_id", "site_id", "building_type"]].drop_duplicates()
    merged = merged.merge(meta, on="building_id", how="left")
    merged = merged.merge(difficulty_labels, on="building_id", how="left")
    merged = merged.merge(cluster_assignments, on="building_id", how="left")
    for piece in mse_pieces:
        merged = merged.merge(piece, on="building_id", how="left")
    merged["mse_diff_patchtst_minus_lgbm_lag"] = merged["mse_patchtst"] - merged["mse_lgbm_lag"]
    merged["cv_diff_patchtst_minus_lgbm_lag"] = merged["patchtst"] - merged["lgbm_lag"]
    return merged.sort_values("mean_true", ascending=False).reset_index(drop=True)


def _compute_overall_per_building_summary(per_building_frame: pd.DataFrame) -> pd.DataFrame:
    summary = (
        per_building_frame.groupby("model", sort=True)
        .agg(
            n_buildings=("building_id", "nunique"),
            mean_cv_rmse=("cv_rmse", "mean"),
            median_cv_rmse=("cv_rmse", "median"),
        )
        .reset_index()
    )
    wide = per_building_frame.pivot(index="building_id", columns="model", values="cv_rmse").reset_index()
    wide["best_model"] = wide[["lgbm", "lgbm_lag", "lstm", "patchtst"]].idxmin(axis=1)
    best_share = (
        wide["best_model"].value_counts(normalize=True, sort=False)
        .rename_axis("model")
        .rename("share_best_buildings")
        .reset_index()
    )
    return summary.merge(best_share, on="model", how="left")


def _plot_stratified_figure(
    type_summary: pd.DataFrame,
    difficulty_summary: pd.DataFrame,
    cluster_summary: pd.DataFrame,
) -> None:
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.2))
    ax_type_row, ax_type_median, ax_diff, ax_cluster = axes.flatten()

    high_support = (
        type_summary.groupby("building_type", sort=False)["n_buildings"]
        .max()
        .sort_values(ascending=False)
    )
    high_support = high_support[high_support >= 10].index.tolist()
    type_medians = (
        type_summary.loc[
            (type_summary["model"] == "lgbm_lag") & (type_summary["building_type"].isin(high_support)),
            ["building_type", "per_building_median_cv_rmse"],
        ]
        .sort_values("per_building_median_cv_rmse")["building_type"]
        .tolist()
    )

    y = np.arange(len(type_medians))
    offsets = {"lgbm_lag": -0.22, "lstm": 0.0, "patchtst": 0.22}
    for model_name in FOCUS_MODELS:
        subset = type_summary[
            (type_summary["model"] == model_name) & (type_summary["building_type"].isin(type_medians))
        ].set_index("building_type").loc[type_medians]
        ax_type_row.scatter(
            subset["rowwise_cv_rmse"],
            y + offsets[model_name],
            s=55,
            color=PALETTE[model_name],
            label=model_name.replace("_", "+") if model_name == "lgbm_lag" else model_name.upper(),
        )
        ax_type_median.scatter(
            subset["per_building_median_cv_rmse"],
            y + offsets[model_name],
            s=55,
            color=PALETTE[model_name],
        )
    ax_type_row.set_yticks(y, type_medians)
    ax_type_median.set_yticks(y, type_medians)
    ax_type_row.set_title("High-support building types\nRow-wise pooled CV(RMSE)")
    ax_type_median.set_title("High-support building types\nMedian per-building CV(RMSE)")
    ax_type_row.set_xlabel("CV(RMSE)")
    ax_type_median.set_xlabel("CV(RMSE)")
    ax_type_row.invert_yaxis()
    ax_type_median.invert_yaxis()
    ax_type_row.grid(axis="x", alpha=0.25)
    ax_type_median.grid(axis="x", alpha=0.25)
    ax_type_row.legend(loc="lower right", frameon=False)

    difficulty_order = ["Q1 easiest", "Q2", "Q3", "Q4 hardest"]
    x = np.arange(len(difficulty_order))
    width = 0.22
    for idx, model_name in enumerate(FOCUS_MODELS):
        subset = (
            difficulty_summary[difficulty_summary["model"] == model_name]
            .set_index("difficulty_q")
            .reindex(difficulty_order)
        )
        ax_diff.bar(
            x + (idx - 1) * width,
            subset["rowwise_cv_rmse"],
            width=width,
            color=PALETTE[model_name],
            label=model_name.replace("_", "+") if model_name == "lgbm_lag" else model_name.upper(),
        )
    ax_diff.set_xticks(x, difficulty_order)
    ax_diff.set_ylabel("Row-wise CV(RMSE)")
    ax_diff.set_title("Difficulty quartiles")
    ax_diff.grid(axis="y", alpha=0.25)

    cluster_order = (
        cluster_summary.groupby("cluster_label", sort=True)["n_buildings"]
        .max()
        .sort_index()
        .index
        .tolist()
    )
    x = np.arange(len(cluster_order))
    for idx, model_name in enumerate(FOCUS_MODELS):
        subset = (
            cluster_summary[cluster_summary["model"] == model_name]
            .set_index("cluster_label")
            .reindex(cluster_order)
        )
        labels = [
            f"C{int(cluster)}\n(n={int(subset.loc[cluster, 'n_buildings'])})"
            for cluster in cluster_order
        ]
        ax_cluster.bar(
            x + (idx - 1) * width,
            subset["rowwise_cv_rmse"],
            width=width,
            color=PALETTE[model_name],
        )
        ax_cluster.set_xticks(x, labels)
    ax_cluster.set_ylabel("Row-wise CV(RMSE)")
    ax_cluster.set_title("Load-profile clusters")
    ax_cluster.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=320, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ensure_phase2_dirs()

    prediction_frame = _load_predictions()
    per_building = _compute_per_building_metrics(prediction_frame)

    difficulty_labels = _compute_difficulty_labels(per_building)
    prediction_with_difficulty = prediction_frame.merge(difficulty_labels, on="building_id", how="inner")
    per_building_with_difficulty = per_building.merge(difficulty_labels, on="building_id", how="inner")

    cluster_assignments, cluster_metrics = _compute_cluster_assignments()
    prediction_with_cluster = prediction_frame.merge(cluster_assignments, on="building_id", how="left")
    per_building_with_cluster = per_building.merge(cluster_assignments, on="building_id", how="left")

    type_summary = _summarize_stratum(prediction_frame, per_building, "building_type")
    site_summary = _summarize_stratum(prediction_frame, per_building, "site_id")
    difficulty_summary = _summarize_stratum(
        prediction_with_difficulty,
        per_building_with_difficulty,
        "difficulty_q",
        ordered_levels=["Q1 easiest", "Q2", "Q3", "Q4 hardest"],
    )
    cluster_summary = _summarize_stratum(
        prediction_with_cluster,
        per_building_with_cluster,
        "cluster_label",
    )

    drivers = _compute_driver_buildings(
        prediction_frame,
        per_building,
        difficulty_labels,
        cluster_assignments,
    )
    overall_per_building = _compute_overall_per_building_summary(per_building)

    type_summary.to_csv(TYPE_SUMMARY_PATH, index=False)
    site_summary.to_csv(SITE_SUMMARY_PATH, index=False)
    difficulty_summary.to_csv(DIFFICULTY_SUMMARY_PATH, index=False)
    cluster_summary.to_csv(CLUSTER_SUMMARY_PATH, index=False)
    cluster_metrics.to_csv(CLUSTER_K_PATH, index=False)
    drivers.to_csv(DRIVER_BUILDINGS_PATH, index=False)
    overall_per_building.to_csv(OVERALL_PER_BUILDING_PATH, index=False)

    _plot_stratified_figure(type_summary, difficulty_summary, cluster_summary)

    print(f"type_summary: {TYPE_SUMMARY_PATH}")
    print(f"site_summary: {SITE_SUMMARY_PATH}")
    print(f"difficulty_summary: {DIFFICULTY_SUMMARY_PATH}")
    print(f"cluster_summary: {CLUSTER_SUMMARY_PATH}")
    print(f"cluster_k_metrics: {CLUSTER_K_PATH}")
    print(f"driver_buildings: {DRIVER_BUILDINGS_PATH}")
    print(f"overall_per_building: {OVERALL_PER_BUILDING_PATH}")
    print(f"figure: {FIGURE_PATH}")


if __name__ == "__main__":
    main()
