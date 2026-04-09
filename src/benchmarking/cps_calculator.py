from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.benchmarking.residual_calculator import RESIDUAL_DIMENSION_COLUMNS


RESIDUAL_TO_SCORE_COLUMN = {
    "annual_mean_residual": "score_annual",
    "worktime_residual": "score_worktime",
    "nighttime_residual": "score_nighttime",
    "summer_residual": "score_summer",
    "winter_residual": "score_winter",
}


def percentile_rank_within_group(values: pd.Series) -> pd.Series:
    valid = values.dropna()
    if valid.empty:
        return pd.Series(np.nan, index=values.index, dtype=np.float64)
    if len(valid) == 1:
        ranked = pd.Series(0.5, index=valid.index, dtype=np.float64)
    else:
        ranks = valid.rank(method="average", ascending=True)
        ranked = (ranks - 1.0) / float(len(valid) - 1)
    return ranked.reindex(values.index)


def build_cps_frame(
    residual_frame: pd.DataFrame,
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    frame = residual_frame.copy()
    weights = weights or {name: 1.0 for name in RESIDUAL_TO_SCORE_COLUMN.values()}

    for residual_col, score_col in RESIDUAL_TO_SCORE_COLUMN.items():
        frame[score_col] = frame.groupby("building_type", sort=False)[residual_col].transform(
            percentile_rank_within_group
        )

    score_columns = list(RESIDUAL_TO_SCORE_COLUMN.values())
    weight_vector = np.asarray([weights[column] for column in score_columns], dtype=np.float64)
    score_matrix = frame[score_columns].to_numpy(dtype=np.float64)

    def _weighted_mean(row: np.ndarray) -> float:
        mask = np.isfinite(row)
        if not mask.any():
            return float("nan")
        return float(np.average(row[mask], weights=weight_vector[mask]))

    frame["cps"] = np.apply_along_axis(_weighted_mean, 1, score_matrix)
    frame["cps_percentile"] = frame.groupby("building_type", sort=False)["cps"].transform(
        percentile_rank_within_group
    )
    frame["type_building_count"] = frame.groupby("building_type", sort=False)["building_id"].transform("size")

    ordered = [
        "building_id",
        "site_id",
        "building_type",
        "floor_area",
        "model_name",
        "type_building_count",
        *RESIDUAL_DIMENSION_COLUMNS,
        *score_columns,
        "cps",
        "cps_percentile",
    ]
    existing = [column for column in ordered if column in frame.columns]
    remaining = [column for column in frame.columns if column not in existing]
    return frame[existing + remaining].sort_values(["building_type", "cps", "building_id"]).reset_index(drop=True)


def plot_cps_distribution_by_type(
    cps_frame: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    plotting = cps_frame.dropna(subset=["building_type", "cps"]).copy()
    counts = plotting.groupby("building_type", sort=True)["building_id"].size()
    order = (
        plotting.groupby("building_type", sort=True)["cps"]
        .median()
        .sort_values()
        .index
        .tolist()
    )
    data = [plotting.loc[plotting["building_type"] == name, "cps"].to_numpy() for name in order]
    labels = [f"{name} (n={int(counts[name])})" for name in order]

    fig_height = max(6.0, 0.4 * len(order) + 2.5)
    plt.figure(figsize=(12, fig_height))
    plt.boxplot(
        data,
        vert=False,
        labels=labels,
        patch_artist=True,
        showfliers=False,
        boxprops={"facecolor": "#dce6f2", "edgecolor": "#44617b"},
        medianprops={"color": "#b23a48", "linewidth": 1.5},
        whiskerprops={"color": "#44617b"},
        capprops={"color": "#44617b"},
    )
    plt.xlabel("Composite Performance Score (lower is more efficient)")
    plt.title("Experiment 3 CPS Distribution by Building Type")
    plt.grid(axis="x", alpha=0.25, linestyle="--")
    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path
