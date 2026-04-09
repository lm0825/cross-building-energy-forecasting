from __future__ import annotations

from pathlib import Path

import pandas as pd


PREDICTION_USECOLS = [
    "building_id",
    "site_id",
    "building_type",
    "timestamp",
    "y_true",
    "y_pred",
]

RESIDUAL_DIMENSION_COLUMNS = [
    "annual_mean_residual",
    "worktime_residual",
    "nighttime_residual",
    "summer_residual",
    "winter_residual",
]


def load_prediction_frame(prediction_path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(
        prediction_path,
        usecols=PREDICTION_USECOLS,
        parse_dates=["timestamp"],
    )
    frame["building_id"] = frame["building_id"].astype(str)
    frame["site_id"] = frame["site_id"].astype(str)
    frame["building_type"] = frame["building_type"].astype(str)
    return frame.sort_values(["building_id", "timestamp"]).reset_index(drop=True)


def apply_low_mean_filter(
    prediction_frame: pd.DataFrame,
    filter_frame: pd.DataFrame,
) -> pd.DataFrame:
    allowed = filter_frame.loc[
        ~filter_frame["exclude_from_benchmarking"],
        ["building_id", "mean_actual", "threshold_quantile", "threshold_value"],
    ].copy()
    allowed["building_id"] = allowed["building_id"].astype(str)
    merged = prediction_frame.merge(
        allowed,
        on="building_id",
        how="inner",
        validate="many_to_one",
    )
    return merged.sort_values(["building_id", "timestamp"]).reset_index(drop=True)


def add_residual_features(prediction_frame: pd.DataFrame) -> pd.DataFrame:
    frame = prediction_frame.copy()
    frame["residual"] = frame["y_true"] - frame["y_pred"]
    frame["hour_of_day"] = frame["timestamp"].dt.hour.astype("int16")
    frame["day_of_week"] = frame["timestamp"].dt.dayofweek.astype("int16")
    frame["month"] = frame["timestamp"].dt.month.astype("int16")
    frame["is_worktime"] = (
        frame["day_of_week"].lt(5)
        & frame["hour_of_day"].ge(9)
        & frame["hour_of_day"].lt(18)
    )
    frame["is_nighttime"] = frame["hour_of_day"].lt(6)
    frame["is_summer"] = frame["month"].isin([6, 7, 8])
    frame["is_winter"] = frame["month"].isin([12, 1, 2])
    return frame


def build_residual_summary(
    prediction_frame: pd.DataFrame,
    metadata_frame: pd.DataFrame,
    model_name: str,
) -> pd.DataFrame:
    frame = add_residual_features(prediction_frame)

    base = (
        frame.groupby("building_id", sort=True)
        .agg(
            site_id=("site_id", "first"),
            building_type=("building_type", "first"),
            n_prediction_rows=("residual", "size"),
            prediction_start=("timestamp", "min"),
            prediction_end=("timestamp", "max"),
            annual_mean_actual=("y_true", "mean"),
            annual_mean_predicted=("y_pred", "mean"),
            annual_mean_residual=("residual", "mean"),
        )
        .reset_index()
    )

    def _merge_mask_stats(mask: pd.Series, residual_col: str, count_col: str) -> None:
        nonlocal base
        subset = frame.loc[mask, ["building_id", "residual"]]
        means = (
            subset.groupby("building_id", sort=True)["residual"]
            .mean()
            .rename(residual_col)
            .reset_index()
        )
        counts = (
            subset.groupby("building_id", sort=True)["residual"]
            .size()
            .rename(count_col)
            .reset_index()
        )
        base = base.merge(means, on="building_id", how="left")
        base = base.merge(counts, on="building_id", how="left")

    _merge_mask_stats(frame["is_worktime"], "worktime_residual", "n_worktime_rows")
    _merge_mask_stats(frame["is_nighttime"], "nighttime_residual", "n_nighttime_rows")
    _merge_mask_stats(frame["is_summer"], "summer_residual", "n_summer_rows")
    _merge_mask_stats(frame["is_winter"], "winter_residual", "n_winter_rows")

    metadata = metadata_frame[
        ["building_id", "floor_area", "site_id", "building_type"]
    ].drop_duplicates(subset=["building_id"])
    metadata["building_id"] = metadata["building_id"].astype(str)
    metadata = metadata.rename(
        columns={
            "site_id": "meta_site_id",
            "building_type": "meta_building_type",
        }
    )

    base = base.merge(metadata, on="building_id", how="left", validate="one_to_one")
    base["site_id"] = base["site_id"].fillna(base["meta_site_id"])
    base["building_type"] = base["building_type"].fillna(base["meta_building_type"])
    base = base.drop(columns=["meta_site_id", "meta_building_type"])
    base["model_name"] = model_name

    ordered_columns = [
        "building_id",
        "site_id",
        "building_type",
        "floor_area",
        "model_name",
        "n_prediction_rows",
        "prediction_start",
        "prediction_end",
        "annual_mean_actual",
        "annual_mean_predicted",
        "annual_mean_residual",
        "worktime_residual",
        "nighttime_residual",
        "summer_residual",
        "winter_residual",
        "n_worktime_rows",
        "n_nighttime_rows",
        "n_summer_rows",
        "n_winter_rows",
    ]
    return base[ordered_columns].sort_values(["building_type", "building_id"]).reset_index(drop=True)

