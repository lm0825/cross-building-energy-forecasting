from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.feature_engineering import (
    add_time_features,
    export_features,
    standardize_gepiii_metadata,
    standardize_gepiii_weather,
)

DATA_DIR = ROOT / "data" / "gepiii"
TABLES_DIR = ROOT / "tables"

TRAIN_PATH = DATA_DIR / "train.csv"
META_PATH = DATA_DIR / "building_metadata.csv"
WEATHER_PATH = DATA_DIR / "weather_train.csv"

FILTERED_METER_IDS_PATH = DATA_DIR / "filtered_meter_ids.csv"
FEATURES_PATH = DATA_DIR / "features_all.parquet"
FILTERING_SUMMARY_PATH = TABLES_DIR / "gepiii_filtering_summary.csv"

FULL_YEAR_INDEX = pd.date_range("2016-01-01 00:00:00", "2016-12-31 23:00:00", freq="h")
FULL_YEAR_HOURS = len(FULL_YEAR_INDEX)


def ensure_output_dirs() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(TRAIN_PATH, parse_dates=["timestamp"])
    metadata = pd.read_csv(META_PATH)
    weather = pd.read_csv(WEATHER_PATH, parse_dates=["timestamp"])
    return train, metadata, weather


def compute_meter_stats(train: pd.DataFrame) -> pd.DataFrame:
    electricity = train[train["meter"] == 0].copy()

    stats = (
        electricity.groupby("building_id")
        .agg(
            site_id=("building_id", lambda _: np.nan),
            observed_hours=("timestamp", "size"),
            first_timestamp=("timestamp", "min"),
            last_timestamp=("timestamp", "max"),
        )
        .reset_index()
    )

    site_map = (
        pd.read_csv(META_PATH, usecols=["building_id", "site_id"])
        .drop_duplicates("building_id")
        .set_index("building_id")["site_id"]
    )
    stats["site_id"] = stats["building_id"].map(site_map)

    stats["coverage_hours_2016"] = (
        (
            stats["last_timestamp"].sub(stats["first_timestamp"]).dt.total_seconds() / 3600
        ).astype("int64")
        + 1
    )
    stats["coverage_full_year_2016"] = stats["coverage_hours_2016"] >= FULL_YEAR_HOURS
    stats["missing_hours_2016"] = FULL_YEAR_HOURS - stats["observed_hours"]
    stats["missing_rate"] = stats["missing_hours_2016"] / FULL_YEAR_HOURS
    return stats.sort_values("building_id").reset_index(drop=True)


def fill_series_gaps(series: pd.Series, short_gap_hours: int = 6) -> pd.Series:
    result = series.copy()
    is_missing = result.isna()
    if not is_missing.any():
        return result

    missing_groups = is_missing.ne(is_missing.shift(fill_value=False)).cumsum()
    valid_groups = (
        pd.DataFrame({"missing": is_missing, "group": missing_groups})
        .groupby("group")
        .agg(is_missing=("missing", "first"), gap_size=("missing", "size"))
    )

    for group_id, row in valid_groups.iterrows():
        if not row["is_missing"]:
            continue

        gap_idx = missing_groups[missing_groups == group_id].index
        start = gap_idx[0]
        end = gap_idx[-1]
        gap_size = int(row["gap_size"])

        left_pos = result.index.get_loc(start) - 1
        right_pos = result.index.get_loc(end) + 1

        has_left = left_pos >= 0 and pd.notna(result.iloc[left_pos])
        has_right = right_pos < len(result) and pd.notna(result.iloc[right_pos])

        if gap_size <= short_gap_hours and has_left and has_right:
            x = np.array([left_pos, right_pos], dtype=float)
            y = np.array([result.iloc[left_pos], result.iloc[right_pos]], dtype=float)
            fill_positions = np.arange(left_pos + 1, right_pos, dtype=float)
            result.loc[gap_idx] = np.interp(fill_positions, x, y)
        else:
            if has_left:
                result.loc[gap_idx] = result.iloc[left_pos]

    return result


def build_filtered_long_frame(train: pd.DataFrame, filtered_ids: list[int]) -> pd.DataFrame:
    electricity = train[(train["meter"] == 0) & (train["building_id"].isin(filtered_ids))].copy()

    frames: list[pd.DataFrame] = []
    for building_id, group in electricity.groupby("building_id", sort=True):
        series = group.set_index("timestamp")["meter_reading"].sort_index()
        full_series = series.reindex(FULL_YEAR_INDEX)
        filled = fill_series_gaps(full_series)

        frame = pd.DataFrame(
            {
                "timestamp": FULL_YEAR_INDEX,
                "building_id": building_id,
                "meter": 0,
                "meter_reading": filled.to_numpy(),
            }
        )
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)


def prepare_weather(weather: pd.DataFrame) -> pd.DataFrame:
    standardized = standardize_gepiii_weather(weather)
    all_sites = sorted(standardized["site_id"].dropna().unique().tolist())
    full_index = pd.MultiIndex.from_product(
        [all_sites, FULL_YEAR_INDEX],
        names=["site_id", "timestamp"],
    )

    weather_full = (
        standardized.set_index(["site_id", "timestamp"])
        .sort_index()
        .reindex(full_index)
        .reset_index()
    )

    fill_columns = ["airTemperature", "dewTemperature", "windSpeed", "cloudCoverage"]
    for column in fill_columns:
        weather_full[column] = weather_full.groupby("site_id")[column].transform(
            lambda s: s.interpolate(limit_direction="both")
        )

    return weather_full


def build_features(
    filtered_long: pd.DataFrame, metadata: pd.DataFrame, weather: pd.DataFrame
) -> pd.DataFrame:
    meta_std = standardize_gepiii_metadata(metadata)
    weather_std = prepare_weather(weather)

    features = filtered_long.merge(meta_std, on="building_id", how="left")
    features = features.merge(weather_std, on=["site_id", "timestamp"], how="left")
    features = add_time_features(features, timestamp_col="timestamp")
    features = features.sort_values(["building_id", "timestamp"]).reset_index(drop=True)
    return features


def build_filtering_summary(
    stats: pd.DataFrame, filtered_meta: pd.DataFrame
) -> pd.DataFrame:
    steps = [
        {
            "step_id": "0_start",
            "rule": "All GEPIII training buildings",
            "buildings_remaining": int(stats["building_id"].nunique()),
            "sites_remaining": int(stats["site_id"].nunique()),
        },
        {
            "step_id": "1_electricity_only",
            "rule": "Keep electricity meters only (meter == 0)",
            "buildings_remaining": int(stats["building_id"].nunique()),
            "sites_remaining": int(stats["site_id"].nunique()),
        },
    ]

    coverage = stats[stats["coverage_full_year_2016"]].copy()
    steps.append(
        {
            "step_id": "2_coverage_ge_12_months",
            "rule": "Keep buildings with 2016 coverage spanning the full year",
            "buildings_remaining": int(coverage["building_id"].nunique()),
            "sites_remaining": int(coverage["site_id"].nunique()),
        }
    )

    missing = coverage[coverage["missing_rate"] <= 0.25].copy()
    steps.append(
        {
            "step_id": "3_missing_rate_le_25pct",
            "rule": "Keep buildings with missing rate <= 25%",
            "buildings_remaining": int(missing["building_id"].nunique()),
            "sites_remaining": int(missing["site_id"].nunique()),
        }
    )

    steps.append(
        {
            "step_id": "4_gap_filling",
            "rule": "Interpolate short gaps and forward-fill gaps longer than 6 hours",
            "buildings_remaining": int(missing["building_id"].nunique()),
            "sites_remaining": int(missing["site_id"].nunique()),
        }
    )

    steps.append(
        {
            "step_id": "5_one_meter_per_building",
            "rule": "Keep one electricity meter per building; GEPIII electricity data is unique by building",
            "buildings_remaining": int(missing["building_id"].nunique()),
            "sites_remaining": int(missing["site_id"].nunique()),
        }
    )

    steps.append(
        {
            "step_id": "6_complete_core_metadata",
            "rule": "Keep buildings with non-null building type, floor area, and site_id",
            "buildings_remaining": int(filtered_meta["building_id"].nunique()),
            "sites_remaining": int(filtered_meta["site_id"].nunique()),
        }
    )

    return pd.DataFrame(steps)


def run() -> dict[str, Path]:
    ensure_output_dirs()
    train, metadata, weather = load_inputs()

    meter_stats = compute_meter_stats(train)
    filtered_meter_ids = meter_stats[
        meter_stats["coverage_full_year_2016"] & (meter_stats["missing_rate"] <= 0.25)
    ].copy()

    filtered_meta = metadata[metadata["building_id"].isin(filtered_meter_ids["building_id"])].copy()
    filtered_meta = filtered_meta[
        filtered_meta["primary_use"].notna()
        & filtered_meta["square_feet"].notna()
        & filtered_meta["site_id"].notna()
    ].copy()

    filtered_meter_ids = filtered_meter_ids[
        filtered_meter_ids["building_id"].isin(filtered_meta["building_id"])
    ].copy()

    filtered_long = build_filtered_long_frame(train, filtered_meter_ids["building_id"].tolist())
    features = build_features(filtered_long, filtered_meta, weather)
    filtering_summary = build_filtering_summary(meter_stats, filtered_meta)

    filtered_meter_ids.sort_values("building_id").to_csv(FILTERED_METER_IDS_PATH, index=False)
    filtering_summary.to_csv(FILTERING_SUMMARY_PATH, index=False)
    export_features(features, FEATURES_PATH)

    return {
        "filtered_meter_ids": FILTERED_METER_IDS_PATH,
        "features_all": FEATURES_PATH,
        "filtering_summary": FILTERING_SUMMARY_PATH,
    }


if __name__ == "__main__":
    outputs = run()
    for name, path in outputs.items():
        print(f"{name}: {path}")
