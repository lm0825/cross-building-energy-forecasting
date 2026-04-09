from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def standardize_bdg2_metadata(metadata: pd.DataFrame) -> pd.DataFrame:
    standardized = metadata.copy()
    if "building_type" not in standardized.columns and "primaryspaceusage" in standardized.columns:
        standardized["building_type"] = standardized["primaryspaceusage"]
    if "floor_area" not in standardized.columns and "sqm" in standardized.columns:
        standardized["floor_area"] = standardized["sqm"]
    standardized["log_floor_area"] = standardized["floor_area"].map(
        lambda x: np.nan if pd.isna(x) or x <= 0 else float(np.log(x))
    )
    return standardized


def standardize_bdg2_weather(weather: pd.DataFrame) -> pd.DataFrame:
    standardized = weather.copy()
    standardized["timestamp"] = pd.to_datetime(standardized["timestamp"])
    return standardized[
        [
            "timestamp",
            "site_id",
            "airTemperature",
            "dewTemperature",
            "windSpeed",
            "cloudCoverage",
        ]
    ].copy()


def standardize_gepiii_metadata(metadata: pd.DataFrame) -> pd.DataFrame:
    standardized = metadata.copy()
    standardized = standardized.rename(
        columns={
            "primary_use": "building_type",
            "square_feet": "floor_area_sqft",
        }
    )
    standardized["floor_area"] = standardized["floor_area_sqft"] * 0.092903
    standardized["log_floor_area"] = standardized["floor_area"].map(
        lambda x: np.nan if pd.isna(x) or x <= 0 else float(np.log(x))
    )
    return standardized


def standardize_gepiii_weather(weather: pd.DataFrame) -> pd.DataFrame:
    standardized = weather.copy()
    standardized["timestamp"] = pd.to_datetime(standardized["timestamp"])
    return standardized.rename(
        columns={
            "air_temperature": "airTemperature",
            "cloud_coverage": "cloudCoverage",
            "dew_temperature": "dewTemperature",
            "wind_speed": "windSpeed",
        }
    )[
        [
            "timestamp",
            "site_id",
            "airTemperature",
            "dewTemperature",
            "windSpeed",
            "cloudCoverage",
        ]
    ].copy()


def add_time_features(frame: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
    enriched = frame.copy()
    timestamps = pd.to_datetime(enriched[timestamp_col])
    enriched["hour_of_day"] = timestamps.dt.hour
    enriched["day_of_week"] = timestamps.dt.dayofweek
    enriched["month"] = timestamps.dt.month
    enriched["is_weekend"] = (enriched["day_of_week"] >= 5).astype("int8")
    # Holiday calendars vary by site/country and should be added once country mapping is finalized.
    enriched["is_holiday"] = pd.NA
    return enriched


def melt_bdg2_electricity_to_long(electricity: pd.DataFrame, building_ids: list[str] | None = None) -> pd.DataFrame:
    columns = ["timestamp"]
    if building_ids is None:
        columns.extend([c for c in electricity.columns if c != "timestamp"])
    else:
        columns.extend(building_ids)

    subset = electricity.loc[:, columns].copy()
    long_frame = subset.melt(
        id_vars="timestamp",
        var_name="building_id",
        value_name="meter_reading",
    )
    return long_frame


def export_features(frame: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    try:
        frame.to_parquet(output_path, index=False)
    except ImportError as exc:
        raise ImportError(
            "Parquet export requires `pyarrow` or `fastparquet`. "
            "Install one of them before generating features_all.parquet."
        ) from exc
