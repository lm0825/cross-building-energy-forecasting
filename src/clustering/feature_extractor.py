from __future__ import annotations

import pickle
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import CLUSTERING_DIR, ensure_phase2_dirs


TRAIN_BUILDING_FEATURES_PATH = CLUSTERING_DIR / "train_building_features.npy"
TRAIN_BUILDING_INDEX_PATH = CLUSTERING_DIR / "train_building_feature_index.csv"
FEATURE_SCALER_PATH = CLUSTERING_DIR / "feature_scaler.pkl"

WORKDAY_COLUMNS = [f"workday_hour_{hour:02d}" for hour in range(24)]
WEEKEND_COLUMNS = [f"weekend_hour_{hour:02d}" for hour in range(24)]
PROFILE_FEATURE_COLUMNS = WORKDAY_COLUMNS + WEEKEND_COLUMNS


def _full_profile_index(building_ids: list[str]) -> pd.MultiIndex:
    return pd.MultiIndex.from_product(
        [building_ids, [0, 1], range(24)],
        names=["building_id", "is_weekend", "hour_of_day"],
    )


def extract_building_profile_features(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        raise ValueError("Cannot extract clustering features from an empty frame.")

    working = frame[["building_id", "timestamp", "meter_reading"]].copy()
    working["timestamp"] = pd.to_datetime(working["timestamp"])
    working["hour_of_day"] = working["timestamp"].dt.hour.astype(np.int8)
    working["is_weekend"] = working["timestamp"].dt.dayofweek.ge(5).astype(np.int8)

    building_ids = sorted(working["building_id"].astype(str).unique().tolist())
    complete = pd.DataFrame(index=_full_profile_index(building_ids)).reset_index()

    grouped = (
        working.groupby(["building_id", "is_weekend", "hour_of_day"], sort=True)["meter_reading"]
        .median()
        .rename("profile_value")
        .reset_index()
    )
    overall_hourly = (
        working.groupby(["building_id", "hour_of_day"], sort=True)["meter_reading"]
        .median()
        .rename("overall_hourly_value")
        .reset_index()
    )
    building_median = (
        working.groupby("building_id", sort=True)["meter_reading"]
        .median()
        .rename("building_median")
        .reset_index()
    )
    global_median = float(working["meter_reading"].median())

    merged = complete.merge(
        grouped,
        on=["building_id", "is_weekend", "hour_of_day"],
        how="left",
    ).merge(
        overall_hourly,
        on=["building_id", "hour_of_day"],
        how="left",
    ).merge(
        building_median,
        on="building_id",
        how="left",
    )
    merged["profile_value"] = (
        merged["profile_value"]
        .fillna(merged["overall_hourly_value"])
        .fillna(merged["building_median"])
        .fillna(global_median)
    )

    label_map = {0: "workday", 1: "weekend"}
    merged["profile_column"] = (
        merged["is_weekend"].map(label_map)
        + "_hour_"
        + merged["hour_of_day"].astype(int).astype(str).str.zfill(2)
    )

    pivot = (
        merged.pivot(index="building_id", columns="profile_column", values="profile_value")
        .reindex(columns=PROFILE_FEATURE_COLUMNS)
        .reset_index()
    )
    return pivot


def fit_profile_scaler(feature_frame: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    scaled = scaler.fit_transform(feature_frame[PROFILE_FEATURE_COLUMNS].to_numpy(dtype=np.float64))
    return scaled.astype(np.float32), scaler


def transform_profile_features(feature_frame: pd.DataFrame, scaler: StandardScaler) -> np.ndarray:
    scaled = scaler.transform(feature_frame[PROFILE_FEATURE_COLUMNS].to_numpy(dtype=np.float64))
    return scaled.astype(np.float32)


def save_profile_scaler(scaler: StandardScaler, path: str | Path = FEATURE_SCALER_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        pickle.dump(scaler, file, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_profile_scaler(path: str | Path = FEATURE_SCALER_PATH) -> StandardScaler:
    with Path(path).open("rb") as file:
        return pickle.load(file)


def save_train_feature_artifacts(
    feature_frame: pd.DataFrame,
    scaled_features: np.ndarray,
    scaler: StandardScaler,
) -> dict[str, Path]:
    ensure_phase2_dirs()
    TRAIN_BUILDING_FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(TRAIN_BUILDING_FEATURES_PATH, scaled_features)
    feature_frame[["building_id"]].to_csv(TRAIN_BUILDING_INDEX_PATH, index=False)
    save_profile_scaler(scaler, FEATURE_SCALER_PATH)
    return {
        "train_building_features": TRAIN_BUILDING_FEATURES_PATH,
        "train_building_feature_index": TRAIN_BUILDING_INDEX_PATH,
        "feature_scaler": FEATURE_SCALER_PATH,
    }
