from __future__ import annotations

import json
import math
import os
import random
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.config import (
    CONTEXT_WINDOW,
    DEFAULT_GPU_INDEX,
    DEFAULT_TORCH_DEVICE,
    EVAL_WINDOW_STRIDE,
    EXP1_PREDICTIONS_DIR,
    MODELS_DIR,
    PREDICTION_HORIZON,
    RANDOM_SEED,
    SEQUENCE_DYNAMIC_COLUMNS,
    SEQUENCE_STATIC_COLUMNS,
    TABULAR_CATEGORICAL_COLUMNS,
    TABULAR_FEATURE_COLUMNS,
    TABULAR_LAG_FEATURE_COLUMNS,
    TRAIN_WINDOW_STRIDE,
    ensure_phase2_dirs,
)


def set_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(payload: object, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if is_dataclass(payload):
        payload = asdict(payload)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=True, indent=2)


def load_feature_frame(feature_path: str | Path, columns: list[str] | None = None) -> pd.DataFrame:
    frame = pd.read_parquet(feature_path, columns=columns)
    if "timestamp" in frame.columns:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    return frame


def temporal_validation_split(
    frame: pd.DataFrame,
    validation_ratio: float = 0.1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    unique_timestamps = np.sort(frame["timestamp"].unique())
    split_idx = max(1, int(math.floor(len(unique_timestamps) * (1 - validation_ratio))))
    cutoff = unique_timestamps[split_idx - 1]
    train = frame[frame["timestamp"] <= cutoff].copy()
    valid = frame[frame["timestamp"] > cutoff].copy()
    if valid.empty:
        valid = train.tail(min(len(train), 10_000)).copy()
        train = train.iloc[: max(len(train) - len(valid), 1)].copy()
    return train, valid


@dataclass
class TargetMinMaxNormalizer:
    global_min: float
    global_max: float
    building_min: dict[str, float]
    building_max: dict[str, float]

    @classmethod
    def fit(
        cls,
        frame: pd.DataFrame,
        building_col: str = "building_id",
        target_col: str = "meter_reading",
    ) -> "TargetMinMaxNormalizer":
        grouped = frame.groupby(building_col)[target_col].agg(["min", "max"])
        return cls(
            global_min=float(frame[target_col].min()),
            global_max=float(frame[target_col].max()),
            building_min={str(idx): float(row["min"]) for idx, row in grouped.iterrows()},
            building_max={str(idx): float(row["max"]) for idx, row in grouped.iterrows()},
        )

    def _get_min_max(self, building_id: str) -> tuple[float, float]:
        low = self.building_min.get(str(building_id), self.global_min)
        high = self.building_max.get(str(building_id), self.global_max)
        if high <= low:
            high = low + 1.0
        return low, high

    def transform(self, values: np.ndarray, building_ids: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        building_ids = pd.Series(np.asarray(building_ids, dtype=object)).astype(str)
        mins = building_ids.map(self.building_min).fillna(self.global_min).to_numpy(dtype=np.float32)
        maxs = building_ids.map(self.building_max).fillna(self.global_max).to_numpy(dtype=np.float32)
        denom = np.where(maxs > mins, maxs - mins, 1.0).astype(np.float32)
        return ((values - mins) / denom).astype(np.float32)

    def inverse_transform(self, values: np.ndarray, building_ids: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        building_ids = pd.Series(np.asarray(building_ids, dtype=object)).astype(str)
        mins = building_ids.map(self.building_min).fillna(self.global_min).to_numpy(dtype=np.float32)
        maxs = building_ids.map(self.building_max).fillna(self.global_max).to_numpy(dtype=np.float32)
        denom = np.where(maxs > mins, maxs - mins, 1.0).astype(np.float32)
        return (values * denom + mins).astype(np.float32)

    def inverse_transform_rows(self, values: np.ndarray, building_ids: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        if values.ndim != 2:
            raise ValueError("inverse_transform_rows expects a 2D array.")
        building_ids = pd.Series(np.asarray(building_ids, dtype=object)).astype(str)
        mins = building_ids.map(self.building_min).fillna(self.global_min).to_numpy(dtype=np.float32)
        maxs = building_ids.map(self.building_max).fillna(self.global_max).to_numpy(dtype=np.float32)
        denom = np.where(maxs > mins, maxs - mins, 1.0).astype(np.float32)
        return (values * denom[:, None] + mins[:, None]).astype(np.float32)


@dataclass
class GlobalTargetMinMaxNormalizer:
    global_min: float
    global_max: float

    @classmethod
    def fit(
        cls,
        frame: pd.DataFrame,
        target_col: str = "meter_reading",
    ) -> "GlobalTargetMinMaxNormalizer":
        return cls(
            global_min=float(frame[target_col].min()),
            global_max=float(frame[target_col].max()),
        )

    def _denom(self) -> np.float32:
        span = self.global_max - self.global_min
        return np.float32(span if span > 0 else 1.0)

    def transform(self, values: np.ndarray, building_ids: np.ndarray | None = None) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        return ((values - np.float32(self.global_min)) / self._denom()).astype(np.float32)

    def inverse_transform(self, values: np.ndarray, building_ids: np.ndarray | None = None) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        return (values * self._denom() + np.float32(self.global_min)).astype(np.float32)

    def inverse_transform_rows(
        self,
        values: np.ndarray,
        building_ids: np.ndarray | None = None,
    ) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        if values.ndim != 2:
            raise ValueError("inverse_transform_rows expects a 2D array.")
        return (values * self._denom() + np.float32(self.global_min)).astype(np.float32)


@dataclass
class GlobalTargetLogNormalizer:
    offset: float = 0.0

    @classmethod
    def fit(
        cls,
        frame: pd.DataFrame,
        target_col: str = "meter_reading",
    ) -> "GlobalTargetLogNormalizer":
        min_value = float(frame[target_col].min())
        offset = -min_value if min_value < 0 else 0.0
        return cls(offset=offset)

    def transform(self, values: np.ndarray, building_ids: np.ndarray | None = None) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        shifted = np.clip(values + np.float32(self.offset), a_min=0.0, a_max=None)
        return np.log1p(shifted).astype(np.float32)

    def inverse_transform(self, values: np.ndarray, building_ids: np.ndarray | None = None) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        restored = np.expm1(values) - np.float32(self.offset)
        return restored.astype(np.float32)

    def inverse_transform_rows(
        self,
        values: np.ndarray,
        building_ids: np.ndarray | None = None,
    ) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        if values.ndim != 2:
            raise ValueError("inverse_transform_rows expects a 2D array.")
        restored = np.expm1(values) - np.float32(self.offset)
        return restored.astype(np.float32)


def add_tabular_lag_features(
    frame: pd.DataFrame,
    *,
    group_column: str = "building_id",
    target_column: str = "meter_reading",
    lag_columns: list[str] | None = None,
) -> pd.DataFrame:
    lag_columns = lag_columns or TABULAR_LAG_FEATURE_COLUMNS
    if all(column in frame.columns for column in lag_columns):
        return frame

    augmented = frame.sort_values([group_column, "timestamp"]).reset_index(drop=True).copy()
    grouped = augmented.groupby(group_column, sort=False)[target_column]
    for column in lag_columns:
        if column in augmented.columns:
            continue
        lag_hours = int(column.removeprefix("lag_").removesuffix("h"))
        augmented[column] = grouped.shift(lag_hours).astype(np.float32)
    return augmented


def prepare_tabular_frame(
    frame: pd.DataFrame,
    *,
    feature_columns: list[str] | None = None,
    categorical_columns: list[str] | None = None,
) -> pd.DataFrame:
    feature_columns = feature_columns or TABULAR_FEATURE_COLUMNS
    categorical_columns = categorical_columns or TABULAR_CATEGORICAL_COLUMNS
    tabular = frame[feature_columns].copy()
    for column in categorical_columns:
        tabular[column] = tabular[column].astype("category")
    return tabular


def fit_category_maps(
    frame: pd.DataFrame,
    categorical_columns: list[str] | None = None,
) -> dict[str, dict[str, int]]:
    categorical_columns = categorical_columns or ["building_type", "site_id"]
    mappings: dict[str, dict[str, int]] = {}
    for column in categorical_columns:
        categories = sorted(frame[column].astype(str).fillna("Unknown").unique().tolist())
        mappings[column] = {value: idx for idx, value in enumerate(categories)}
    return mappings


def apply_category_maps(
    frame: pd.DataFrame,
    mappings: dict[str, dict[str, int]],
) -> pd.DataFrame:
    encoded = frame.copy()
    for column, mapping in mappings.items():
        encoded[f"{column}_code"] = (
            encoded[column]
            .astype(str)
            .fillna("Unknown")
            .map(mapping)
            .fillna(-1)
            .astype("int32")
        )
    return encoded


def _fill_numeric_by_group(
    frame: pd.DataFrame,
    group_col: str,
    columns: list[str],
) -> pd.DataFrame:
    filled = frame.copy()
    for column in columns:
        if column not in filled.columns:
            continue
        series = filled[column]
        if not pd.api.types.is_numeric_dtype(series):
            continue
        if not series.isna().any():
            continue
        filled[column] = filled.groupby(group_col, sort=False)[column].transform(
            lambda s: s.interpolate(limit_direction="both")
        )
        if filled[column].isna().any():
            filled[column] = filled.groupby(group_col, sort=False)[column].transform(
                lambda s: s.fillna(s.median())
            )
        if filled[column].isna().any():
            global_fill = float(filled[column].median()) if filled[column].notna().any() else 0.0
            filled[column] = filled[column].fillna(global_fill)
    return filled


def sanitize_sequence_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return _fill_numeric_by_group(
        frame,
        group_col="building_id",
        columns=["meter_reading", *SEQUENCE_DYNAMIC_COLUMNS, *SEQUENCE_STATIC_COLUMNS],
    )


def validate_finite_frame(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    issues: list[str] = []
    for column in columns:
        if column not in frame.columns:
            continue
        series = frame[column]
        nan_count = int(series.isna().sum())
        if nan_count:
            issues.append(f"{column}:nan={nan_count}")
            continue
        if pd.api.types.is_numeric_dtype(series):
            values = series.to_numpy(dtype=np.float32, copy=False)
            inf_count = int(np.isinf(values).sum())
            if inf_count:
                issues.append(f"{column}:inf={inf_count}")
    if issues:
        joined = ", ".join(issues)
        raise RuntimeError(f"{label} contains non-finite values after sanitization: {joined}")


def _sample_indices(num_samples: int, max_samples: int | None, seed: int) -> np.ndarray:
    indices = np.arange(num_samples, dtype=np.int64)
    if max_samples is None or max_samples >= num_samples:
        return indices
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(indices, size=max_samples, replace=False))


class WindowedSequenceDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        normalizer: TargetMinMaxNormalizer | GlobalTargetMinMaxNormalizer | GlobalTargetLogNormalizer,
        category_maps: dict[str, dict[str, int]],
        context_window: int = CONTEXT_WINDOW,
        horizon: int = PREDICTION_HORIZON,
        stride: int = TRAIN_WINDOW_STRIDE,
        max_windows: int | None = None,
        seed: int = RANDOM_SEED,
    ) -> None:
        self.context_window = context_window
        self.horizon = horizon
        self.sample_meta: list[dict[str, object]] = []
        self.inputs: list[np.ndarray] = []
        self.targets: list[np.ndarray] = []

        encoded = apply_category_maps(frame, category_maps)
        encoded = encoded.sort_values(["building_id", "timestamp"]).reset_index(drop=True)
        encoded = sanitize_sequence_frame(encoded)
        validate_finite_frame(
            encoded,
            columns=["meter_reading", *SEQUENCE_DYNAMIC_COLUMNS, *SEQUENCE_STATIC_COLUMNS],
            label="WindowedSequenceDataset input frame",
        )

        for building_id, group in encoded.groupby("building_id", sort=True):
            if len(group) < context_window + horizon:
                continue

            dynamic_values = group[SEQUENCE_DYNAMIC_COLUMNS].to_numpy(dtype=np.float32)
            static_values = (
                group[SEQUENCE_STATIC_COLUMNS]
                .iloc[0]
                .to_numpy(dtype=np.float32)
            )
            targets = group["meter_reading"].to_numpy(dtype=np.float32)
            timestamps = group["timestamp"].to_numpy()
            site_id = str(group["site_id"].iloc[0])
            building_type = str(group["building_type"].iloc[0])

            normalized_targets = normalizer.transform(
                targets,
                np.array([building_id] * len(targets), dtype=object),
            )

            for start in range(0, len(group) - context_window - horizon + 1, stride):
                history_target = normalized_targets[start : start + context_window][:, None]
                history_dynamic = dynamic_values[start : start + context_window]
                history_static = np.repeat(
                    static_values[None, :],
                    repeats=context_window,
                    axis=0,
                )
                x = np.concatenate([history_target, history_dynamic, history_static], axis=1)
                y = normalized_targets[
                    start + context_window : start + context_window + horizon
                ]
                forecast_timestamps = timestamps[
                    start + context_window : start + context_window + horizon
                ]

                self.inputs.append(x.astype(np.float32))
                self.targets.append(y.astype(np.float32))
                self.sample_meta.append(
                    {
                        "building_id": str(building_id),
                        "site_id": site_id,
                        "building_type": building_type,
                        "forecast_timestamps": forecast_timestamps,
                    }
                )

        keep = _sample_indices(len(self.inputs), max_windows, seed)
        self.inputs = [self.inputs[idx] for idx in keep]
        self.targets = [self.targets[idx] for idx in keep]
        self.sample_meta = [self.sample_meta[idx] for idx in keep]

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        return (
            torch.tensor(self.inputs[index], dtype=torch.float32),
            torch.tensor(self.targets[index], dtype=torch.float32),
            index,
        )


def predictions_to_frame(
    dataset: WindowedSequenceDataset,
    predictions: np.ndarray,
    truths: np.ndarray,
    split_name: str,
    model_name: str,
    fold_id: str | None = None,
) -> pd.DataFrame:
    if predictions.shape != truths.shape:
        raise ValueError(
            f"Predictions/truths shape mismatch: {predictions.shape} vs {truths.shape}"
        )
    if predictions.shape[0] != len(dataset.sample_meta):
        raise ValueError(
            f"Prediction rows do not match sample metadata: {predictions.shape[0]} vs {len(dataset.sample_meta)}"
        )

    horizon = predictions.shape[1]
    building_ids = np.repeat([meta["building_id"] for meta in dataset.sample_meta], horizon)
    site_ids = np.repeat([meta["site_id"] for meta in dataset.sample_meta], horizon)
    building_types = np.repeat([meta["building_type"] for meta in dataset.sample_meta], horizon)
    timestamps = np.concatenate(
        [np.asarray(meta["forecast_timestamps"]) for meta in dataset.sample_meta]
    )

    return pd.DataFrame(
        {
            "split_name": split_name,
            "model": model_name,
            "fold_id": fold_id,
            "building_id": building_ids,
            "site_id": site_ids,
            "building_type": building_types,
            "timestamp": pd.to_datetime(timestamps),
            "y_true": truths.reshape(-1).astype(np.float32),
            "y_pred": predictions.reshape(-1).astype(np.float32),
        }
    )


def save_predictions(frame: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    frame.to_csv(path, index=False)


def default_device(requested: str | None = None) -> str:
    if requested:
        return requested
    configured = os.getenv("BUILDING_BDG2_DEVICE")
    if configured:
        return configured
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        gpu_index = int(os.getenv("BUILDING_BDG2_GPU_INDEX", str(DEFAULT_GPU_INDEX)))
        if 0 <= gpu_index < device_count:
            return f"cuda:{gpu_index}"
        return DEFAULT_TORCH_DEVICE if device_count > DEFAULT_GPU_INDEX else "cuda:0"
    return "cpu"


def default_prediction_path(model_name: str, split_name: str) -> Path:
    ensure_phase2_dirs()
    return EXP1_PREDICTIONS_DIR / f"{model_name}_{split_name}.csv"


def default_model_path(model_name: str, split_name: str, suffix: str) -> Path:
    ensure_phase2_dirs()
    split_dir = MODELS_DIR / split_name
    if split_dir.exists() and not os.access(split_dir, os.W_OK):
        split_dir = MODELS_DIR / "_rerun" / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    return split_dir / f"{model_name}.{suffix}"
