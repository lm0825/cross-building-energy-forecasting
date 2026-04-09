from __future__ import annotations

from dataclasses import dataclass, asdict
import json
import logging
from pathlib import Path
import re

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.config import (
    CONTEXT_WINDOW,
    EVAL_WINDOW_STRIDE,
    PREDICTION_HORIZON,
    RANDOM_SEED,
    TABULAR_CATEGORICAL_COLUMNS,
    TABULAR_FEATURE_COLUMNS,
    TABULAR_LAG_FEATURE_COLUMNS,
)
from src.models.common import (
    TargetMinMaxNormalizer,
    GlobalTargetLogNormalizer,
    GlobalTargetMinMaxNormalizer,
    prepare_tabular_frame,
    save_json,
)

LOGGER = logging.getLogger(__name__)


@dataclass
class LightGBMConfig:
    objective: str = "regression_l2"
    learning_rate: float = 0.05
    n_estimators: int = 600
    num_leaves: int = 63
    min_child_samples: int = 100
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.0
    reg_lambda: float = 0.0
    target_transform: str = "log1p"
    random_state: int = RANDOM_SEED
    n_jobs: int = -1
    use_lag_features: bool = False
    lag_feature_columns: list[str] | None = None
    lag_feature_steps: dict[str, int] | None = None


SUMMARY_FEATURE_PATTERN = re.compile(r"^hist_(mean|std|min|max)_(\d+)([hd])$")


def _ordered_time_folds(
    frame: pd.DataFrame,
    n_splits: int = 5,
) -> list[tuple[pd.Series, pd.Series]]:
    timestamps = np.sort(frame["timestamp"].unique())
    fold_size = max(1, len(timestamps) // (n_splits + 1))
    folds: list[tuple[pd.Series, pd.Series]] = []
    for fold_idx in range(n_splits):
        train_end = fold_size * (fold_idx + 1)
        valid_end = min(train_end + fold_size, len(timestamps))
        if valid_end <= train_end:
            break
        train_ts = timestamps[:train_end]
        valid_ts = timestamps[train_end:valid_end]
        train_mask = frame["timestamp"].isin(train_ts)
        valid_mask = frame["timestamp"].isin(valid_ts)
        folds.append((train_mask, valid_mask))
    return folds


def _sample_frame(
    frame: pd.DataFrame,
    max_rows: int | None,
    random_state: int,
) -> pd.DataFrame:
    if max_rows is None or len(frame) <= max_rows:
        return frame.copy()
    return frame.sample(n=max_rows, random_state=random_state).sort_values("timestamp").copy()


class LightGBMExperimentModel:
    def __init__(self, config: LightGBMConfig | None = None) -> None:
        self.config = config or LightGBMConfig()
        self.model: lgb.LGBMRegressor | None = None
        self.normalizer: TargetMinMaxNormalizer | GlobalTargetMinMaxNormalizer | GlobalTargetLogNormalizer | None = None

    def _lgbm_params(self) -> dict[str, object]:
        params = asdict(self.config)
        params.pop("target_transform", None)
        params.pop("use_lag_features", None)
        params.pop("lag_feature_columns", None)
        params.pop("lag_feature_steps", None)
        return params

    def _feature_columns(self) -> list[str]:
        if not self.config.use_lag_features:
            return list(TABULAR_FEATURE_COLUMNS)
        lag_columns = self.config.lag_feature_columns or list(TABULAR_LAG_FEATURE_COLUMNS)
        return list(TABULAR_FEATURE_COLUMNS) + list(lag_columns)

    def _categorical_columns(self) -> list[str]:
        return list(TABULAR_CATEGORICAL_COLUMNS)

    def _resolved_model_name(self) -> str:
        return "lgbm_lag" if self.config.use_lag_features else "lgbm"

    def _target_history_specs(self) -> tuple[dict[str, int], list[tuple[str, str, int]]]:
        if not self.config.use_lag_features:
            return {}, []

        feature_columns = self.config.lag_feature_columns or list(TABULAR_LAG_FEATURE_COLUMNS)
        lag_steps_override = self.config.lag_feature_steps or {}
        lag_steps: dict[str, int] = {}
        summary_specs: list[tuple[str, str, int]] = []

        for column in feature_columns:
            if column in lag_steps_override:
                lag_steps[column] = int(lag_steps_override[column])
                continue

            summary_match = SUMMARY_FEATURE_PATTERN.match(column)
            if summary_match is not None:
                stat = str(summary_match.group(1))
                window = int(summary_match.group(2))
                unit = str(summary_match.group(3))
                step = window * 24 if unit == "d" else window
                summary_specs.append((column, stat, step))
                continue

            lag_match = re.match(r"^lag_(\d+)([hd])$", column)
            if lag_match is not None:
                lag = int(lag_match.group(1))
                unit = str(lag_match.group(2))
                lag_steps[column] = lag * 24 if unit == "d" else lag
                continue

            raise ValueError(
                f"Unsupported target-derived feature column for rollout prediction: {column}"
            )

        return lag_steps, summary_specs

    def _build_normalizer(
        self,
        train_frame: pd.DataFrame,
    ) -> GlobalTargetMinMaxNormalizer | GlobalTargetLogNormalizer:
        if self.config.target_transform == "minmax":
            return GlobalTargetMinMaxNormalizer.fit(train_frame)
        if self.config.target_transform == "log1p":
            return GlobalTargetLogNormalizer.fit(train_frame)
        raise ValueError(f"Unsupported LightGBM target_transform: {self.config.target_transform}")

    def tune_on_tsplit(
        self,
        train_frame: pd.DataFrame,
        max_rows: int = 500_000,
    ) -> LightGBMConfig:
        LOGGER.info(
            "LightGBM tuning start rows=%s sampled_max_rows=%s n_jobs=%s",
            len(train_frame),
            max_rows,
            self.config.n_jobs,
        )
        sampled = _sample_frame(train_frame, max_rows=max_rows, random_state=self.config.random_state)
        folds = _ordered_time_folds(sampled, n_splits=5)
        grid = [
            {"learning_rate": 0.05, "num_leaves": 63, "min_child_samples": 100},
            {"learning_rate": 0.05, "num_leaves": 127, "min_child_samples": 200},
            {"learning_rate": 0.03, "num_leaves": 63, "min_child_samples": 200},
        ]

        best_params = asdict(self.config)
        best_score = np.inf

        for params in grid:
            fold_scores: list[float] = []
            for train_mask, valid_mask in folds:
                fold_train = sampled.loc[train_mask].copy()
                fold_valid = sampled.loc[valid_mask].copy()

                normalizer = self._build_normalizer(fold_train)
                y_train = normalizer.transform(
                    fold_train["meter_reading"].to_numpy(dtype=np.float32),
                )
                y_valid = fold_valid["meter_reading"].to_numpy(dtype=np.float32)

                model_params = (best_params | params).copy()
                model_params.pop("target_transform", None)
                model_params.pop("use_lag_features", None)
                model_params.pop("lag_feature_columns", None)
                model_params.pop("lag_feature_steps", None)
                model = lgb.LGBMRegressor(**model_params)
                model.fit(
                    prepare_tabular_frame(
                        fold_train,
                        feature_columns=self._feature_columns(),
                        categorical_columns=self._categorical_columns(),
                    ),
                    y_train,
                    categorical_feature=self._categorical_columns(),
                )
                pred_norm = model.predict(
                    prepare_tabular_frame(
                        fold_valid,
                        feature_columns=self._feature_columns(),
                        categorical_columns=self._categorical_columns(),
                    )
                )
                pred = normalizer.inverse_transform(
                    pred_norm.astype(np.float32),
                )
                rmse = float(np.sqrt(np.mean((pred - y_valid) ** 2)))
                fold_scores.append(rmse)

            score = float(np.mean(fold_scores))
            LOGGER.info("LightGBM tuning params=%s mean_rmse=%.6f", params, score)
            if score < best_score:
                best_score = score
                best_params = best_params | params

        self.config = LightGBMConfig(**best_params)
        LOGGER.info("LightGBM tuning best_params=%s best_rmse=%.6f", asdict(self.config), best_score)
        return self.config

    def fit(self, train_frame: pd.DataFrame) -> "LightGBMExperimentModel":
        LOGGER.info(
            "LightGBM fit start rows=%s buildings=%s n_jobs=%s",
            len(train_frame),
            train_frame["building_id"].nunique(),
            self.config.n_jobs,
        )
        self.normalizer = self._build_normalizer(train_frame)
        y_train = self.normalizer.transform(
            train_frame["meter_reading"].to_numpy(dtype=np.float32),
        )
        self.model = lgb.LGBMRegressor(**self._lgbm_params())
        self.model.fit(
            prepare_tabular_frame(
                train_frame,
                feature_columns=self._feature_columns(),
                categorical_columns=self._categorical_columns(),
            ),
            y_train,
            categorical_feature=self._categorical_columns(),
        )
        LOGGER.info("LightGBM fit finished estimators=%s", self.config.n_estimators)
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self.model is None or self.normalizer is None:
            raise RuntimeError("Model must be fitted before prediction.")
        pred_norm = self.model.predict(
            prepare_tabular_frame(
                frame,
                feature_columns=self._feature_columns(),
                categorical_columns=self._categorical_columns(),
            )
        ).astype(np.float32)
        if isinstance(self.normalizer, TargetMinMaxNormalizer):
            pred = self.normalizer.inverse_transform(
                pred_norm,
                frame["building_id"].astype(str).to_numpy(dtype=object),
            )
        else:
            pred = self.normalizer.inverse_transform(pred_norm)
        if not np.isfinite(pred).all():
            raise RuntimeError("LightGBM prediction contains non-finite values.")
        return pred

    def predict_frame(
        self,
        frame: pd.DataFrame,
        *,
        split_name: str,
        fold_id: str | None = None,
        model_name: str | None = None,
        context_window: int = CONTEXT_WINDOW,
        horizon: int = PREDICTION_HORIZON,
        stride: int = EVAL_WINDOW_STRIDE,
    ) -> pd.DataFrame:
        if self.model is None or self.normalizer is None:
            raise RuntimeError("Model must be fitted before prediction.")
        if horizon <= 0 or stride <= 0 or context_window < 0:
            raise ValueError("context_window, horizon, and stride must be non-negative with horizon/stride > 0.")

        resolved_model_name = model_name or self._resolved_model_name()
        lag_steps, summary_specs = self._target_history_specs()
        max_target_lookback = max(
            [context_window, *lag_steps.values(), *(window for _, _, window in summary_specs)],
            default=context_window,
        )

        ordered = frame.sort_values(["building_id", "timestamp"]).reset_index(drop=True)
        per_building_frames: list[pd.DataFrame] = []

        for building_id, group in ordered.groupby("building_id", sort=True):
            building_frame = group.reset_index(drop=True)
            n_rows = len(building_frame)
            if n_rows < max_target_lookback + horizon:
                continue

            origins = np.arange(max_target_lookback, n_rows - horizon + 1, stride, dtype=np.int32)
            if origins.size == 0:
                continue

            future_positions = origins[:, None] + np.arange(horizon, dtype=np.int32)[None, :]
            y_true = building_frame["meter_reading"].to_numpy(dtype=np.float32)[future_positions]

            if not lag_steps and not summary_specs:
                future_rows = building_frame.iloc[future_positions.reshape(-1)].copy()
                y_pred = self.predict(future_rows).reshape(-1, horizon).astype(np.float32)
            else:
                history_length = int(max_target_lookback)
                target_values = building_frame["meter_reading"].to_numpy(dtype=np.float32)
                history_windows = np.lib.stride_tricks.sliding_window_view(target_values, history_length)[
                    origins - history_length
                ].astype(np.float32, copy=True)
                state = np.full((len(origins), history_length + horizon), np.nan, dtype=np.float32)
                state[:, :history_length] = history_windows
                y_pred = np.empty((len(origins), horizon), dtype=np.float32)

                for step_idx in range(horizon):
                    rows = building_frame.iloc[future_positions[:, step_idx]].copy()

                    for column, lag_hours in lag_steps.items():
                        if lag_hours > step_idx:
                            values = state[:, history_length - lag_hours + step_idx]
                        else:
                            values = y_pred[:, step_idx - lag_hours]
                        rows[column] = values.astype(np.float32)

                    for column, stat, window_hours in summary_specs:
                        end_idx = history_length + step_idx
                        window_values = state[:, end_idx - window_hours : end_idx]
                        if stat == "mean":
                            derived = np.mean(window_values, axis=1)
                        elif stat == "std":
                            derived = np.std(window_values, axis=1, ddof=0)
                        elif stat == "min":
                            derived = np.min(window_values, axis=1)
                        elif stat == "max":
                            derived = np.max(window_values, axis=1)
                        else:
                            raise ValueError(f"Unsupported summary stat: {stat}")
                        rows[column] = derived.astype(np.float32)

                    step_pred = self.predict(rows).astype(np.float32)
                    y_pred[:, step_idx] = step_pred
                    state[:, history_length + step_idx] = step_pred

            flat_rows = building_frame.iloc[future_positions.reshape(-1)].copy()
            flat_rows["split_name"] = split_name
            flat_rows["model"] = resolved_model_name
            flat_rows["fold_id"] = fold_id
            flat_rows["y_true"] = y_true.reshape(-1).astype(np.float32)
            flat_rows["y_pred"] = y_pred.reshape(-1).astype(np.float32)
            per_building_frames.append(
                flat_rows[
                    [
                        "split_name",
                        "model",
                        "fold_id",
                        "building_id",
                        "site_id",
                        "building_type",
                        "timestamp",
                        "y_true",
                        "y_pred",
                    ]
                ].copy()
            )

        if not per_building_frames:
            raise RuntimeError(
                f"No forecast windows generated for LightGBM split={split_name} model={resolved_model_name}."
            )

        return pd.concat(per_building_frames, ignore_index=True)

    def save(self, path: str | Path) -> None:
        if self.model is None or self.normalizer is None:
            raise RuntimeError("Model must be fitted before saving.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.booster_.save_model(str(path))
        save_json(asdict(self.config), path.with_suffix(".config.json"))
        save_json(asdict(self.normalizer), path.with_suffix(".normalizer.json"))

    @classmethod
    def load(cls, path: str | Path) -> "LightGBMExperimentModel":
        path = Path(path)
        config_payload = json.loads(path.with_suffix(".config.json").read_text(encoding="utf-8"))
        normalizer_payload = json.loads(path.with_suffix(".normalizer.json").read_text(encoding="utf-8"))
        if "target_transform" not in config_payload:
            if {"global_min", "global_max"}.issubset(normalizer_payload):
                config_payload["target_transform"] = "minmax"
            else:
                config_payload["target_transform"] = "log1p"
        config = LightGBMConfig(**(asdict(LightGBMConfig()) | config_payload))
        if config.target_transform == "minmax":
            if {"building_min", "building_max"}.issubset(normalizer_payload):
                normalizer = TargetMinMaxNormalizer(**normalizer_payload)
            else:
                normalizer = GlobalTargetMinMaxNormalizer(**normalizer_payload)
        elif config.target_transform == "log1p":
            normalizer = GlobalTargetLogNormalizer(**normalizer_payload)
        else:
            raise ValueError(f"Unsupported LightGBM target_transform: {config.target_transform}")

        loaded = cls(config=config)
        loaded.model = lgb.Booster(model_file=str(path))
        loaded.normalizer = normalizer
        return loaded
