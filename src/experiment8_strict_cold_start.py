from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import json
import logging
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (
    EVAL_WINDOW_STRIDE,
    FEATURES_BDG2_PATH,
    FIGURES_DIR,
    LOGS_DIR,
    MODELS_DIR,
    PREDICTION_HORIZON,
    SEQUENCE_DYNAMIC_COLUMNS,
    SEQUENCE_STATIC_COLUMNS,
    TABLES_DIR,
    TABULAR_CATEGORICAL_COLUMNS,
    TABULAR_FEATURE_COLUMNS,
    ensure_phase2_dirs,
)
from src.data_splitting import B_SPLIT_PATH, load_pickle, save_split_artifacts, unpack_mask
from src.models.common import (
    add_tabular_lag_features,
    apply_category_maps,
    fit_category_maps,
    prepare_tabular_frame,
    sanitize_sequence_frame,
    set_seed,
    validate_finite_frame,
)
from src.models.lgbm_model import LightGBMConfig, LightGBMExperimentModel
from src.models.lstm_model import LSTMConfig, LSTMExperimentModel
from src.models.patchtst_model import PatchTSTConfig, PatchTSTExperimentModel
from src.runtime import apply_runtime_limits, format_bytes

LOGGER = logging.getLogger(__name__)

SPLIT_NAME = "b_split_scs"
MODEL_ORDER = ["lgbm", "lgbm_lag", "lstm", "patchtst"]
SEED_DEFAULT = [7, 42, 123]
BUDGET_DAYS_DEFAULT = [0, 1, 3, 7, 14, 30]
LAG_COLUMNS = ["lag_1h", "lag_24h", "lag_168h"]


def _path(stem: str, suffix: str | None = None) -> Path:
    suffix = suffix or ""
    if suffix and not suffix.startswith("_"):
        suffix = f"_{suffix}"
    return TABLES_DIR / f"{stem}{suffix}.csv"


def _figure_path(stem: str, suffix: str | None = None) -> Path:
    suffix = suffix or ""
    if suffix and not suffix.startswith("_"):
        suffix = f"_{suffix}"
    return FIGURES_DIR / f"{stem}{suffix}.png"


def configure_logging(log_file: str | Path | None = None) -> Path | None:
    ensure_phase2_dirs()
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    resolved_path: Path | None = None
    if log_file is not None:
        resolved_path = Path(log_file)
        if not resolved_path.is_absolute():
            resolved_path = ROOT / resolved_path
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(resolved_path, encoding="utf-8"))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )
    return resolved_path


def _load_tuned_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_tuned_json(
    model_name: str,
    *,
    split_name: str | None = None,
    fold_id: str | None = None,
) -> dict[str, object]:
    tuning_dir = MODELS_DIR / "_tuning"
    candidates: list[Path] = []
    if split_name is not None and fold_id is not None:
        candidates.append(tuning_dir / f"{model_name}.{split_name}.{fold_id}.config.json")
    if split_name is not None:
        candidates.append(tuning_dir / f"{model_name}.{split_name}.config.json")
    candidates.append(tuning_dir / f"{model_name}.config.json")
    for candidate in candidates:
        if candidate.exists():
            return _load_tuned_json(candidate)
    return _load_tuned_json(candidates[-1])


def _load_feature_frame() -> pd.DataFrame:
    columns = list(
        dict.fromkeys(
            ["building_id", "site_id", "building_type", "timestamp", "meter_reading"]
            + list(TABULAR_FEATURE_COLUMNS)
        )
    )
    frame = pd.read_parquet(FEATURES_BDG2_PATH, columns=columns)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame = frame.sort_values(["building_id", "timestamp"]).reset_index(drop=True)
    frame = add_tabular_lag_features(frame, lag_columns=LAG_COLUMNS)
    LOGGER.info(
        "Loaded BDG2 feature frame rows=%s buildings=%s sites=%s",
        len(frame),
        frame["building_id"].nunique(),
        frame["site_id"].nunique(),
    )
    return frame


def _select_rows(frame: pd.DataFrame, packed_mask: dict[str, object]) -> pd.DataFrame:
    return frame.loc[unpack_mask(packed_mask)].reset_index(drop=True)


def _tabular_config(
    model_name: str,
    *,
    seed: int,
    cpu_threads: int | None,
    split_name: str,
) -> LightGBMConfig:
    payload = _resolve_tuned_json("lgbm_lag" if model_name == "lgbm_lag" else "lgbm", split_name=split_name)
    defaults = asdict(LightGBMConfig())
    config = LightGBMConfig(**(defaults | payload))
    config.random_state = int(seed)
    config.n_jobs = cpu_threads if cpu_threads is not None else config.n_jobs
    config.target_transform = "log1p"
    config.use_lag_features = model_name == "lgbm_lag"
    config.lag_feature_columns = list(LAG_COLUMNS) if model_name == "lgbm_lag" else None
    config.lag_feature_steps = {column: int(column.removeprefix("lag_").removesuffix("h")) for column in LAG_COLUMNS}
    return config


def _lstm_config(*, seed: int, split_name: str) -> LSTMConfig:
    payload = _resolve_tuned_json("lstm", split_name=split_name)
    defaults = asdict(LSTMConfig())
    config = LSTMConfig(**(defaults | payload))
    config.random_seed = int(seed)
    return config


def _patchtst_config(*, seed: int, split_name: str) -> PatchTSTConfig:
    payload = _resolve_tuned_json("patchtst", split_name=split_name)
    defaults = asdict(PatchTSTConfig())
    config = PatchTSTConfig(**(defaults | payload))
    config.random_seed = int(seed)
    config.patch_len = min(config.patch_len, config.context_window)
    config.patch_stride = min(config.patch_stride, config.patch_len)
    return config


def _prepare_tabular_cold_start_frame(
    test_frame: pd.DataFrame,
    *,
    budget_hours: int,
    fill_value: float,
) -> pd.DataFrame:
    if budget_hours < 0:
        raise ValueError("budget_hours must be non-negative.")

    prepared = test_frame.sort_values(["building_id", "timestamp"]).reset_index(drop=True).copy()
    first_ts = prepared.groupby("building_id", sort=False)["timestamp"].transform("min")
    elapsed_hours = (
        (prepared["timestamp"] - first_ts).dt.total_seconds().div(3600.0).round().astype("int32")
    )
    prepared["elapsed_hours"] = elapsed_hours

    lag_hours_map = {column: int(column.removeprefix("lag_").removesuffix("h")) for column in LAG_COLUMNS}
    for column, lag_hours in lag_hours_map.items():
        values = prepared[column].astype(np.float32).copy()
        unavailable = prepared["elapsed_hours"] < lag_hours
        over_budget = budget_hours < lag_hours
        if over_budget:
            values[:] = np.float32(fill_value)
        else:
            values.loc[unavailable] = np.float32(fill_value)
            values = values.fillna(np.float32(fill_value))
        prepared[column] = values.astype(np.float32)

    return prepared.drop(columns=["elapsed_hours"])


def _predict_tabular_budget(
    model_name: str,
    model: LightGBMExperimentModel,
    test_frame: pd.DataFrame,
    *,
    budget_hours: int,
    fill_value: float,
    seed: int,
) -> pd.DataFrame:
    if model_name == "lgbm":
        result = model.predict_frame(
            test_frame,
            split_name=SPLIT_NAME,
            model_name=model_name,
            context_window=0,
            horizon=PREDICTION_HORIZON,
            stride=EVAL_WINDOW_STRIDE,
        ).copy()
        result["seed"] = int(seed)
        result["budget_hours"] = int(budget_hours)
        result["budget_days"] = float(budget_hours) / 24.0
        return result

    lag_hours_map = {column: int(column.removeprefix("lag_").removesuffix("h")) for column in LAG_COLUMNS}
    horizon = int(PREDICTION_HORIZON)
    stride = int(EVAL_WINDOW_STRIDE)
    frames: list[pd.DataFrame] = []

    ordered = test_frame.sort_values(["building_id", "timestamp"]).reset_index(drop=True)
    for _, group in ordered.groupby("building_id", sort=True):
        building_frame = group.reset_index(drop=True)
        n_rows = len(building_frame)
        if n_rows < horizon:
            continue

        origins = np.arange(0, n_rows - horizon + 1, stride, dtype=np.int32)
        if origins.size == 0:
            continue

        future_positions = origins[:, None] + np.arange(horizon, dtype=np.int32)[None, :]
        targets = building_frame["meter_reading"].to_numpy(dtype=np.float32)
        truths = targets[future_positions]
        preds = np.empty((len(origins), horizon), dtype=np.float32)

        for step_idx in range(horizon):
            rows = building_frame.iloc[future_positions[:, step_idx]].copy()
            current_positions = origins + step_idx
            oldest_allowed = origins - int(budget_hours)

            for column, lag_hours in lag_hours_map.items():
                lag_positions = current_positions - lag_hours
                values = np.full(len(origins), np.float32(fill_value), dtype=np.float32)

                history_mask = lag_positions < origins
                valid_history = history_mask & (lag_positions >= 0) & (lag_positions >= oldest_allowed)
                if np.any(valid_history):
                    values[valid_history] = targets[lag_positions[valid_history]]

                prediction_mask = ~history_mask
                if np.any(prediction_mask):
                    pred_offsets = lag_positions[prediction_mask] - origins[prediction_mask]
                    values[prediction_mask] = preds[prediction_mask, pred_offsets]

                rows[column] = values.astype(np.float32)

            preds[:, step_idx] = model.predict(rows).astype(np.float32)

        flat_rows = building_frame.iloc[future_positions.reshape(-1)].copy()
        flat_rows["split_name"] = SPLIT_NAME
        flat_rows["model"] = model_name
        flat_rows["seed"] = int(seed)
        flat_rows["budget_hours"] = int(budget_hours)
        flat_rows["budget_days"] = float(budget_hours) / 24.0
        flat_rows["y_true"] = truths.reshape(-1).astype(np.float32)
        flat_rows["y_pred"] = preds.reshape(-1).astype(np.float32)
        frames.append(
            flat_rows[
                [
                    "split_name",
                    "model",
                    "seed",
                    "budget_hours",
                    "budget_days",
                    "building_id",
                    "site_id",
                    "building_type",
                    "timestamp",
                    "y_true",
                    "y_pred",
                ]
            ].copy()
        )

    if not frames:
        raise RuntimeError(f"No strict cold-start tabular windows generated for model={model_name}.")
    return pd.concat(frames, ignore_index=True)


def _sequence_prediction_frame(
    predictions: np.ndarray,
    truths: np.ndarray,
    sample_meta: list[dict[str, object]],
    *,
    model_name: str,
    seed: int,
    budget_hours: int,
) -> pd.DataFrame:
    horizon = predictions.shape[1]
    building_ids = np.repeat([meta["building_id"] for meta in sample_meta], horizon)
    site_ids = np.repeat([meta["site_id"] for meta in sample_meta], horizon)
    building_types = np.repeat([meta["building_type"] for meta in sample_meta], horizon)
    timestamps = np.concatenate([np.asarray(meta["forecast_timestamps"]) for meta in sample_meta])

    return pd.DataFrame(
        {
            "split_name": SPLIT_NAME,
            "model": model_name,
            "seed": int(seed),
            "budget_hours": int(budget_hours),
            "budget_days": float(budget_hours) / 24.0,
            "building_id": building_ids,
            "site_id": site_ids,
            "building_type": building_types,
            "timestamp": pd.to_datetime(timestamps),
            "y_true": truths.reshape(-1).astype(np.float32),
            "y_pred": predictions.reshape(-1).astype(np.float32),
        }
    )


def _predict_sequence_budget(
    model_name: str,
    model: LSTMExperimentModel | PatchTSTExperimentModel,
    test_frame: pd.DataFrame,
    *,
    budget_hours: int,
    fill_value: float,
    seed: int,
) -> pd.DataFrame:
    if model.model is None or model.normalizer is None or model.category_maps is None:
        raise RuntimeError("Sequence model must be fitted before cold-start prediction.")

    encoded = apply_category_maps(test_frame, model.category_maps)
    encoded = encoded.sort_values(["building_id", "timestamp"]).reset_index(drop=True)
    encoded = sanitize_sequence_frame(encoded)
    validate_finite_frame(
        encoded,
        columns=["meter_reading", *SEQUENCE_DYNAMIC_COLUMNS, *SEQUENCE_STATIC_COLUMNS],
        label=f"{model_name} strict cold-start frame",
    )

    fill_norm = float(model.normalizer.transform(np.asarray([fill_value], dtype=np.float32))[0])
    inputs: list[np.ndarray] = []
    truths_norm: list[np.ndarray] = []
    sample_meta: list[dict[str, object]] = []

    context_window = int(model.config.context_window)
    horizon = int(model.config.prediction_horizon)
    stride = int(model.config.eval_stride)

    for building_id, group in encoded.groupby("building_id", sort=True):
        group = group.sort_values("timestamp").reset_index(drop=True)
        if len(group) < horizon:
            continue

        dynamic_values = group[SEQUENCE_DYNAMIC_COLUMNS].to_numpy(dtype=np.float32)
        static_values = group[SEQUENCE_STATIC_COLUMNS].iloc[0].to_numpy(dtype=np.float32)
        targets = group["meter_reading"].to_numpy(dtype=np.float32)
        timestamps = group["timestamp"].to_numpy()
        site_id = str(group["site_id"].iloc[0])
        building_type = str(group["building_type"].iloc[0])
        normalized_targets = model.normalizer.transform(targets)

        for start in range(0, len(group) - horizon + 1, stride):
            x = np.empty(
                (context_window, 1 + len(SEQUENCE_DYNAMIC_COLUMNS) + len(SEQUENCE_STATIC_COLUMNS)),
                dtype=np.float32,
            )
            oldest_allowed_idx = start - budget_hours
            for pos, hist_idx in enumerate(range(start - context_window, start)):
                if hist_idx < 0:
                    hist_dynamic = dynamic_values[0]
                    hist_target = fill_norm
                else:
                    hist_dynamic = dynamic_values[hist_idx]
                    if hist_idx < oldest_allowed_idx:
                        hist_target = fill_norm
                    else:
                        hist_target = float(normalized_targets[hist_idx])
                x[pos, 0] = hist_target
                x[pos, 1 : 1 + len(SEQUENCE_DYNAMIC_COLUMNS)] = hist_dynamic
                x[pos, 1 + len(SEQUENCE_DYNAMIC_COLUMNS) :] = static_values

            truth = normalized_targets[start : start + horizon].astype(np.float32)
            forecast_timestamps = timestamps[start : start + horizon]
            inputs.append(x)
            truths_norm.append(truth)
            sample_meta.append(
                {
                    "building_id": str(building_id),
                    "site_id": site_id,
                    "building_type": building_type,
                    "forecast_timestamps": forecast_timestamps,
                }
            )

    if not inputs:
        raise RuntimeError(f"No strict cold-start windows generated for model={model_name}.")

    x_array = np.stack(inputs).astype(np.float32)
    truth_array = np.stack(truths_norm).astype(np.float32)
    preds: list[np.ndarray] = []
    model.model.eval()
    with torch.no_grad():
        batch_size = int(model.config.batch_size)
        for start in range(0, len(x_array), batch_size):
            x_batch = torch.tensor(x_array[start : start + batch_size], dtype=torch.float32, device=model.device)
            pred = model.model(x_batch).detach().cpu().numpy()
            preds.append(pred.astype(np.float32))
    pred_norm = np.vstack(preds)
    building_ids = np.array([meta["building_id"] for meta in sample_meta], dtype=object)
    pred_denorm = model.normalizer.inverse_transform_rows(pred_norm, building_ids)
    truth_denorm = model.normalizer.inverse_transform_rows(truth_array, building_ids)
    return _sequence_prediction_frame(
        pred_denorm,
        truth_denorm,
        sample_meta,
        model_name=model_name,
        seed=seed,
        budget_hours=budget_hours,
    )


def _per_building_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    errors = predictions["y_pred"].to_numpy(dtype=np.float64) - predictions["y_true"].to_numpy(dtype=np.float64)
    working = predictions[["split_name", "model", "seed", "budget_hours", "budget_days", "building_id", "site_id", "building_type"]].copy()
    working["sum_true"] = predictions["y_true"].to_numpy(dtype=np.float64)
    working["sum_sq_err"] = errors ** 2
    working["n_rows"] = 1
    aggregated = (
        working.groupby(
            ["split_name", "model", "seed", "budget_hours", "budget_days", "building_id", "site_id", "building_type"],
            as_index=False,
            sort=False,
        )
        .agg(sum_true=("sum_true", "sum"), sum_sq_err=("sum_sq_err", "sum"), n_rows=("n_rows", "sum"))
    )
    aggregated["mean_true"] = aggregated["sum_true"] / aggregated["n_rows"]
    aggregated["rmse"] = np.sqrt(aggregated["sum_sq_err"] / aggregated["n_rows"])
    aggregated["cv_rmse"] = aggregated["rmse"] / aggregated["mean_true"]
    return aggregated


def _pooled_metrics(predictions: pd.DataFrame, per_building: pd.DataFrame) -> dict[str, object]:
    errors = predictions["y_pred"].to_numpy(dtype=np.float64) - predictions["y_true"].to_numpy(dtype=np.float64)
    y_true = predictions["y_true"].to_numpy(dtype=np.float64)
    mean_true = float(y_true.mean())
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    cv_rmse = float(rmse / mean_true) if mean_true else np.nan
    return {
        "split_name": SPLIT_NAME,
        "model": str(predictions["model"].iloc[0]),
        "seed": int(predictions["seed"].iloc[0]),
        "budget_hours": int(predictions["budget_hours"].iloc[0]),
        "budget_days": float(predictions["budget_days"].iloc[0]),
        "pooled_cv_rmse": cv_rmse,
        "pooled_rmse": rmse,
        "mean_per_building_cv_rmse": float(per_building["cv_rmse"].mean()),
        "median_per_building_cv_rmse": float(per_building["cv_rmse"].median()),
        "p25_per_building_cv_rmse": float(per_building["cv_rmse"].quantile(0.25)),
        "p75_per_building_cv_rmse": float(per_building["cv_rmse"].quantile(0.75)),
        "n_buildings": int(per_building["building_id"].nunique()),
        "n_rows": int(len(predictions)),
    }


def _summarize(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    summary = (
        seed_metrics.groupby(["model", "budget_hours", "budget_days"], as_index=False)
        .agg(
            n_seeds=("seed", "nunique"),
            pooled_cv_rmse_mean=("pooled_cv_rmse", "mean"),
            pooled_cv_rmse_sd=("pooled_cv_rmse", "std"),
            median_per_building_cv_rmse_mean=("median_per_building_cv_rmse", "mean"),
            median_per_building_cv_rmse_sd=("median_per_building_cv_rmse", "std"),
        )
    )
    model_order = {name: idx for idx, name in enumerate(MODEL_ORDER)}
    summary = summary.sort_values(
        ["model", "budget_hours"],
        key=lambda s: s.map(model_order).fillna(s) if s.name == "model" else s,
    ).reset_index(drop=True)
    return summary


def _plot_recovery_curve(summary: pd.DataFrame, *, output_suffix: str | None) -> Path:
    path = _figure_path("strict_cold_start_recovery_curve", output_suffix)
    model_labels = {
        "lgbm": "LightGBM",
        "lgbm_lag": "LightGBM+lag",
        "lstm": "LSTM",
        "patchtst": "PatchTST",
    }
    colors = {
        "lgbm": "#2f6690",
        "lgbm_lag": "#5f8f3b",
        "lstm": "#81b29a",
        "patchtst": "#d1495b",
    }
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)
    metrics = [
        ("pooled_cv_rmse_mean", "pooled_cv_rmse_sd", "Pooled CV(RMSE)"),
        ("median_per_building_cv_rmse_mean", "median_per_building_cv_rmse_sd", "Median per-building CV(RMSE)"),
    ]
    for ax, (mean_col, sd_col, ylabel) in zip(axes, metrics, strict=False):
        for model_name in MODEL_ORDER:
            group = summary[summary["model"] == model_name].sort_values("budget_days")
            if group.empty:
                continue
            x = group["budget_days"].to_numpy(dtype=np.float64)
            y = group[mean_col].to_numpy(dtype=np.float64)
            sd = group[sd_col].fillna(0.0).to_numpy(dtype=np.float64)
            ax.plot(x, y, marker="o", linewidth=2, label=model_labels[model_name], color=colors[model_name])
            ax.fill_between(x, y - sd, y + sd, alpha=0.15, color=colors[model_name])
        ax.set_xlabel("Observed history days")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)
    return path


def run_strict_cold_start(
    *,
    models: list[str],
    budget_days: list[int],
    seeds: list[int],
    device: str | None,
    cpu_threads: int | None,
    output_suffix: str | None,
) -> dict[str, Path]:
    ensure_phase2_dirs()
    set_seed()
    if not B_SPLIT_PATH.exists():
        save_split_artifacts()
    split_artifact = load_pickle(B_SPLIT_PATH)
    frame = _load_feature_frame()
    train_frame = _select_rows(frame, split_artifact["train_mask"])
    test_frame = _select_rows(frame, split_artifact["test_mask"])
    fill_value = float(train_frame["meter_reading"].mean())
    LOGGER.info(
        "Strict cold-start setup train_rows=%s test_rows=%s test_buildings=%s fill_value=%.6f budgets_days=%s seeds=%s",
        len(train_frame),
        len(test_frame),
        test_frame["building_id"].nunique(),
        fill_value,
        budget_days,
        seeds,
    )

    seed_rows: list[dict[str, object]] = []
    per_building_rows: list[pd.DataFrame] = []

    for seed in seeds:
        for model_name in models:
            started = time.time()
            LOGGER.info("Fitting strict cold-start base model=%s seed=%s", model_name, seed)
            if model_name in {"lgbm", "lgbm_lag"}:
                fitted_model: object = LightGBMExperimentModel(
                    config=_tabular_config(model_name, seed=seed, cpu_threads=cpu_threads, split_name="b_split")
                ).fit(train_frame)
            elif model_name == "lstm":
                fitted_model = LSTMExperimentModel(
                    config=_lstm_config(seed=seed, split_name="b_split"),
                    device=device,
                ).fit(train_frame)
            elif model_name == "patchtst":
                fitted_model = PatchTSTExperimentModel(
                    config=_patchtst_config(seed=seed, split_name="b_split"),
                    device=device,
                ).fit(train_frame)
            else:
                raise ValueError(f"Unsupported model: {model_name}")

            for budget_day in budget_days:
                budget_hours = int(budget_day) * 24
                LOGGER.info(
                    "Running strict cold-start model=%s seed=%s budget_days=%s",
                    model_name,
                    seed,
                    budget_day,
                )
                if model_name in {"lgbm", "lgbm_lag"}:
                    predictions = _predict_tabular_budget(
                        model_name,
                        fitted_model,
                        test_frame,
                        budget_hours=budget_hours,
                        fill_value=fill_value,
                        seed=seed,
                    )
                else:
                    predictions = _predict_sequence_budget(
                        model_name,
                        fitted_model,
                        test_frame,
                        budget_hours=budget_hours,
                        fill_value=fill_value,
                        seed=seed,
                    )
                per_building = _per_building_metrics(predictions)
                pooled = _pooled_metrics(predictions, per_building)
                seed_rows.append(pooled)
                per_building_rows.append(per_building)
                LOGGER.info(
                    "Finished strict cold-start model=%s seed=%s budget_days=%s pooled_cv_rmse=%.6f median_per_building=%.6f",
                    model_name,
                    seed,
                    budget_day,
                    pooled["pooled_cv_rmse"],
                    pooled["median_per_building_cv_rmse"],
                )

            LOGGER.info(
                "Completed fitted model=%s seed=%s total_elapsed=%.1fs",
                model_name,
                seed,
                time.time() - started,
            )
            del fitted_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    seed_metrics = pd.DataFrame(seed_rows)
    per_building = pd.concat(per_building_rows, ignore_index=True)
    summary = _summarize(seed_metrics)

    seed_path = _path("strict_cold_start_seed_metrics", output_suffix)
    per_building_path = _path("strict_cold_start_per_building_metrics", output_suffix)
    summary_path = _path("strict_cold_start_summary", output_suffix)
    fig_path = _plot_recovery_curve(summary, output_suffix=output_suffix)

    seed_metrics.to_csv(seed_path, index=False)
    per_building.to_csv(per_building_path, index=False)
    summary.to_csv(summary_path, index=False)
    LOGGER.info("Saved strict cold-start seed metrics to %s", seed_path)
    LOGGER.info("Saved strict cold-start per-building metrics to %s", per_building_path)
    LOGGER.info("Saved strict cold-start summary to %s", summary_path)
    LOGGER.info("Saved strict cold-start recovery curve to %s", fig_path)

    return {
        "seed_metrics": seed_path,
        "per_building": per_building_path,
        "summary": summary_path,
        "figure": fig_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict cold-start recovery experiment for BDG2 B-split.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODEL_ORDER),
        choices=MODEL_ORDER,
        help="Models to evaluate.",
    )
    parser.add_argument(
        "--budget-days",
        nargs="+",
        type=int,
        default=list(BUDGET_DAYS_DEFAULT),
        help="Observed history budgets in days.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(SEED_DEFAULT),
        help="Model random seeds.",
    )
    parser.add_argument("--device", default=None, help="Torch device, e.g. cuda:0.")
    parser.add_argument("--max-cpu-threads", type=int, default=None)
    parser.add_argument("--cpu-fraction", type=float, default=None)
    parser.add_argument("--memory-fraction", type=float, default=None)
    parser.add_argument("--output-suffix", default=None)
    parser.add_argument("--log-file", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    resolved_log = configure_logging(args.log_file)
    limits = apply_runtime_limits(
        max_cpu_threads=args.max_cpu_threads,
        cpu_fraction=args.cpu_fraction,
        memory_fraction=args.memory_fraction,
    )
    LOGGER.info(
        "Strict cold-start runtime cpu_threads=%s memory_limit=%s applied=%s log=%s",
        limits.cpu_threads,
        format_bytes(limits.memory_bytes),
        limits.memory_limit_applied,
        resolved_log,
    )
    outputs = run_strict_cold_start(
        models=args.models,
        budget_days=sorted(set(args.budget_days)),
        seeds=args.seeds,
        device=args.device,
        cpu_threads=limits.cpu_threads,
        output_suffix=args.output_suffix,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
