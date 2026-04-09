from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import logging
from pathlib import Path
import sys
import time

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.common import default_device, save_predictions, set_seed
from src.models.lgbm_model import LightGBMConfig, LightGBMExperimentModel
from src.models.lstm_model import LSTMConfig, LSTMExperimentModel
from src.models.patchtst_model import PatchTSTConfig, PatchTSTExperimentModel
from src.runtime import apply_runtime_limits, format_bytes

LOGGER = logging.getLogger(__name__)

HEEW_DATA_PATH = ROOT / "data" / "heww" / "heew_daily_energy_weather.csv"
HEEW_RESULTS_DIR = ROOT / "results" / "heew_predictions"
HEEW_TABLES_DIR = ROOT / "tables"
HEEW_FIGURES_DIR = ROOT / "figures" / "paper"
HEEW_LOGS_DIR = ROOT / "logs"

HEEW_FILTER_SUMMARY_PATH = HEEW_TABLES_DIR / "heew_replication_filter_summary.csv"
HEEW_DATASET_SUMMARY_PATH = HEEW_TABLES_DIR / "heew_replication_dataset_summary.csv"
HEEW_METRICS_PATH = HEEW_TABLES_DIR / "heew_replication_metrics.csv"
HEEW_PER_BUILDING_PATH = HEEW_TABLES_DIR / "heew_replication_per_building_metrics.csv"
HEEW_WIN_COUNTS_PATH = HEEW_TABLES_DIR / "heew_replication_win_counts.csv"
HEEW_BSPLIT_SEED_METRICS_PATH = HEEW_TABLES_DIR / "heew_replication_bsplit_seed_metrics.csv"
HEEW_BSPLIT_SEED_SUMMARY_PATH = HEEW_TABLES_DIR / "heew_replication_bsplit_seed_summary.csv"
HEEW_FIGURE_PATH = HEEW_FIGURES_DIR / "paper_fig14_heew_ranking_shift.png"
HEEW_CONFIG_PATH = HEEW_TABLES_DIR / "heew_replication_model_configs.csv"

MODEL_ORDER = ["lgbm", "lgbm_lag", "lstm", "patchtst"]
MODEL_LABELS = {
    "lgbm": "LightGBM",
    "lgbm_lag": "LightGBM+lag",
    "lstm": "LSTM",
    "patchtst": "PatchTST",
}
MODEL_COLORS = {
    "lgbm": "#3b6c8f",
    "lgbm_lag": "#5f8f3b",
    "lstm": "#b44b5c",
    "patchtst": "#2f7d6d",
}

CANONICAL_BSPLIT_SEED = 42
BSPLIT_SEEDS = [7, 42, 123, 256, 512]
HEEW_TEST_YEAR = 2022
HEEW_CONTEXT_DAYS = 28
HEEW_HORIZON_DAYS = 7
HEEW_LAG_MAP = {"lag_1h": 1, "lag_24h": 7, "lag_168h": 28}


def configure_logging(log_file: str | Path | None = None) -> Path | None:
    HEEW_LOGS_DIR.mkdir(parents=True, exist_ok=True)
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


def _ensure_dirs() -> None:
    HEEW_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    HEEW_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    HEEW_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    HEEW_LOGS_DIR.mkdir(parents=True, exist_ok=True)


def _load_heew_raw() -> pd.DataFrame:
    LOGGER.info("Loading HEEW daily data from %s", HEEW_DATA_PATH)
    frame = pd.read_csv(HEEW_DATA_PATH, parse_dates=["date"])
    frame = frame.sort_values(["building_id", "date"]).reset_index(drop=True)
    return frame


def _filter_heew(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    grouped = raw.groupby("building_id", sort=True)
    summary = grouped.agg(
        n_rows=("date", "size"),
        start_date=("date", "min"),
        end_date=("date", "max"),
        electricity_missing_rate=("electricity", lambda s: float(s.isna().mean())),
        weather_missing_rate=(
            "temp_avg",
            lambda s: float(
                raw.loc[s.index, ["temp_avg", "dew_point_avg", "humidity_avg", "wind_speed_avg"]]
                .isna()
                .mean()
                .mean()
            ),
        ),
        nonzero_share=("electricity", lambda s: float((s.fillna(0) > 0).mean())),
        target_std=("electricity", lambda s: float(s.std(ddof=0))),
    ).reset_index()
    summary["coverage_days"] = (summary["end_date"] - summary["start_date"]).dt.days + 1
    summary["passes_filter"] = (
        (summary["coverage_days"] >= 730)
        & (summary["electricity_missing_rate"] <= 0.05)
        & (summary["weather_missing_rate"] <= 0.05)
        & (summary["nonzero_share"] >= 0.80)
        & (summary["target_std"] > 1.0)
    )
    filtered_ids = summary.loc[summary["passes_filter"], "building_id"].tolist()
    filtered = raw[raw["building_id"].isin(filtered_ids)].copy()
    LOGGER.info(
        "HEEW filter retained buildings=%s/%s rows=%s",
        len(filtered_ids),
        raw["building_id"].nunique(),
        len(filtered),
    )
    return filtered, summary


def _prepare_model_frame(raw: pd.DataFrame) -> pd.DataFrame:
    frame = raw.copy()
    frame["timestamp"] = pd.to_datetime(frame["date"])
    frame["meter_reading"] = frame["electricity"].astype(np.float32)
    frame["hour_of_day"] = np.int16(0)
    frame["day_of_week"] = frame["timestamp"].dt.dayofweek.astype(np.int16)
    frame["month"] = frame["timestamp"].dt.month.astype(np.int16)
    frame["is_weekend"] = (frame["day_of_week"] >= 5).astype(np.int16)
    frame["airTemperature"] = frame["temp_avg"].astype(np.float32)
    frame["dewTemperature"] = frame["dew_point_avg"].astype(np.float32)
    frame["windSpeed"] = frame["wind_speed_avg"].astype(np.float32)
    frame["cloudCoverage"] = frame["humidity_avg"].astype(np.float32)
    frame["log_floor_area"] = np.float32(0.0)
    frame["building_type"] = "Unknown"
    frame["site_id"] = "HEEW"
    columns = [
        "building_id",
        "site_id",
        "building_type",
        "timestamp",
        "meter_reading",
        "hour_of_day",
        "day_of_week",
        "month",
        "is_weekend",
        "airTemperature",
        "dewTemperature",
        "windSpeed",
        "cloudCoverage",
        "log_floor_area",
    ]
    prepared = frame[columns].sort_values(["building_id", "timestamp"]).reset_index(drop=True)
    return prepared


def _add_heew_lags(frame: pd.DataFrame) -> pd.DataFrame:
    augmented = frame.sort_values(["building_id", "timestamp"]).reset_index(drop=True).copy()
    grouped = augmented.groupby("building_id", sort=False)["meter_reading"]
    for lag_col, lag_days in HEEW_LAG_MAP.items():
        augmented[lag_col] = grouped.shift(lag_days).astype(np.float32)
    return augmented


def _build_t_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = frame[frame["timestamp"].dt.year < HEEW_TEST_YEAR].copy()
    test = frame[frame["timestamp"].dt.year == HEEW_TEST_YEAR].copy()
    return train, test


def _build_b_split(frame: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    building_ids = sorted(frame["building_id"].astype(str).unique().tolist())
    rng = np.random.default_rng(seed)
    n_test = max(2, int(np.floor(len(building_ids) * 0.2)))
    held_out = sorted(rng.choice(np.array(building_ids, dtype=object), size=n_test, replace=False).tolist())
    train = frame[~frame["building_id"].isin(held_out)].copy()
    test = frame[frame["building_id"].isin(held_out)].copy()
    return train, test, held_out


def _compute_rowwise_metrics(prediction_frame: pd.DataFrame) -> dict[str, float]:
    y_true = prediction_frame["y_true"].to_numpy(dtype=np.float64)
    y_pred = prediction_frame["y_pred"].to_numpy(dtype=np.float64)
    mae = float(np.mean(np.abs(y_pred - y_true)))
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    mean_y = float(np.mean(y_true))
    cv_rmse = float(rmse / mean_y) if mean_y else np.nan
    return {
        "mae": mae,
        "rmse": rmse,
        "cv_rmse": cv_rmse,
        "n_rows": int(len(prediction_frame)),
    }


def _compute_per_building_metrics(prediction_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for building_id, group in prediction_frame.groupby("building_id", sort=True):
        y_true = group["y_true"].to_numpy(dtype=np.float64)
        y_pred = group["y_pred"].to_numpy(dtype=np.float64)
        rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
        mean_y = float(np.mean(y_true))
        rows.append(
            {
                "building_id": str(building_id),
                "mean_load": mean_y,
                "cv_rmse": float(rmse / mean_y) if mean_y else np.nan,
                "n_rows": int(len(group)),
            }
        )
    return pd.DataFrame(rows)


def _summarize_per_building(per_building: pd.DataFrame) -> dict[str, float]:
    values = per_building["cv_rmse"].to_numpy(dtype=np.float64)
    q1, q3 = np.quantile(values, [0.25, 0.75])
    return {
        "n_buildings": int(len(per_building)),
        "mean_per_building_cv_rmse": float(np.mean(values)),
        "median_per_building_cv_rmse": float(np.median(values)),
        "q1_per_building_cv_rmse": float(q1),
        "q3_per_building_cv_rmse": float(q3),
    }


def _build_model_specs(cpu_threads: int | None, device: str) -> dict[str, object]:
    return {
        "lgbm": LightGBMConfig(
            n_jobs=cpu_threads if cpu_threads is not None else -1,
            random_state=42,
            target_transform="log1p",
            use_lag_features=False,
        ),
        "lgbm_lag": LightGBMConfig(
            n_jobs=cpu_threads if cpu_threads is not None else -1,
            random_state=42,
            target_transform="log1p",
            use_lag_features=True,
            lag_feature_steps=HEEW_LAG_MAP,
        ),
        "lstm": LSTMConfig(
            context_window=HEEW_CONTEXT_DAYS,
            prediction_horizon=HEEW_HORIZON_DAYS,
            hidden_size=64,
            num_layers=2,
            dropout=0.1,
            learning_rate=1e-3,
            batch_size=64,
            max_epochs=12,
            patience=3,
            train_stride=HEEW_HORIZON_DAYS,
            eval_stride=HEEW_HORIZON_DAYS,
            max_train_windows=20_000,
            max_eval_windows=None,
            target_transform="log1p",
            random_seed=42,
        ),
        "patchtst": PatchTSTConfig(
            context_window=HEEW_CONTEXT_DAYS,
            prediction_horizon=HEEW_HORIZON_DAYS,
            patch_len=7,
            patch_stride=7,
            d_model=64,
            n_heads=4,
            num_layers=2,
            dropout=0.1,
            learning_rate=3e-4,
            batch_size=64,
            max_epochs=12,
            patience=3,
            train_stride=HEEW_HORIZON_DAYS,
            eval_stride=HEEW_HORIZON_DAYS,
            max_train_windows=20_000,
            max_eval_windows=None,
            target_transform="log1p",
            random_seed=42,
        ),
    }


def _fit_predict_model(
    model_name: str,
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    split_name: str,
    model_specs: dict[str, object],
    device: str,
) -> pd.DataFrame:
    LOGGER.info(
        "HEEW run start model=%s split=%s train_rows=%s test_rows=%s train_buildings=%s test_buildings=%s",
        model_name,
        split_name,
        len(train_frame),
        len(test_frame),
        train_frame["building_id"].nunique(),
        test_frame["building_id"].nunique(),
    )
    if model_name in {"lgbm", "lgbm_lag"}:
        train_ready = train_frame if model_name == "lgbm" else _add_heew_lags(train_frame)
        model = LightGBMExperimentModel(config=LightGBMConfig(**asdict(model_specs[model_name])))
        model.fit(train_ready)
        test_ready = test_frame if model_name == "lgbm" else _add_heew_lags(test_frame)
        return model.predict_frame(
            test_ready,
            split_name=split_name,
            model_name=model_name,
            context_window=HEEW_CONTEXT_DAYS,
            horizon=HEEW_HORIZON_DAYS,
            stride=HEEW_HORIZON_DAYS,
        )

    if model_name == "lstm":
        model = LSTMExperimentModel(config=LSTMConfig(**asdict(model_specs[model_name])), device=device)
        model.fit(train_frame)
        return model.predict_frame(test_frame, split_name=split_name)

    if model_name == "patchtst":
        model = PatchTSTExperimentModel(config=PatchTSTConfig(**asdict(model_specs[model_name])), device=device)
        model.fit(train_frame)
        return model.predict_frame(test_frame, split_name=split_name)

    raise ValueError(f"Unsupported model: {model_name}")


def _run_split(
    frame: pd.DataFrame,
    split_name: str,
    models: list[str],
    model_specs: dict[str, object],
    device: str,
    bsplit_seed: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if split_name == "t_split":
        train_frame, test_frame = _build_t_split(frame)
        held_out_buildings: list[str] = []
    elif split_name == "b_split":
        if bsplit_seed is None:
            raise ValueError("bsplit_seed is required for B-split.")
        train_frame, test_frame, held_out_buildings = _build_b_split(frame, seed=bsplit_seed)
    else:
        raise ValueError(f"Unsupported split: {split_name}")

    metrics_rows: list[dict[str, object]] = []
    per_building_rows: list[pd.DataFrame] = []
    for model_name in models:
        prediction_frame = _fit_predict_model(
            model_name=model_name,
            train_frame=train_frame,
            test_frame=test_frame,
            split_name=split_name,
            model_specs=model_specs,
            device=device,
        )
        suffix = split_name if bsplit_seed is None else f"{split_name}_seed{bsplit_seed}"
        save_predictions(prediction_frame, HEEW_RESULTS_DIR / f"{model_name}_{suffix}.csv")
        row_metrics = _compute_rowwise_metrics(prediction_frame)
        per_building = _compute_per_building_metrics(prediction_frame)
        per_building_summary = _summarize_per_building(per_building)
        metrics_rows.append(
            {
                "dataset": "HEEW",
                "split_name": split_name,
                "split_seed": bsplit_seed,
                "model": model_name,
                "held_out_buildings": ",".join(held_out_buildings) if held_out_buildings else "",
                **row_metrics,
                **per_building_summary,
            }
        )
        enriched_per_building = per_building.copy()
        enriched_per_building["dataset"] = "HEEW"
        enriched_per_building["split_name"] = split_name
        enriched_per_building["split_seed"] = bsplit_seed
        enriched_per_building["model"] = model_name
        per_building_rows.append(enriched_per_building)

    return pd.DataFrame(metrics_rows), pd.concat(per_building_rows, ignore_index=True)


def _compute_win_counts(per_building: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    grouped = per_building.groupby(["split_name", "split_seed", "building_id"], dropna=False, sort=True)
    for (split_name, split_seed, building_id), group in grouped:
        model_scores = {
            str(row["model"]): float(row["cv_rmse"])
            for _, row in group.iterrows()
            if pd.notna(row["cv_rmse"])
        }
        if not model_scores:
            continue
        best_model = min(model_scores, key=model_scores.get)
        rows.append(
            {
                "split_name": split_name,
                "split_seed": split_seed,
                "building_id": building_id,
                "best_model": best_model,
            }
        )
    best_frame = pd.DataFrame(rows)
    counts = (
        best_frame.groupby(["split_name", "split_seed", "best_model"], dropna=False)
        .size()
        .rename("n_best_buildings")
        .reset_index()
        .rename(columns={"best_model": "model"})
    )
    totals = (
        best_frame.groupby(["split_name", "split_seed"], dropna=False)
        .size()
        .rename("n_buildings")
        .reset_index()
    )
    counts = counts.merge(totals, on=["split_name", "split_seed"], how="left")
    counts["share_best_buildings"] = counts["n_best_buildings"] / counts["n_buildings"]
    split_keys = (
        per_building[["split_name", "split_seed"]]
        .drop_duplicates()
        .sort_values(["split_name", "split_seed"], na_position="first")
        .reset_index(drop=True)
    )
    model_frame = pd.DataFrame({"model": MODEL_ORDER})
    split_keys["_merge_key"] = 1
    model_frame["_merge_key"] = 1
    full_index = split_keys.merge(model_frame, on="_merge_key", how="inner").drop(columns="_merge_key")
    counts = full_index.merge(counts, on=["split_name", "split_seed", "model"], how="left")
    counts["n_buildings"] = counts["n_buildings"].fillna(
        counts.groupby(["split_name", "split_seed"], dropna=False)["n_buildings"].transform("max")
    )
    counts["n_best_buildings"] = counts["n_best_buildings"].fillna(0).astype(int)
    counts["share_best_buildings"] = counts["share_best_buildings"].fillna(0.0)
    return counts.sort_values(["split_name", "split_seed", "model"]).reset_index(drop=True)


def _summarize_bsplit_repeats(metrics: pd.DataFrame, win_counts: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    bsplit_metrics = metrics[metrics["split_name"] == "b_split"].copy()
    bsplit_wins = win_counts[win_counts["split_name"] == "b_split"].copy()
    best_by_seed = (
        bsplit_metrics.sort_values(["split_seed", "cv_rmse", "model"])
        .groupby("split_seed", as_index=False)
        .first()[["split_seed", "model"]]
        .rename(columns={"model": "best_model"})
    )
    for model_name, group in bsplit_metrics.groupby("model", sort=False):
        wins = bsplit_wins[bsplit_wins["model"] == model_name]
        rows.append(
            {
                "dataset": "HEEW",
                "model": model_name,
                "n_split_seeds": int(group["split_seed"].nunique()),
                "pooled_cv_rmse_mean": float(group["cv_rmse"].mean()),
                "pooled_cv_rmse_sd": float(group["cv_rmse"].std(ddof=1)),
                "median_per_building_cv_rmse_mean": float(group["median_per_building_cv_rmse"].mean()),
                "median_per_building_cv_rmse_sd": float(group["median_per_building_cv_rmse"].std(ddof=1)),
                "mean_share_best_buildings": float(wins["share_best_buildings"].mean()) if not wins.empty else np.nan,
                "best_rank_frequency": int((best_by_seed["best_model"] == model_name).sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("model").reset_index(drop=True)


def _write_dataset_summary(filtered: pd.DataFrame, filter_summary: pd.DataFrame) -> None:
    t_train, t_test = _build_t_split(filtered)
    b_train, b_test, held_out = _build_b_split(filtered, seed=CANONICAL_BSPLIT_SEED)
    summary = pd.DataFrame(
        [
            {
                "dataset": "HEEW",
                "frequency": "daily",
                "target_variable": "whole-building electricity",
                "n_buildings_retained": int(filtered["building_id"].nunique()),
                "date_start": str(filtered["timestamp"].min().date()),
                "date_end": str(filtered["timestamp"].max().date()),
                "t_split_train_rows": int(len(t_train)),
                "t_split_test_rows": int(len(t_test)),
                "b_split_train_buildings": int(b_train["building_id"].nunique()),
                "b_split_test_buildings": int(b_test["building_id"].nunique()),
                "b_split_test_ids": ",".join(held_out),
            }
        ]
    )
    filter_rows = [
        {"step": "Raw daily electricity-weather records", "n_buildings": int(filter_summary["building_id"].nunique())},
        {"step": "Coverage >= 2 years", "n_buildings": int((filter_summary["coverage_days"] >= 730).sum())},
        {"step": "Electricity missing rate <= 5%", "n_buildings": int((filter_summary["electricity_missing_rate"] <= 0.05).sum())},
        {"step": "Weather missing rate <= 5%", "n_buildings": int((filter_summary["weather_missing_rate"] <= 0.05).sum())},
        {"step": "Nonzero electricity share >= 80%", "n_buildings": int((filter_summary["nonzero_share"] >= 0.80).sum())},
        {"step": "Target std > 1.0", "n_buildings": int((filter_summary["target_std"] > 1.0).sum())},
        {"step": "Final retained buildings", "n_buildings": int(filter_summary["passes_filter"].sum())},
    ]
    pd.DataFrame(filter_rows).to_csv(HEEW_FILTER_SUMMARY_PATH, index=False)
    summary.to_csv(HEEW_DATASET_SUMMARY_PATH, index=False)


def _build_heew_figure(metrics: pd.DataFrame) -> Path:
    canonical = metrics[metrics["split_seed"].isna()].copy()
    pooled = canonical[["split_name", "model", "cv_rmse"]].copy()
    per_building = canonical[["split_name", "model", "median_per_building_cv_rmse"]].copy()

    def _prepare_rank_frame(frame: pd.DataFrame, value_col: str) -> pd.DataFrame:
        prepared = frame.copy()
        prepared["rank"] = prepared.groupby("split_name")[value_col].rank(method="min", ascending=True)
        return prepared

    pooled = _prepare_rank_frame(pooled, "cv_rmse")
    per_building = _prepare_rank_frame(per_building, "median_per_building_cv_rmse")

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.6), sharey=True)
    for ax, frame, value_col, title in [
        (axes[0], pooled, "cv_rmse", "Pooled row-wise CV(RMSE) rank"),
        (axes[1], per_building, "median_per_building_cv_rmse", "Median per-building CV(RMSE) rank"),
    ]:
        for model_name in MODEL_ORDER:
            subset = frame[frame["model"] == model_name].set_index("split_name")
            x = np.array([0, 1], dtype=np.float64)
            y = np.array([subset.loc["t_split", "rank"], subset.loc["b_split", "rank"]], dtype=np.float64)
            values = np.array([subset.loc["t_split", value_col], subset.loc["b_split", value_col]], dtype=np.float64)
            ax.plot(
                x,
                y,
                color=MODEL_COLORS[model_name],
                marker="o",
                linewidth=2.0,
                markersize=6.0,
            )
            ax.text(
                x[0] - 0.03,
                y[0],
                f"{MODEL_LABELS[model_name]}\n{values[0]:.3f}",
                ha="right",
                va="center",
                fontsize=8.1,
                color=MODEL_COLORS[model_name],
            )
            ax.text(
                x[1] + 0.03,
                y[1],
                f"{MODEL_LABELS[model_name]}\n{values[1]:.3f}",
                ha="left",
                va="center",
                fontsize=8.1,
                color=MODEL_COLORS[model_name],
            )
        ax.set_xticks([0, 1], ["T-split", "B-split"])
        ax.set_xlim(-0.35, 1.35)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2, linestyle="--")
        ax.invert_yaxis()
    axes[0].set_ylabel("Rank (1 = best)")
    axes[0].set_yticks([1, 2, 3, 4])
    fig.tight_layout()
    fig.savefig(HEEW_FIGURE_PATH, dpi=320, facecolor="white")
    plt.close(fig)
    return HEEW_FIGURE_PATH


def _write_model_config_table(model_specs: dict[str, object]) -> None:
    rows: list[dict[str, object]] = []
    for model_name, config in model_specs.items():
        payload = asdict(config)
        payload["model"] = model_name
        rows.append(payload)
    pd.DataFrame(rows).to_csv(HEEW_CONFIG_PATH, index=False)


def run_heew_replication(
    models: list[str] | None = None,
    cpu_fraction: float | None = None,
    max_cpu_threads: int | None = None,
    device: str | None = None,
) -> dict[str, Path]:
    _ensure_dirs()
    set_seed(42)
    runtime_limits = apply_runtime_limits(max_cpu_threads=max_cpu_threads, cpu_fraction=cpu_fraction)
    resolved_device = device or "cpu"
    LOGGER.info(
        "HEEW replication runtime limits cpu_threads=%s memory_limit=%s device=%s",
        runtime_limits.cpu_threads,
        format_bytes(runtime_limits.memory_bytes),
        resolved_device,
    )
    started_at = time.time()
    raw = _load_heew_raw()
    filtered_raw, filter_summary = _filter_heew(raw)
    frame = _prepare_model_frame(filtered_raw)
    _write_dataset_summary(frame, filter_summary)

    models = models or MODEL_ORDER
    model_specs = _build_model_specs(runtime_limits.cpu_threads, resolved_device)
    _write_model_config_table(model_specs)

    canonical_metrics: list[pd.DataFrame] = []
    canonical_per_building: list[pd.DataFrame] = []
    t_metrics, t_per_building = _run_split(
        frame=frame,
        split_name="t_split",
        models=models,
        model_specs=model_specs,
        device=resolved_device,
    )
    canonical_metrics.append(t_metrics)
    canonical_per_building.append(t_per_building)

    b_metrics, b_per_building = _run_split(
        frame=frame,
        split_name="b_split",
        models=models,
        model_specs=model_specs,
        device=resolved_device,
        bsplit_seed=CANONICAL_BSPLIT_SEED,
    )
    b_metrics["split_seed"] = np.nan
    b_per_building["split_seed"] = np.nan
    canonical_metrics.append(b_metrics)
    canonical_per_building.append(b_per_building)

    repeated_metrics: list[pd.DataFrame] = []
    repeated_per_building: list[pd.DataFrame] = []
    for split_seed in BSPLIT_SEEDS:
        seed_metrics, seed_per_building = _run_split(
            frame=frame,
            split_name="b_split",
            models=models,
            model_specs=model_specs,
            device=resolved_device,
            bsplit_seed=split_seed,
        )
        repeated_metrics.append(seed_metrics)
        repeated_per_building.append(seed_per_building)

    metrics = pd.concat(canonical_metrics + repeated_metrics, ignore_index=True)
    per_building = pd.concat(canonical_per_building + repeated_per_building, ignore_index=True)
    win_counts = _compute_win_counts(per_building)
    bsplit_seed_summary = _summarize_bsplit_repeats(metrics, win_counts)

    metrics.to_csv(HEEW_METRICS_PATH, index=False)
    per_building.to_csv(HEEW_PER_BUILDING_PATH, index=False)
    win_counts.to_csv(HEEW_WIN_COUNTS_PATH, index=False)
    metrics[(metrics["split_name"] == "b_split") & metrics["split_seed"].notna()].to_csv(
        HEEW_BSPLIT_SEED_METRICS_PATH,
        index=False,
    )
    bsplit_seed_summary.to_csv(HEEW_BSPLIT_SEED_SUMMARY_PATH, index=False)
    _build_heew_figure(metrics)

    elapsed = time.time() - started_at
    LOGGER.info("HEEW replication finished in %.1f seconds", elapsed)
    return {
        "heew_filter_summary": HEEW_FILTER_SUMMARY_PATH,
        "heew_dataset_summary": HEEW_DATASET_SUMMARY_PATH,
        "heew_metrics": HEEW_METRICS_PATH,
        "heew_per_building": HEEW_PER_BUILDING_PATH,
        "heew_win_counts": HEEW_WIN_COUNTS_PATH,
        "heew_bsplit_seed_metrics": HEEW_BSPLIT_SEED_METRICS_PATH,
        "heew_bsplit_seed_summary": HEEW_BSPLIT_SEED_SUMMARY_PATH,
        "heew_figure": HEEW_FIGURE_PATH,
        "heew_model_configs": HEEW_CONFIG_PATH,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Independent HEEW replication experiment.")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_ORDER,
        default=MODEL_ORDER,
        help="Models to run.",
    )
    parser.add_argument("--device", default="cpu", help="Torch device for sequence models.")
    parser.add_argument("--cpu-fraction", type=float, default=None, help="CPU thread fraction.")
    parser.add_argument("--max-cpu-threads", type=int, default=None, help="Maximum CPU threads.")
    parser.add_argument("--log-file", type=str, default=None, help="Optional log file path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.log_file)
    LOGGER.info("HEEW replication started at %s", datetime.now().isoformat())
    run_heew_replication(
        models=args.models,
        cpu_fraction=args.cpu_fraction,
        max_cpu_threads=args.max_cpu_threads,
        device=args.device,
    )


if __name__ == "__main__":
    main()
