from __future__ import annotations

import argparse
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

from src.config import FEATURES_BDG2_PATH, FIGURES_DIR, LOGS_DIR, RESULTS_DIR, TABLES_DIR, ensure_phase2_dirs
from src.data_splitting import B_SPLIT_PATH, S_SPLIT_PATH, T_SPLIT_PATH, load_pickle, unpack_mask
from src.metrics import summarize_metrics
from src.models.common import save_predictions
from src.runtime import apply_runtime_limits, format_bytes


LOGGER = logging.getLogger(__name__)

BASELINE_PREDICTIONS_DIR = RESULTS_DIR / "exp1_baseline_predictions"
BASELINE_METRICS_PATH = TABLES_DIR / "exp1_baseline_metrics.csv"
BASELINE_METRICS_PIVOT_PATH = TABLES_DIR / "exp1_baseline_metrics_pivot.csv"
BASELINE_FIG_PATH = FIGURES_DIR / "exp1_baseline_cv_rmse.png"


def configure_logging(log_file: str | Path | None = None) -> Path | None:
    ensure_phase2_dirs()
    BASELINE_PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
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


def _load_full_features() -> pd.DataFrame:
    frame = pd.read_parquet(FEATURES_BDG2_PATH)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame = frame.sort_values(["building_id", "timestamp"]).reset_index(drop=True)
    frame["hour_of_week"] = frame["timestamp"].dt.dayofweek.astype("int16") * 24 + frame["timestamp"].dt.hour.astype("int16")
    return frame


def _select_rows(frame: pd.DataFrame, packed_mask: dict[str, object]) -> pd.DataFrame:
    mask = unpack_mask(packed_mask)
    return frame.loc[mask].reset_index(drop=True).copy()


def _attach_full_frame_lags(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    grouped = enriched.groupby("building_id", sort=False)["meter_reading"]
    enriched["lag_1h"] = grouped.shift(1)
    enriched["lag_24h"] = grouped.shift(24)
    return enriched


def _build_naive_predictions(
    test_frame: pd.DataFrame,
    train_frame: pd.DataFrame,
    split_name: str,
    fold_id: str | None = None,
) -> pd.DataFrame:
    building_train_mean = (
        train_frame.groupby("building_id", sort=False)["meter_reading"].mean().rename("building_train_mean")
    )
    global_train_mean = float(train_frame["meter_reading"].mean())
    baseline = test_frame.copy()
    baseline = baseline.merge(
        building_train_mean.reset_index(),
        on="building_id",
        how="left",
    )
    baseline["y_pred"] = baseline["lag_24h"].fillna(baseline["lag_1h"])
    baseline["y_pred"] = baseline["y_pred"].fillna(baseline["building_train_mean"])
    baseline["y_pred"] = baseline["y_pred"].fillna(global_train_mean)
    baseline["split_name"] = split_name
    baseline["model"] = "naive"
    baseline["fold_id"] = fold_id
    baseline["y_true"] = baseline["meter_reading"]
    return baseline[
        ["split_name", "model", "fold_id", "building_id", "site_id", "building_type", "timestamp", "y_true", "y_pred"]
    ].copy()


def _build_single_building_profile_predictions(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
) -> pd.DataFrame:
    profile = (
        train_frame.groupby(["building_id", "hour_of_week"], sort=False)["meter_reading"]
        .mean()
        .rename("profile_pred")
        .reset_index()
    )
    building_mean = (
        train_frame.groupby("building_id", sort=False)["meter_reading"]
        .mean()
        .rename("building_train_mean")
        .reset_index()
    )
    baseline = test_frame.merge(profile, on=["building_id", "hour_of_week"], how="left")
    baseline = baseline.merge(building_mean, on="building_id", how="left")
    baseline["y_pred"] = baseline["profile_pred"].fillna(baseline["building_train_mean"])
    baseline["split_name"] = "t_split"
    baseline["model"] = "single_building"
    baseline["fold_id"] = None
    baseline["y_true"] = baseline["meter_reading"]
    return baseline[
        ["split_name", "model", "fold_id", "building_id", "site_id", "building_type", "timestamp", "y_true", "y_pred"]
    ].copy()


def _plot_baseline_cv_rmse(metrics_frame: pd.DataFrame) -> None:
    if metrics_frame.empty:
        return
    pivot = metrics_frame.pivot(index="split_name", columns="model", values="cv_rmse")
    split_order = [name for name in ["t_split", "b_split", "s_split"] if name in pivot.index]
    model_order = [name for name in ["naive", "single_building"] if name in pivot.columns]
    pivot = pivot.reindex(index=split_order, columns=model_order)
    x = np.arange(len(split_order), dtype=np.float64)
    width = 0.35 if model_order else 0.8

    plt.figure(figsize=(8, 5))
    for idx, model_name in enumerate(model_order):
        offsets = x + (idx - (len(model_order) - 1) / 2) * width
        values = pivot[model_name].to_numpy(dtype=np.float64)
        bars = plt.bar(offsets, values, width=width, label=model_name)
        for bar, value in zip(bars, values, strict=False):
            if np.isnan(value):
                continue
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                value,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.xticks(x, split_order)
    plt.ylabel("CV(RMSE)")
    plt.title("Supplementary Baselines for Experiment 1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(BASELINE_FIG_PATH, dpi=200)
    plt.close()


def run_supplementary_baselines() -> dict[str, Path]:
    ensure_phase2_dirs()
    BASELINE_PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    started_at = time.time()

    frame = _attach_full_frame_lags(_load_full_features())
    artifacts = {
        "t_split": load_pickle(T_SPLIT_PATH),
        "b_split": load_pickle(B_SPLIT_PATH),
        "s_split": load_pickle(S_SPLIT_PATH),
    }

    outputs: dict[str, Path] = {}
    metrics_rows: list[pd.DataFrame] = []

    for split_name in ["t_split", "b_split"]:
        artifact = artifacts[split_name]
        train_frame = _select_rows(frame, artifact["train_mask"])
        test_frame = _select_rows(frame, artifact["test_mask"])
        naive_pred = _build_naive_predictions(test_frame=test_frame, train_frame=train_frame, split_name=split_name)
        naive_path = BASELINE_PREDICTIONS_DIR / f"naive_{split_name}.csv"
        save_predictions(naive_pred, naive_path)
        outputs[f"naive_{split_name}"] = naive_path
        metrics_rows.append(summarize_metrics(naive_pred[["split_name", "model", "y_true", "y_pred"]], ["split_name", "model"]))

        if split_name == "t_split":
            single_pred = _build_single_building_profile_predictions(train_frame=train_frame, test_frame=test_frame)
            single_path = BASELINE_PREDICTIONS_DIR / "single_building_t_split.csv"
            save_predictions(single_pred, single_path)
            outputs["single_building_t_split"] = single_path
            metrics_rows.append(
                summarize_metrics(single_pred[["split_name", "model", "y_true", "y_pred"]], ["split_name", "model"])
            )

    s_artifact = artifacts["s_split"]
    naive_folds: list[pd.DataFrame] = []
    for fold in s_artifact["folds"]:
        train_frame = _select_rows(frame, fold["train_mask"])
        test_frame = _select_rows(frame, fold["test_mask"])
        naive_folds.append(
            _build_naive_predictions(
                test_frame=test_frame,
                train_frame=train_frame,
                split_name="s_split",
                fold_id=str(fold["fold_id"]),
            )
        )
    naive_s = pd.concat(naive_folds, ignore_index=True)
    naive_s_path = BASELINE_PREDICTIONS_DIR / "naive_s_split.csv"
    save_predictions(naive_s, naive_s_path)
    outputs["naive_s_split"] = naive_s_path
    metrics_rows.append(summarize_metrics(naive_s[["split_name", "model", "y_true", "y_pred"]], ["split_name", "model"]))

    metrics_frame = pd.concat(metrics_rows, ignore_index=True).sort_values(["split_name", "model"]).reset_index(drop=True)
    pivot = metrics_frame.pivot(index="split_name", columns="model", values=["mae", "rmse", "cv_rmse"])
    pivot.columns = [f"{model}_{metric}" for metric, model in pivot.columns]
    pivot = pivot.reset_index()

    metrics_frame.to_csv(BASELINE_METRICS_PATH, index=False)
    pivot.to_csv(BASELINE_METRICS_PIVOT_PATH, index=False)
    _plot_baseline_cv_rmse(metrics_frame)

    outputs["exp1_baseline_metrics"] = BASELINE_METRICS_PATH
    outputs["exp1_baseline_metrics_pivot"] = BASELINE_METRICS_PIVOT_PATH
    outputs["exp1_baseline_cv_rmse"] = BASELINE_FIG_PATH
    LOGGER.info("Supplementary baselines finished in %.1f seconds", time.time() - started_at)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run supplementary Naive and Single-building baselines for Experiment 1.")
    parser.add_argument(
        "--log-file",
        default=None,
        help="Optional log file path. Relative paths are resolved from the repository root.",
    )
    parser.add_argument(
        "--max-cpu-threads",
        type=int,
        default=None,
        help="Cap CPU thread usage for pandas, NumPy, and BLAS/OpenMP.",
    )
    parser.add_argument(
        "--cpu-fraction",
        type=float,
        default=None,
        help="Cap CPU usage as a fraction of total logical CPUs, for example 0.7.",
    )
    parser.add_argument(
        "--memory-fraction",
        type=float,
        default=None,
        help="Cap this process to a fraction of total system memory on supported platforms.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    resolved_log = configure_logging(args.log_file)
    if resolved_log is None:
        resolved_log = configure_logging(LOGS_DIR / f"exp1_supplementary_baselines_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    LOGGER.info("Logging to %s", resolved_log)
    limits = apply_runtime_limits(
        max_cpu_threads=args.max_cpu_threads,
        cpu_fraction=args.cpu_fraction,
        memory_fraction=args.memory_fraction,
    )
    LOGGER.info(
        "Runtime limits cpu_threads=%s memory_limit=%s applied=%s",
        limits.cpu_threads if limits.cpu_threads is not None else "unlimited",
        format_bytes(limits.memory_bytes),
        limits.memory_limit_applied,
    )
    outputs = run_supplementary_baselines()
    for name, path in outputs.items():
        print(f"{name}: {path}")
