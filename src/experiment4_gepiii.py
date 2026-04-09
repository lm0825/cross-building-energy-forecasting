from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import gc
import json
import logging
import os
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

from src.config import (
    EXP4_PREDICTIONS_DIR,
    FEATURES_GEPIII_PATH,
    FIGURES_DIR,
    LOGS_DIR,
    MODELS_DIR,
    RANDOM_SEED,
    RESULTS_DIR,
    SPLITS_DIR,
    TABLES_DIR,
    ensure_phase2_dirs,
)
from src.data_splitting import (
    load_split_frame,
    load_pickle,
    make_b_split,
    pack_mask,
    save_pickle,
    unpack_mask,
)
from src.metrics import summarize_metrics
from src.models.common import add_tabular_lag_features, save_json, save_predictions, set_seed
from src.models.lgbm_model import LightGBMConfig, LightGBMExperimentModel
from src.models.lstm_model import LSTMConfig, LSTMExperimentModel
from src.models.patchtst_model import PatchTSTExperimentModel
from src.runtime import apply_runtime_limits, format_bytes


LOGGER = logging.getLogger(__name__)

GEPIII_T_SPLIT_PATH = SPLITS_DIR / "gepiii_t_split_index.pkl"
GEPIII_B_SPLIT_PATH = SPLITS_DIR / "gepiii_b_split_index.pkl"

EXP4_METRICS_PATH = RESULTS_DIR / "exp4_gepiii_metrics.csv"
EXP4_CROSS_DATASET_PATH = TABLES_DIR / "exp4_cross_dataset_comparison.csv"
EXP4_DROP_FIG_PATH = FIGURES_DIR / "exp4_performance_drop_comparison.png"


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


def _load_full_features(with_lag_features: bool = False) -> pd.DataFrame:
    LOGGER.info("Loading GEPIII features from %s", FEATURES_GEPIII_PATH)
    frame = pd.read_parquet(FEATURES_GEPIII_PATH)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame = frame.sort_values(["building_id", "timestamp"]).reset_index(drop=True)
    if with_lag_features:
        frame = add_tabular_lag_features(frame)
    LOGGER.info(
        "Loaded GEPIII frame rows=%s buildings=%s sites=%s date_range=%s to %s lag_features=%s",
        len(frame),
        frame["building_id"].nunique(),
        frame["site_id"].nunique(),
        frame["timestamp"].min(),
        frame["timestamp"].max(),
        with_lag_features,
    )
    return frame


def _select_rows(frame: pd.DataFrame, packed_mask: dict[str, object]) -> pd.DataFrame:
    mask = unpack_mask(packed_mask)
    return frame.loc[mask].reset_index(drop=True)


def make_gepiii_t_split(frame: pd.DataFrame, train_fraction: float = 0.7) -> dict[str, object]:
    timestamps = np.sort(frame["timestamp"].unique())
    split_idx = max(1, int(np.floor(len(timestamps) * train_fraction)))
    cutoff = timestamps[split_idx - 1]
    train_mask = frame["timestamp"] <= cutoff
    test_mask = frame["timestamp"] > cutoff
    return {
        "split_name": "t_split",
        "description": "First 70% timestamps train / last 30% timestamps test within GEPIII 2016 train set",
        "num_rows": int(len(frame)),
        "train_fraction": float(train_fraction),
        "train_end_timestamp": pd.Timestamp(cutoff).isoformat(),
        "train_mask": pack_mask(train_mask.to_numpy()),
        "test_mask": pack_mask(test_mask.to_numpy()),
    }


def save_gepiii_split_artifacts() -> dict[str, Path]:
    ensure_phase2_dirs()
    split_frame = load_split_frame(FEATURES_GEPIII_PATH)
    t_split = make_gepiii_t_split(split_frame)
    b_split = make_b_split(split_frame, test_fraction=0.2, random_seed=RANDOM_SEED)
    save_pickle(t_split, GEPIII_T_SPLIT_PATH)
    save_pickle(b_split, GEPIII_B_SPLIT_PATH)
    return {
        "gepiii_t_split": GEPIII_T_SPLIT_PATH,
        "gepiii_b_split": GEPIII_B_SPLIT_PATH,
    }


def _resolve_model_dir(*parts: str) -> Path:
    base = MODELS_DIR / "exp4" / "gepiii"
    if base.exists() and not os.access(base, os.W_OK):
        base = MODELS_DIR / "_rerun" / "exp4" / "gepiii"
    path = base.joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save_tuned_config(
    model_name: str,
    config: object,
    *,
    split_name: str,
) -> Path:
    tuning_dir = MODELS_DIR / "_tuning_gepiii"
    tuning_dir.mkdir(parents=True, exist_ok=True)
    path = tuning_dir / f"{model_name}.{split_name}.config.json"
    save_json(asdict(config), path)
    if split_name == "t_split":
        save_json(asdict(config), tuning_dir / f"{model_name}.config.json")
    return path


def _fit_and_predict_model(
    model_name: str,
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    split_name: str,
    model_config: object | None = None,
    device: str | None = None,
    cpu_threads: int | None = None,
) -> pd.DataFrame:
    LOGGER.info(
        "Starting GEPIII model=%s split=%s train_rows=%s test_rows=%s train_buildings=%s test_buildings=%s",
        model_name,
        split_name,
        len(train_frame),
        len(test_frame),
        train_frame["building_id"].nunique(),
        test_frame["building_id"].nunique(),
    )
    if model_name in {"lgbm", "lgbm_lag"}:
        config = (
            LightGBMConfig(**asdict(model_config))
            if model_config is not None
            else LightGBMConfig(n_jobs=cpu_threads) if cpu_threads is not None else LightGBMConfig()
        )
        config.target_transform = "log1p"
        config.use_lag_features = model_name == "lgbm_lag"
        model = LightGBMExperimentModel(config=config)
        model.fit(train_frame)
        prediction_frame = model.predict_frame(
            test_frame,
            split_name=split_name,
            model_name=model_name,
        ).drop(columns=["fold_id"])
        model.save(_resolve_model_dir(split_name, model_name) / "lgbm.txt")
        return prediction_frame

    if model_name == "lstm":
        if model_config is None:
            config = LSTMConfig(target_transform="log1p")
        else:
            config = LSTMConfig(**asdict(model_config))
            config.target_transform = "log1p"
        model = LSTMExperimentModel(
            config=config,
            device=device,
        )
        model.fit(train_frame)
        model.save(_resolve_model_dir(split_name, model_name) / "lstm.pt")
        frame = model.predict_frame(test_frame, split_name=split_name, fold_id=None)
        return frame.drop(columns=["fold_id"])

    if model_name == "patchtst":
        model = PatchTSTExperimentModel(
            config=None if model_config is None else type(model_config)(**asdict(model_config)),
            device=device,
        )
        model.fit(train_frame)
        model.save(_resolve_model_dir(split_name, model_name) / "patchtst.pt")
        frame = model.predict_frame(test_frame, split_name=split_name, fold_id=None)
        return frame.drop(columns=["fold_id"])

    raise ValueError(f"Unsupported model name: {model_name}")


def _tune_model_config(
    model_name: str,
    train_frame: pd.DataFrame,
    *,
    split_name: str,
    device: str | None,
    cpu_threads: int | None,
) -> tuple[object, Path]:
    LOGGER.info(
        "Tuning GEPIII model=%s on train partition for split=%s rows=%s buildings=%s",
        model_name,
        split_name,
        len(train_frame),
        train_frame["building_id"].nunique(),
    )
    if model_name in {"lgbm", "lgbm_lag"}:
        model = LightGBMExperimentModel(
            config=LightGBMConfig(
                n_jobs=cpu_threads,
                target_transform="log1p",
                use_lag_features=model_name == "lgbm_lag",
            )
            if cpu_threads is not None
            else LightGBMConfig(target_transform="log1p", use_lag_features=model_name == "lgbm_lag")
        )
    elif model_name == "lstm":
        model = LSTMExperimentModel(config=LSTMConfig(target_transform="log1p"), device=device)
    elif model_name == "patchtst":
        model = PatchTSTExperimentModel(device=device)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    config = model.tune_on_tsplit(train_frame)
    frozen_config = type(config)(**asdict(config))
    path = _save_tuned_config(model_name, frozen_config, split_name=split_name)
    LOGGER.info("Saved GEPIII tuned config model=%s split=%s path=%s values=%s", model_name, split_name, path, asdict(frozen_config))
    return frozen_config, path


def _compute_gepiii_metrics(prediction_paths: dict[str, Path]) -> Path:
    rows: list[dict[str, object]] = []
    known_models = ["lgbm_lag", "patchtst", "lgbm", "lstm"]
    for key, path in sorted(prediction_paths.items()):
        if not key.endswith("_predictions"):
            continue
        frame = pd.read_csv(path)
        stem = key.removesuffix("_predictions")
        split_name = None
        model = None
        for candidate in known_models:
            suffix = f"_{candidate}"
            if stem.endswith(suffix):
                split_name = stem[: -len(suffix)]
                model = candidate
                break
        if split_name is None or model is None:
            raise ValueError(f"Could not parse split/model from prediction key: {key}")
        metrics = summarize_metrics(frame[["y_true", "y_pred"]]).iloc[0].to_dict()
        rows.append(
            {
                "split_name": split_name,
                "model": model,
                **metrics,
            }
        )

    overall = pd.DataFrame(rows)
    if EXP4_METRICS_PATH.exists():
        existing_wide = pd.read_csv(EXP4_METRICS_PATH)
        existing_rows: list[dict[str, object]] = []
        metric_names = ["mae", "rmse", "cv_rmse"]
        for _, record in existing_wide.iterrows():
            split_name = record["split_name"]
            discovered_models = sorted(
                {
                    column[: -len(f"_{metric}")]
                    for metric in metric_names
                    for column in existing_wide.columns
                    if column.endswith(f"_{metric}")
                }
            )
            for model in discovered_models:
                row = {"split_name": split_name, "model": model}
                missing_metric = False
                for metric in metric_names:
                    column = f"{model}_{metric}"
                    if column not in existing_wide.columns:
                        missing_metric = True
                        break
                    row[metric] = record[column]
                if not missing_metric:
                    existing_rows.append(row)

        existing = pd.DataFrame(existing_rows)
        if not overall.empty and not existing.empty:
            current_keys = overall[["split_name", "model"]].drop_duplicates().assign(_current=1)
            existing = existing.merge(current_keys, on=["split_name", "model"], how="left")
            existing = existing.loc[existing["_current"].isna()].drop(columns="_current")
        if not existing.empty:
            overall = pd.concat([existing, overall], ignore_index=True)
    overall = overall.sort_values(["split_name", "model"]).reset_index(drop=True)
    metrics_table = (
        overall.pivot(index="split_name", columns="model", values=["mae", "rmse", "cv_rmse"])
        .sort_index(axis=1)
    )
    metrics_table.columns = [f"{model}_{metric}" for metric, model in metrics_table.columns]
    metrics_table = metrics_table.reset_index()
    metrics_table.to_csv(EXP4_METRICS_PATH, index=False)
    return EXP4_METRICS_PATH


def _flatten_dataset_metrics(metrics_table: pd.DataFrame, dataset_name: str) -> dict[str, object]:
    row: dict[str, object] = {"dataset": dataset_name}
    for _, record in metrics_table.iterrows():
        split_name = str(record["split_name"])
        for column, value in record.items():
            if column == "split_name":
                continue
            row[f"{split_name}_{column}"] = value
    return row


def _plot_drop_comparison(comparison: pd.DataFrame) -> Path:
    models = [
        model
        for model in ["lgbm", "lgbm_lag", "lstm", "patchtst"]
        if f"t_split_{model}_cv_rmse" in comparison.columns
        and f"b_split_{model}_cv_rmse" in comparison.columns
    ]
    datasets = comparison["dataset"].tolist()
    x = np.arange(len(models), dtype=np.float64)
    width = 0.35 if datasets else 0.8

    if not models:
        plt.figure(figsize=(8, 3))
        plt.text(0.5, 0.5, "No shared models available for drop comparison.", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(EXP4_DROP_FIG_PATH, dpi=200)
        plt.close()
        return EXP4_DROP_FIG_PATH

    display_names = {
        "lgbm": "LightGBM",
        "lgbm_lag": "LightGBM+lag",
        "lstm": "LSTM",
        "patchtst": "PatchTST",
    }

    plt.figure(figsize=(10, 6))
    for idx, dataset in enumerate(datasets):
        row = comparison.loc[comparison["dataset"] == dataset].iloc[0]
        drops = []
        for model in models:
            t_value = float(row[f"t_split_{model}_cv_rmse"])
            b_value = float(row[f"b_split_{model}_cv_rmse"])
            drops.append((b_value - t_value) / t_value if t_value else np.nan)
        offsets = x + (idx - (len(datasets) - 1) / 2) * width
        bars = plt.bar(offsets, drops, width=width, label=dataset)
        for bar, value in zip(bars, drops, strict=False):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                value,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.xticks(x, [display_names.get(model, model) for model in models], rotation=15)
    plt.ylabel("Relative CV(RMSE) Increase: (B - T) / T")
    plt.title("Experiment 4 Generalization Drop Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(EXP4_DROP_FIG_PATH, dpi=200)
    plt.close()
    return EXP4_DROP_FIG_PATH


def _build_cross_dataset_comparison() -> dict[str, Path]:
    bdg2_metrics = pd.read_csv(TABLES_DIR / "exp1_metrics.csv")
    bdg2_metrics = bdg2_metrics[bdg2_metrics["split_name"].isin(["t_split", "b_split"])].copy()
    gepiii_metrics = pd.read_csv(EXP4_METRICS_PATH)
    gepiii_metrics = gepiii_metrics[gepiii_metrics["split_name"].isin(["t_split", "b_split"])].copy()

    comparison = pd.DataFrame(
        [
            _flatten_dataset_metrics(bdg2_metrics, "BDG2"),
            _flatten_dataset_metrics(gepiii_metrics, "GEPIII"),
        ]
    )
    comparison.to_csv(EXP4_CROSS_DATASET_PATH, index=False)
    _plot_drop_comparison(comparison)
    return {
        "exp4_cross_dataset_comparison": EXP4_CROSS_DATASET_PATH,
        "exp4_performance_drop_comparison": EXP4_DROP_FIG_PATH,
    }


def run_experiment4(
    models: list[str] | None = None,
    skip_training: bool = False,
    device: str | None = None,
    cpu_threads: int | None = None,
) -> dict[str, Path]:
    ensure_phase2_dirs()
    set_seed()
    started_at = time.time()

    if not GEPIII_T_SPLIT_PATH.exists() or not GEPIII_B_SPLIT_PATH.exists():
        LOGGER.info("GEPIII split artifacts missing. Building fresh split files.")
        outputs = save_gepiii_split_artifacts()
    else:
        outputs = {
            "gepiii_t_split": GEPIII_T_SPLIT_PATH,
            "gepiii_b_split": GEPIII_B_SPLIT_PATH,
        }
        LOGGER.info("Using existing GEPIII split artifacts from %s", SPLITS_DIR)

    if skip_training:
        return outputs

    frame = _load_full_features(with_lag_features="lgbm_lag" in models)
    t_split_artifact = load_pickle(GEPIII_T_SPLIT_PATH)
    b_split_artifact = load_pickle(GEPIII_B_SPLIT_PATH)

    models = models or ["lgbm", "lgbm_lag", "lstm", "patchtst"]
    prediction_outputs: dict[str, Path] = {}
    for split_name, artifact in [("t_split", t_split_artifact), ("b_split", b_split_artifact)]:
        train_frame = _select_rows(frame, artifact["train_mask"])
        test_frame = _select_rows(frame, artifact["test_mask"])
        for model_name in models:
            model_config, config_path = _tune_model_config(
                model_name=model_name,
                train_frame=train_frame,
                split_name=split_name,
                device=device,
                cpu_threads=cpu_threads,
            )
            outputs[f"{split_name}_{model_name}_tuning_config"] = config_path
            prediction_frame = _fit_and_predict_model(
                model_name=model_name,
                train_frame=train_frame,
                test_frame=test_frame,
                split_name=split_name,
                model_config=model_config,
                device=device,
                cpu_threads=cpu_threads,
            )
            path = EXP4_PREDICTIONS_DIR / f"{model_name}_{split_name}.csv"
            save_predictions(prediction_frame, path)
            prediction_outputs[f"{split_name}_{model_name}_predictions"] = path
            outputs[f"{split_name}_{model_name}_predictions"] = path
            del prediction_frame
            gc.collect()
        del train_frame
        del test_frame
        gc.collect()

    outputs["exp4_gepiii_metrics"] = _compute_gepiii_metrics(prediction_outputs)
    outputs.update(_build_cross_dataset_comparison())
    LOGGER.info("Experiment 4 finished in %.1f seconds", time.time() - started_at)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run experiment 4 for GEPIII cross-dataset validation.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["lgbm", "lgbm_lag", "lstm", "patchtst"],
        choices=["lgbm", "lgbm_lag", "lstm", "patchtst"],
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Only prepare GEPIII split artifacts without fitting models.",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Optional log file path. Relative paths are resolved from the repository root.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device for neural models, for example cuda:0 or cpu.",
    )
    parser.add_argument(
        "--max-cpu-threads",
        type=int,
        default=None,
        help="Cap CPU thread usage for LightGBM, PyTorch, OpenMP, BLAS, and parquet reads.",
    )
    parser.add_argument(
        "--cpu-fraction",
        type=float,
        default=None,
        help="Cap CPU thread usage as a fraction of total logical CPUs, for example 0.7.",
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
        resolved_log = configure_logging(LOGS_DIR / f"exp4_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
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
    outputs = run_experiment4(
        models=args.models,
        skip_training=args.skip_training,
        device=args.device,
        cpu_threads=limits.cpu_threads,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")
