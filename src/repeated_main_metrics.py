from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import gc
import json
import logging
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (
    FEATURES_BDG2_PATH,
    FEATURES_GEPIII_PATH,
    LOGS_DIR,
    MODELS_DIR,
    TABLES_DIR,
    ensure_phase2_dirs,
)
from src.data_splitting import B_SPLIT_PATH, S_SPLIT_PATH, T_SPLIT_PATH, load_pickle, unpack_mask
from src.experiment4_gepiii import GEPIII_B_SPLIT_PATH, GEPIII_T_SPLIT_PATH
from src.metrics import summarize_metrics
from src.models.common import add_tabular_lag_features, set_seed
from src.models.lgbm_model import LightGBMConfig, LightGBMExperimentModel
from src.models.lstm_model import LSTMConfig, LSTMExperimentModel
from src.models.patchtst_model import PatchTSTConfig, PatchTSTExperimentModel
from src.runtime import apply_runtime_limits, format_bytes


LOGGER = logging.getLogger(__name__)

PER_BUILDING_KEYS = ["building_id", "site_id", "building_type"]


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


def _load_full_features(dataset: str, with_lag_features: bool = False) -> pd.DataFrame:
    feature_path = FEATURES_BDG2_PATH if dataset == "bdg2" else FEATURES_GEPIII_PATH
    frame = pd.read_parquet(feature_path)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame = frame.sort_values(["building_id", "timestamp"]).reset_index(drop=True)
    if with_lag_features:
        frame = add_tabular_lag_features(frame)
    LOGGER.info(
        "Loaded dataset=%s rows=%s buildings=%s sites=%s lag_features=%s",
        dataset,
        len(frame),
        frame["building_id"].nunique(),
        frame["site_id"].nunique(),
        with_lag_features,
    )
    return frame


def _load_split_artifacts(dataset: str) -> dict[str, dict[str, object]]:
    if dataset == "bdg2":
        return {
            "t_split": load_pickle(T_SPLIT_PATH),
            "b_split": load_pickle(B_SPLIT_PATH),
            "s_split": load_pickle(S_SPLIT_PATH),
        }
    return {
        "t_split": load_pickle(GEPIII_T_SPLIT_PATH),
        "b_split": load_pickle(GEPIII_B_SPLIT_PATH),
    }


def _select_rows(frame: pd.DataFrame, packed_mask: dict[str, object]) -> pd.DataFrame:
    mask = unpack_mask(packed_mask)
    return frame.loc[mask].reset_index(drop=True)


def _tuning_dir(dataset: str) -> Path:
    return MODELS_DIR / ("_tuning" if dataset == "bdg2" else "_tuning_gepiii")


def _resolve_tuning_path(
    dataset: str,
    model_name: str,
    *,
    split_name: str | None = None,
    fold_id: str | None = None,
) -> Path:
    tuning_dir = _tuning_dir(dataset)
    candidates: list[Path] = []
    if split_name is not None and fold_id is not None:
        candidates.append(tuning_dir / f"{model_name}.{split_name}.{fold_id}.config.json")
    if split_name is not None:
        candidates.append(tuning_dir / f"{model_name}.{split_name}.config.json")
    candidates.append(tuning_dir / f"{model_name}.config.json")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[-1]


def _load_lgbm_config(
    dataset: str,
    cpu_threads: int | None = None,
    model_name: str = "lgbm",
    *,
    split_name: str | None = None,
    fold_id: str | None = None,
) -> LightGBMConfig:
    tuning_path = _resolve_tuning_path(dataset, model_name, split_name=split_name, fold_id=fold_id)
    if tuning_path.exists():
        payload = json.loads(tuning_path.read_text(encoding="utf-8"))
        payload["target_transform"] = "log1p"
        if cpu_threads is not None:
            payload["n_jobs"] = cpu_threads
        return LightGBMConfig(**payload)
    return (
        LightGBMConfig(n_jobs=cpu_threads, target_transform="log1p")
        if cpu_threads is not None
        else LightGBMConfig(target_transform="log1p")
    )


def _load_lstm_config(
    dataset: str,
    *,
    split_name: str | None = None,
    fold_id: str | None = None,
) -> LSTMConfig:
    tuning_path = _resolve_tuning_path(dataset, "lstm", split_name=split_name, fold_id=fold_id)
    if tuning_path.exists():
        payload = json.loads(tuning_path.read_text(encoding="utf-8"))
        return LSTMConfig(**payload)
    return LSTMConfig()


def _load_patchtst_config(
    dataset: str,
    *,
    split_name: str | None = None,
    fold_id: str | None = None,
) -> PatchTSTConfig:
    tuning_path = _resolve_tuning_path(dataset, "patchtst", split_name=split_name, fold_id=fold_id)
    if tuning_path.exists():
        payload = json.loads(tuning_path.read_text(encoding="utf-8"))
        return PatchTSTConfig(**payload)
    return PatchTSTConfig()


def _evaluate_single_run(
    dataset: str,
    model_name: str,
    split_name: str,
    random_seed: int,
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    cpu_threads: int | None,
    device: str | None,
    fold_id: str | None = None,
) -> dict[str, object]:
    LOGGER.info(
        "Repeated run dataset=%s split=%s model=%s seed=%s train_rows=%s test_rows=%s",
        dataset,
        split_name,
        model_name,
        random_seed,
        len(train_frame),
        len(test_frame),
    )
    set_seed(random_seed)

    if model_name in {"lgbm", "lgbm_lag"}:
        payload = asdict(
            _load_lgbm_config(
                dataset,
                cpu_threads=cpu_threads,
                model_name=model_name,
                split_name=split_name,
                fold_id=fold_id,
            )
        )
        payload["random_state"] = random_seed
        if cpu_threads is not None:
            payload["n_jobs"] = cpu_threads
        payload["target_transform"] = "log1p"
        payload["use_lag_features"] = model_name == "lgbm_lag"
        model = LightGBMExperimentModel(config=LightGBMConfig(**payload))
        model.fit(train_frame)
        prediction_frame = model.predict_frame(
            test_frame,
            split_name=split_name,
            fold_id=fold_id,
            model_name=model_name,
        ).copy()
    elif model_name == "lstm":
        payload = asdict(_load_lstm_config(dataset, split_name=split_name, fold_id=fold_id))
        payload["random_seed"] = random_seed
        model = LSTMExperimentModel(config=LSTMConfig(**payload), device=device)
        model.fit(train_frame)
        prediction_frame = model.predict_frame(test_frame, split_name=split_name, fold_id=fold_id).copy()
    elif model_name == "patchtst":
        payload = asdict(_load_patchtst_config(dataset, split_name=split_name, fold_id=fold_id))
        payload["random_seed"] = random_seed
        model = PatchTSTExperimentModel(config=PatchTSTConfig(**payload), device=device)
        model.fit(train_frame)
        prediction_frame = model.predict_frame(test_frame, split_name=split_name, fold_id=fold_id).copy()
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    metrics = summarize_metrics(prediction_frame[["y_true", "y_pred"]]).iloc[0].to_dict()
    prediction_frame["dataset"] = dataset
    prediction_frame["random_seed"] = int(random_seed)

    per_building = _compute_per_building_metrics(prediction_frame)
    per_building_summary = _summarize_per_building_metrics(per_building)

    return {
        "metrics": {
            "dataset": dataset,
            "split_name": split_name,
            "model": model_name,
            "random_seed": int(random_seed),
            **metrics,
            **per_building_summary,
        },
        "prediction_frame": prediction_frame,
        "per_building": per_building,
    }


def _compute_per_building_metrics(prediction_frame: pd.DataFrame) -> pd.DataFrame:
    working = prediction_frame[PER_BUILDING_KEYS + ["y_true", "y_pred"]].copy()
    errors = working["y_pred"].to_numpy(dtype=np.float64) - working["y_true"].to_numpy(dtype=np.float64)
    working["sum_true"] = working["y_true"].to_numpy(dtype=np.float64)
    working["sum_abs_err"] = np.abs(errors)
    working["sum_sq_err"] = errors ** 2
    grouped = (
        working.groupby(PER_BUILDING_KEYS, sort=False, as_index=False)
        .agg(
            sum_true=("sum_true", "sum"),
            sum_abs_err=("sum_abs_err", "sum"),
            sum_sq_err=("sum_sq_err", "sum"),
            n_rows=("y_true", "size"),
        )
    )
    grouped["mean_true"] = grouped["sum_true"] / grouped["n_rows"]
    grouped["mae"] = grouped["sum_abs_err"] / grouped["n_rows"]
    grouped["rmse"] = np.sqrt(grouped["sum_sq_err"] / grouped["n_rows"])
    grouped["cv_rmse"] = grouped["rmse"] / grouped["mean_true"]
    grouped = grouped.replace([np.inf, -np.inf], np.nan)
    grouped = grouped[grouped["mean_true"] >= 1.0].copy()
    grouped = grouped.dropna(subset=["cv_rmse"]).reset_index(drop=True)
    return grouped


def _summarize_per_building_metrics(per_building: pd.DataFrame) -> dict[str, float | int]:
    values = per_building["cv_rmse"].to_numpy(dtype=np.float64)
    return {
        "n_buildings": int(len(per_building)),
        "mean_per_building_cv_rmse": float(np.mean(values)),
        "median_per_building_cv_rmse": float(np.median(values)),
        "p90_per_building_cv_rmse": float(np.quantile(values, 0.90)),
    }


def _compute_share_best(seed_metrics: pd.DataFrame, per_building: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    grouped = per_building.groupby(["dataset", "split_name", "random_seed"], sort=False)
    for (dataset, split_name, random_seed), group in grouped:
        pivot = group.pivot(index="building_id", columns="model", values="cv_rmse").dropna()
        if pivot.empty:
            continue
        best_model = pivot.idxmin(axis=1)
        shares = best_model.value_counts(normalize=True)
        for model_name in (
            seed_metrics.loc[
                (seed_metrics["dataset"] == dataset)
                & (seed_metrics["split_name"] == split_name)
                & (seed_metrics["random_seed"] == random_seed),
                "model",
            ]
            .drop_duplicates()
            .tolist()
        ):
            rows.append(
                {
                    "dataset": dataset,
                    "split_name": split_name,
                    "model": model_name,
                    "random_seed": int(random_seed),
                    "share_best_buildings": float(shares.get(model_name, 0.0)),
                }
            )
    return pd.DataFrame(rows)


def _summarize_repeat_metrics(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    metric_cols = [
        "mae",
        "rmse",
        "cv_rmse",
        "mean_per_building_cv_rmse",
        "median_per_building_cv_rmse",
        "p90_per_building_cv_rmse",
    ]
    for (dataset, split_name, model), group in seed_metrics.groupby(["dataset", "split_name", "model"], sort=True):
        row: dict[str, object] = {
            "dataset": dataset,
            "split_name": split_name,
            "model": model,
            "n_seeds": int(len(group)),
            "n_buildings": int(group["n_buildings"].iloc[0]),
        }
        for metric in metric_cols:
            values = group[metric].to_numpy(dtype=np.float64)
            mean = float(values.mean())
            std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            half_width = 1.96 * std / np.sqrt(len(values)) if len(values) > 1 else 0.0
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            row[f"{metric}_ci95_low"] = mean - half_width
            row[f"{metric}_ci95_high"] = mean + half_width
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["dataset", "split_name", "model"]).reset_index(drop=True)


def run_repeated_main_metrics(
    dataset: str,
    models: list[str],
    splits: list[str],
    seeds: list[int],
    cpu_threads: int | None,
    device: str | None,
    output_suffix: str = "",
) -> dict[str, Path]:
    ensure_phase2_dirs()
    started_at = time.time()
    frame = _load_full_features(dataset, with_lag_features="lgbm_lag" in models)
    artifacts = _load_split_artifacts(dataset)

    suffix = output_suffix.strip()
    if suffix and not suffix.startswith("_"):
        suffix = f"_{suffix}"

    seed_path = TABLES_DIR / f"repeated_main_metrics_{dataset}{suffix}.csv"
    summary_path = TABLES_DIR / f"repeated_main_metrics_{dataset}{suffix}_summary.csv"
    per_building_path = TABLES_DIR / f"repeated_main_metrics_{dataset}{suffix}_per_building.csv"

    def _write_progress(current_rows: list[dict[str, object]], current_per_building: list[pd.DataFrame]) -> None:
        if not current_rows:
            return
        seed_metrics = pd.DataFrame(current_rows).sort_values(["split_name", "model", "random_seed"]).reset_index(drop=True)
        per_building = (
            pd.concat(current_per_building, ignore_index=True)
            .sort_values(["split_name", "model", "random_seed", "building_id"])
            .reset_index(drop=True)
        )
        share_best = _compute_share_best(seed_metrics, per_building)
        summary = _summarize_repeat_metrics(seed_metrics)
        if not share_best.empty:
            share_summary = (
                share_best.groupby(["dataset", "split_name", "model"], as_index=False)
                .agg(
                    share_best_buildings_mean=("share_best_buildings", "mean"),
                    share_best_buildings_sd=("share_best_buildings", "std"),
                )
            )
            summary = summary.merge(share_summary, on=["dataset", "split_name", "model"], how="left")
        seed_metrics.to_csv(seed_path, index=False)
        per_building.to_csv(per_building_path, index=False)
        summary.to_csv(summary_path, index=False)

    rows: list[dict[str, object]] = []
    per_building_frames: list[pd.DataFrame] = []
    for split_name in splits:
        artifact = artifacts[split_name]
        if split_name != "s_split":
            train_frame = _select_rows(frame, artifact["train_mask"])
            test_frame = _select_rows(frame, artifact["test_mask"])
            try:
                for model_name in models:
                    for seed in seeds:
                        run_output = _evaluate_single_run(
                            dataset=dataset,
                            model_name=model_name,
                            split_name=split_name,
                            random_seed=seed,
                            train_frame=train_frame,
                            test_frame=test_frame,
                            cpu_threads=cpu_threads,
                            device=device,
                        )
                        rows.append(run_output["metrics"])
                        per_building = run_output["per_building"].copy()
                        per_building["dataset"] = dataset
                        per_building["split_name"] = split_name
                        per_building["model"] = model_name
                        per_building["random_seed"] = int(seed)
                        per_building_frames.append(per_building)
                        _write_progress(rows, per_building_frames)
                        del run_output["prediction_frame"]
                        gc.collect()
            finally:
                del train_frame
                del test_frame
                gc.collect()
            continue

        if dataset != "bdg2":
            raise ValueError("S-split repeated evaluation is only defined for BDG2.")

        for model_name in models:
            for seed in seeds:
                fold_predictions: list[pd.DataFrame] = []
                try:
                    for fold in artifact["folds"]:
                        train_frame = _select_rows(frame, fold["train_mask"])
                        test_frame = _select_rows(frame, fold["test_mask"])
                        try:
                            run_output = _evaluate_single_run(
                                dataset=dataset,
                                model_name=model_name,
                                split_name="s_split",
                                random_seed=seed,
                                train_frame=train_frame,
                                test_frame=test_frame,
                                cpu_threads=cpu_threads,
                                device=device,
                                fold_id=str(fold["fold_id"]),
                            )
                            fold_predictions.append(run_output["prediction_frame"].copy())
                            del run_output
                        finally:
                            del train_frame
                            del test_frame
                            gc.collect()

                    combined_prediction = pd.concat(fold_predictions, ignore_index=True)
                    metrics = summarize_metrics(combined_prediction[["y_true", "y_pred"]]).iloc[0].to_dict()
                    per_building = _compute_per_building_metrics(combined_prediction)
                    per_building_summary = _summarize_per_building_metrics(per_building)
                    rows.append(
                        {
                            "dataset": dataset,
                            "split_name": "s_split",
                            "model": model_name,
                            "random_seed": int(seed),
                            **metrics,
                            **per_building_summary,
                        }
                    )
                    per_building = per_building.copy()
                    per_building["dataset"] = dataset
                    per_building["split_name"] = "s_split"
                    per_building["model"] = model_name
                    per_building["random_seed"] = int(seed)
                    per_building_frames.append(per_building)
                    _write_progress(rows, per_building_frames)
                finally:
                    del fold_predictions
                    gc.collect()

    seed_metrics = pd.DataFrame(rows).sort_values(["split_name", "model", "random_seed"]).reset_index(drop=True)
    per_building = (
        pd.concat(per_building_frames, ignore_index=True)
        .sort_values(["split_name", "model", "random_seed", "building_id"])
        .reset_index(drop=True)
    )
    summary = _summarize_repeat_metrics(seed_metrics)
    share_best = _compute_share_best(seed_metrics, per_building)
    if not share_best.empty:
        share_summary = (
            share_best.groupby(["dataset", "split_name", "model"], as_index=False)
            .agg(
                share_best_buildings_mean=("share_best_buildings", "mean"),
                share_best_buildings_sd=("share_best_buildings", "std"),
            )
        )
        summary = summary.merge(share_summary, on=["dataset", "split_name", "model"], how="left")
    seed_metrics.to_csv(seed_path, index=False)
    per_building.to_csv(per_building_path, index=False)
    summary.to_csv(summary_path, index=False)
    LOGGER.info("Repeated main-metric evaluation finished in %.1f seconds", time.time() - started_at)
    return {
        f"repeated_main_metrics_{dataset}": seed_path,
        f"repeated_main_metrics_{dataset}_summary": summary_path,
        f"repeated_main_metrics_{dataset}_per_building": per_building_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run repeated-seed main metrics with confidence intervals.")
    parser.add_argument(
        "--dataset",
        choices=["bdg2", "gepiii"],
        default="bdg2",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["lgbm", "lgbm_lag", "lstm", "patchtst"],
        default=["lgbm", "lstm", "patchtst"],
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["t_split", "b_split", "s_split"],
        default=["t_split", "b_split"],
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[7, 42, 123],
    )
    parser.add_argument(
        "--log-file",
        default=None,
    )
    parser.add_argument(
        "--device",
        default=None,
    )
    parser.add_argument(
        "--max-cpu-threads",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--cpu-fraction",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--memory-fraction",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--output-suffix",
        default="",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    resolved_log = configure_logging(args.log_file)
    if resolved_log is None:
        resolved_log = configure_logging(
            LOGS_DIR / f"repeated_main_metrics_{args.dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
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
    outputs = run_repeated_main_metrics(
        dataset=args.dataset,
        models=args.models,
        splits=args.splits,
        seeds=args.seeds,
        cpu_threads=limits.cpu_threads,
        device=args.device,
        output_suffix=args.output_suffix,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")
