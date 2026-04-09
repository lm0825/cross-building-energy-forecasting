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

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (
    FEATURES_BDG2_PATH,
    FEATURES_GEPIII_PATH,
    LOGS_DIR,
    MODELS_DIR,
    RESULTS_DIR,
    TABLES_DIR,
    ensure_phase2_dirs,
)
from src.data_splitting import B_SPLIT_PATH, T_SPLIT_PATH, load_pickle, unpack_mask
from src.experiment4_gepiii import GEPIII_B_SPLIT_PATH, GEPIII_T_SPLIT_PATH
from src.metrics import summarize_metrics
from src.models.common import save_predictions, set_seed
from src.models.lgbm_model import LightGBMConfig, LightGBMExperimentModel
from src.models.lstm_model import LSTMConfig, LSTMExperimentModel
from src.models.patchtst_model import PatchTSTConfig, PatchTSTExperimentModel
from src.runtime import apply_runtime_limits, format_bytes


LOGGER = logging.getLogger(__name__)

ABLATION_RESULTS_DIR = RESULTS_DIR / "ablations" / "target_transforms"


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


def _load_full_features(dataset: str) -> pd.DataFrame:
    feature_path = FEATURES_BDG2_PATH if dataset == "bdg2" else FEATURES_GEPIII_PATH
    frame = pd.read_parquet(feature_path)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame = frame.sort_values(["building_id", "timestamp"]).reset_index(drop=True)
    LOGGER.info(
        "Loaded dataset=%s rows=%s buildings=%s sites=%s",
        dataset,
        len(frame),
        frame["building_id"].nunique(),
        frame["site_id"].nunique(),
    )
    return frame


def _load_split_artifacts(dataset: str) -> dict[str, dict[str, object]]:
    if dataset == "bdg2":
        return {
            "t_split": load_pickle(T_SPLIT_PATH),
            "b_split": load_pickle(B_SPLIT_PATH),
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
    *,
    split_name: str | None = None,
) -> LightGBMConfig:
    tuning_path = _resolve_tuning_path(dataset, "lgbm", split_name=split_name)
    if tuning_path.exists():
        payload = json.loads(tuning_path.read_text(encoding="utf-8"))
        if cpu_threads is not None:
            payload["n_jobs"] = cpu_threads
        return LightGBMConfig(**payload)
    return LightGBMConfig(n_jobs=cpu_threads) if cpu_threads is not None else LightGBMConfig()


def _load_lstm_config(dataset: str, *, split_name: str | None = None) -> LSTMConfig:
    tuning_path = _resolve_tuning_path(dataset, "lstm", split_name=split_name)
    if tuning_path.exists():
        payload = json.loads(tuning_path.read_text(encoding="utf-8"))
        return LSTMConfig(**payload)
    return LSTMConfig()


def _load_patchtst_config(dataset: str, *, split_name: str | None = None) -> PatchTSTConfig:
    tuning_path = _resolve_tuning_path(dataset, "patchtst", split_name=split_name)
    if tuning_path.exists():
        payload = json.loads(tuning_path.read_text(encoding="utf-8"))
        return PatchTSTConfig(**payload)
    return PatchTSTConfig()


def _build_prediction_frame_for_lgbm(
    model: LightGBMExperimentModel,
    test_frame: pd.DataFrame,
    split_name: str,
) -> pd.DataFrame:
    return model.predict_frame(
        test_frame,
        split_name=split_name,
        model_name="lgbm",
    ).drop(columns=["fold_id"])


def _run_single(
    dataset: str,
    model_name: str,
    split_name: str,
    target_transform: str,
    random_seed: int,
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    cpu_threads: int | None,
    device: str | None,
) -> pd.DataFrame:
    if model_name == "lgbm":
        payload = asdict(_load_lgbm_config(dataset, cpu_threads=cpu_threads, split_name=split_name))
        payload["target_transform"] = target_transform
        payload["random_state"] = random_seed
        if cpu_threads is not None:
            payload["n_jobs"] = cpu_threads
        model = LightGBMExperimentModel(config=LightGBMConfig(**payload))
        model.fit(train_frame)
        return _build_prediction_frame_for_lgbm(model, test_frame, split_name)

    if model_name == "lstm":
        payload = asdict(_load_lstm_config(dataset, split_name=split_name))
        payload["target_transform"] = target_transform
        payload["random_seed"] = random_seed
        model = LSTMExperimentModel(config=LSTMConfig(**payload), device=device)
        model.fit(train_frame)
        return model.predict_frame(test_frame, split_name=split_name)

    if model_name == "patchtst":
        payload = asdict(_load_patchtst_config(dataset, split_name=split_name))
        payload["target_transform"] = target_transform
        payload["random_seed"] = random_seed
        model = PatchTSTExperimentModel(config=PatchTSTConfig(**payload), device=device)
        model.fit(train_frame)
        return model.predict_frame(test_frame, split_name=split_name)

    raise ValueError(f"Unsupported model name: {model_name}")


def run_ablation(
    dataset: str,
    models: list[str],
    splits: list[str],
    target_transforms: list[str],
    random_seed: int,
    cpu_threads: int | None,
    device: str | None,
) -> dict[str, Path]:
    ensure_phase2_dirs()
    set_seed(random_seed)
    started_at = time.time()
    frame = _load_full_features(dataset)
    artifacts = _load_split_artifacts(dataset)

    output_dir = ABLATION_RESULTS_DIR / dataset
    prediction_dir = output_dir / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for split_name in splits:
        artifact = artifacts[split_name]
        train_frame = _select_rows(frame, artifact["train_mask"])
        test_frame = _select_rows(frame, artifact["test_mask"])
        try:
            for model_name in models:
                for target_transform in target_transforms:
                    LOGGER.info(
                        "Running ablation dataset=%s split=%s model=%s target_transform=%s seed=%s",
                        dataset,
                        split_name,
                        model_name,
                        target_transform,
                        random_seed,
                    )
                    prediction_frame = _run_single(
                        dataset=dataset,
                        model_name=model_name,
                        split_name=split_name,
                        target_transform=target_transform,
                        random_seed=random_seed,
                        train_frame=train_frame,
                        test_frame=test_frame,
                        cpu_threads=cpu_threads,
                        device=device,
                    )
                    pred_path = prediction_dir / f"{model_name}_{split_name}_{target_transform}.csv"
                    save_predictions(prediction_frame, pred_path)
                    metrics = summarize_metrics(prediction_frame[["y_true", "y_pred"]]).iloc[0].to_dict()
                    rows.append(
                        {
                            "dataset": dataset,
                            "split_name": split_name,
                            "model": model_name,
                            "target_transform": target_transform,
                            "random_seed": int(random_seed),
                            **metrics,
                            "prediction_path": str(pred_path),
                        }
                    )
                    del prediction_frame
                    gc.collect()
        finally:
            del train_frame
            del test_frame
            gc.collect()

    result_frame = pd.DataFrame(rows).sort_values(
        ["split_name", "model", "target_transform"]
    ).reset_index(drop=True)
    result_path = TABLES_DIR / f"ablation_target_transforms_{dataset}.csv"
    result_frame.to_csv(result_path, index=False)

    pivot = result_frame.pivot(
        index=["split_name", "model"],
        columns="target_transform",
        values=["cv_rmse", "rmse", "mae"],
    ).sort_index(axis=1)
    pivot.columns = [f"{metric}_{transform}" for metric, transform in pivot.columns]
    pivot = pivot.reset_index()
    if "cv_rmse_log1p" in pivot.columns and "cv_rmse_minmax" in pivot.columns:
        pivot["cv_rmse_delta_log1p_minus_minmax"] = pivot["cv_rmse_log1p"] - pivot["cv_rmse_minmax"]
    if "rmse_log1p" in pivot.columns and "rmse_minmax" in pivot.columns:
        pivot["rmse_delta_log1p_minus_minmax"] = pivot["rmse_log1p"] - pivot["rmse_minmax"]
    if "mae_log1p" in pivot.columns and "mae_minmax" in pivot.columns:
        pivot["mae_delta_log1p_minus_minmax"] = pivot["mae_log1p"] - pivot["mae_minmax"]
    pivot_path = TABLES_DIR / f"ablation_target_transforms_{dataset}_pivot.csv"
    pivot.to_csv(pivot_path, index=False)

    LOGGER.info("Target-transform ablation finished in %.1f seconds", time.time() - started_at)
    return {
        f"ablation_target_transforms_{dataset}": result_path,
        f"ablation_target_transforms_{dataset}_pivot": pivot_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run target-transform ablations for selected models.")
    parser.add_argument(
        "--dataset",
        choices=["bdg2", "gepiii"],
        default="bdg2",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["lgbm", "lstm", "patchtst"],
        default=["lgbm", "lstm", "patchtst"],
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["t_split", "b_split"],
        default=["t_split", "b_split"],
    )
    parser.add_argument(
        "--target-transforms",
        nargs="+",
        choices=["minmax", "log1p"],
        default=["minmax", "log1p"],
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    resolved_log = configure_logging(args.log_file)
    if resolved_log is None:
        resolved_log = configure_logging(
            LOGS_DIR / f"ablation_target_transforms_{args.dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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
    outputs = run_ablation(
        dataset=args.dataset,
        models=args.models,
        splits=args.splits,
        target_transforms=args.target_transforms,
        random_seed=args.random_seed,
        cpu_threads=limits.cpu_threads,
        device=args.device,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")
