from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import gc
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
    EXP1_PREDICTIONS_DIR,
    LOGS_DIR,
    MODELS_DIR,
    SPLITS_DIR,
    ensure_phase2_dirs,
)
from src.data_splitting import (
    B_SPLIT_PATH,
    S_SPLIT_PATH,
    T_SPLIT_PATH,
    load_pickle,
    save_split_artifacts,
    unpack_mask,
)
from src.metrics import compute_exp1_outputs
from src.models.common import (
    add_tabular_lag_features,
    default_model_path,
    save_json,
    save_predictions,
    set_seed,
)
from src.models.lgbm_model import LightGBMConfig, LightGBMExperimentModel
from src.models.lstm_model import LSTMConfig, LSTMExperimentModel
from src.models.patchtst_model import PatchTSTExperimentModel
from src.runtime import apply_runtime_limits, format_bytes
from src.site_analysis import run_site_analysis

LOGGER = logging.getLogger(__name__)


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
    LOGGER.info("Loading features from %s", FEATURES_BDG2_PATH)
    frame = pd.read_parquet(FEATURES_BDG2_PATH)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame = frame.sort_values(["building_id", "timestamp"]).reset_index(drop=True)
    if with_lag_features:
        frame = add_tabular_lag_features(frame)
    LOGGER.info(
        "Loaded feature frame: rows=%s buildings=%s sites=%s date_range=%s to %s lag_features=%s",
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


def _fit_and_predict_model(
    model_name: str,
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    split_name: str,
    fold_id: str | None = None,
    model_config: object | None = None,
    device: str | None = None,
    cpu_threads: int | None = None,
) -> pd.DataFrame:
    run_label = split_name if fold_id is None else f"{split_name}:{fold_id}"
    LOGGER.info(
        "Starting model=%s run=%s train_rows=%s test_rows=%s train_buildings=%s test_buildings=%s",
        model_name,
        run_label,
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
            fold_id=fold_id,
            model_name=model_name,
        )
        model_path = default_model_path(model_name, split_name if fold_id is None else f"{split_name}_{fold_id}", "txt")
        model.save(model_path)
        LOGGER.info("Finished model=%s run=%s saved_model=%s", model_name, run_label, model_path)
        return prediction_frame

    if model_name == "lstm":
        model = LSTMExperimentModel(
            config=None if model_config is None else type(model_config)(**asdict(model_config)),
            device=device,
        )
        model.fit(train_frame)
        model_path = default_model_path(model_name, split_name if fold_id is None else f"{split_name}_{fold_id}", "pt")
        model.save(model_path)
        LOGGER.info("Finished model=%s run=%s saved_model=%s", model_name, run_label, model_path)
        return model.predict_frame(test_frame, split_name=split_name, fold_id=fold_id)

    if model_name == "patchtst":
        model = PatchTSTExperimentModel(
            config=None if model_config is None else type(model_config)(**asdict(model_config)),
            device=device,
        )
        model.fit(train_frame)
        model_path = default_model_path(model_name, split_name if fold_id is None else f"{split_name}_{fold_id}", "pt")
        model.save(model_path)
        LOGGER.info("Finished model=%s run=%s saved_model=%s", model_name, run_label, model_path)
        return model.predict_frame(test_frame, split_name=split_name, fold_id=fold_id)

    raise ValueError(f"Unsupported model name: {model_name}")


def _save_tuned_config(
    model_name: str,
    config: object,
    *,
    split_name: str,
    fold_id: str | None = None,
) -> Path:
    tuning_dir = MODELS_DIR / "_tuning"
    tuning_dir.mkdir(parents=True, exist_ok=True)
    suffix = f".{split_name}"
    if fold_id is not None:
        suffix = f"{suffix}.{fold_id}"
    path = tuning_dir / f"{model_name}{suffix}.config.json"
    save_json(asdict(config), path)
    if split_name == "t_split" and fold_id is None:
        save_json(asdict(config), tuning_dir / f"{model_name}.config.json")
    return path


def _tune_model_config(
    model_name: str,
    train_frame: pd.DataFrame,
    *,
    split_name: str,
    fold_id: str | None,
    device: str | None,
    cpu_threads: int | None,
) -> tuple[object, Path]:
    tuning_label = split_name if fold_id is None else f"{split_name}:{fold_id}"
    LOGGER.info(
        "Tuning model=%s on train partition for run=%s rows=%s buildings=%s",
        model_name,
        tuning_label,
        len(train_frame),
        train_frame["building_id"].nunique(),
    )
    if model_name in {"lgbm", "lgbm_lag"}:
        config = LightGBMConfig(n_jobs=cpu_threads) if cpu_threads is not None else LightGBMConfig()
        config.target_transform = "log1p"
        config.use_lag_features = model_name == "lgbm_lag"
        model = LightGBMExperimentModel(config=config)
    elif model_name == "lstm":
        model = LSTMExperimentModel(
            config=LSTMConfig(target_transform="log1p"),
            device=device,
        )
    elif model_name == "patchtst":
        model = PatchTSTExperimentModel(device=device)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    config = model.tune_on_tsplit(train_frame)
    frozen_config = type(config)(**asdict(config))
    config_path = _save_tuned_config(
        model_name,
        frozen_config,
        split_name=split_name,
        fold_id=fold_id,
    )
    LOGGER.info("Saved tuned config model=%s run=%s path=%s values=%s", model_name, tuning_label, config_path, asdict(frozen_config))
    return frozen_config, config_path


def run_experiment1(
    models: list[str] | None = None,
    splits: list[str] | None = None,
    skip_training: bool = False,
    device: str | None = None,
    cpu_threads: int | None = None,
) -> dict[str, Path]:
    ensure_phase2_dirs()
    set_seed()
    started_at = time.time()
    if not T_SPLIT_PATH.exists() or not B_SPLIT_PATH.exists() or not S_SPLIT_PATH.exists():
        LOGGER.info("Split artifacts missing. Building fresh split files.")
        save_split_artifacts()
    else:
        LOGGER.info("Using existing split artifacts from %s", SPLITS_DIR)

    models = models or ["lgbm", "lstm", "patchtst"]
    splits = splits or ["t_split", "b_split", "s_split"]
    LOGGER.info("Experiment config models=%s splits=%s skip_training=%s", models, splits, skip_training)
    frame = _load_full_features(with_lag_features="lgbm_lag" in models)

    outputs: dict[str, Path] = {}
    artifacts = {
        "t_split": load_pickle(T_SPLIT_PATH),
        "b_split": load_pickle(B_SPLIT_PATH),
        "s_split": load_pickle(S_SPLIT_PATH),
    }

    if skip_training:
        outputs.update(
            {
                "t_split": T_SPLIT_PATH,
                "b_split": B_SPLIT_PATH,
                "s_split": S_SPLIT_PATH,
            }
        )
        LOGGER.info("Skip-training mode finished. Split artifacts are ready.")
        return outputs

    for split_name in splits:
        if split_name in {"t_split", "b_split"}:
            split_artifact = artifacts[split_name]
            train_frame = _select_rows(frame, split_artifact["train_mask"])
            test_frame = _select_rows(frame, split_artifact["test_mask"])
            LOGGER.info(
                "Prepared split=%s train_rows=%s test_rows=%s",
                split_name,
                len(train_frame),
                len(test_frame),
            )
            for model_name in models:
                model_config, config_path = _tune_model_config(
                    model_name=model_name,
                    train_frame=train_frame,
                    split_name=split_name,
                    fold_id=None,
                    device=device,
                    cpu_threads=cpu_threads,
                )
                outputs[f"{model_name}_{split_name}_tuning_config"] = config_path
                prediction_frame = _fit_and_predict_model(
                    model_name=model_name,
                    train_frame=train_frame,
                    test_frame=test_frame,
                    split_name=split_name,
                    model_config=model_config,
                    device=device,
                    cpu_threads=cpu_threads,
                )
                path = EXP1_PREDICTIONS_DIR / f"{model_name}_{split_name}.csv"
                save_predictions(prediction_frame, path)
                outputs[f"{model_name}_{split_name}"] = path
                LOGGER.info("Saved predictions to %s", path)
                del prediction_frame
                gc.collect()
            del train_frame
            del test_frame
            gc.collect()
        elif split_name == "s_split":
            split_artifact = artifacts["s_split"]
            LOGGER.info("Prepared split=s_split folds=%s", len(split_artifact["folds"]))
            for model_name in models:
                frames: list[pd.DataFrame] = []
                for fold in split_artifact["folds"]:
                    LOGGER.info("Running S-split fold=%s model=%s", fold["fold_id"], model_name)
                    train_frame = _select_rows(frame, fold["train_mask"])
                    test_frame = _select_rows(frame, fold["test_mask"])
                    model_config, config_path = _tune_model_config(
                        model_name=model_name,
                        train_frame=train_frame,
                        split_name="s_split",
                        fold_id=str(fold["fold_id"]),
                        device=device,
                        cpu_threads=cpu_threads,
                    )
                    outputs[f"{model_name}_s_split_{fold['fold_id']}_tuning_config"] = config_path
                    frames.append(
                        _fit_and_predict_model(
                            model_name=model_name,
                            train_frame=train_frame,
                            test_frame=test_frame,
                            split_name="s_split",
                            fold_id=str(fold["fold_id"]),
                            model_config=model_config,
                            device=device,
                            cpu_threads=cpu_threads,
                        )
                    )
                    del train_frame
                    del test_frame
                    gc.collect()
                combined = pd.concat(frames, ignore_index=True)
                path = EXP1_PREDICTIONS_DIR / f"{model_name}_s_split.csv"
                save_predictions(combined, path)
                outputs[f"{model_name}_s_split"] = path
                LOGGER.info("Saved combined S-split predictions to %s", path)
                del combined
                del frames
                gc.collect()
        else:
            raise ValueError(f"Unknown split: {split_name}")

    metric_outputs = compute_exp1_outputs(EXP1_PREDICTIONS_DIR, s_split_path=S_SPLIT_PATH)
    outputs.update(metric_outputs)
    LOGGER.info("Metric artifacts: %s", {key: str(value) for key, value in metric_outputs.items()})
    site_outputs = run_site_analysis()
    outputs.update(site_outputs)
    LOGGER.info("Site-analysis artifacts: %s", {key: str(value) for key, value in site_outputs.items()})
    LOGGER.info("Experiment finished in %.1f seconds", time.time() - started_at)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run experiment 1 for cross-building generalization.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["lgbm", "lstm", "patchtst"],
        choices=["lgbm", "lgbm_lag", "lstm", "patchtst"],
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["t_split", "b_split", "s_split"],
        choices=["t_split", "b_split", "s_split"],
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Only prepare directories and split artifacts without training models.",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Optional log file path. Relative paths are resolved from the repository root.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device for neural models, for example cuda:1 or cpu. LightGBM remains CPU-based.",
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
        help="Cap CPU thread usage as a fraction of total logical CPUs, for example 0.5.",
    )
    parser.add_argument(
        "--memory-fraction",
        type=float,
        default=None,
        help="Cap this process to a fraction of total system memory on supported platforms, for example 0.5.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    resolved_log = configure_logging(args.log_file)
    if resolved_log is None:
        default_log = LOGS_DIR / f"exp1_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        resolved_log = configure_logging(default_log)
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
    outputs = run_experiment1(
        models=args.models,
        splits=args.splits,
        skip_training=args.skip_training,
        device=args.device,
        cpu_threads=limits.cpu_threads,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")
