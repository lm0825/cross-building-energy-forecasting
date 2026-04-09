from __future__ import annotations

import argparse
import gc
from dataclasses import asdict
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import pickle
import re
import sys
import time

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, silhouette_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clustering.feature_extractor import (
    PROFILE_FEATURE_COLUMNS,
    extract_building_profile_features,
    fit_profile_scaler,
    save_train_feature_artifacts,
    transform_profile_features,
)
from src.config import (
    CLUSTERING_DIR,
    EXP2_PREDICTIONS_DIR,
    FEATURES_BDG2_PATH,
    FIGURES_DIR,
    LOGS_DIR,
    MODELS_DIR,
    RANDOM_SEED,
    RESULTS_DIR,
    TABLES_DIR,
    ensure_phase2_dirs,
)
from src.data_splitting import B_SPLIT_PATH, load_pickle, unpack_mask
from src.metrics import summarize_metrics
from src.models.common import add_tabular_lag_features, save_predictions, set_seed
from src.models.lgbm_model import LightGBMConfig, LightGBMExperimentModel
from src.models.lstm_model import LSTMConfig, LSTMExperimentModel
from src.models.patchtst_model import PatchTSTConfig, PatchTSTExperimentModel
from src.runtime import apply_runtime_limits, format_bytes


KMEANS_K_METRICS_PATH = TABLES_DIR / "kmeans_k_metrics.csv"
KMEANS_ELBOW_PATH = FIGURES_DIR / "exp2_kmeans_elbow.png"
KMEANS_MODEL_PATH = CLUSTERING_DIR / "kmeans_model.pkl"
TRAIN_CLUSTER_LABELS_PATH = CLUSTERING_DIR / "train_building_cluster_labels.csv"
TEST_CLUSTER_ASSIGNMENTS_PATH = CLUSTERING_DIR / "test_building_cluster_assignments.csv"
EXP2_STRATEGY_COMPARISON_PATH = RESULTS_DIR / "exp2_strategy_comparison.csv"
EXP2_GROUPING_ACCURACY_PATH = TABLES_DIR / "exp2_grouping_accuracy.csv"
EXP2_COLD_START_METRICS_PATH = TABLES_DIR / "exp2_cold_start_metrics.csv"
EXP2_ACCURACY_FIG_PATH = FIGURES_DIR / "exp2_accuracy_vs_N.png"

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


def _load_feature_frame(with_lag_features: bool = False) -> pd.DataFrame:
    frame = pd.read_parquet(FEATURES_BDG2_PATH)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame = frame.sort_values(["building_id", "timestamp"]).reset_index(drop=True)
    if with_lag_features:
        frame = add_tabular_lag_features(frame)
    return frame


def _select_rows(frame: pd.DataFrame, packed_mask: dict[str, object]) -> pd.DataFrame:
    mask = unpack_mask(packed_mask)
    return frame.loc[mask].reset_index(drop=True)


def _slugify(value: object) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", str(value)).strip("_").lower()
    return slug or "group"


def _resolve_model_dir(*parts: str) -> Path:
    base = MODELS_DIR / "exp2"
    if base.exists() and not os.access(base, os.W_OK):
        base = MODELS_DIR / "_rerun" / "exp2"
    path = base.joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _resolve_tuning_path(
    model_name: str,
    *,
    split_name: str | None = None,
    fold_id: str | None = None,
) -> Path:
    tuning_dir = MODELS_DIR / "_tuning"
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
    cpu_threads: int | None = None,
    *,
    model_name: str = "lgbm",
    split_name: str | None = None,
) -> LightGBMConfig:
    tuning_path = _resolve_tuning_path(model_name, split_name=split_name)
    if tuning_path.exists():
        payload = json.loads(tuning_path.read_text(encoding="utf-8"))
        payload["target_transform"] = "log1p"
        if cpu_threads is not None:
            payload["n_jobs"] = cpu_threads
        return LightGBMConfig(**payload)
    config = (
        LightGBMConfig(n_jobs=cpu_threads, target_transform="log1p")
        if cpu_threads is not None
        else LightGBMConfig(target_transform="log1p")
    )
    config.use_lag_features = model_name == "lgbm_lag"
    return config


def _load_lstm_config(*, split_name: str | None = None) -> LSTMConfig:
    tuning_path = _resolve_tuning_path("lstm", split_name=split_name)
    if tuning_path.exists():
        payload = json.loads(tuning_path.read_text(encoding="utf-8"))
        return LSTMConfig(**payload)
    return LSTMConfig()


def _load_patchtst_config(*, split_name: str | None = None) -> PatchTSTConfig:
    tuning_path = _resolve_tuning_path("patchtst", split_name=split_name)
    if tuning_path.exists():
        payload = json.loads(tuning_path.read_text(encoding="utf-8"))
        return PatchTSTConfig(**payload)
    return PatchTSTConfig()


def _fit_experiment_model(
    model_name: str,
    train_frame: pd.DataFrame,
    model_config: object | None,
    cpu_threads: int | None = None,
    device: str | None = None,
) -> LightGBMExperimentModel | LSTMExperimentModel | PatchTSTExperimentModel:
    if model_name in {"lgbm", "lgbm_lag"}:
        config = (
            LightGBMConfig(**asdict(model_config))
            if model_config is not None
            else _load_lgbm_config(cpu_threads=cpu_threads, model_name=model_name, split_name="b_split")
        )
        config.target_transform = "log1p"
        config.use_lag_features = model_name == "lgbm_lag"
        model = LightGBMExperimentModel(config=config)
        model.fit(train_frame)
        return model

    if model_name == "lstm":
        config = (
            LSTMConfig(**asdict(model_config))
            if model_config is not None
            else _load_lstm_config(split_name="b_split")
        )
        model = LSTMExperimentModel(config=config, device=device)
        model.fit(train_frame)
        return model

    if model_name == "patchtst":
        config = (
            PatchTSTConfig(**asdict(model_config))
            if model_config is not None
            else _load_patchtst_config(split_name="b_split")
        )
        model = PatchTSTExperimentModel(config=config, device=device)
        model.fit(train_frame)
        return model

    raise ValueError(f"Unsupported model name: {model_name}")


def _prediction_frame(
    frame: pd.DataFrame,
    predictions: np.ndarray,
    strategy: str,
    assigned_group: pd.Series,
    fallback_to_all_mix: pd.Series | None = None,
) -> pd.DataFrame:
    output = frame[
        ["building_id", "site_id", "building_type", "timestamp", "meter_reading"]
    ].copy()
    output["strategy"] = strategy
    output["assigned_group"] = assigned_group.astype(str).to_numpy()
    output["fallback_to_all_mix"] = (
        fallback_to_all_mix.astype(bool).to_numpy()
        if fallback_to_all_mix is not None
        else False
    )
    output["y_true"] = output["meter_reading"]
    output["y_pred"] = predictions.astype(np.float32)
    return output.drop(columns=["meter_reading"])


def _prediction_frame_for_model(
    model_name: str,
    model: LightGBMExperimentModel | LSTMExperimentModel | PatchTSTExperimentModel,
    frame: pd.DataFrame,
    strategy: str,
    assigned_group: pd.Series,
    fallback_to_all_mix: pd.Series | None = None,
) -> pd.DataFrame:
    if model_name in {"lgbm", "lgbm_lag"}:
        building_lookup = pd.DataFrame(
            {
                "building_id": frame["building_id"].astype(str).to_numpy(),
                "assigned_group": assigned_group.astype(str).to_numpy(),
                "fallback_to_all_mix": (
                    fallback_to_all_mix.astype(bool).to_numpy()
                    if fallback_to_all_mix is not None
                    else False
                ),
            }
        ).drop_duplicates(subset=["building_id"])
        prediction_frame = model.predict_frame(
            frame,
            split_name="b_split",
            model_name=model_name,
        ).copy()
        prediction_frame["building_id"] = prediction_frame["building_id"].astype(str)
        prediction_frame = prediction_frame.merge(
            building_lookup,
            on="building_id",
            how="left",
            validate="many_to_one",
        )
        prediction_frame["strategy"] = strategy
        return prediction_frame[
            [
                "building_id",
                "site_id",
                "building_type",
                "timestamp",
                "strategy",
                "assigned_group",
                "fallback_to_all_mix",
                "y_true",
                "y_pred",
            ]
        ].copy()

    if model_name == "lstm":
        building_lookup = pd.DataFrame(
            {
                "building_id": frame["building_id"].astype(str).to_numpy(),
                "assigned_group": assigned_group.astype(str).to_numpy(),
                "fallback_to_all_mix": (
                    fallback_to_all_mix.astype(bool).to_numpy()
                    if fallback_to_all_mix is not None
                    else False
                ),
            }
        ).drop_duplicates(subset=["building_id"])
        prediction_frame = model.predict_frame(frame, split_name="b_split").copy()
        prediction_frame["building_id"] = prediction_frame["building_id"].astype(str)
        prediction_frame = prediction_frame.merge(
            building_lookup,
            on="building_id",
            how="left",
            validate="many_to_one",
        )
        prediction_frame["strategy"] = strategy
        return prediction_frame[
            [
                "building_id",
                "site_id",
                "building_type",
                "timestamp",
                "strategy",
                "assigned_group",
                "fallback_to_all_mix",
                "y_true",
                "y_pred",
            ]
        ].copy()

    if model_name == "patchtst":
        building_lookup = pd.DataFrame(
            {
                "building_id": frame["building_id"].astype(str).to_numpy(),
                "assigned_group": assigned_group.astype(str).to_numpy(),
                "fallback_to_all_mix": (
                    fallback_to_all_mix.astype(bool).to_numpy()
                    if fallback_to_all_mix is not None
                    else False
                ),
            }
        ).drop_duplicates(subset=["building_id"])
        prediction_frame = model.predict_frame(frame, split_name="b_split").copy()
        prediction_frame["building_id"] = prediction_frame["building_id"].astype(str)
        prediction_frame = prediction_frame.merge(
            building_lookup,
            on="building_id",
            how="left",
            validate="many_to_one",
        )
        prediction_frame["strategy"] = strategy
        return prediction_frame[
            [
                "building_id",
                "site_id",
                "building_type",
                "timestamp",
                "strategy",
                "assigned_group",
                "fallback_to_all_mix",
                "y_true",
                "y_pred",
            ]
        ].copy()

    raise ValueError(f"Unsupported model name: {model_name}")


def _evaluate_k_grid(
    scaled_features: np.ndarray,
    k_values: list[int],
) -> tuple[pd.DataFrame, int]:
    rows: list[dict[str, object]] = []
    for k in k_values:
        model = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=20)
        labels = model.fit_predict(scaled_features)
        rows.append(
            {
                "k": int(k),
                "inertia": float(model.inertia_),
                "silhouette_score": float(silhouette_score(scaled_features, labels)),
                "calinski_harabasz_score": float(calinski_harabasz_score(scaled_features, labels)),
            }
        )

    metrics = pd.DataFrame(rows).sort_values("k").reset_index(drop=True)
    best = metrics.sort_values(
        ["silhouette_score", "calinski_harabasz_score", "k"],
        ascending=[False, False, True],
    ).iloc[0]
    selected_k = int(best["k"])
    metrics["selected"] = metrics["k"] == selected_k
    return metrics, selected_k


def _plot_kmeans_elbow(metrics: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(metrics["k"], metrics["inertia"], marker="o", linewidth=2)
    plt.xlabel("K")
    plt.ylabel("Inertia")
    plt.title("Experiment 2 K-means Elbow Curve")
    plt.xticks(metrics["k"])
    selected = metrics.loc[metrics["selected"]].iloc[0]
    plt.scatter([selected["k"]], [selected["inertia"]], color="#d1495b", zorder=3)
    plt.tight_layout()
    plt.savefig(KMEANS_ELBOW_PATH, dpi=200)
    plt.close()


def _fit_kmeans_model(
    scaled_train_features: np.ndarray,
    train_feature_frame: pd.DataFrame,
    selected_k: int,
) -> tuple[KMeans, pd.DataFrame]:
    model = KMeans(n_clusters=selected_k, random_state=RANDOM_SEED, n_init=20)
    labels = model.fit_predict(scaled_train_features)
    label_frame = train_feature_frame[["building_id"]].copy()
    label_frame["cluster_label"] = labels.astype(int)
    return model, label_frame


def _save_kmeans_artifacts(model: KMeans, label_frame: pd.DataFrame, selected_k: int) -> dict[str, Path]:
    payload = {
        "model": model,
        "selected_k": int(selected_k),
        "feature_columns": PROFILE_FEATURE_COLUMNS,
        "random_seed": int(RANDOM_SEED),
    }
    with KMEANS_MODEL_PATH.open("wb") as file:
        pickle.dump(payload, file, protocol=pickle.HIGHEST_PROTOCOL)
    label_frame.to_csv(TRAIN_CLUSTER_LABELS_PATH, index=False)
    return {
        "kmeans_model": KMEANS_MODEL_PATH,
        "train_building_cluster_labels": TRAIN_CLUSTER_LABELS_PATH,
    }


def _assign_test_clusters(
    test_frame: pd.DataFrame,
    model: KMeans,
    scaler,
) -> pd.DataFrame:
    full_feature_frame = extract_building_profile_features(test_frame)
    scaled_full = transform_profile_features(full_feature_frame, scaler)
    full_assignments = pd.DataFrame(
        {
            "building_id": full_feature_frame["building_id"],
            "oracle_cluster_label": model.predict(scaled_full).astype(int),
        }
    )

    rows: list[pd.DataFrame] = [full_assignments]
    for days in [3, 7, 14]:
        history = _initial_history_slice(test_frame, days)
        history_features = extract_building_profile_features(history)
        scaled_history = transform_profile_features(history_features, scaler)
        partial = pd.DataFrame(
            {
                "building_id": history_features["building_id"],
                f"cold_start_cluster_{days}d": model.predict(scaled_history).astype(int),
            }
        )
        rows.append(partial)

    assignments = rows[0]
    for partial in rows[1:]:
        assignments = assignments.merge(partial, on="building_id", how="left")
    assignments.to_csv(TEST_CLUSTER_ASSIGNMENTS_PATH, index=False)
    return assignments


def _fit_group_models(
    train_frame: pd.DataFrame,
    group_assignments: pd.Series,
    model_name: str,
    model_config: object | None,
    strategy_dir: str,
    cpu_threads: int | None = None,
    device: str | None = None,
    ) -> dict[str, LightGBMExperimentModel | LSTMExperimentModel | PatchTSTExperimentModel]:
    grouped_frame = train_frame.copy()
    grouped_frame["_group_key"] = group_assignments.astype(str).to_numpy()
    models: dict[str, LightGBMExperimentModel | LSTMExperimentModel | PatchTSTExperimentModel] = {}
    for group_key, subset in grouped_frame.groupby("_group_key", sort=True):
        model = _fit_experiment_model(
            model_name=model_name,
            train_frame=subset.drop(columns="_group_key"),
            model_config=model_config,
            cpu_threads=cpu_threads,
            device=device,
        )
        group_dir = _resolve_model_dir(strategy_dir, _slugify(group_key))
        if model_name in {"lgbm", "lgbm_lag"}:
            model.save(group_dir / f"{model_name}.txt")
        elif model_name == "lstm":
            model.save(group_dir / "lstm.pt")
        elif model_name == "patchtst":
            model.save(group_dir / "patchtst.pt")
        else:
            raise ValueError(f"Unsupported model name: {model_name}")
        models[str(group_key)] = model
        LOGGER.info(
            "Experiment 2 model=%s strategy=%s group=%s rows=%s buildings=%s",
            model_name,
            strategy_dir,
            group_key,
            len(subset),
            subset["building_id"].nunique(),
        )
        gc.collect()
    return models


def _predict_group_strategy(
    model_name: str,
    test_frame: pd.DataFrame,
    assignments: pd.Series,
    strategy: str,
    models: dict[str, LightGBMExperimentModel | LSTMExperimentModel | PatchTSTExperimentModel],
    all_mix_model: LightGBMExperimentModel | LSTMExperimentModel | PatchTSTExperimentModel,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    assigned = assignments.astype(str).reset_index(drop=True)
    test_with_assignments = test_frame.copy()
    test_with_assignments["_assigned_group"] = assigned.to_numpy()
    for group_key, subset in test_with_assignments.groupby("_assigned_group", sort=False):
        model = models.get(str(group_key), all_mix_model)
        fallback = str(group_key) not in models
        frames.append(
            _prediction_frame_for_model(
                model_name=model_name,
                model=model,
                frame=subset.drop(columns="_assigned_group"),
                strategy=strategy,
                assigned_group=pd.Series([group_key] * len(subset)),
                fallback_to_all_mix=pd.Series([fallback] * len(subset)),
            )
        )
        gc.collect()
    return pd.concat(frames, ignore_index=True).sort_values(["building_id", "timestamp"]).reset_index(drop=True)


def _initial_history_slice(frame: pd.DataFrame, n_days: int) -> pd.DataFrame:
    starts = frame.groupby("building_id")["timestamp"].transform("min")
    cutoff = starts + pd.to_timedelta(n_days, unit="D")
    return frame[frame["timestamp"] < cutoff].copy()


def _future_slice(frame: pd.DataFrame, n_days: int) -> pd.DataFrame:
    starts = frame.groupby("building_id")["timestamp"].transform("min")
    cutoff = starts + pd.to_timedelta(n_days, unit="D")
    return frame[frame["timestamp"] >= cutoff].copy()


def _subset_prediction_frame(prediction_frame: pd.DataFrame, subset_frame: pd.DataFrame) -> pd.DataFrame:
    keys = subset_frame[["building_id", "timestamp"]].copy()
    merged = keys.merge(
        prediction_frame,
        on=["building_id", "timestamp"],
        how="left",
        validate="many_to_one",
    )
    return merged


def _plot_cold_start_accuracy(accuracy_frame: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(accuracy_frame["history_days"], accuracy_frame["accuracy"], marker="o", linewidth=2)
    for _, row in accuracy_frame.iterrows():
        plt.text(row["history_days"], row["accuracy"], f"{row['accuracy']:.3f}", ha="center", va="bottom")
    plt.xlabel("History Days")
    plt.ylabel("Cluster Assignment Accuracy")
    plt.title("Cold-start Cluster Accuracy vs History Length")
    plt.xticks(accuracy_frame["history_days"])
    plt.ylim(0.0, 1.05)
    plt.tight_layout()
    plt.savefig(EXP2_ACCURACY_FIG_PATH, dpi=200)
    plt.close()


def _compute_cold_start_outputs(
    model_name: str,
    test_frame: pd.DataFrame,
    all_mix_predictions: pd.DataFrame,
    cluster_models: dict[str, LightGBMExperimentModel | LSTMExperimentModel | PatchTSTExperimentModel],
    all_mix_model: LightGBMExperimentModel | LSTMExperimentModel | PatchTSTExperimentModel,
    assignment_frame: pd.DataFrame,
) -> dict[str, Path]:
    accuracy_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []

    oracle_assignments = assignment_frame.set_index("building_id")["oracle_cluster_label"].astype(str)
    cold_assignment_lookup = assignment_frame.set_index("building_id")

    for history_days in [3, 7, 14]:
        assignment_col = f"cold_start_cluster_{history_days}d"
        assigned = cold_assignment_lookup[assignment_col].astype(str)
        comparable = oracle_assignments.index.intersection(assigned.index)
        accuracy = float((oracle_assignments.loc[comparable] == assigned.loc[comparable]).mean())
        accuracy_rows.append(
            {
                "history_days": int(history_days),
                "n_buildings": int(len(comparable)),
                "accuracy": accuracy,
            }
        )

        future = _future_slice(test_frame, history_days)
        future_assignments = future["building_id"].map(assigned)
        cold_predictions = _predict_group_strategy(
            model_name=model_name,
            test_frame=future,
            assignments=future_assignments,
            strategy=f"cold_start_cluster_group_{history_days}d",
            models=cluster_models,
            all_mix_model=all_mix_model,
        )
        cold_path = EXP2_PREDICTIONS_DIR / f"cold_start_cluster_group_{history_days}d_{model_name}.csv"
        save_predictions(cold_predictions, cold_path)

        baseline_subset = _subset_prediction_frame(all_mix_predictions, future)
        baseline_subset["strategy"] = f"all_mix_after_{history_days}d"
        baseline_subset["assigned_group"] = "all_mix"
        baseline_subset["fallback_to_all_mix"] = False

        for strategy_name, frame in [
            (f"all_mix_after_{history_days}d", baseline_subset),
            (f"cold_start_cluster_group_{history_days}d", cold_predictions),
        ]:
            metrics = summarize_metrics(frame[["y_true", "y_pred"]]).iloc[0].to_dict()
            metric_rows.append(
                {
                    "model": model_name,
                    "history_days": int(history_days),
                    "strategy": strategy_name,
                    **metrics,
                }
            )

    accuracy_frame = pd.DataFrame(accuracy_rows)
    metrics_frame = pd.DataFrame(metric_rows)
    if EXP2_COLD_START_METRICS_PATH.exists():
        existing_metrics = pd.read_csv(EXP2_COLD_START_METRICS_PATH)
        if "model" not in existing_metrics.columns:
            existing_metrics["model"] = "lgbm"
        existing_metrics = existing_metrics.loc[existing_metrics["model"] != model_name].copy()
        metrics_frame = pd.concat([existing_metrics, metrics_frame], ignore_index=True)
    metrics_frame = metrics_frame.sort_values(["model", "history_days", "strategy"]).reset_index(drop=True)
    accuracy_frame.to_csv(EXP2_GROUPING_ACCURACY_PATH, index=False)
    metrics_frame.to_csv(EXP2_COLD_START_METRICS_PATH, index=False)
    _plot_cold_start_accuracy(accuracy_frame)
    return {
        "exp2_grouping_accuracy": EXP2_GROUPING_ACCURACY_PATH,
        "exp2_cold_start_metrics": EXP2_COLD_START_METRICS_PATH,
        "exp2_accuracy_vs_n": EXP2_ACCURACY_FIG_PATH,
    }


def run_experiment2(
    models: list[str] | None = None,
    k_values: list[int] | None = None,
    selected_k: int | None = None,
    cpu_threads: int | None = None,
    device: str | None = None,
) -> dict[str, Path]:
    ensure_phase2_dirs()
    set_seed()
    started_at = time.time()
    models = models or ["lgbm"]
    k_values = k_values or [3, 4, 5, 6, 7, 8]

    frame = _load_feature_frame(with_lag_features="lgbm_lag" in models)
    b_split = load_pickle(B_SPLIT_PATH)
    train_frame = _select_rows(frame, b_split["train_mask"])
    test_frame = _select_rows(frame, b_split["test_mask"])
    LOGGER.info(
        "Experiment 2 start train_rows=%s test_rows=%s train_buildings=%s test_buildings=%s",
        len(train_frame),
        len(test_frame),
        train_frame["building_id"].nunique(),
        test_frame["building_id"].nunique(),
    )

    outputs: dict[str, Path] = {}

    train_feature_frame = extract_building_profile_features(train_frame)
    scaled_train_features, scaler = fit_profile_scaler(train_feature_frame)
    outputs.update(save_train_feature_artifacts(train_feature_frame, scaled_train_features, scaler))

    k_metrics, auto_selected_k = _evaluate_k_grid(scaled_train_features, k_values)
    resolved_k = int(selected_k) if selected_k is not None else auto_selected_k
    k_metrics["selected"] = k_metrics["k"] == resolved_k
    k_metrics.to_csv(KMEANS_K_METRICS_PATH, index=False)
    _plot_kmeans_elbow(k_metrics)
    outputs["kmeans_k_metrics"] = KMEANS_K_METRICS_PATH
    outputs["exp2_kmeans_elbow"] = KMEANS_ELBOW_PATH
    LOGGER.info("Experiment 2 selected_k=%s", resolved_k)

    kmeans_model, train_label_frame = _fit_kmeans_model(scaled_train_features, train_feature_frame, resolved_k)
    outputs.update(_save_kmeans_artifacts(kmeans_model, train_label_frame, resolved_k))

    assignment_frame = _assign_test_clusters(test_frame, kmeans_model, scaler)
    outputs["test_building_cluster_assignments"] = TEST_CLUSTER_ASSIGNMENTS_PATH

    strategy_metric_frames: list[pd.DataFrame] = []
    existing_strategy_metrics = (
        pd.read_csv(EXP2_STRATEGY_COMPARISON_PATH)
        if EXP2_STRATEGY_COMPARISON_PATH.exists()
        else None
    )

    train_cluster_lookup = train_label_frame.set_index("building_id")["cluster_label"].astype(str)
    oracle_cluster_lookup = assignment_frame.set_index("building_id")["oracle_cluster_label"].astype(str)

    for model_name in models:
        LOGGER.info("Experiment 2 running model=%s", model_name)
        if model_name in {"lgbm", "lgbm_lag"}:
            model_config = _load_lgbm_config(cpu_threads=cpu_threads, model_name=model_name)
        elif model_name == "lstm":
            model_config = _load_lstm_config()
        elif model_name == "patchtst":
            model_config = _load_patchtst_config()
        else:
            raise ValueError(f"Unsupported model name: {model_name}")

        all_mix_model = _fit_experiment_model(
            model_name=model_name,
            train_frame=train_frame,
            model_config=model_config,
            cpu_threads=cpu_threads,
            device=device,
        )
        all_mix_dir = _resolve_model_dir("all_mix")
        if model_name in {"lgbm", "lgbm_lag"}:
            all_mix_model.save(all_mix_dir / f"{model_name}.txt")
        elif model_name == "lstm":
            all_mix_model.save(all_mix_dir / "lstm.pt")
        else:
            all_mix_model.save(all_mix_dir / "patchtst.pt")
        all_mix_predictions = _prediction_frame_for_model(
            model_name=model_name,
            model=all_mix_model,
            frame=test_frame,
            strategy="all_mix",
            assigned_group=pd.Series(["all_mix"] * len(test_frame)),
        )
        all_mix_path = EXP2_PREDICTIONS_DIR / f"all_mix_{model_name}.csv"
        save_predictions(all_mix_predictions, all_mix_path)
        outputs[f"all_mix_{model_name}_predictions"] = all_mix_path

        meta_models = _fit_group_models(
            train_frame,
            group_assignments=train_frame["building_type"],
            model_name=model_name,
            model_config=model_config,
            strategy_dir="meta_group",
            cpu_threads=cpu_threads,
            device=device,
        )
        meta_predictions = _predict_group_strategy(
            model_name=model_name,
            test_frame=test_frame,
            assignments=test_frame["building_type"],
            strategy="meta_group",
            models=meta_models,
            all_mix_model=all_mix_model,
        )
        meta_path = EXP2_PREDICTIONS_DIR / f"meta_group_{model_name}.csv"
        save_predictions(meta_predictions, meta_path)
        outputs[f"meta_group_{model_name}_predictions"] = meta_path

        cluster_models = _fit_group_models(
            train_frame,
            group_assignments=train_frame["building_id"].map(train_cluster_lookup),
            model_name=model_name,
            model_config=model_config,
            strategy_dir="cluster_group",
            cpu_threads=cpu_threads,
            device=device,
        )
        cluster_predictions = _predict_group_strategy(
            model_name=model_name,
            test_frame=test_frame,
            assignments=test_frame["building_id"].map(oracle_cluster_lookup),
            strategy="cluster_group",
            models=cluster_models,
            all_mix_model=all_mix_model,
        )
        cluster_path = EXP2_PREDICTIONS_DIR / f"cluster_group_{model_name}.csv"
        save_predictions(cluster_predictions, cluster_path)
        outputs[f"cluster_group_{model_name}_predictions"] = cluster_path

        strategy_frame = pd.concat(
            [all_mix_predictions, meta_predictions, cluster_predictions],
            ignore_index=True,
        )
        strategy_metrics = summarize_metrics(strategy_frame, ["strategy"])
        strategy_metrics["model"] = model_name
        strategy_metric_frames.append(strategy_metrics)

        if model_name in {"lgbm", "lgbm_lag", "lstm", "patchtst"}:
            cold_start_outputs = _compute_cold_start_outputs(
                model_name=model_name,
                test_frame=test_frame,
                all_mix_predictions=all_mix_predictions,
                cluster_models=cluster_models,
                all_mix_model=all_mix_model,
                assignment_frame=assignment_frame,
            )
            outputs.update(cold_start_outputs)

        del all_mix_model
        del meta_models
        del cluster_models
        del all_mix_predictions
        del meta_predictions
        del cluster_predictions
        del strategy_frame
        gc.collect()

    if strategy_metric_frames:
        strategy_metrics = pd.concat(strategy_metric_frames, ignore_index=True)
        if existing_strategy_metrics is not None and "model" in existing_strategy_metrics.columns:
            keep = existing_strategy_metrics[
                ~existing_strategy_metrics["model"].isin(strategy_metrics["model"].unique())
            ]
            strategy_metrics = pd.concat([keep, strategy_metrics], ignore_index=True)
        strategy_metrics = strategy_metrics.sort_values(["model", "strategy"]).reset_index(drop=True)
        strategy_metrics.to_csv(EXP2_STRATEGY_COMPARISON_PATH, index=False)
        outputs["exp2_strategy_comparison"] = EXP2_STRATEGY_COMPARISON_PATH

    LOGGER.info("Experiment 2 finished in %.1f seconds", time.time() - started_at)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run experiment 2 for building similarity grouping.")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["lgbm", "lgbm_lag", "lstm", "patchtst"],
        default=["lgbm"],
        help="Experiment 2 models to run.",
    )
    parser.add_argument(
        "--k-values",
        nargs="+",
        type=int,
        default=[3, 4, 5, 6, 7, 8],
        help="Candidate K values evaluated before selecting the final K-means model.",
    )
    parser.add_argument(
        "--selected-k",
        type=int,
        default=None,
        help="Override automatic K selection with a fixed value.",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Optional log file path. Relative paths are resolved from the repository root.",
    )
    parser.add_argument(
        "--max-cpu-threads",
        type=int,
        default=None,
        help="Cap CPU thread usage for K-means, LightGBM, and BLAS/OpenMP.",
    )
    parser.add_argument(
        "--cpu-fraction",
        type=float,
        default=None,
        help="Cap CPU thread usage as a fraction of total logical CPUs, for example 0.7.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device for sequence models, for example cuda:0 or cpu.",
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
        resolved_log = configure_logging(LOGS_DIR / f"exp2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
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
    outputs = run_experiment2(
        models=args.models,
        k_values=args.k_values,
        selected_k=args.selected_k,
        cpu_threads=limits.cpu_threads,
        device=args.device,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")
