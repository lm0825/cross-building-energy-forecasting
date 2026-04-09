from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import gc
import logging
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clustering.feature_extractor import (
    extract_building_profile_features,
    fit_profile_scaler,
    transform_profile_features,
)
from src.config import FEATURES_BDG2_PATH, LOGS_DIR, TABLES_DIR, ensure_phase2_dirs
from src.data_splitting import B_SPLIT_PATH, load_pickle, unpack_mask
from src.experiment2 import (
    _fit_experiment_model,
    _future_slice,
    _initial_history_slice,
    _load_lgbm_config,
    _load_lstm_config,
    _load_patchtst_config,
    _predict_group_strategy,
    _prediction_frame_for_model,
    _subset_prediction_frame,
)
from src.metrics import summarize_metrics
from src.models.common import add_tabular_lag_features, set_seed
from src.models.lgbm_model import LightGBMConfig
from src.models.lstm_model import LSTMConfig
from src.models.patchtst_model import PatchTSTConfig
from src.runtime import apply_runtime_limits, format_bytes


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
    LOGGER.info(
        "Loaded BDG2 features rows=%s buildings=%s sites=%s",
        len(frame),
        frame["building_id"].nunique(),
        frame["site_id"].nunique(),
    )
    return frame


def _select_rows(frame: pd.DataFrame, packed_mask: dict[str, object]) -> pd.DataFrame:
    mask = unpack_mask(packed_mask)
    return frame.loc[mask].reset_index(drop=True)


def _resolve_selected_k(selected_k: int | None) -> int:
    if selected_k is not None:
        return int(selected_k)

    metrics_path = TABLES_DIR / "kmeans_k_metrics.csv"
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        if "selected" in metrics.columns and metrics["selected"].astype(bool).any():
            return int(metrics.loc[metrics["selected"].astype(bool), "k"].iloc[0])
    return 3


def _default_output_label(models: list[str], output_label: str | None) -> str:
    if output_label is not None:
        return output_label
    ordered = [model for model in ["lgbm", "lgbm_lag", "lstm", "patchtst"] if model in models]
    return "_".join(ordered) if ordered else "none"


def _seeded_model_config(
    model_name: str,
    random_seed: int,
    cpu_threads: int | None,
) -> LightGBMConfig | LSTMConfig | PatchTSTConfig:
    if model_name in {"lgbm", "lgbm_lag"}:
        payload = asdict(_load_lgbm_config(cpu_threads=cpu_threads, model_name=model_name))
        payload["random_state"] = random_seed
        if cpu_threads is not None:
            payload["n_jobs"] = cpu_threads
        payload["use_lag_features"] = model_name == "lgbm_lag"
        return LightGBMConfig(**payload)

    if model_name == "lstm":
        payload = asdict(_load_lstm_config())
        payload["random_seed"] = random_seed
        return LSTMConfig(**payload)

    if model_name == "patchtst":
        payload = asdict(_load_patchtst_config())
        payload["random_seed"] = random_seed
        return PatchTSTConfig(**payload)

    raise ValueError(f"Unsupported model name: {model_name}")


def _fit_group_models_no_save(
    train_frame: pd.DataFrame,
    group_assignments: pd.Series,
    model_name: str,
    model_config: object,
    cpu_threads: int | None,
    device: str | None,
):
    grouped_frame = train_frame.copy()
    grouped_frame["_group_key"] = group_assignments.astype(str).to_numpy()
    models: dict[str, object] = {}
    for group_key, subset in grouped_frame.groupby("_group_key", sort=True):
        models[str(group_key)] = _fit_experiment_model(
            model_name=model_name,
            train_frame=subset.drop(columns="_group_key"),
            model_config=model_config,
            cpu_threads=cpu_threads,
            device=device,
        )
        LOGGER.info(
            "Repeated Experiment 2 model=%s group=%s rows=%s buildings=%s",
            model_name,
            group_key,
            len(subset),
            subset["building_id"].nunique(),
        )
        gc.collect()
    return models


def _fit_kmeans_assignments(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    selected_k: int,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_feature_frame = extract_building_profile_features(train_frame)
    scaled_train_features, scaler = fit_profile_scaler(train_feature_frame)
    kmeans_model = KMeans(n_clusters=selected_k, random_state=random_seed, n_init=20)
    labels = kmeans_model.fit_predict(scaled_train_features)

    train_label_frame = train_feature_frame[["building_id"]].copy()
    train_label_frame["cluster_label"] = labels.astype(int)

    cluster_sizes = (
        train_label_frame["cluster_label"]
        .value_counts(sort=False)
        .rename_axis("cluster_label")
        .reset_index(name="n_train_buildings")
        .sort_values("cluster_label")
        .reset_index(drop=True)
    )
    cluster_sizes["random_seed"] = int(random_seed)
    cluster_sizes["selected_k"] = int(selected_k)

    full_feature_frame = extract_building_profile_features(test_frame)
    scaled_full = transform_profile_features(full_feature_frame, scaler)
    assignment_frame = pd.DataFrame(
        {
            "building_id": full_feature_frame["building_id"],
            "oracle_cluster_label": kmeans_model.predict(scaled_full).astype(int),
        }
    )

    for days in [3, 7, 14]:
        history = _initial_history_slice(test_frame, days)
        history_features = extract_building_profile_features(history)
        scaled_history = transform_profile_features(history_features, scaler)
        partial = pd.DataFrame(
            {
                "building_id": history_features["building_id"],
                f"cold_start_cluster_{days}d": kmeans_model.predict(scaled_history).astype(int),
            }
        )
        assignment_frame = assignment_frame.merge(partial, on="building_id", how="left")

    return train_label_frame, assignment_frame, cluster_sizes


def _summarize_seed_metrics(seed_frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    metric_cols = [col for col in ["mae", "rmse", "cv_rmse", "accuracy"] if col in seed_frame.columns]
    rows: list[dict[str, object]] = []
    for group_key, group in seed_frame.groupby(group_cols, sort=True):
        values = group_key if isinstance(group_key, tuple) else (group_key,)
        row = {col: value for col, value in zip(group_cols, values, strict=False)}
        if "random_seed" in group.columns:
            row["n_seeds"] = int(group["random_seed"].nunique())
        if "n_rows" in group.columns:
            row["n_rows"] = int(group["n_rows"].median())
        if "n_buildings" in group.columns:
            row["n_buildings"] = int(group["n_buildings"].median())
        for metric in metric_cols:
            metric_values = group[metric].to_numpy(dtype=np.float64)
            mean = float(metric_values.mean())
            std = float(metric_values.std(ddof=1)) if len(metric_values) > 1 else 0.0
            half_width = 1.96 * std / np.sqrt(len(metric_values)) if len(metric_values) > 1 else 0.0
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            row[f"{metric}_ci95_low"] = mean - half_width
            row[f"{metric}_ci95_high"] = mean + half_width
            row[f"{metric}_min"] = float(metric_values.min())
            row[f"{metric}_max"] = float(metric_values.max())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def _output_paths(output_label: str) -> dict[str, Path]:
    suffix = f"_{output_label}" if output_label else ""
    return {
        "strategy_seed": TABLES_DIR / f"repeated_exp2_strategy_metrics{suffix}.csv",
        "strategy_summary": TABLES_DIR / f"repeated_exp2_strategy_metrics{suffix}_summary.csv",
        "cold_seed": TABLES_DIR / f"repeated_exp2_cold_start_metrics{suffix}.csv",
        "cold_summary": TABLES_DIR / f"repeated_exp2_cold_start_metrics{suffix}_summary.csv",
        "accuracy_seed": TABLES_DIR / f"repeated_exp2_grouping_accuracy{suffix}.csv",
        "accuracy_summary": TABLES_DIR / f"repeated_exp2_grouping_accuracy{suffix}_summary.csv",
        "cluster_sizes": TABLES_DIR / f"repeated_exp2_cluster_sizes{suffix}.csv",
    }


def run_repeated_exp2_metrics(
    models: list[str],
    seeds: list[int],
    selected_k: int | None,
    cpu_threads: int | None,
    device: str | None,
    output_label: str | None,
) -> dict[str, Path]:
    ensure_phase2_dirs()
    started_at = time.time()
    resolved_k = _resolve_selected_k(selected_k)
    label = _default_output_label(models, output_label)

    frame = _load_feature_frame(with_lag_features="lgbm_lag" in models)
    split_artifact = load_pickle(B_SPLIT_PATH)
    train_frame = _select_rows(frame, split_artifact["train_mask"])
    test_frame = _select_rows(frame, split_artifact["test_mask"])
    LOGGER.info(
        "Repeated Experiment 2 setup models=%s seeds=%s selected_k=%s train_rows=%s test_rows=%s",
        models,
        seeds,
        resolved_k,
        len(train_frame),
        len(test_frame),
    )

    strategy_rows: list[dict[str, object]] = []
    cold_rows: list[dict[str, object]] = []
    accuracy_rows: list[dict[str, object]] = []
    cluster_size_rows: list[pd.DataFrame] = []

    try:
        for random_seed in seeds:
            LOGGER.info("Repeated Experiment 2 seed=%s", random_seed)
            set_seed(random_seed)

            train_label_frame, assignment_frame, cluster_sizes = _fit_kmeans_assignments(
                train_frame=train_frame,
                test_frame=test_frame,
                selected_k=resolved_k,
                random_seed=random_seed,
            )
            cluster_size_rows.append(cluster_sizes)

            train_cluster_lookup = train_label_frame.set_index("building_id")["cluster_label"].astype(str)
            oracle_cluster_lookup = assignment_frame.set_index("building_id")["oracle_cluster_label"].astype(str)
            cold_assignment_lookup = assignment_frame.set_index("building_id")

            for history_days in [3, 7, 14]:
                assignment_col = f"cold_start_cluster_{history_days}d"
                assigned = cold_assignment_lookup[assignment_col].astype(str)
                comparable = oracle_cluster_lookup.index.intersection(assigned.index)
                accuracy_rows.append(
                    {
                        "random_seed": int(random_seed),
                        "selected_k": int(resolved_k),
                        "history_days": int(history_days),
                        "n_buildings": int(len(comparable)),
                        "accuracy": float((oracle_cluster_lookup.loc[comparable] == assigned.loc[comparable]).mean()),
                    }
                )

            for model_name in models:
                LOGGER.info("Repeated Experiment 2 running model=%s seed=%s", model_name, random_seed)
                model_config = _seeded_model_config(
                    model_name=model_name,
                    random_seed=random_seed,
                    cpu_threads=cpu_threads,
                )

                all_mix_model = _fit_experiment_model(
                    model_name=model_name,
                    train_frame=train_frame,
                    model_config=model_config,
                    cpu_threads=cpu_threads,
                    device=device,
                )
                all_mix_predictions = _prediction_frame_for_model(
                    model_name=model_name,
                    model=all_mix_model,
                    frame=test_frame,
                    strategy="all_mix",
                    assigned_group=pd.Series(["all_mix"] * len(test_frame)),
                )

                meta_models = _fit_group_models_no_save(
                    train_frame=train_frame,
                    group_assignments=train_frame["building_type"],
                    model_name=model_name,
                    model_config=model_config,
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

                cluster_models = _fit_group_models_no_save(
                    train_frame=train_frame,
                    group_assignments=train_frame["building_id"].map(train_cluster_lookup),
                    model_name=model_name,
                    model_config=model_config,
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

                strategy_frame = pd.concat(
                    [all_mix_predictions, meta_predictions, cluster_predictions],
                    ignore_index=True,
                )
                strategy_metrics = summarize_metrics(strategy_frame, ["strategy"]).sort_values("strategy")
                for record in strategy_metrics.to_dict(orient="records"):
                    strategy_rows.append(
                        {
                            "model": model_name,
                            "random_seed": int(random_seed),
                            "selected_k": int(resolved_k),
                            **record,
                        }
                    )

                for history_days in [3, 7, 14]:
                    assignment_col = f"cold_start_cluster_{history_days}d"
                    assigned = cold_assignment_lookup[assignment_col].astype(str)
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
                    baseline_subset = _subset_prediction_frame(all_mix_predictions, future)
                    baseline_subset["strategy"] = f"all_mix_after_{history_days}d"
                    baseline_subset["assigned_group"] = "all_mix"
                    baseline_subset["fallback_to_all_mix"] = False

                    for strategy_name, metric_frame in [
                        (f"all_mix_after_{history_days}d", baseline_subset),
                        (f"cold_start_cluster_group_{history_days}d", cold_predictions),
                    ]:
                        metrics = summarize_metrics(metric_frame[["y_true", "y_pred"]]).iloc[0].to_dict()
                        cold_rows.append(
                            {
                                "model": model_name,
                                "random_seed": int(random_seed),
                                "selected_k": int(resolved_k),
                                "history_days": int(history_days),
                                "strategy": strategy_name,
                                **metrics,
                            }
                        )

                del all_mix_model
                del meta_models
                del cluster_models
                del all_mix_predictions
                del meta_predictions
                del cluster_predictions
                del strategy_frame
                gc.collect()
    finally:
        del train_frame
        del test_frame
        del frame
        gc.collect()

    strategy_seed = pd.DataFrame(strategy_rows).sort_values(
        ["model", "strategy", "random_seed"]
    ).reset_index(drop=True)
    cold_seed = pd.DataFrame(cold_rows).sort_values(
        ["model", "history_days", "strategy", "random_seed"]
    ).reset_index(drop=True)
    accuracy_seed = pd.DataFrame(accuracy_rows).sort_values(
        ["history_days", "random_seed"]
    ).drop_duplicates(subset=["history_days", "random_seed"]).reset_index(drop=True)
    cluster_sizes = pd.concat(cluster_size_rows, ignore_index=True).sort_values(
        ["random_seed", "cluster_label"]
    ).reset_index(drop=True)

    strategy_summary = _summarize_seed_metrics(strategy_seed, ["model", "strategy"])
    cold_summary = _summarize_seed_metrics(cold_seed, ["model", "history_days", "strategy"])
    accuracy_summary = _summarize_seed_metrics(accuracy_seed, ["history_days"])

    paths = _output_paths(label)
    strategy_seed.to_csv(paths["strategy_seed"], index=False)
    strategy_summary.to_csv(paths["strategy_summary"], index=False)
    cold_seed.to_csv(paths["cold_seed"], index=False)
    cold_summary.to_csv(paths["cold_summary"], index=False)
    accuracy_seed.to_csv(paths["accuracy_seed"], index=False)
    accuracy_summary.to_csv(paths["accuracy_summary"], index=False)
    cluster_sizes.to_csv(paths["cluster_sizes"], index=False)

    LOGGER.info("Repeated Experiment 2 finished in %.1f seconds", time.time() - started_at)
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run repeated-seed Experiment 2 strategy comparisons.")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["lgbm", "lgbm_lag", "lstm", "patchtst"],
        default=["lgbm", "lstm", "patchtst"],
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[7, 42, 123],
    )
    parser.add_argument(
        "--selected-k",
        type=int,
        default=None,
        help="Cluster count for repeated runs. Defaults to the canonical selected K if available.",
    )
    parser.add_argument(
        "--output-label",
        default=None,
        help="Optional suffix for output CSV names, useful for per-model parallel runs.",
    )
    parser.add_argument(
        "--log-file",
        default=None,
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
            LOGS_DIR / f"repeated_exp2_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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
    outputs = run_repeated_exp2_metrics(
        models=args.models,
        seeds=args.seeds,
        selected_k=args.selected_k,
        cpu_threads=limits.cpu_threads,
        device=args.device,
        output_label=args.output_label,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")
