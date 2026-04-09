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
    LOGS_DIR,
    MODELS_DIR,
    TABLES_DIR,
    TABULAR_FEATURE_COLUMNS,
    ensure_phase2_dirs,
)
from src.data_splitting import (
    B_SPLIT_PATH,
    T_SPLIT_PATH,
    load_pickle,
    save_split_artifacts,
    unpack_mask,
)
from src.models.common import (
    add_tabular_lag_features,
    set_seed,
)
from src.models.lgbm_model import LightGBMConfig, LightGBMExperimentModel
from src.models.lstm_model import LSTMConfig, LSTMExperimentModel
from src.models.patchtst_model import PatchTSTConfig, PatchTSTExperimentModel
from src.runtime import apply_runtime_limits, format_bytes

LOGGER = logging.getLogger(__name__)

TABULAR_LAG_COLUMNS = ["lag_1h", "lag_24h", "lag_168h"]
SUMMARY_WINDOWS = [24, 72, 168]
SUMMARY_STATS = ["mean", "std", "min", "max"]
SEED_DEFAULT = [7, 42, 123]
SPLIT_ORDER = ["t_split", "b_split"]
SPEC_ORDER = [
    "lgbm_no_history",
    "lgbm_sparse",
    "lgbm_dense",
    "lstm_ctx24",
    "lstm_ctx72",
    "lstm_ctx168",
    "patchtst_ctx24",
    "patchtst_ctx72",
    "patchtst_ctx168",
]


def _summary_columns() -> list[str]:
    return [
        f"hist_{stat}_{window}h"
        for window in SUMMARY_WINDOWS
        for stat in SUMMARY_STATS
    ]


SPEC_SPECS: dict[str, dict[str, object]] = {
    "lgbm_no_history": {
        "family": "tabular",
        "label": "LightGBM (0 lag)",
        "history_budget": "no_history",
        "context_hours": 0,
        "feature_columns": [],
    },
    "lgbm_sparse": {
        "family": "tabular",
        "label": "LightGBM sparse lag",
        "history_budget": "sparse_lag",
        "context_hours": 168,
        "feature_columns": list(TABULAR_LAG_COLUMNS),
    },
    "lgbm_dense": {
        "family": "tabular",
        "label": "LightGBM dense summary",
        "history_budget": "dense_summary",
        "context_hours": 168,
        "feature_columns": list(TABULAR_LAG_COLUMNS) + _summary_columns(),
    },
    "lstm_ctx24": {
        "family": "lstm",
        "label": "LSTM 24h context",
        "history_budget": "context_24h",
        "context_hours": 24,
    },
    "lstm_ctx72": {
        "family": "lstm",
        "label": "LSTM 72h context",
        "history_budget": "context_72h",
        "context_hours": 72,
    },
    "lstm_ctx168": {
        "family": "lstm",
        "label": "LSTM 168h context",
        "history_budget": "context_168h",
        "context_hours": 168,
    },
    "patchtst_ctx24": {
        "family": "patchtst",
        "label": "PatchTST 24h context",
        "history_budget": "context_24h",
        "context_hours": 24,
    },
    "patchtst_ctx72": {
        "family": "patchtst",
        "label": "PatchTST 72h context",
        "history_budget": "context_72h",
        "context_hours": 72,
    },
    "patchtst_ctx168": {
        "family": "patchtst",
        "label": "PatchTST 168h context",
        "history_budget": "context_168h",
        "context_hours": 168,
    },
}


def _path(stem: str, suffix: str | None = None) -> Path:
    suffix = suffix or ""
    if suffix and not suffix.startswith("_"):
        suffix = f"_{suffix}"
    return TABLES_DIR / f"{stem}{suffix}.csv"


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


def _add_history_summary_features(
    frame: pd.DataFrame,
    *,
    group_column: str = "building_id",
    target_column: str = "meter_reading",
) -> pd.DataFrame:
    required_columns = _summary_columns()
    if all(column in frame.columns for column in required_columns):
        return frame

    augmented = frame.sort_values([group_column, "timestamp"]).reset_index(drop=True).copy()
    shifted = augmented.groupby(group_column, sort=False)[target_column].shift(1)
    group_keys = augmented[group_column]

    for window in SUMMARY_WINDOWS:
        rolling = shifted.groupby(group_keys, sort=False).rolling(window=window, min_periods=1)
        augmented[f"hist_mean_{window}h"] = (
            rolling.mean().reset_index(level=0, drop=True).astype(np.float32)
        )
        augmented[f"hist_std_{window}h"] = (
            rolling.std(ddof=0).reset_index(level=0, drop=True).fillna(0.0).astype(np.float32)
        )
        augmented[f"hist_min_{window}h"] = (
            rolling.min().reset_index(level=0, drop=True).astype(np.float32)
        )
        augmented[f"hist_max_{window}h"] = (
            rolling.max().reset_index(level=0, drop=True).astype(np.float32)
        )

    return augmented


def _load_feature_frame(selected_specs: list[str]) -> pd.DataFrame:
    load_columns = ["building_id", "timestamp", "meter_reading"] + list(TABULAR_FEATURE_COLUMNS)
    frame = pd.read_parquet(FEATURES_BDG2_PATH, columns=load_columns)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame = frame.sort_values(["building_id", "timestamp"]).reset_index(drop=True)

    need_lag = any(
        bool(SPEC_SPECS[name].get("feature_columns"))
        for name in selected_specs
        if SPEC_SPECS[name]["family"] == "tabular"
    )
    need_summary = any(
        "hist_mean_24h" in SPEC_SPECS[name].get("feature_columns", [])
        for name in selected_specs
        if SPEC_SPECS[name]["family"] == "tabular"
    )

    if need_lag:
        frame = add_tabular_lag_features(frame, lag_columns=TABULAR_LAG_COLUMNS)
    if need_summary:
        frame = _add_history_summary_features(frame)

    LOGGER.info(
        "Loaded BDG2 feature frame rows=%s buildings=%s lag_features=%s summary_features=%s",
        len(frame),
        frame["building_id"].nunique(),
        need_lag,
        need_summary,
    )
    return frame


def _select_rows(frame: pd.DataFrame, packed_mask: dict[str, object]) -> pd.DataFrame:
    return frame.loc[unpack_mask(packed_mask)].reset_index(drop=True)


def _tabular_config(
    spec_name: str,
    *,
    seed: int,
    cpu_threads: int | None,
    split_name: str,
) -> LightGBMConfig:
    spec = SPEC_SPECS[spec_name]
    config_name = "lgbm" if spec_name == "lgbm_no_history" else "lgbm_lag"
    payload = _resolve_tuned_json(config_name, split_name=split_name)
    defaults = asdict(LightGBMConfig())
    config = LightGBMConfig(**(defaults | payload))
    config.random_state = int(seed)
    config.n_jobs = cpu_threads if cpu_threads is not None else config.n_jobs
    config.target_transform = "log1p"
    feature_columns = list(spec.get("feature_columns", []))
    config.use_lag_features = bool(feature_columns)
    config.lag_feature_columns = feature_columns if feature_columns else None
    return config


def _lstm_config(spec_name: str, *, seed: int, split_name: str) -> LSTMConfig:
    payload = _resolve_tuned_json("lstm", split_name=split_name)
    defaults = asdict(LSTMConfig())
    config = LSTMConfig(**(defaults | payload))
    config.context_window = int(SPEC_SPECS[spec_name]["context_hours"])
    config.random_seed = int(seed)
    return config


def _patch_config(spec_name: str, *, seed: int, split_name: str) -> PatchTSTConfig:
    payload = _resolve_tuned_json("patchtst", split_name=split_name)
    defaults = asdict(PatchTSTConfig())
    config = PatchTSTConfig(**(defaults | payload))
    config.context_window = int(SPEC_SPECS[spec_name]["context_hours"])
    config.patch_len = min(config.patch_len, config.context_window)
    config.patch_stride = min(config.patch_stride, config.patch_len)
    config.random_seed = int(seed)
    return config


def _prediction_frame_for_tabular(
    test_frame: pd.DataFrame,
    predictions: np.ndarray,
    *,
    split_name: str,
    spec_name: str,
    seed: int,
) -> pd.DataFrame:
    result = test_frame[["building_id", "site_id", "building_type", "timestamp"]].copy()
    result["split_name"] = split_name
    result["spec_name"] = spec_name
    result["seed"] = int(seed)
    result["y_true"] = test_frame["meter_reading"].to_numpy(dtype=np.float32)
    result["y_pred"] = np.asarray(predictions, dtype=np.float32)
    return result


def _per_building_metrics(
    predictions: pd.DataFrame,
    *,
    split_name: str,
    spec_name: str,
    seed: int,
) -> pd.DataFrame:
    errors = predictions["y_pred"].to_numpy(dtype=np.float64) - predictions["y_true"].to_numpy(dtype=np.float64)
    working = predictions[["building_id", "site_id", "building_type"]].copy()
    working["sum_true"] = predictions["y_true"].to_numpy(dtype=np.float64)
    working["sum_sq_err"] = errors ** 2
    working["n_rows"] = 1

    aggregated = (
        working.groupby(["building_id", "site_id", "building_type"], as_index=False, sort=False)
        .agg(
            sum_true=("sum_true", "sum"),
            sum_sq_err=("sum_sq_err", "sum"),
            n_rows=("n_rows", "sum"),
        )
    )
    aggregated["mean_true"] = aggregated["sum_true"] / aggregated["n_rows"]
    aggregated["rmse"] = np.sqrt(aggregated["sum_sq_err"] / aggregated["n_rows"])
    aggregated["cv_rmse"] = aggregated["rmse"] / aggregated["mean_true"]
    aggregated["split_name"] = split_name
    aggregated["spec_name"] = spec_name
    aggregated["seed"] = int(seed)
    aggregated["family"] = SPEC_SPECS[spec_name]["family"]
    aggregated["history_budget"] = SPEC_SPECS[spec_name]["history_budget"]
    aggregated["context_hours"] = int(SPEC_SPECS[spec_name]["context_hours"])
    return aggregated[
        [
            "split_name",
            "spec_name",
            "seed",
            "family",
            "history_budget",
            "context_hours",
            "building_id",
            "site_id",
            "building_type",
            "n_rows",
            "mean_true",
            "rmse",
            "cv_rmse",
        ]
    ]


def _pooled_metrics(
    predictions: pd.DataFrame,
    per_building: pd.DataFrame,
    *,
    split_name: str,
    spec_name: str,
    seed: int,
) -> dict[str, object]:
    errors = predictions["y_pred"].to_numpy(dtype=np.float64) - predictions["y_true"].to_numpy(dtype=np.float64)
    y_true = predictions["y_true"].to_numpy(dtype=np.float64)
    n_rows = int(len(predictions))
    mean_true = float(y_true.mean()) if n_rows else np.nan
    mae = float(np.abs(errors).mean()) if n_rows else np.nan
    rmse = float(np.sqrt(np.mean(errors ** 2))) if n_rows else np.nan
    cv_rmse = float(rmse / mean_true) if mean_true else np.nan
    return {
        "split_name": split_name,
        "spec_name": spec_name,
        "seed": int(seed),
        "family": SPEC_SPECS[spec_name]["family"],
        "history_budget": SPEC_SPECS[spec_name]["history_budget"],
        "context_hours": int(SPEC_SPECS[spec_name]["context_hours"]),
        "pooled_mae": mae,
        "pooled_rmse": rmse,
        "pooled_cv_rmse": cv_rmse,
        "mean_per_building_cv_rmse": float(per_building["cv_rmse"].mean()),
        "median_per_building_cv_rmse": float(per_building["cv_rmse"].median()),
        "p25_per_building_cv_rmse": float(per_building["cv_rmse"].quantile(0.25)),
        "p75_per_building_cv_rmse": float(per_building["cv_rmse"].quantile(0.75)),
        "p90_per_building_cv_rmse": float(per_building["cv_rmse"].quantile(0.90)),
        "n_buildings": int(per_building["building_id"].nunique()),
        "n_rows": n_rows,
    }


def _run_single_split(
    frame: pd.DataFrame,
    split_name: str,
    split_artifact: dict[str, object],
    spec_name: str,
    seed: int,
    *,
    device: str | None,
    cpu_threads: int | None,
) -> tuple[dict[str, object], pd.DataFrame]:
    spec = SPEC_SPECS[spec_name]
    set_seed(seed)
    started = time.time()
    train_frame = _select_rows(frame, split_artifact["train_mask"])
    test_frame = _select_rows(frame, split_artifact["test_mask"])
    family = str(spec["family"])

    if family == "tabular":
        model = LightGBMExperimentModel(
            config=_tabular_config(spec_name, seed=seed, cpu_threads=cpu_threads, split_name=split_name)
        )
        model.fit(train_frame)
        predictions = model.predict_frame(
            test_frame,
            split_name=split_name,
            model_name=spec_name,
            context_window=int(spec["context_hours"]),
            horizon=24,
            stride=24,
        )
        predictions["spec_name"] = spec_name
        predictions["seed"] = int(seed)
    elif family == "lstm":
        model = LSTMExperimentModel(
            config=_lstm_config(spec_name, seed=seed, split_name=split_name),
            device=device,
        )
        model.fit(train_frame)
        predictions = model.predict_frame(test_frame, split_name=split_name)
        predictions["spec_name"] = spec_name
        predictions["seed"] = int(seed)
    elif family == "patchtst":
        model = PatchTSTExperimentModel(
            config=_patch_config(spec_name, seed=seed, split_name=split_name),
            device=device,
        )
        model.fit(train_frame)
        predictions = model.predict_frame(test_frame, split_name=split_name)
        predictions["spec_name"] = spec_name
        predictions["seed"] = int(seed)
    else:
        raise ValueError(f"Unsupported family: {family}")

    per_building = _per_building_metrics(
        predictions,
        split_name=split_name,
        spec_name=spec_name,
        seed=seed,
    )
    pooled = _pooled_metrics(
        predictions,
        per_building,
        split_name=split_name,
        spec_name=spec_name,
        seed=seed,
    )
    LOGGER.info(
        "Finished spec=%s split=%s seed=%s pooled_cv_rmse=%.6f median_per_building=%.6f elapsed=%.1fs",
        spec_name,
        split_name,
        seed,
        pooled["pooled_cv_rmse"],
        pooled["median_per_building_cv_rmse"],
        time.time() - started,
    )

    del model
    del train_frame
    del test_frame
    gc.collect()
    return pooled, per_building


def _summarize_seed_metrics(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    summary = (
        seed_metrics.groupby(
            ["split_name", "spec_name", "family", "history_budget", "context_hours"],
            as_index=False,
        )
        .agg(
            n_seeds=("seed", "nunique"),
            pooled_cv_rmse_mean=("pooled_cv_rmse", "mean"),
            pooled_cv_rmse_sd=("pooled_cv_rmse", "std"),
            median_per_building_cv_rmse_mean=("median_per_building_cv_rmse", "mean"),
            median_per_building_cv_rmse_sd=("median_per_building_cv_rmse", "std"),
            mean_per_building_cv_rmse_mean=("mean_per_building_cv_rmse", "mean"),
            mean_per_building_cv_rmse_sd=("mean_per_building_cv_rmse", "std"),
            p90_per_building_cv_rmse_mean=("p90_per_building_cv_rmse", "mean"),
        )
    )
    split_rank = {name: idx for idx, name in enumerate(SPLIT_ORDER)}
    spec_rank = {name: idx for idx, name in enumerate(SPEC_ORDER)}
    summary = summary.sort_values(
        ["split_name", "spec_name"],
        key=lambda col: col.map(split_rank | spec_rank).fillna(999),
    ).reset_index(drop=True)
    return summary


def _compute_improvements(summary: pd.DataFrame) -> pd.DataFrame:
    baseline = (
        summary.loc[summary["spec_name"] == "lgbm_no_history", ["split_name", "pooled_cv_rmse_mean", "median_per_building_cv_rmse_mean"]]
        .rename(
            columns={
                "pooled_cv_rmse_mean": "baseline_pooled_cv_rmse",
                "median_per_building_cv_rmse_mean": "baseline_median_per_building_cv_rmse",
            }
        )
    )
    merged = summary.merge(baseline, on="split_name", how="left")
    merged["pooled_improvement_vs_no_history_pct"] = 100.0 * (
        merged["baseline_pooled_cv_rmse"] - merged["pooled_cv_rmse_mean"]
    ) / merged["baseline_pooled_cv_rmse"]
    merged["median_improvement_vs_no_history_pct"] = 100.0 * (
        merged["baseline_median_per_building_cv_rmse"] - merged["median_per_building_cv_rmse_mean"]
    ) / merged["baseline_median_per_building_cv_rmse"]
    return merged


def _compute_share_best(per_building: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    spec_rank = {name: idx for idx, name in enumerate(SPEC_ORDER)}
    for (split_name, seed), group in per_building.groupby(["split_name", "seed"], sort=False):
        pivot = (
            group.pivot(index="building_id", columns="spec_name", values="cv_rmse")
            .reindex(columns=SPEC_ORDER)
            .dropna(axis=1, how="all")
        )
        if pivot.empty:
            continue
        best = pivot.idxmin(axis=1)
        shares = best.value_counts(normalize=True)
        for spec_name in pivot.columns:
            rows.append(
                {
                    "split_name": split_name,
                    "seed": int(seed),
                    "spec_name": spec_name,
                    "share_best_buildings": float(shares.get(spec_name, 0.0)),
                    "order": spec_rank.get(spec_name, 999),
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    summary = (
        frame.groupby(["split_name", "spec_name"], as_index=False)
        .agg(
            share_best_mean=("share_best_buildings", "mean"),
            share_best_sd=("share_best_buildings", "std"),
        )
    )
    return summary


def _write_outputs(
    seed_metrics: pd.DataFrame,
    per_building: pd.DataFrame,
    *,
    output_suffix: str | None,
) -> None:
    seed_path = _path("information_budget_seed_metrics", output_suffix)
    per_building_path = _path("information_budget_per_building_metrics", output_suffix)
    summary_path = _path("information_budget_summary_main", output_suffix)
    improvement_path = _path("information_budget_improvement_vs_no_history", output_suffix)
    share_best_path = _path("information_budget_share_best", output_suffix)
    config_path = _path("information_budget_config_definitions", output_suffix)

    config_rows = [
        {
            "spec_name": spec_name,
            "label": spec["label"],
            "family": spec["family"],
            "history_budget": spec["history_budget"],
            "context_hours": spec["context_hours"],
            "feature_columns": ",".join(spec.get("feature_columns", [])),
        }
        for spec_name, spec in SPEC_SPECS.items()
        if spec_name in seed_metrics["spec_name"].unique()
    ]
    pd.DataFrame(config_rows).to_csv(config_path, index=False)

    summary = _summarize_seed_metrics(seed_metrics)
    share_best = _compute_share_best(per_building)
    if not share_best.empty:
        summary = summary.merge(share_best, on=["split_name", "spec_name"], how="left")
    improvements = _compute_improvements(summary)

    seed_metrics.to_csv(seed_path, index=False)
    per_building.to_csv(per_building_path, index=False)
    summary.to_csv(summary_path, index=False)
    improvements.to_csv(improvement_path, index=False)
    if not share_best.empty:
        share_best.to_csv(share_best_path, index=False)

    LOGGER.info("Saved seed metrics to %s", seed_path)
    LOGGER.info("Saved per-building metrics to %s", per_building_path)
    LOGGER.info("Saved summary metrics to %s", summary_path)
    LOGGER.info("Saved improvement table to %s", improvement_path)
    if not share_best.empty:
        LOGGER.info("Saved share-best summary to %s", share_best_path)


def _aggregate_suffixes(suffixes: list[str], output_suffix: str | None) -> None:
    seed_frames: list[pd.DataFrame] = []
    building_frames: list[pd.DataFrame] = []
    for suffix in suffixes:
        seed_path = _path("information_budget_seed_metrics", suffix)
        building_path = _path("information_budget_per_building_metrics", suffix)
        if not seed_path.exists() or not building_path.exists():
            raise FileNotFoundError(f"Missing partial output for suffix={suffix}: {seed_path} or {building_path}")
        seed_frames.append(pd.read_csv(seed_path))
        building_frames.append(pd.read_csv(building_path))

    seed_metrics = (
        pd.concat(seed_frames, ignore_index=True)
        .drop_duplicates(subset=["split_name", "spec_name", "seed"])
        .reset_index(drop=True)
    )
    per_building = (
        pd.concat(building_frames, ignore_index=True)
        .drop_duplicates(subset=["split_name", "spec_name", "seed", "building_id"])
        .reset_index(drop=True)
    )
    _write_outputs(seed_metrics, per_building, output_suffix=output_suffix)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run BDG2 information-budget sensitivity for tabular and sequence models."
    )
    parser.add_argument(
        "--specs",
        nargs="+",
        default=SPEC_ORDER,
        choices=SPEC_ORDER,
        help="Budget specifications to run.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=SPLIT_ORDER,
        choices=SPLIT_ORDER,
        help="Split protocols to evaluate.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=SEED_DEFAULT,
        help="Training random seeds.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device for sequence models, e.g. cuda:0. Ignored by tabular specs.",
    )
    parser.add_argument(
        "--output-suffix",
        default=None,
        help="Optional suffix for partial outputs, e.g. lgbm or lstm.",
    )
    parser.add_argument(
        "--aggregate-from-suffixes",
        nargs="+",
        default=None,
        help="If set, skip training and aggregate previously written partial outputs.",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Optional log file path.",
    )
    parser.add_argument(
        "--max-cpu-threads",
        type=int,
        default=None,
        help="Cap CPU thread usage.",
    )
    parser.add_argument(
        "--cpu-fraction",
        type=float,
        default=None,
        help="Cap CPU thread usage as a fraction of total logical CPUs.",
    )
    parser.add_argument(
        "--memory-fraction",
        type=float,
        default=None,
        help="Cap process memory as a fraction of total system memory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    resolved_log = configure_logging(args.log_file)
    if resolved_log is None:
        resolved_log = configure_logging(LOGS_DIR / f"exp7_information_budget_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
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

    ensure_phase2_dirs()
    started = time.time()

    if args.aggregate_from_suffixes:
        _aggregate_suffixes(args.aggregate_from_suffixes, output_suffix=args.output_suffix)
        LOGGER.info("Aggregation finished in %.1fs", time.time() - started)
        return

    if not T_SPLIT_PATH.exists() or not B_SPLIT_PATH.exists():
        LOGGER.info("Split artifacts missing. Building fresh split files.")
        save_split_artifacts()

    frame = _load_feature_frame(args.specs)
    artifacts = {
        "t_split": load_pickle(T_SPLIT_PATH),
        "b_split": load_pickle(B_SPLIT_PATH),
    }

    seed_rows: list[dict[str, object]] = []
    per_building_frames: list[pd.DataFrame] = []
    for spec_name in args.specs:
        for seed in args.seeds:
            for split_name in args.splits:
                pooled, per_building = _run_single_split(
                    frame=frame,
                    split_name=split_name,
                    split_artifact=artifacts[split_name],
                    spec_name=spec_name,
                    seed=int(seed),
                    device=args.device,
                    cpu_threads=limits.cpu_threads,
                )
                seed_rows.append(pooled)
                per_building_frames.append(per_building)

    seed_metrics = pd.DataFrame(seed_rows)
    per_building = pd.concat(per_building_frames, ignore_index=True)
    _write_outputs(seed_metrics, per_building, output_suffix=args.output_suffix)
    LOGGER.info("Information-budget sensitivity finished in %.1fs", time.time() - started)


if __name__ == "__main__":
    main()
