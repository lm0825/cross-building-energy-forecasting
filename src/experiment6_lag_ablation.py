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
    S_SPLIT_PATH,
    T_SPLIT_PATH,
    load_pickle,
    save_split_artifacts,
    unpack_mask,
)
from src.models.common import add_tabular_lag_features, set_seed
from src.models.lgbm_model import LightGBMConfig, LightGBMExperimentModel
from src.runtime import apply_runtime_limits, format_bytes

LOGGER = logging.getLogger(__name__)

SEED_METRICS_PATH = TABLES_DIR / "lag_ablation_seed_metrics.csv"
PER_BUILDING_PATH = TABLES_DIR / "lag_ablation_per_building_metrics.csv"
SUMMARY_MAIN_PATH = TABLES_DIR / "lag_ablation_summary_main.csv"
IMPROVEMENT_PATH = TABLES_DIR / "lag_ablation_improvement_vs_nolag.csv"
SHARE_BEST_PATH = TABLES_DIR / "lag_ablation_share_best.csv"
CONFIG_PATH = TABLES_DIR / "lag_ablation_config_definitions.csv"

CONFIG_SPECS: list[dict[str, object]] = [
    {"config_name": "C0", "stage": "stage1", "lags": []},
    {"config_name": "A1", "stage": "stage1", "lags": [1]},
    {"config_name": "A2", "stage": "stage1", "lags": [24]},
    {"config_name": "A3", "stage": "stage1", "lags": [168]},
    {"config_name": "B1", "stage": "stage1", "lags": [1, 24]},
    {"config_name": "B2", "stage": "stage1", "lags": [1, 168]},
    {"config_name": "B3", "stage": "stage1", "lags": [24, 168]},
    {"config_name": "C1", "stage": "stage1", "lags": [1, 24, 168]},
    {"config_name": "D1", "stage": "stage2", "lags": [1, 2, 3, 24, 168]},
    {"config_name": "D2", "stage": "stage2", "lags": [1, 24, 48, 168]},
    {"config_name": "D3", "stage": "stage2", "lags": [1, 24, 168, 336]},
    {"config_name": "D4", "stage": "stage2", "lags": [1, 2, 3, 24, 48, 168, 336]},
]
CONFIG_INDEX = {str(spec["config_name"]): spec for spec in CONFIG_SPECS}
CONFIG_ORDER = [str(spec["config_name"]) for spec in CONFIG_SPECS]
SPLIT_ORDER = ["t_split", "b_split", "s_split"]


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


def _lag_columns_from_hours(hours: list[int]) -> list[str]:
    return [f"lag_{int(hour)}h" for hour in hours]


def _lag_label(hours: list[int]) -> str:
    if not hours:
        return "none"
    return "+".join(f"t-{hour}" for hour in hours)


def _resolve_tuning_path(
    model_name: str,
    *,
    split_name: str,
    fold_id: str | None = None,
) -> Path:
    tuning_dir = MODELS_DIR / "_tuning"
    candidates: list[Path] = []
    if fold_id is not None:
        candidates.append(tuning_dir / f"{model_name}.{split_name}.{fold_id}.config.json")
    candidates.append(tuning_dir / f"{model_name}.{split_name}.config.json")
    candidates.append(tuning_dir / f"{model_name}.config.json")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[-1]


def _load_tuned_config(path: Path) -> LightGBMConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    defaults = asdict(LightGBMConfig())
    return LightGBMConfig(**(defaults | payload))


def _load_feature_frame(required_lag_columns: list[str]) -> pd.DataFrame:
    load_columns = ["building_id", "timestamp", "meter_reading"] + list(TABULAR_FEATURE_COLUMNS)
    LOGGER.info(
        "Loading BDG2 feature frame from %s with %s custom lag columns",
        FEATURES_BDG2_PATH,
        len(required_lag_columns),
    )
    frame = pd.read_parquet(FEATURES_BDG2_PATH, columns=load_columns)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame = frame.sort_values(["building_id", "timestamp"]).reset_index(drop=True)
    if required_lag_columns:
        frame = add_tabular_lag_features(frame, lag_columns=required_lag_columns)
    LOGGER.info(
        "Loaded feature frame rows=%s buildings=%s columns=%s lag_columns=%s",
        len(frame),
        frame["building_id"].nunique(),
        len(frame.columns),
        required_lag_columns,
    )
    return frame


def _select_rows(frame: pd.DataFrame, packed_mask: dict[str, object]) -> pd.DataFrame:
    mask = unpack_mask(packed_mask)
    return frame.loc[mask].reset_index(drop=True)


def _fit_config(
    train_frame: pd.DataFrame,
    lag_hours: list[int],
    seed: int,
    cpu_threads: int | None,
    *,
    split_name: str,
    fold_id: str | None = None,
) -> LightGBMExperimentModel:
    if lag_hours:
        base_config = _load_tuned_config(_resolve_tuning_path("lgbm_lag", split_name=split_name, fold_id=fold_id))
    else:
        base_config = _load_tuned_config(_resolve_tuning_path("lgbm", split_name=split_name, fold_id=fold_id))

    base_config.random_state = int(seed)
    base_config.n_jobs = cpu_threads if cpu_threads is not None else base_config.n_jobs
    base_config.target_transform = "log1p"
    base_config.use_lag_features = bool(lag_hours)
    base_config.lag_feature_columns = _lag_columns_from_hours(lag_hours) if lag_hours else None
    base_config.lag_feature_steps = (
        {
            column: int(hour)
            for column, hour in zip(_lag_columns_from_hours(lag_hours), lag_hours, strict=True)
        }
        if lag_hours
        else None
    )

    model = LightGBMExperimentModel(config=base_config)
    model.fit(train_frame)
    return model


def _per_building_metrics(
    prediction_frame: pd.DataFrame,
    split_name: str,
    config_name: str,
    stage: str,
    seed: int,
    lag_hours: list[int],
) -> pd.DataFrame:
    y_true = prediction_frame["y_true"].to_numpy(dtype=np.float64)
    y_pred = prediction_frame["y_pred"].to_numpy(dtype=np.float64)
    errors = y_pred - y_true

    working = prediction_frame[["building_id", "site_id", "building_type"]].copy()
    working["sum_true"] = y_true
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
    aggregated["config_name"] = config_name
    aggregated["stage"] = stage
    aggregated["seed"] = int(seed)
    aggregated["lag_hours"] = _lag_label(lag_hours)
    aggregated["lag_count"] = int(len(lag_hours))
    return aggregated[
        [
            "split_name",
            "config_name",
            "stage",
            "seed",
            "lag_hours",
            "lag_count",
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
    per_building: pd.DataFrame,
    sum_true: float,
    sum_abs_err: float,
    sum_sq_err: float,
    n_rows: int,
    split_name: str,
    config_name: str,
    stage: str,
    seed: int,
    lag_hours: list[int],
) -> dict[str, object]:
    mean_true = sum_true / n_rows if n_rows else np.nan
    mae = sum_abs_err / n_rows if n_rows else np.nan
    rmse = float(np.sqrt(sum_sq_err / n_rows)) if n_rows else np.nan
    cv_rmse = float(rmse / mean_true) if mean_true else np.nan
    return {
        "split_name": split_name,
        "config_name": config_name,
        "stage": stage,
        "seed": int(seed),
        "lag_hours": _lag_label(lag_hours),
        "lag_count": int(len(lag_hours)),
        "pooled_mae": float(mae),
        "pooled_rmse": float(rmse),
        "pooled_cv_rmse": float(cv_rmse),
        "mean_per_building_cv_rmse": float(per_building["cv_rmse"].mean()),
        "median_per_building_cv_rmse": float(per_building["cv_rmse"].median()),
        "p25_per_building_cv_rmse": float(per_building["cv_rmse"].quantile(0.25)),
        "p75_per_building_cv_rmse": float(per_building["cv_rmse"].quantile(0.75)),
        "p90_per_building_cv_rmse": float(per_building["cv_rmse"].quantile(0.90)),
        "n_buildings": int(per_building["building_id"].nunique()),
        "n_rows": int(n_rows),
    }


def _run_single_split(
    frame: pd.DataFrame,
    split_name: str,
    split_artifact: dict[str, object],
    config_name: str,
    stage: str,
    lag_hours: list[int],
    seed: int,
    cpu_threads: int | None,
) -> tuple[dict[str, object], pd.DataFrame]:
    set_seed(seed)
    split_started = time.time()
    LOGGER.info(
        "Running lag ablation config=%s split=%s seed=%s lag_hours=%s",
        config_name,
        split_name,
        seed,
        lag_hours,
    )

    if split_name in {"t_split", "b_split"}:
        train_frame = _select_rows(frame, split_artifact["train_mask"])
        test_frame = _select_rows(frame, split_artifact["test_mask"])
        model = _fit_config(
            train_frame=train_frame,
            lag_hours=lag_hours,
            seed=seed,
            cpu_threads=cpu_threads,
            split_name=split_name,
        )
        prediction_frame = model.predict_frame(
            test_frame,
            split_name=split_name,
            model_name="lgbm_lag" if lag_hours else "lgbm",
        )
        y_true = prediction_frame["y_true"].to_numpy(dtype=np.float64)
        errors = prediction_frame["y_pred"].to_numpy(dtype=np.float64) - y_true
        per_building = _per_building_metrics(
            prediction_frame=prediction_frame,
            split_name=split_name,
            config_name=config_name,
            stage=stage,
            seed=seed,
            lag_hours=lag_hours,
        )
        pooled = _pooled_metrics(
            per_building=per_building,
            sum_true=float(y_true.sum()),
            sum_abs_err=float(np.abs(errors).sum()),
            sum_sq_err=float((errors ** 2).sum()),
            n_rows=len(prediction_frame),
            split_name=split_name,
            config_name=config_name,
            stage=stage,
            seed=seed,
            lag_hours=lag_hours,
        )
        del model
        del train_frame
        del test_frame
        del prediction_frame
        gc.collect()
    elif split_name == "s_split":
        total_true = 0.0
        total_abs_err = 0.0
        total_sq_err = 0.0
        total_rows = 0
        per_building_frames: list[pd.DataFrame] = []
        for fold in split_artifact["folds"]:
            LOGGER.info(
                "Running S-split fold=%s config=%s seed=%s",
                fold["fold_id"],
                config_name,
                seed,
            )
            train_frame = _select_rows(frame, fold["train_mask"])
            test_frame = _select_rows(frame, fold["test_mask"])
            model = _fit_config(
                train_frame=train_frame,
                lag_hours=lag_hours,
                seed=seed,
                cpu_threads=cpu_threads,
                split_name=split_name,
                fold_id=str(fold["fold_id"]),
            )
            prediction_frame = model.predict_frame(
                test_frame,
                split_name=split_name,
                model_name="lgbm_lag" if lag_hours else "lgbm",
            )
            y_true = prediction_frame["y_true"].to_numpy(dtype=np.float64)
            errors = prediction_frame["y_pred"].to_numpy(dtype=np.float64) - y_true
            total_true += float(y_true.sum())
            total_abs_err += float(np.abs(errors).sum())
            total_sq_err += float((errors ** 2).sum())
            total_rows += int(len(prediction_frame))
            per_building_frames.append(
                _per_building_metrics(
                    prediction_frame=prediction_frame,
                    split_name=split_name,
                    config_name=config_name,
                    stage=stage,
                    seed=seed,
                    lag_hours=lag_hours,
                )
            )
            del model
            del train_frame
            del test_frame
            del prediction_frame
            gc.collect()

        per_building = pd.concat(per_building_frames, ignore_index=True)
        pooled = _pooled_metrics(
            per_building=per_building,
            sum_true=total_true,
            sum_abs_err=total_abs_err,
            sum_sq_err=total_sq_err,
            n_rows=total_rows,
            split_name=split_name,
            config_name=config_name,
            stage=stage,
            seed=seed,
            lag_hours=lag_hours,
        )
        del per_building_frames
        gc.collect()
    else:
        raise ValueError(f"Unsupported split: {split_name}")

    LOGGER.info(
        "Finished config=%s split=%s seed=%s pooled_cv_rmse=%.6f median_per_building=%.6f elapsed=%.1fs",
        config_name,
        split_name,
        seed,
        pooled["pooled_cv_rmse"],
        pooled["median_per_building_cv_rmse"],
        time.time() - split_started,
    )
    return pooled, per_building


def _compute_share_best(per_building: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (split_name, seed), group in per_building.groupby(["split_name", "seed"], sort=False):
        pivot = (
            group.pivot(index="building_id", columns="config_name", values="cv_rmse")
            .reindex(columns=CONFIG_ORDER)
            .dropna(axis=1, how="all")
        )
        if pivot.empty:
            continue
        best = pivot.idxmin(axis=1)
        shares = best.value_counts(normalize=True)
        for config_name in pivot.columns:
            rows.append(
                {
                    "split_name": split_name,
                    "seed": int(seed),
                    "config_name": config_name,
                    "share_best_buildings": float(shares.get(config_name, 0.0)),
                }
            )
    share_best = pd.DataFrame(rows)
    if share_best.empty:
        return share_best
    summary = (
        share_best.groupby(["split_name", "config_name"], as_index=False)
        .agg(
            share_best_mean=("share_best_buildings", "mean"),
            share_best_sd=("share_best_buildings", "std"),
        )
    )
    return summary


def _summarize_seed_metrics(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    summary = (
        seed_metrics.groupby(["split_name", "config_name", "stage", "lag_hours", "lag_count"], as_index=False)
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
    return summary.sort_values(
        ["split_name", "config_name"],
        key=lambda col: col.map({**{v: i for i, v in enumerate(SPLIT_ORDER)}, **{v: i for i, v in enumerate(CONFIG_ORDER)}}).fillna(999),
    ).reset_index(drop=True)


def _compute_improvements(summary: pd.DataFrame) -> pd.DataFrame:
    baseline = (
        summary.loc[summary["config_name"] == "C0", ["split_name", "pooled_cv_rmse_mean", "median_per_building_cv_rmse_mean"]]
        .rename(
            columns={
                "pooled_cv_rmse_mean": "baseline_pooled_cv_rmse",
                "median_per_building_cv_rmse_mean": "baseline_median_per_building_cv_rmse",
            }
        )
    )
    merged = summary.merge(baseline, on="split_name", how="left")
    merged["pooled_improvement_vs_c0_pct"] = 100.0 * (
        merged["baseline_pooled_cv_rmse"] - merged["pooled_cv_rmse_mean"]
    ) / merged["baseline_pooled_cv_rmse"]
    merged["median_per_building_improvement_vs_c0_pct"] = 100.0 * (
        merged["baseline_median_per_building_cv_rmse"] - merged["median_per_building_cv_rmse_mean"]
    ) / merged["baseline_median_per_building_cv_rmse"]
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BDG2 lag-combination ablation for parity robustness.")
    parser.add_argument(
        "--configs",
        nargs="+",
        default=CONFIG_ORDER,
        choices=CONFIG_ORDER,
        help="Lag configurations to run.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["t_split", "b_split"],
        choices=SPLIT_ORDER,
        help="Split protocols to evaluate.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[7, 42, 123],
        help="Training random seeds.",
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
        help="Cap this process to a fraction of total system memory on supported platforms.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    resolved_log = configure_logging(args.log_file)
    if resolved_log is None:
        default_log = LOGS_DIR / f"exp6_lag_ablation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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

    ensure_phase2_dirs()
    started_at = time.time()
    if not T_SPLIT_PATH.exists() or not B_SPLIT_PATH.exists() or not S_SPLIT_PATH.exists():
        LOGGER.info("Split artifacts missing. Building fresh split files.")
        save_split_artifacts()

    selected_specs = [CONFIG_INDEX[name] for name in args.configs]
    required_lag_columns = sorted(
        {
            column
            for spec in selected_specs
            for column in _lag_columns_from_hours(list(spec["lags"]))
        },
        key=lambda column: int(column.removeprefix("lag_").removesuffix("h")),
    )
    frame = _load_feature_frame(required_lag_columns=required_lag_columns)
    artifacts = {
        "t_split": load_pickle(T_SPLIT_PATH),
        "b_split": load_pickle(B_SPLIT_PATH),
        "s_split": load_pickle(S_SPLIT_PATH),
    }

    config_rows = [
        {
            "config_name": str(spec["config_name"]),
            "stage": str(spec["stage"]),
            "lag_hours": _lag_label(list(spec["lags"])),
            "lag_count": int(len(spec["lags"])),
        }
        for spec in selected_specs
    ]
    pd.DataFrame(config_rows).to_csv(CONFIG_PATH, index=False)

    seed_rows: list[dict[str, object]] = []
    per_building_frames: list[pd.DataFrame] = []
    for spec in selected_specs:
        config_name = str(spec["config_name"])
        stage = str(spec["stage"])
        lag_hours = list(spec["lags"])
        for seed in args.seeds:
            for split_name in args.splits:
                pooled, per_building = _run_single_split(
                    frame=frame,
                    split_name=split_name,
                    split_artifact=artifacts[split_name],
                    config_name=config_name,
                    stage=stage,
                    lag_hours=lag_hours,
                    seed=int(seed),
                    cpu_threads=limits.cpu_threads,
                )
                seed_rows.append(pooled)
                per_building_frames.append(per_building)
                pd.DataFrame(seed_rows).to_csv(SEED_METRICS_PATH, index=False)
                pd.concat(per_building_frames, ignore_index=True).to_csv(PER_BUILDING_PATH, index=False)

    seed_metrics = pd.DataFrame(seed_rows)
    per_building = pd.concat(per_building_frames, ignore_index=True)
    summary_main = _summarize_seed_metrics(seed_metrics)
    share_best = _compute_share_best(per_building)
    if not share_best.empty:
        summary_main = summary_main.merge(share_best, on=["split_name", "config_name"], how="left")
    improvements = _compute_improvements(summary_main)

    seed_metrics.to_csv(SEED_METRICS_PATH, index=False)
    per_building.to_csv(PER_BUILDING_PATH, index=False)
    summary_main.to_csv(SUMMARY_MAIN_PATH, index=False)
    improvements.to_csv(IMPROVEMENT_PATH, index=False)
    if not share_best.empty:
        share_best.to_csv(SHARE_BEST_PATH, index=False)

    LOGGER.info("Lag ablation finished in %.1fs", time.time() - started_at)
    LOGGER.info("Saved config definitions to %s", CONFIG_PATH)
    LOGGER.info("Saved seed metrics to %s", SEED_METRICS_PATH)
    LOGGER.info("Saved per-building metrics to %s", PER_BUILDING_PATH)
    LOGGER.info("Saved summary metrics to %s", SUMMARY_MAIN_PATH)
    LOGGER.info("Saved improvement table to %s", IMPROVEMENT_PATH)
    if not share_best.empty:
        LOGGER.info("Saved share-best summary to %s", SHARE_BEST_PATH)


if __name__ == "__main__":
    main()
