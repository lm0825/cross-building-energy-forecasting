from __future__ import annotations

import argparse
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

from src.experiment5_heew import (
    BSPLIT_SEEDS,
    CANONICAL_BSPLIT_SEED,
    HEEW_FIGURES_DIR,
    HEEW_LOGS_DIR,
    HEEW_TABLES_DIR,
    MODEL_COLORS,
    _build_b_split,
    _build_model_specs,
    _build_t_split,
    _compute_per_building_metrics,
    _compute_rowwise_metrics,
    _filter_heew,
    _load_heew_raw,
    _prepare_model_frame,
)
from src.models.lgbm_model import LightGBMConfig, LightGBMExperimentModel
from src.runtime import apply_runtime_limits, format_bytes

LOGGER = logging.getLogger(__name__)

CONFIG_SPECS = [
    {"config_name": "C0", "lag_days": []},
    {"config_name": "A1", "lag_days": [1]},
    {"config_name": "A2", "lag_days": [7]},
    {"config_name": "A3", "lag_days": [28]},
    {"config_name": "B1", "lag_days": [1, 7]},
    {"config_name": "B2", "lag_days": [1, 28]},
    {"config_name": "B3", "lag_days": [7, 28]},
    {"config_name": "C1", "lag_days": [1, 7, 28]},
]

CONFIG_ORDER = [spec["config_name"] for spec in CONFIG_SPECS]
CONFIG_LABELS = {
    "C0": "none",
    "A1": "{t-1d}",
    "A2": "{t-7d}",
    "A3": "{t-28d}",
    "B1": "{t-1d,t-7d}",
    "B2": "{t-1d,t-28d}",
    "B3": "{t-7d,t-28d}",
    "C1": "{t-1d,t-7d,t-28d}",
}

SUMMARY_PATH = HEEW_TABLES_DIR / "heew_lag_ablation_summary.csv"
SEED_METRICS_PATH = HEEW_TABLES_DIR / "heew_lag_ablation_seed_metrics.csv"
PER_BUILDING_PATH = HEEW_TABLES_DIR / "heew_lag_ablation_per_building.csv"
FIGURE_PATH = HEEW_FIGURES_DIR / "paper_figS3_heew_lag_ablation.png"


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


def _add_custom_lags(frame: pd.DataFrame, lag_days: list[int]) -> tuple[pd.DataFrame, list[str]]:
    augmented = frame.sort_values(["building_id", "timestamp"]).reset_index(drop=True).copy()
    grouped = augmented.groupby("building_id", sort=False)["meter_reading"]
    lag_columns: list[str] = []
    for lag_day in lag_days:
        col = f"lag_{lag_day}d"
        augmented[col] = grouped.shift(lag_day).astype(np.float32)
        lag_columns.append(col)
    return augmented, lag_columns


def _fit_predict_lgbm(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    split_name: str,
    lag_days: list[int],
    cpu_threads: int | None,
) -> pd.DataFrame:
    model_specs = _build_model_specs(cpu_threads=cpu_threads, device="cpu")
    if lag_days:
        train_ready, lag_columns = _add_custom_lags(train_frame, lag_days)
        test_ready, _ = _add_custom_lags(test_frame, lag_days)
        cfg = LightGBMConfig(**model_specs["lgbm_lag"].__dict__)
        cfg.use_lag_features = True
        cfg.lag_feature_columns = lag_columns
        cfg.lag_feature_steps = {column: int(day) for column, day in zip(lag_columns, lag_days, strict=True)}
    else:
        train_ready = train_frame.copy()
        test_ready = test_frame.copy()
        cfg = LightGBMConfig(**model_specs["lgbm"].__dict__)
        cfg.use_lag_features = False
        cfg.lag_feature_columns = None
        cfg.lag_feature_steps = None
    model = LightGBMExperimentModel(config=cfg)
    model.fit(train_ready)
    return model.predict_frame(
        test_ready,
        split_name=split_name,
        model_name="lgbm_lag" if lag_days else "lgbm",
        context_window=HEEW_CONTEXT_DAYS,
        horizon=HEEW_HORIZON_DAYS,
        stride=HEEW_HORIZON_DAYS,
    ).drop(columns=["fold_id"]).reset_index(drop=True)


def _summarize_per_building_frame(per_building: pd.DataFrame) -> dict[str, float]:
    values = per_building["cv_rmse"].to_numpy(dtype=np.float64)
    return {
        "n_buildings": int(len(per_building)),
        "mean_per_building_cv_rmse": float(np.mean(values)),
        "median_per_building_cv_rmse": float(np.median(values)),
        "q1_per_building_cv_rmse": float(np.quantile(values, 0.25)),
        "q3_per_building_cv_rmse": float(np.quantile(values, 0.75)),
    }


def _run_config(
    frame: pd.DataFrame,
    split_name: str,
    split_seed: int | None,
    config_name: str,
    lag_days: list[int],
    cpu_threads: int | None,
) -> tuple[dict[str, object], pd.DataFrame]:
    if split_name == "t_split":
        train_frame, test_frame = _build_t_split(frame)
        held_out = []
    elif split_name == "b_split":
        if split_seed is None:
            raise ValueError("split_seed is required for HEEW B-split lag ablation.")
        train_frame, test_frame, held_out = _build_b_split(frame, split_seed)
    else:
        raise ValueError(f"Unsupported split: {split_name}")

    prediction_frame = _fit_predict_lgbm(
        train_frame=train_frame,
        test_frame=test_frame,
        split_name=split_name,
        lag_days=lag_days,
        cpu_threads=cpu_threads,
    )
    row_metrics = _compute_rowwise_metrics(prediction_frame)
    per_building = _compute_per_building_metrics(prediction_frame)
    per_building_summary = _summarize_per_building_frame(per_building)
    per_building = per_building.copy()
    per_building["split_name"] = split_name
    per_building["split_seed"] = split_seed
    per_building["config_name"] = config_name
    per_building["lag_label"] = CONFIG_LABELS[config_name]

    row = {
        "dataset": "HEEW",
        "split_name": split_name,
        "split_seed": split_seed,
        "config_name": config_name,
        "lag_label": CONFIG_LABELS[config_name],
        "lag_days": ",".join(str(day) for day in lag_days),
        "lag_count": int(len(lag_days)),
        "held_out_buildings": ",".join(held_out),
        **row_metrics,
        **per_building_summary,
    }
    return row, per_building


def _build_summary(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (split_name, config_name, lag_label), group in seed_metrics.groupby(
        ["split_name", "config_name", "lag_label"], sort=False
    ):
        base = {
            "split_name": split_name,
            "config_name": config_name,
            "lag_label": lag_label,
            "n_runs": int(len(group)),
            "lag_count": int(group["lag_count"].iloc[0]),
            "pooled_cv_rmse_mean": float(group["cv_rmse"].mean()),
            "pooled_cv_rmse_sd": float(group["cv_rmse"].std(ddof=1)) if len(group) > 1 else 0.0,
            "median_per_building_cv_rmse_mean": float(group["median_per_building_cv_rmse"].mean()),
            "median_per_building_cv_rmse_sd": float(group["median_per_building_cv_rmse"].std(ddof=1))
            if len(group) > 1
            else 0.0,
            "pooled_cv_rmse_min": float(group["cv_rmse"].min()),
            "pooled_cv_rmse_max": float(group["cv_rmse"].max()),
        }
        rows.append(base)
    summary = pd.DataFrame(rows)

    for split_name in summary["split_name"].unique():
        baseline = summary.loc[
            (summary["split_name"] == split_name) & (summary["config_name"] == "C0"),
            "pooled_cv_rmse_mean",
        ].iloc[0]
        baseline_med = summary.loc[
            (summary["split_name"] == split_name) & (summary["config_name"] == "C0"),
            "median_per_building_cv_rmse_mean",
        ].iloc[0]
        mask = summary["split_name"] == split_name
        summary.loc[mask, "pooled_improvement_vs_c0_pct"] = (
            100.0 * (baseline - summary.loc[mask, "pooled_cv_rmse_mean"]) / baseline
        )
        summary.loc[mask, "median_improvement_vs_c0_pct"] = (
            100.0 * (baseline_med - summary.loc[mask, "median_per_building_cv_rmse_mean"]) / baseline_med
        )
    summary["config_name"] = pd.Categorical(summary["config_name"], categories=CONFIG_ORDER, ordered=True)
    return summary.sort_values(["split_name", "config_name"]).reset_index(drop=True)


def _plot_summary(summary: pd.DataFrame) -> Path:
    HEEW_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), constrained_layout=True)
    x = np.arange(len(CONFIG_ORDER))

    for ax, value_col, ylabel, title in [
        (axes[0], "pooled_cv_rmse_mean", "CV(RMSE)", "Pooled CV(RMSE)"),
        (axes[1], "median_per_building_cv_rmse_mean", "Median per-building CV(RMSE)", "Median per-building CV(RMSE)"),
    ]:
        for split_name, color in [("t_split", MODEL_COLORS["lgbm_lag"]), ("b_split", MODEL_COLORS["lgbm"])]:
            subset = summary[summary["split_name"] == split_name].set_index("config_name").loc[CONFIG_ORDER]
            y = subset[value_col].to_numpy(dtype=float)
            sd_col = "pooled_cv_rmse_sd" if value_col == "pooled_cv_rmse_mean" else "median_per_building_cv_rmse_sd"
            yerr = subset[sd_col].to_numpy(dtype=float)
            label = "HEEW T-split" if split_name == "t_split" else "HEEW B-split (5 seeds)"
            ax.plot(x, y, marker="o", linewidth=2.0, color=color, label=label)
            if np.any(yerr > 0):
                ax.fill_between(x, y - yerr, y + yerr, color=color, alpha=0.16)
        ax.set_xticks(x)
        ax.set_xticklabels(CONFIG_ORDER)
        ax.set_xlabel("Lag configuration")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)

    axes[0].legend(frameon=False, loc="upper right")
    fig.savefig(FIGURE_PATH, dpi=320, bbox_inches="tight")
    plt.close(fig)
    return FIGURE_PATH


def run_heew_lag_ablation(
    cpu_threads: int | None,
    log_file: str | Path | None = None,
) -> dict[str, Path]:
    resolved_log = configure_logging(log_file)
    limits = apply_runtime_limits(max_cpu_threads=cpu_threads)
    LOGGER.info(
        "Runtime limits cpu_threads=%s memory_limit=%s applied=%s",
        limits.cpu_threads,
        format_bytes(limits.memory_bytes),
        limits.memory_limit_applied,
    )

    started = time.time()
    raw = _load_heew_raw()
    filtered, _ = _filter_heew(raw)
    frame = _prepare_model_frame(filtered)

    metric_rows: list[dict[str, object]] = []
    per_building_frames: list[pd.DataFrame] = []

    for spec in CONFIG_SPECS:
        config_name = spec["config_name"]
        lag_days = list(spec["lag_days"])
        LOGGER.info("HEEW lag ablation start config=%s lag_days=%s", config_name, lag_days)
        row, per_building = _run_config(
            frame=frame,
            split_name="t_split",
            split_seed=None,
            config_name=config_name,
            lag_days=lag_days,
            cpu_threads=cpu_threads,
        )
        metric_rows.append(row)
        per_building_frames.append(per_building)
        for split_seed in BSPLIT_SEEDS:
            row, per_building = _run_config(
                frame=frame,
                split_name="b_split",
                split_seed=split_seed,
                config_name=config_name,
                lag_days=lag_days,
                cpu_threads=cpu_threads,
            )
            metric_rows.append(row)
            per_building_frames.append(per_building)

    seed_metrics = pd.DataFrame(metric_rows)
    per_building_metrics = pd.concat(per_building_frames, ignore_index=True)
    summary = _build_summary(seed_metrics)

    seed_metrics.to_csv(SEED_METRICS_PATH, index=False)
    per_building_metrics.to_csv(PER_BUILDING_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)
    figure_path = _plot_summary(summary)

    elapsed = time.time() - started
    LOGGER.info("HEEW lag ablation finished in %.1f seconds", elapsed)
    return {
        "seed_metrics": SEED_METRICS_PATH,
        "per_building": PER_BUILDING_PATH,
        "summary": SUMMARY_PATH,
        "figure": figure_path,
        "log_file": resolved_log,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run stage-one HEEW lag-ablation for the parity-check baseline.")
    parser.add_argument("--max-cpu-threads", type=int, default=None)
    parser.add_argument("--log-file", type=str, default="logs/exp5_heew_lag_ablation.log")
    args = parser.parse_args()
    run_heew_lag_ablation(cpu_threads=args.max_cpu_threads, log_file=args.log_file)


if __name__ == "__main__":
    main()
