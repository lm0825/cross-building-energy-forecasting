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

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.benchmark_filtering import build_low_mean_building_filter
from src.benchmarking.cps_calculator import build_cps_frame
from src.benchmarking.residual_calculator import apply_low_mean_filter, build_residual_summary
from src.config import (
    FEATURES_BDG2_PATH,
    FIGURES_DIR,
    LOGS_DIR,
    MODELS_DIR,
    TABLES_DIR,
    ensure_phase2_dirs,
)
from src.data_splitting import B_SPLIT_PATH, load_pickle, unpack_mask
from src.experiment3 import (
    _build_eui_frame,
    _build_eui_vs_cps_frame,
    _compute_spearman_by_type,
    _load_b_split_actual_frame,
    _load_filtered_metadata,
)
from src.models.lgbm_model import LightGBMConfig, LightGBMExperimentModel
from src.models.lstm_model import LSTMConfig, LSTMExperimentModel
from src.models.patchtst_model import PatchTSTConfig, PatchTSTExperimentModel
from src.models.common import add_tabular_lag_features
from src.runtime import apply_runtime_limits, format_bytes


LOGGER = logging.getLogger(__name__)

TUNING_DIR = MODELS_DIR / "_tuning"


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
    frame = pd.read_parquet(FEATURES_BDG2_PATH)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame = frame.sort_values(["building_id", "timestamp"]).reset_index(drop=True)
    if with_lag_features:
        frame = add_tabular_lag_features(frame)
    LOGGER.info(
        "Loaded BDG2 features rows=%s buildings=%s sites=%s lag_features=%s",
        len(frame),
        frame["building_id"].nunique(),
        frame["site_id"].nunique(),
        with_lag_features,
    )
    return frame


def _select_rows(frame: pd.DataFrame, packed_mask: dict[str, object]) -> pd.DataFrame:
    mask = unpack_mask(packed_mask)
    return frame.loc[mask].reset_index(drop=True)


def _resolve_tuning_path(
    model_name: str,
    *,
    split_name: str | None = None,
    fold_id: str | None = None,
) -> Path:
    candidates: list[Path] = []
    if split_name is not None and fold_id is not None:
        candidates.append(TUNING_DIR / f"{model_name}.{split_name}.{fold_id}.config.json")
    if split_name is not None:
        candidates.append(TUNING_DIR / f"{model_name}.{split_name}.config.json")
    candidates.append(TUNING_DIR / f"{model_name}.config.json")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[-1]


def _load_lgbm_config(
    cpu_threads: int | None = None,
    model_name: str = "lgbm",
    *,
    split_name: str | None = None,
) -> LightGBMConfig:
    tuning_path = _resolve_tuning_path(model_name, split_name=split_name)
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


def _load_lstm_config(*, split_name: str | None = None) -> LSTMConfig:
    payload = json.loads(_resolve_tuning_path("lstm", split_name=split_name).read_text(encoding="utf-8"))
    return LSTMConfig(**payload)


def _load_patchtst_config(*, split_name: str | None = None) -> PatchTSTConfig:
    payload = json.loads(_resolve_tuning_path("patchtst", split_name=split_name).read_text(encoding="utf-8"))
    return PatchTSTConfig(**payload)


def _fit_and_predict_b_split(
    model_name: str,
    random_seed: int,
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    cpu_threads: int | None,
    device: str | None,
) -> pd.DataFrame:
    LOGGER.info(
        "Repeated Experiment 3 run model=%s seed=%s train_rows=%s test_rows=%s",
        model_name,
        random_seed,
        len(train_frame),
        len(test_frame),
    )
    if model_name in {"lgbm", "lgbm_lag"}:
        payload = asdict(_load_lgbm_config(cpu_threads=cpu_threads, model_name=model_name, split_name="b_split"))
        payload["random_state"] = random_seed
        if cpu_threads is not None:
            payload["n_jobs"] = cpu_threads
        payload["target_transform"] = "log1p"
        payload["use_lag_features"] = model_name == "lgbm_lag"
        model = LightGBMExperimentModel(config=LightGBMConfig(**payload))
        model.fit(train_frame)
        return model.predict_frame(
            test_frame,
            split_name="b_split",
            fold_id=None,
            model_name=model_name,
        )

    if model_name == "lstm":
        payload = asdict(_load_lstm_config(split_name="b_split"))
        payload["random_seed"] = random_seed
        model = LSTMExperimentModel(config=LSTMConfig(**payload), device=device)
        model.fit(train_frame)
        return model.predict_frame(test_frame, split_name="b_split", fold_id=None)

    if model_name == "patchtst":
        payload = asdict(_load_patchtst_config(split_name="b_split"))
        payload["random_seed"] = random_seed
        model = PatchTSTExperimentModel(config=PatchTSTConfig(**payload), device=device)
        model.fit(train_frame)
        return model.predict_frame(test_frame, split_name="b_split", fold_id=None)

    raise ValueError(f"Unsupported model name: {model_name}")


def _summarize_single_seed(
    model_name: str,
    random_seed: int,
    prediction_frame: pd.DataFrame,
    metadata: pd.DataFrame,
    low_mean_detail: pd.DataFrame,
    eui_frame: pd.DataFrame,
) -> tuple[dict[str, object], pd.DataFrame]:
    filtered_prediction_frame = apply_low_mean_filter(prediction_frame, low_mean_detail)
    residual_frame = build_residual_summary(
        prediction_frame=filtered_prediction_frame,
        metadata_frame=metadata,
        model_name=model_name,
    )
    cps_frame = build_cps_frame(residual_frame)
    ranking_frame = _build_eui_vs_cps_frame(cps_frame, eui_frame)
    spearman_frame = _compute_spearman_by_type(ranking_frame)

    overall = spearman_frame.loc[spearman_frame["building_type"] == "ALL"].iloc[0]
    summary_row = {
        "model_name": model_name,
        "random_seed": int(random_seed),
        "n_buildings": int(overall["n_buildings"]),
        "overall_spearman_rho": float(overall["spearman_rho"]) if pd.notna(overall["spearman_rho"]) else np.nan,
        "overall_p_value": float(overall["p_value"]) if pd.notna(overall["p_value"]) else np.nan,
        "top20_eui_count": int(overall["eui_top20_count"]),
        "worst20_cps_count": int(overall["cps_worst20_count"]),
        "strict_typeA_count": int((ranking_frame["eui_top20_flag"] & ranking_frame["cps_worst20_flag"]).sum()),
        "strict_typeB_count": int((ranking_frame["eui_worst20_flag"] & ranking_frame["cps_best20_flag"]).sum()),
        "mean_cps_percentile_top20_eui": float(ranking_frame.loc[ranking_frame["eui_top20_flag"], "cps_percentile"].mean()),
        "mean_cps_percentile_all": float(ranking_frame["cps_percentile"].mean()),
    }

    by_type = spearman_frame.loc[spearman_frame["building_type"] != "ALL"].copy()
    by_type["model_name"] = model_name
    by_type["random_seed"] = int(random_seed)
    return summary_row, by_type.reset_index(drop=True)


def _summarize_seed_distribution(seed_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model_name, group in seed_frame.groupby("model_name", sort=False):
        values = group["overall_spearman_rho"].to_numpy(dtype=np.float64)
        mean = float(values.mean())
        std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        half_width = 1.96 * std / np.sqrt(len(values)) if len(values) > 1 else 0.0
        rows.append(
            {
                "model_name": model_name,
                "n_seeds": int(len(group)),
                "overall_spearman_rho_mean": mean,
                "overall_spearman_rho_std": std,
                "overall_spearman_rho_ci95_low": mean - half_width,
                "overall_spearman_rho_ci95_high": mean + half_width,
                "overall_spearman_rho_min": float(values.min()),
                "overall_spearman_rho_max": float(values.max()),
                "strict_typeA_count_mean": float(group["strict_typeA_count"].mean()),
                "strict_typeB_count_mean": float(group["strict_typeB_count"].mean()),
                "mean_cps_percentile_top20_eui_mean": float(group["mean_cps_percentile_top20_eui"].mean()),
                "mean_cps_percentile_all_mean": float(group["mean_cps_percentile_all"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _output_paths(output_suffix: str = "") -> dict[str, Path]:
    suffix = output_suffix.strip()
    if suffix and not suffix.startswith("_"):
        suffix = f"_{suffix}"
    return {
        "seed": TABLES_DIR / f"exp3_seed_sensitivity{suffix}.csv",
        "summary": TABLES_DIR / f"exp3_seed_sensitivity_summary{suffix}.csv",
        "by_type": TABLES_DIR / f"exp3_seed_sensitivity_by_type{suffix}.csv",
        "figure": FIGURES_DIR / f"exp3_model_sensitivity_intervals{suffix}.png",
    }


def _plot_seed_sensitivity_intervals(seed_frame: pd.DataFrame, summary_frame: pd.DataFrame, figure_path: Path) -> Path:
    order = [
        model for model in ["lgbm", "lgbm_lag", "lstm", "patchtst"]
        if model in summary_frame["model_name"].tolist()
    ]
    x = np.arange(len(order), dtype=np.float64)
    colors = {
        "lgbm": "#44617b",
        "lgbm_lag": "#1f7a8c",
        "lstm": "#b23a48",
        "patchtst": "#2f6c8f",
    }

    plt.figure(figsize=(9, 5.5))
    for idx, model_name in enumerate(order):
        seed_subset = seed_frame.loc[seed_frame["model_name"] == model_name].copy()
        jitter = np.linspace(-0.08, 0.08, num=len(seed_subset)) if len(seed_subset) > 1 else np.array([0.0])
        plt.scatter(
            np.full(len(seed_subset), x[idx]) + jitter,
            seed_subset["overall_spearman_rho"],
            s=44,
            color=colors[model_name],
            alpha=0.75,
            edgecolors="white",
            linewidths=0.6,
            zorder=3,
        )

        summary = summary_frame.loc[summary_frame["model_name"] == model_name].iloc[0]
        mean = float(summary["overall_spearman_rho_mean"])
        low = float(summary["overall_spearman_rho_ci95_low"])
        high = float(summary["overall_spearman_rho_ci95_high"])
        plt.errorbar(
            x[idx],
            mean,
            yerr=np.array([[mean - low], [high - mean]]),
            fmt="o",
            color="black",
            ecolor="black",
            elinewidth=1.4,
            capsize=5,
            markersize=6,
            zorder=4,
        )

    plt.axhline(0.0, color="#555555", linestyle="--", linewidth=1.0, alpha=0.8)
    plt.xticks(x, order)
    plt.ylabel("Overall Spearman rho between EUI rank and CPS rank")
    plt.title("Experiment 3 Multi-Seed Model Sensitivity")
    plt.grid(axis="y", alpha=0.2, linestyle="--")
    plt.tight_layout()
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(figure_path, dpi=200)
    plt.close()
    return figure_path


def run_repeated_exp3_sensitivity(
    models: list[str],
    seeds: list[int],
    cpu_threads: int | None,
    device: str | None,
    output_suffix: str = "",
) -> dict[str, Path]:
    ensure_phase2_dirs()
    started_at = time.time()
    output_paths = _output_paths(output_suffix)

    frame = _load_full_features(with_lag_features="lgbm_lag" in models)
    split_artifact = load_pickle(B_SPLIT_PATH)
    train_frame = _select_rows(frame, split_artifact["train_mask"])
    test_frame = _select_rows(frame, split_artifact["test_mask"])

    metadata = _load_filtered_metadata()
    low_mean_detail, low_mean_summary = build_low_mean_building_filter()
    actual_b_split = _load_b_split_actual_frame()
    allowed_buildings = low_mean_detail.loc[
        ~low_mean_detail["exclude_from_benchmarking"],
        ["building_id"],
    ].copy()
    eui_frame = _build_eui_frame(actual_b_split, allowed_buildings)

    LOGGER.info(
        "Repeated Experiment 3 setup train_rows=%s test_rows=%s seeds=%s benchmark_threshold=%.6f excluded=%s/%s",
        len(train_frame),
        len(test_frame),
        seeds,
        float(low_mean_summary.iloc[0]["threshold_value"]),
        int(low_mean_summary.iloc[0]["n_buildings_excluded"]),
        int(low_mean_summary.iloc[0]["n_buildings_total"]),
    )

    seed_rows: list[dict[str, object]] = []
    by_type_frames: list[pd.DataFrame] = []

    def _write_progress() -> None:
        if not seed_rows:
            return
        seed_frame = pd.DataFrame(seed_rows).sort_values(["model_name", "random_seed"]).reset_index(drop=True)
        by_type_frame = pd.concat(by_type_frames, ignore_index=True).sort_values(
            ["model_name", "random_seed", "building_type"]
        ).reset_index(drop=True)
        summary_frame = _summarize_seed_distribution(seed_frame).sort_values("model_name").reset_index(drop=True)
        seed_frame.to_csv(output_paths["seed"], index=False)
        summary_frame.to_csv(output_paths["summary"], index=False)
        by_type_frame.to_csv(output_paths["by_type"], index=False)
        _plot_seed_sensitivity_intervals(seed_frame, summary_frame, output_paths["figure"])

    try:
        for model_name in models:
            for random_seed in seeds:
                prediction_frame = _fit_and_predict_b_split(
                    model_name=model_name,
                    random_seed=random_seed,
                    train_frame=train_frame,
                    test_frame=test_frame,
                    cpu_threads=cpu_threads,
                    device=device,
                )
                summary_row, by_type = _summarize_single_seed(
                    model_name=model_name,
                    random_seed=random_seed,
                    prediction_frame=prediction_frame,
                    metadata=metadata,
                    low_mean_detail=low_mean_detail,
                    eui_frame=eui_frame,
                )
                seed_rows.append(summary_row)
                by_type_frames.append(by_type)
                LOGGER.info(
                    "Repeated Experiment 3 model=%s seed=%s overall_rho=%.6f p=%.6g",
                    model_name,
                    random_seed,
                    summary_row["overall_spearman_rho"],
                    summary_row["overall_p_value"],
                )
                _write_progress()
                del prediction_frame
                gc.collect()
    finally:
        del train_frame
        del test_frame
        del frame
        gc.collect()

    seed_frame = pd.DataFrame(seed_rows).sort_values(["model_name", "random_seed"]).reset_index(drop=True)
    by_type_frame = pd.concat(by_type_frames, ignore_index=True).sort_values(
        ["model_name", "random_seed", "building_type"]
    ).reset_index(drop=True)
    summary_frame = _summarize_seed_distribution(seed_frame).sort_values("model_name").reset_index(drop=True)

    seed_frame.to_csv(output_paths["seed"], index=False)
    summary_frame.to_csv(output_paths["summary"], index=False)
    by_type_frame.to_csv(output_paths["by_type"], index=False)
    _plot_seed_sensitivity_intervals(seed_frame, summary_frame, output_paths["figure"])

    LOGGER.info("Repeated Experiment 3 sensitivity finished in %.1f seconds", time.time() - started_at)
    return {
        "exp3_seed_sensitivity": output_paths["seed"],
        "exp3_seed_sensitivity_summary": output_paths["summary"],
        "exp3_seed_sensitivity_by_type": output_paths["by_type"],
        "exp3_model_sensitivity_intervals": output_paths["figure"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run repeated-seed Experiment 3 CPS-EUI sensitivity checks.")
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
            LOGS_DIR / f"repeated_exp3_sensitivity_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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
    outputs = run_repeated_exp3_sensitivity(
        models=args.models,
        seeds=args.seeds,
        cpu_threads=limits.cpu_threads,
        device=args.device,
        output_suffix=args.output_suffix,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")
