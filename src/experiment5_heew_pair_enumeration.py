from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
from itertools import combinations
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
    HEEW_FIGURES_DIR,
    HEEW_LOGS_DIR,
    HEEW_TABLES_DIR,
    MODEL_COLORS,
    MODEL_LABELS,
    MODEL_ORDER,
    _build_model_specs,
    _compute_per_building_metrics,
    _compute_rowwise_metrics,
    _filter_heew,
    _fit_predict_model,
    _load_heew_raw,
    _prepare_model_frame,
    _summarize_per_building,
    configure_logging,
)
from src.models.common import set_seed
from src.runtime import apply_runtime_limits, format_bytes

LOGGER = logging.getLogger(__name__)

PAIR_METRICS_PATH = HEEW_TABLES_DIR / "heew_pair_enumeration_metrics.csv"
PAIR_PER_BUILDING_PATH = HEEW_TABLES_DIR / "heew_pair_enumeration_per_building_metrics.csv"
PAIR_SUMMARY_PATH = HEEW_TABLES_DIR / "heew_pair_enumeration_summary.csv"
PAIR_WIN_COUNTS_PATH = HEEW_TABLES_DIR / "heew_pair_enumeration_win_counts.csv"
PAIR_PAIRWISE_PATH = HEEW_TABLES_DIR / "heew_pair_enumeration_pairwise.csv"
PAIR_DOMINANCE_PATH = HEEW_TABLES_DIR / "heew_pair_enumeration_dominance_summary.csv"
PAIR_FIGURE_PATH = HEEW_FIGURES_DIR / "paper_figS2_heew_pair_enumeration.png"


def _ensure_dirs() -> None:
    HEEW_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    HEEW_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    HEEW_LOGS_DIR.mkdir(parents=True, exist_ok=True)


def _all_pairs(building_ids: list[str]) -> list[tuple[str, str]]:
    ordered = sorted(str(x) for x in building_ids)
    return [(left, right) for left, right in combinations(ordered, 2)]


def _assign_dominance_bins(pair_frame: pd.DataFrame) -> pd.DataFrame:
    ranked = pair_frame.copy()
    ranked["dominance_bin"] = pd.qcut(
        ranked["pair_max_load_share"],
        q=3,
        labels=["Lower-dominance tercile", "Middle-dominance tercile", "Higher-dominance tercile"],
        duplicates="drop",
    )
    return ranked


def _build_pair_figure(pair_metrics: pd.DataFrame) -> Path:
    metrics = pair_metrics.copy()
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.8))
    panels = [
        ("cv_rmse", "Pooled CV(RMSE) across 45 held-out pairs"),
        ("median_per_building_cv_rmse", "Median per-building CV(RMSE) across 45 held-out pairs"),
    ]

    for ax, (value_col, title) in zip(axes, panels):
        positions = np.arange(1, len(MODEL_ORDER) + 1)
        data = [metrics.loc[metrics["model"] == model_name, value_col].to_numpy(dtype=np.float64) for model_name in MODEL_ORDER]
        box = ax.boxplot(
            data,
            positions=positions,
            widths=0.58,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#222222", "linewidth": 1.6},
            whiskerprops={"color": "#444444", "linewidth": 1.1},
            capprops={"color": "#444444", "linewidth": 1.1},
        )
        for patch, model_name in zip(box["boxes"], MODEL_ORDER):
            patch.set_facecolor(MODEL_COLORS[model_name])
            patch.set_alpha(0.75)
            patch.set_edgecolor("#333333")
            patch.set_linewidth(1.0)

        rng = np.random.default_rng(42)
        for pos, model_name in zip(positions, MODEL_ORDER):
            values = metrics.loc[metrics["model"] == model_name, value_col].to_numpy(dtype=np.float64)
            jitter = rng.uniform(-0.14, 0.14, size=len(values))
            ax.scatter(
                np.full_like(values, pos, dtype=np.float64) + jitter,
                values,
                s=18,
                alpha=0.38,
                color=MODEL_COLORS[model_name],
                edgecolor="none",
            )

        ax.set_xticks(positions, [MODEL_LABELS[m] for m in MODEL_ORDER], rotation=12, ha="right")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2, linestyle="--")

    axes[0].set_ylabel("CV(RMSE)")
    fig.tight_layout()
    fig.savefig(PAIR_FIGURE_PATH, dpi=320, facecolor="white")
    plt.close(fig)
    return PAIR_FIGURE_PATH


def _run_pair_enumeration(
    frame: pd.DataFrame,
    models: list[str],
    model_specs: dict[str, object],
    device: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    building_ids = sorted(frame["building_id"].astype(str).unique().tolist())
    held_out_pairs = _all_pairs(building_ids)
    LOGGER.info("Enumerating %s HEEW held-out pairs from %s retained buildings", len(held_out_pairs), len(building_ids))

    metrics_rows: list[dict[str, object]] = []
    per_building_rows: list[pd.DataFrame] = []
    for pair_index, held_out_pair in enumerate(held_out_pairs, start=1):
        held_out_set = set(held_out_pair)
        train_frame = frame[~frame["building_id"].isin(held_out_set)].copy()
        test_frame = frame[frame["building_id"].isin(held_out_set)].copy()

        pair_actual = (
            test_frame.groupby("building_id", sort=True)["meter_reading"]
            .sum()
            .rename("total_actual_load")
            .reset_index()
        )
        total_pair_load = float(pair_actual["total_actual_load"].sum())
        max_pair_load = float(pair_actual["total_actual_load"].max())
        max_pair_share = float(max_pair_load / total_pair_load) if total_pair_load else np.nan
        dominant_building = str(
            pair_actual.sort_values(["total_actual_load", "building_id"], ascending=[False, True]).iloc[0]["building_id"]
        )
        pair_id = "|".join(held_out_pair)

        LOGGER.info(
            "HEEW pair %s/%s held_out=%s train_rows=%s test_rows=%s",
            pair_index,
            len(held_out_pairs),
            pair_id,
            len(train_frame),
            len(test_frame),
        )

        for model_name in models:
            prediction_frame = _fit_predict_model(
                model_name=model_name,
                train_frame=train_frame,
                test_frame=test_frame,
                split_name="b_split_pair_enum",
                model_specs=model_specs,
                device=device,
            )
            row_metrics = _compute_rowwise_metrics(prediction_frame)
            per_building = _compute_per_building_metrics(prediction_frame)
            per_building_summary = _summarize_per_building(per_building)

            metrics_rows.append(
                {
                    "dataset": "HEEW",
                    "pair_index": pair_index,
                    "pair_id": pair_id,
                    "held_out_buildings": pair_id,
                    "n_pair_buildings": len(held_out_pair),
                    "dominant_building": dominant_building,
                    "pair_total_actual_load": total_pair_load,
                    "pair_max_actual_load": max_pair_load,
                    "pair_max_load_share": max_pair_share,
                    "model": model_name,
                    **row_metrics,
                    **per_building_summary,
                }
            )

            enriched = per_building.copy()
            enriched["dataset"] = "HEEW"
            enriched["pair_index"] = pair_index
            enriched["pair_id"] = pair_id
            enriched["held_out_buildings"] = pair_id
            enriched["dominant_building"] = dominant_building
            enriched["pair_total_actual_load"] = total_pair_load
            enriched["pair_max_actual_load"] = max_pair_load
            enriched["pair_max_load_share"] = max_pair_share
            enriched["model"] = model_name
            per_building_rows.append(enriched)

    metrics = pd.DataFrame(metrics_rows)
    metrics = _assign_dominance_bins(metrics)
    per_building = pd.concat(per_building_rows, ignore_index=True)
    per_building = per_building.merge(
        metrics[["pair_index", "pair_id", "pair_max_load_share", "dominance_bin"]].drop_duplicates(),
        on=["pair_index", "pair_id", "pair_max_load_share"],
        how="left",
    )
    return metrics, per_building


def _summarize_pair_metrics(pair_metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    best_pooled = (
        pair_metrics.sort_values(["pair_index", "cv_rmse", "model"])
        .groupby("pair_index", as_index=False)
        .first()[["pair_index", "model"]]
        .rename(columns={"model": "best_model_pooled"})
    )
    best_per_building = (
        pair_metrics.sort_values(["pair_index", "median_per_building_cv_rmse", "model"])
        .groupby("pair_index", as_index=False)
        .first()[["pair_index", "model"]]
        .rename(columns={"model": "best_model_per_building"})
    )
    best = best_pooled.merge(best_per_building, on="pair_index", how="inner")

    summary_rows: list[dict[str, object]] = []
    for model_name, group in pair_metrics.groupby("model", sort=False):
        pooled = group["cv_rmse"].to_numpy(dtype=np.float64)
        per_building = group["median_per_building_cv_rmse"].to_numpy(dtype=np.float64)
        pooled_q1, pooled_q3, pooled_p90 = np.quantile(pooled, [0.25, 0.75, 0.90])
        per_q1, per_q3, per_p90 = np.quantile(per_building, [0.25, 0.75, 0.90])
        summary_rows.append(
            {
                "dataset": "HEEW",
                "model": model_name,
                "n_pairs": int(group["pair_index"].nunique()),
                "pooled_cv_rmse_mean": float(np.mean(pooled)),
                "pooled_cv_rmse_median": float(np.median(pooled)),
                "pooled_cv_rmse_q1": float(pooled_q1),
                "pooled_cv_rmse_q3": float(pooled_q3),
                "pooled_cv_rmse_iqr": float(pooled_q3 - pooled_q1),
                "pooled_cv_rmse_p90": float(pooled_p90),
                "median_per_building_cv_rmse_mean": float(np.mean(per_building)),
                "median_per_building_cv_rmse_median": float(np.median(per_building)),
                "median_per_building_cv_rmse_q1": float(per_q1),
                "median_per_building_cv_rmse_q3": float(per_q3),
                "median_per_building_cv_rmse_iqr": float(per_q3 - per_q1),
                "median_per_building_cv_rmse_p90": float(per_p90),
                "pooled_best_pair_count": int((best["best_model_pooled"] == model_name).sum()),
                "per_building_best_pair_count": int((best["best_model_per_building"] == model_name).sum()),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values("model").reset_index(drop=True)
    summary["pooled_best_pair_share"] = summary["pooled_best_pair_count"] / summary["n_pairs"]
    summary["per_building_best_pair_share"] = summary["per_building_best_pair_count"] / summary["n_pairs"]
    return summary, best


def _pairwise_win_summary(pair_metrics: pd.DataFrame, reference_model: str = "lgbm_lag") -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    ref = pair_metrics[pair_metrics["model"] == reference_model][
        ["pair_index", "cv_rmse", "median_per_building_cv_rmse"]
    ].rename(
        columns={
            "cv_rmse": "ref_pooled",
            "median_per_building_cv_rmse": "ref_per_building",
        }
    )
    for competitor in [m for m in MODEL_ORDER if m != reference_model]:
        comp = pair_metrics[pair_metrics["model"] == competitor][
            ["pair_index", "cv_rmse", "median_per_building_cv_rmse"]
        ].rename(
            columns={
                "cv_rmse": "comp_pooled",
                "median_per_building_cv_rmse": "comp_per_building",
            }
        )
        merged = ref.merge(comp, on="pair_index", how="inner")
        pooled_diff = merged["ref_pooled"] - merged["comp_pooled"]
        per_diff = merged["ref_per_building"] - merged["comp_per_building"]
        rows.append(
            {
                "dataset": "HEEW",
                "reference_model": reference_model,
                "competitor_model": competitor,
                "n_pairs": int(len(merged)),
                "reference_wins_pooled": int((pooled_diff < 0).sum()),
                "competitor_wins_pooled": int((pooled_diff > 0).sum()),
                "ties_pooled": int((np.isclose(pooled_diff, 0.0)).sum()),
                "reference_win_share_pooled": float((pooled_diff < 0).mean()),
                "median_pooled_difference_ref_minus_comp": float(np.median(pooled_diff)),
                "reference_wins_per_building": int((per_diff < 0).sum()),
                "competitor_wins_per_building": int((per_diff > 0).sum()),
                "ties_per_building": int((np.isclose(per_diff, 0.0)).sum()),
                "reference_win_share_per_building": float((per_diff < 0).mean()),
                "median_per_building_difference_ref_minus_comp": float(np.median(per_diff)),
            }
        )
    return pd.DataFrame(rows)


def _dominance_summary(pair_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (dominance_bin, model_name), group in pair_metrics.groupby(["dominance_bin", "model"], sort=False, observed=True):
        pooled = group["cv_rmse"].to_numpy(dtype=np.float64)
        per_building = group["median_per_building_cv_rmse"].to_numpy(dtype=np.float64)
        rows.append(
            {
                "dataset": "HEEW",
                "dominance_bin": str(dominance_bin),
                "model": model_name,
                "n_pairs": int(group["pair_index"].nunique()),
                "median_pair_max_load_share": float(np.median(group["pair_max_load_share"].to_numpy(dtype=np.float64))),
                "pooled_cv_rmse_median": float(np.median(pooled)),
                "pooled_cv_rmse_q1": float(np.quantile(pooled, 0.25)),
                "pooled_cv_rmse_q3": float(np.quantile(pooled, 0.75)),
                "median_per_building_cv_rmse_median": float(np.median(per_building)),
                "median_per_building_cv_rmse_q1": float(np.quantile(per_building, 0.25)),
                "median_per_building_cv_rmse_q3": float(np.quantile(per_building, 0.75)),
            }
        )
    return pd.DataFrame(rows)


def run_heew_pair_enumeration(
    models: list[str] | None = None,
    cpu_fraction: float | None = None,
    max_cpu_threads: int | None = None,
    device: str | None = None,
) -> dict[str, Path]:
    _ensure_dirs()
    set_seed(42)
    runtime_limits = apply_runtime_limits(max_cpu_threads=max_cpu_threads, cpu_fraction=cpu_fraction)
    resolved_device = device or "cpu"
    LOGGER.info(
        "HEEW pair enumeration runtime limits cpu_threads=%s memory_limit=%s device=%s",
        runtime_limits.cpu_threads,
        format_bytes(runtime_limits.memory_bytes),
        resolved_device,
    )

    started_at = time.time()
    raw = _load_heew_raw()
    filtered_raw, _ = _filter_heew(raw)
    frame = _prepare_model_frame(filtered_raw)

    models = models or MODEL_ORDER
    model_specs = _build_model_specs(runtime_limits.cpu_threads, resolved_device)
    pair_metrics, pair_per_building = _run_pair_enumeration(
        frame=frame,
        models=models,
        model_specs=model_specs,
        device=resolved_device,
    )
    pair_summary, pair_best = _summarize_pair_metrics(pair_metrics)
    pair_pairwise = _pairwise_win_summary(pair_metrics)
    dominance_summary = _dominance_summary(pair_metrics)

    pair_metrics.to_csv(PAIR_METRICS_PATH, index=False)
    pair_per_building.to_csv(PAIR_PER_BUILDING_PATH, index=False)
    pair_summary.to_csv(PAIR_SUMMARY_PATH, index=False)
    pair_best.to_csv(PAIR_WIN_COUNTS_PATH, index=False)
    pair_pairwise.to_csv(PAIR_PAIRWISE_PATH, index=False)
    dominance_summary.to_csv(PAIR_DOMINANCE_PATH, index=False)
    _build_pair_figure(pair_metrics)

    elapsed = time.time() - started_at
    LOGGER.info("HEEW pair enumeration finished in %.1f seconds", elapsed)
    return {
        "pair_metrics": PAIR_METRICS_PATH,
        "pair_per_building": PAIR_PER_BUILDING_PATH,
        "pair_summary": PAIR_SUMMARY_PATH,
        "pair_best": PAIR_WIN_COUNTS_PATH,
        "pair_pairwise": PAIR_PAIRWISE_PATH,
        "dominance_summary": PAIR_DOMINANCE_PATH,
        "pair_figure": PAIR_FIGURE_PATH,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enumerate all HEEW held-out building pairs.")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_ORDER,
        default=MODEL_ORDER,
        help="Models to run.",
    )
    parser.add_argument("--device", default="cpu", help="Torch device for sequence models.")
    parser.add_argument("--cpu-fraction", type=float, default=None, help="CPU thread fraction.")
    parser.add_argument("--max-cpu-threads", type=int, default=None, help="Maximum CPU threads.")
    parser.add_argument("--log-file", type=str, default=None, help="Optional log file path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.log_file)
    LOGGER.info("HEEW pair enumeration started at %s", datetime.now().isoformat())
    run_heew_pair_enumeration(
        models=args.models,
        cpu_fraction=args.cpu_fraction,
        max_cpu_threads=args.max_cpu_threads,
        device=args.device,
    )


if __name__ == "__main__":
    main()
