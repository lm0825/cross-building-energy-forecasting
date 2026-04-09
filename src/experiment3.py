from __future__ import annotations

import argparse
from datetime import datetime
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
from src.benchmarking.cps_calculator import (
    build_cps_frame,
    plot_cps_distribution_by_type,
)
from src.benchmarking.residual_calculator import (
    RESIDUAL_DIMENSION_COLUMNS,
    apply_low_mean_filter,
    build_residual_summary,
    load_prediction_frame,
)
from src.config import (
    FEATURES_BDG2_PATH,
    FILTERED_META_BDG2_PATH,
    FIGURES_DIR,
    LOGS_DIR,
    RESULTS_DIR,
    TABLES_DIR,
    ensure_phase2_dirs,
)
from src.data_splitting import B_SPLIT_PATH, load_pickle, unpack_mask
from src.runtime import apply_runtime_limits, format_bytes

try:
    from scipy.stats import spearmanr
except Exception:  # pragma: no cover - fallback only
    spearmanr = None


LOGGER = logging.getLogger(__name__)

EXP1_METRICS_PATH = TABLES_DIR / "exp1_metrics.csv"
EXP1_PREDICTIONS_DIR = RESULTS_DIR / "exp1_predictions"

EXP3_RESIDUALS_PATH = RESULTS_DIR / "exp3_residuals_per_building.csv"
EXP3_CPS_PATH = RESULTS_DIR / "exp3_cps_per_building.csv"
EXP3_EUI_VS_CPS_PATH = RESULTS_DIR / "exp3_eui_vs_cps_ranking.csv"

EXP3_CPS_DISTRIBUTION_FIG_PATH = FIGURES_DIR / "exp3_cps_distribution_by_type.png"
EXP3_RANK_SCATTER_FIG_PATH = FIGURES_DIR / "exp3_rank_scatter.png"
EXP3_TOP20_EUI_CPS_DIST_FIG_PATH = FIGURES_DIR / "exp3_top20_eui_cps_dist.png"
EXP3_CASE_STUDY_TYPE_A_FIG_PATH = FIGURES_DIR / "exp3_case_study_typeA.png"
EXP3_CASE_STUDY_TYPE_B_FIG_PATH = FIGURES_DIR / "exp3_case_study_typeB.png"

EXP3_SPEARMAN_PATH = TABLES_DIR / "exp3_spearman_by_type.csv"
EXP3_CASE_STUDY_PATH = TABLES_DIR / "exp3_case_study_buildings.csv"


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


def _select_best_b_split_model(metrics_path: str | Path = EXP1_METRICS_PATH) -> str:
    metrics = pd.read_csv(metrics_path)
    row = metrics.loc[metrics["split_name"] == "b_split"]
    if row.empty:
        raise RuntimeError("Could not find B-split metrics in exp1_metrics.csv.")
    candidates = {
        column.removesuffix("_cv_rmse"): float(row.iloc[0][column])
        for column in row.columns
        if column.endswith("_cv_rmse")
    }
    model_name = min(candidates, key=candidates.get)
    LOGGER.info("Experiment 3 selected best B-split model=%s metrics=%s", model_name, candidates)
    return model_name


def _output_paths(output_suffix: str = "") -> dict[str, Path]:
    suffix = output_suffix.strip()
    if suffix and not suffix.startswith("_"):
        suffix = f"_{suffix}"
    return {
        "exp3_residuals_per_building": RESULTS_DIR / f"exp3_residuals_per_building{suffix}.csv",
        "exp3_cps_per_building": RESULTS_DIR / f"exp3_cps_per_building{suffix}.csv",
        "exp3_eui_vs_cps_ranking": RESULTS_DIR / f"exp3_eui_vs_cps_ranking{suffix}.csv",
        "exp3_cps_distribution_by_type": FIGURES_DIR / f"exp3_cps_distribution_by_type{suffix}.png",
        "exp3_rank_scatter": FIGURES_DIR / f"exp3_rank_scatter{suffix}.png",
        "exp3_top20_eui_cps_dist": FIGURES_DIR / f"exp3_top20_eui_cps_dist{suffix}.png",
        "exp3_case_study_typeA": FIGURES_DIR / f"exp3_case_study_typeA{suffix}.png",
        "exp3_case_study_typeB": FIGURES_DIR / f"exp3_case_study_typeB{suffix}.png",
        "exp3_spearman_by_type": TABLES_DIR / f"exp3_spearman_by_type{suffix}.csv",
        "exp3_case_study_buildings": TABLES_DIR / f"exp3_case_study_buildings{suffix}.csv",
    }


def _load_filtered_metadata() -> pd.DataFrame:
    meta = pd.read_csv(FILTERED_META_BDG2_PATH)
    meta["building_id"] = meta["building_id"].astype(str)
    meta["site_id"] = meta["site_id"].astype(str)
    meta["building_type"] = meta["building_type"].astype(str)
    return meta


def _load_b_split_actual_frame() -> pd.DataFrame:
    artifact = load_pickle(B_SPLIT_PATH)
    mask = unpack_mask(artifact["test_mask"])
    frame = pd.read_parquet(
        FEATURES_BDG2_PATH,
        columns=["building_id", "site_id", "building_type", "floor_area", "meter_reading", "timestamp"],
    )
    frame["building_id"] = frame["building_id"].astype(str)
    frame["site_id"] = frame["site_id"].astype(str)
    frame["building_type"] = frame["building_type"].astype(str)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    return frame.loc[mask].reset_index(drop=True)


def _percentile_rank(series: pd.Series) -> pd.Series:
    valid = series.dropna()
    if valid.empty:
        return pd.Series(np.nan, index=series.index, dtype=np.float64)
    if len(valid) == 1:
        ranked = pd.Series(0.5, index=valid.index, dtype=np.float64)
    else:
        ranked = (valid.rank(method="average", ascending=True) - 1.0) / float(len(valid) - 1)
    return ranked.reindex(series.index)


def _build_eui_frame(
    actual_frame: pd.DataFrame,
    allowed_buildings: pd.DataFrame,
) -> pd.DataFrame:
    frame = actual_frame.merge(
        allowed_buildings[["building_id"]],
        on="building_id",
        how="inner",
        validate="many_to_one",
    )
    eui = (
        frame.groupby("building_id", sort=True)
        .agg(
            site_id=("site_id", "first"),
            building_type=("building_type", "first"),
            floor_area=("floor_area", "first"),
            total_actual=("meter_reading", "sum"),
            actual_start=("timestamp", "min"),
            actual_end=("timestamp", "max"),
        )
        .reset_index()
    )
    eui["eui"] = eui["total_actual"] / eui["floor_area"].replace({0.0: np.nan})
    return eui


def _build_eui_vs_cps_frame(
    cps_frame: pd.DataFrame,
    eui_frame: pd.DataFrame,
) -> pd.DataFrame:
    ranking = cps_frame.merge(
        eui_frame[["building_id", "eui", "total_actual", "actual_start", "actual_end"]],
        on="building_id",
        how="left",
        validate="one_to_one",
    )
    ranking["eui_percentile"] = ranking.groupby("building_type", sort=False)["eui"].transform(_percentile_rank)
    ranking["cps_percentile"] = ranking.groupby("building_type", sort=False)["cps"].transform(_percentile_rank)
    ranking["eui_top20_flag"] = ranking["eui_percentile"] <= 0.2
    ranking["cps_worst20_flag"] = ranking["cps_percentile"] >= 0.8
    ranking["cps_best20_flag"] = ranking["cps_percentile"] <= 0.2
    ranking["eui_worst20_flag"] = ranking["eui_percentile"] >= 0.8
    ranking["type_a_gap"] = ranking["cps_percentile"] - ranking["eui_percentile"]
    ranking["type_b_gap"] = ranking["eui_percentile"] - ranking["cps_percentile"]
    ranking["type_building_count"] = ranking.groupby("building_type", sort=False)["building_id"].transform("size")
    return ranking.sort_values(["building_type", "building_id"]).reset_index(drop=True)


def _compute_spearman_by_type(ranking_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    def _append_row(building_type: str, subset: pd.DataFrame) -> None:
        valid = subset.dropna(subset=["eui_percentile", "cps_percentile"])
        rho = float("nan")
        p_value = float("nan")
        if len(valid) >= 3:
            if spearmanr is not None:
                rho, p_value = spearmanr(valid["eui_percentile"], valid["cps_percentile"])
                rho = float(rho)
                p_value = float(p_value)
            else:
                rho = float(valid["eui_percentile"].corr(valid["cps_percentile"], method="spearman"))
        rows.append(
            {
                "building_type": building_type,
                "n_buildings": int(len(valid)),
                "spearman_rho": rho,
                "p_value": p_value,
                "eui_top20_count": int(valid["eui_top20_flag"].sum()),
                "cps_worst20_count": int(valid["cps_worst20_flag"].sum()),
            }
        )

    _append_row("ALL", ranking_frame)
    for building_type, subset in ranking_frame.groupby("building_type", sort=True):
        _append_row(str(building_type), subset)

    return pd.DataFrame(rows)


def _pick_diverse_cases(frame: pd.DataFrame, score_column: str, target_cases: int = 3) -> pd.DataFrame:
    ordered = frame.sort_values(score_column, ascending=False).reset_index(drop=True)
    selected_rows: list[pd.Series] = []
    seen_types: set[str] = set()

    for _, row in ordered.iterrows():
        if row["building_type"] in seen_types:
            continue
        selected_rows.append(row)
        seen_types.add(str(row["building_type"]))
        if len(selected_rows) >= target_cases:
            break

    if len(selected_rows) < target_cases:
        chosen_ids = {str(row["building_id"]) for row in selected_rows}
        for _, row in ordered.iterrows():
            if str(row["building_id"]) in chosen_ids:
                continue
            selected_rows.append(row)
            if len(selected_rows) >= target_cases:
                break

    if not selected_rows:
        return ordered.head(0).copy()
    return pd.DataFrame(selected_rows).reset_index(drop=True)


def _select_case_studies(ranking_frame: pd.DataFrame) -> pd.DataFrame:
    strict_a = ranking_frame.loc[
        ranking_frame["eui_top20_flag"] & ranking_frame["cps_worst20_flag"]
    ].copy()
    strict_a["discordance_gap"] = strict_a["type_a_gap"]
    if len(strict_a) < 2:
        strict_a = ranking_frame.loc[ranking_frame["type_a_gap"] > 0].copy()
        strict_a["discordance_gap"] = strict_a["type_a_gap"]
    type_a = _pick_diverse_cases(strict_a, "discordance_gap", target_cases=3)
    type_a["case_type"] = "A"

    strict_b = ranking_frame.loc[
        ranking_frame["eui_worst20_flag"] & ranking_frame["cps_best20_flag"]
    ].copy()
    strict_b["discordance_gap"] = strict_b["type_b_gap"]
    if len(strict_b) < 2:
        strict_b = ranking_frame.loc[ranking_frame["type_b_gap"] > 0].copy()
        strict_b["discordance_gap"] = strict_b["type_b_gap"]
    type_b = _pick_diverse_cases(strict_b, "discordance_gap", target_cases=3)
    type_b["case_type"] = "B"

    case_columns = [
        "case_type",
        "building_id",
        "site_id",
        "building_type",
        "floor_area",
        "model_name",
        "eui",
        "cps",
        "eui_percentile",
        "cps_percentile",
        "discordance_gap",
        *RESIDUAL_DIMENSION_COLUMNS,
    ]
    combined = pd.concat([type_a, type_b], ignore_index=True)
    combined = combined.drop_duplicates(subset=["case_type", "building_id"])
    return combined[case_columns].reset_index(drop=True)


def _plot_rank_scatter(
    ranking_frame: pd.DataFrame,
    case_frame: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    plt.figure(figsize=(8, 8))
    plt.scatter(
        ranking_frame["eui_percentile"],
        ranking_frame["cps_percentile"],
        s=28,
        alpha=0.35,
        color="#6d7b8d",
        edgecolors="none",
        label="All Buildings",
    )

    for case_type, color in [("A", "#b23a48"), ("B", "#2f6c8f")]:
        subset = case_frame.loc[case_frame["case_type"] == case_type]
        if subset.empty:
            continue
        plt.scatter(
            subset["eui_percentile"],
            subset["cps_percentile"],
            s=70,
            color=color,
            edgecolors="black",
            linewidths=0.5,
            label=f"Case {case_type}",
        )
        for _, row in subset.iterrows():
            plt.annotate(
                str(row["building_id"]),
                (row["eui_percentile"], row["cps_percentile"]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
                color=color,
            )

    plt.plot([0, 1], [0, 1], linestyle="--", color="#444444", linewidth=1.0, alpha=0.6)
    plt.axvline(0.2, linestyle=":", color="#888888", linewidth=1.0)
    plt.axvline(0.8, linestyle=":", color="#888888", linewidth=1.0)
    plt.axhline(0.2, linestyle=":", color="#888888", linewidth=1.0)
    plt.axhline(0.8, linestyle=":", color="#888888", linewidth=1.0)
    plt.xlabel("EUI Percentile Within Building Type (lower is better)")
    plt.ylabel("CPS Percentile Within Building Type (lower is better)")
    plt.title("Experiment 3 EUI Rank vs CPS Rank")
    plt.xlim(-0.02, 1.02)
    plt.ylim(-0.02, 1.02)
    plt.legend(loc="upper left")
    plt.grid(alpha=0.2, linestyle="--")
    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def _plot_top20_eui_cps_distribution(
    ranking_frame: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    all_scores = ranking_frame["cps_percentile"].dropna().to_numpy()
    top20_scores = ranking_frame.loc[ranking_frame["eui_top20_flag"], "cps_percentile"].dropna().to_numpy()
    bins = np.linspace(0.0, 1.0, 21)

    plt.figure(figsize=(9, 5))
    plt.hist(all_scores, bins=bins, alpha=0.45, color="#8da9c4", density=True, label="All Buildings")
    plt.hist(top20_scores, bins=bins, alpha=0.65, color="#b23a48", density=True, label="Top 20% by EUI")
    plt.axvline(float(np.mean(all_scores)), color="#4a5d73", linestyle="--", linewidth=1.5)
    if len(top20_scores) > 0:
        plt.axvline(float(np.mean(top20_scores)), color="#7f1d1d", linestyle="--", linewidth=1.5)
    plt.xlabel("CPS Percentile Within Building Type")
    plt.ylabel("Density")
    plt.title("Experiment 3 CPS Distribution for Buildings Ranked Best by EUI")
    plt.legend()
    plt.grid(axis="y", alpha=0.2, linestyle="--")
    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def _plot_case_study_grid(
    case_frame: pd.DataFrame,
    prediction_frame: pd.DataFrame,
    output_path: str | Path,
    title: str,
) -> Path:
    if case_frame.empty:
        plt.figure(figsize=(8, 3))
        plt.text(0.5, 0.5, "No case-study buildings were selected.", ha="center", va="center")
        plt.axis("off")
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close()
        return output_path

    subset = prediction_frame.loc[prediction_frame["building_id"].isin(case_frame["building_id"])].copy()
    subset["date"] = subset["timestamp"].dt.floor("D")
    daily = (
        subset.groupby(["building_id", "date"], sort=True)
        .agg(actual=("y_true", "mean"), predicted=("y_pred", "mean"))
        .reset_index()
    )
    daily["residual"] = daily["actual"] - daily["predicted"]

    n_rows = len(case_frame)
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, max(3.5 * n_rows, 4.5)), squeeze=False)
    fig.suptitle(title, fontsize=14, y=0.995)

    for row_idx, (_, case_row) in enumerate(case_frame.iterrows()):
        building_id = str(case_row["building_id"])
        building_daily = daily.loc[daily["building_id"] == building_id].copy()
        ax_load = axes[row_idx, 0]
        ax_resid = axes[row_idx, 1]

        ax_load.plot(building_daily["date"], building_daily["actual"], color="#1f4e79", linewidth=1.1, label="Actual")
        ax_load.plot(building_daily["date"], building_daily["predicted"], color="#b23a48", linewidth=1.0, label="Predicted")
        ax_load.set_title(f"{building_id} | {case_row['building_type']}")
        ax_load.set_ylabel("Daily Mean Load")
        ax_load.grid(alpha=0.2, linestyle="--")

        ax_resid.plot(building_daily["date"], building_daily["residual"], color="#4f772d", linewidth=1.0)
        ax_resid.axhline(0.0, color="#555555", linestyle="--", linewidth=1.0)
        ax_resid.set_title(
            f"Residual | EUI={case_row['eui_percentile']:.2f}, CPS={case_row['cps_percentile']:.2f}"
        )
        ax_resid.set_ylabel("Daily Mean Residual")
        ax_resid.grid(alpha=0.2, linestyle="--")

        if row_idx == 0:
            ax_load.legend(loc="upper right")

    for ax in axes[-1, :]:
        ax.set_xlabel("Date")

    plt.tight_layout(rect=[0, 0, 1, 0.985])
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def run_experiment3(
    model_name: str | None = None,
    output_suffix: str = "",
) -> dict[str, Path]:
    ensure_phase2_dirs()
    started_at = time.time()
    output_paths = _output_paths(output_suffix)
    resolved_model = model_name or _select_best_b_split_model()
    prediction_path = EXP1_PREDICTIONS_DIR / f"{resolved_model}_b_split.csv"
    if not prediction_path.exists():
        raise FileNotFoundError(f"Missing Experiment 1 prediction file: {prediction_path}")

    LOGGER.info("Experiment 3 using model=%s prediction_path=%s", resolved_model, prediction_path)
    prediction_frame = load_prediction_frame(prediction_path)
    low_mean_detail, low_mean_summary = build_low_mean_building_filter()
    LOGGER.info(
        "Benchmark filter threshold=%.6f excluded=%s/%s",
        float(low_mean_summary.iloc[0]["threshold_value"]),
        int(low_mean_summary.iloc[0]["n_buildings_excluded"]),
        int(low_mean_summary.iloc[0]["n_buildings_total"]),
    )
    filtered_prediction_frame = apply_low_mean_filter(prediction_frame, low_mean_detail)
    metadata = _load_filtered_metadata()

    residual_frame = build_residual_summary(
        prediction_frame=filtered_prediction_frame,
        metadata_frame=metadata,
        model_name=resolved_model,
    )
    residual_frame.to_csv(output_paths["exp3_residuals_per_building"], index=False)

    cps_frame = build_cps_frame(residual_frame)
    cps_frame.to_csv(output_paths["exp3_cps_per_building"], index=False)
    plot_cps_distribution_by_type(cps_frame, output_paths["exp3_cps_distribution_by_type"])

    actual_b_split = _load_b_split_actual_frame()
    allowed_buildings = low_mean_detail.loc[~low_mean_detail["exclude_from_benchmarking"], ["building_id"]].copy()
    eui_frame = _build_eui_frame(actual_b_split, allowed_buildings)
    ranking_frame = _build_eui_vs_cps_frame(cps_frame, eui_frame)
    ranking_frame.to_csv(output_paths["exp3_eui_vs_cps_ranking"], index=False)

    spearman_frame = _compute_spearman_by_type(ranking_frame)
    spearman_frame.to_csv(output_paths["exp3_spearman_by_type"], index=False)

    case_frame = _select_case_studies(ranking_frame)
    case_frame.to_csv(output_paths["exp3_case_study_buildings"], index=False)

    _plot_rank_scatter(ranking_frame, case_frame, output_paths["exp3_rank_scatter"])
    _plot_top20_eui_cps_distribution(ranking_frame, output_paths["exp3_top20_eui_cps_dist"])
    _plot_case_study_grid(
        case_frame.loc[case_frame["case_type"] == "A"].reset_index(drop=True),
        filtered_prediction_frame,
        output_paths["exp3_case_study_typeA"],
        title="Experiment 3 Case Studies: Type A (Good EUI, Poor CPS)",
    )
    _plot_case_study_grid(
        case_frame.loc[case_frame["case_type"] == "B"].reset_index(drop=True),
        filtered_prediction_frame,
        output_paths["exp3_case_study_typeB"],
        title="Experiment 3 Case Studies: Type B (Poor EUI, Good CPS)",
    )

    outputs = output_paths
    LOGGER.info("Experiment 3 finished in %.1f seconds", time.time() - started_at)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run experiment 3 for dynamic benchmarking against static EUI.")
    parser.add_argument(
        "--model-name",
        choices=["lgbm", "lgbm_lag", "lstm", "patchtst"],
        default=None,
        help="Override automatic best-model selection from Experiment 1 B-split.",
    )
    parser.add_argument(
        "--output-suffix",
        default="",
        help="Optional suffix appended to Exp3 output files, for example lgbm_lag.",
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
        help="Cap CPU thread usage for pandas, NumPy, and BLAS/OpenMP.",
    )
    parser.add_argument(
        "--cpu-fraction",
        type=float,
        default=None,
        help="Cap CPU usage as a fraction of total logical CPUs, for example 0.7.",
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
        resolved_log = configure_logging(LOGS_DIR / f"exp3_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
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
    outputs = run_experiment3(model_name=args.model_name, output_suffix=args.output_suffix)
    for name, path in outputs.items():
        print(f"{name}: {path}")
