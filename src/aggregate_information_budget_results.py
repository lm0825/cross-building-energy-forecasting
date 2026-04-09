from __future__ import annotations

import argparse
import re
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import TABLES_DIR
from src.experiment7_information_budget import SPEC_ORDER, SPEC_SPECS


LOG_PATTERN = re.compile(
    r"Finished spec=(?P<spec_name>\S+)\s+split=(?P<split_name>\S+)\s+seed=(?P<seed>\d+)\s+"
    r"pooled_cv_rmse=(?P<pooled_cv_rmse>[0-9.]+)\s+median_per_building=(?P<median_per_building>[0-9.]+)"
)


def parse_log_seed_metrics(path: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = LOG_PATTERN.search(line)
        if not match:
            continue
        spec_name = match.group("spec_name")
        spec = SPEC_SPECS[spec_name]
        rows.append(
            {
                "split_name": match.group("split_name"),
                "spec_name": spec_name,
                "seed": int(match.group("seed")),
                "family": spec["family"],
                "history_budget": spec["history_budget"],
                "context_hours": int(spec["context_hours"]),
                "pooled_cv_rmse": float(match.group("pooled_cv_rmse")),
                "median_per_building_cv_rmse": float(match.group("median_per_building")),
            }
        )
    return pd.DataFrame(rows)


def load_seed_metrics(csv_paths: list[Path], log_paths: list[Path]) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for csv_path in csv_paths:
        frame = pd.read_csv(csv_path)
        keep = [
            "split_name",
            "spec_name",
            "seed",
            "family",
            "history_budget",
            "context_hours",
            "pooled_cv_rmse",
            "median_per_building_cv_rmse",
        ]
        parts.append(frame[keep].copy())
    for log_path in log_paths:
        parts.append(parse_log_seed_metrics(log_path))
    if not parts:
        raise ValueError("No seed metrics sources were provided.")
    combined = pd.concat(parts, ignore_index=True)
    combined = combined.drop_duplicates(subset=["split_name", "spec_name", "seed"], keep="last")
    spec_rank = {name: idx for idx, name in enumerate(SPEC_ORDER)}
    split_rank = {"t_split": 0, "b_split": 1, "s_split": 2}
    combined = combined.sort_values(
        ["split_name", "spec_name", "seed"],
        key=lambda col: col.map(split_rank | spec_rank).fillna(999),
    ).reset_index(drop=True)
    return combined


def summarize(seed_metrics: pd.DataFrame) -> pd.DataFrame:
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
        )
    )
    spec_rank = {name: idx for idx, name in enumerate(SPEC_ORDER)}
    split_rank = {"t_split": 0, "b_split": 1, "s_split": 2}
    summary = summary.sort_values(
        ["split_name", "spec_name"],
        key=lambda col: col.map(split_rank | spec_rank).fillna(999),
    ).reset_index(drop=True)
    return summary


def compute_improvements(summary: pd.DataFrame) -> pd.DataFrame:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate information-budget results from CSV outputs and/or logs.")
    parser.add_argument("--seed-csv", action="append", default=[], help="Seed-metrics CSV path. Can be repeated.")
    parser.add_argument("--log", action="append", default=[], help="Experiment log to parse. Can be repeated.")
    parser.add_argument("--output-suffix", default="", help="Suffix for output tables.")
    args = parser.parse_args()

    csv_paths = [Path(path) if Path(path).is_absolute() else ROOT / path for path in args.seed_csv]
    log_paths = [Path(path) if Path(path).is_absolute() else ROOT / path for path in args.log]

    seed_metrics = load_seed_metrics(csv_paths, log_paths)
    summary = summarize(seed_metrics)
    improvements = compute_improvements(summary)

    suffix = args.output_suffix
    if suffix and not suffix.startswith("_"):
        suffix = f"_{suffix}"

    seed_path = TABLES_DIR / f"information_budget_seed_metrics_combined{suffix}.csv"
    summary_path = TABLES_DIR / f"information_budget_summary_main_combined{suffix}.csv"
    improvements_path = TABLES_DIR / f"information_budget_improvement_vs_no_history_combined{suffix}.csv"

    seed_metrics.to_csv(seed_path, index=False)
    summary.to_csv(summary_path, index=False)
    improvements.to_csv(improvements_path, index=False)

    print(seed_path)
    print(summary_path)
    print(improvements_path)


if __name__ == "__main__":
    main()
