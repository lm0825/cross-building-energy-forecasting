from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import FEATURES_BDG2_PATH, TABLES_DIR, ensure_phase2_dirs


DEFAULT_LOW_MEAN_QUANTILE = 0.05
BENCHMARK_LOW_MEAN_FILTER_PATH = TABLES_DIR / "benchmark_low_mean_filter.csv"
BENCHMARK_LOW_MEAN_SUMMARY_PATH = TABLES_DIR / "benchmark_low_mean_filter_summary.csv"


def build_low_mean_building_filter(
    feature_path: str | Path = FEATURES_BDG2_PATH,
    quantile: float = DEFAULT_LOW_MEAN_QUANTILE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.read_parquet(feature_path, columns=["building_id", "meter_reading"])
    mean_frame = (
        frame.groupby("building_id", sort=True)["meter_reading"]
        .mean()
        .rename("mean_actual")
        .reset_index()
        .sort_values(["mean_actual", "building_id"])
        .reset_index(drop=True)
    )

    threshold = float(mean_frame["mean_actual"].quantile(quantile))
    mean_frame["exclude_from_benchmarking"] = mean_frame["mean_actual"] < threshold
    mean_frame["threshold_quantile"] = quantile
    mean_frame["threshold_value"] = threshold

    summary = pd.DataFrame(
        [
            {
                "threshold_quantile": quantile,
                "threshold_value": threshold,
                "n_buildings_total": int(len(mean_frame)),
                "n_buildings_excluded": int(mean_frame["exclude_from_benchmarking"].sum()),
                "excluded_fraction": float(mean_frame["exclude_from_benchmarking"].mean()),
            }
        ]
    )
    return mean_frame, summary


def export_low_mean_building_filter(
    feature_path: str | Path = FEATURES_BDG2_PATH,
    quantile: float = DEFAULT_LOW_MEAN_QUANTILE,
) -> dict[str, Path]:
    ensure_phase2_dirs()
    detail, summary = build_low_mean_building_filter(feature_path=feature_path, quantile=quantile)
    detail.to_csv(BENCHMARK_LOW_MEAN_FILTER_PATH, index=False)
    summary.to_csv(BENCHMARK_LOW_MEAN_SUMMARY_PATH, index=False)
    return {
        "benchmark_low_mean_filter": BENCHMARK_LOW_MEAN_FILTER_PATH,
        "benchmark_low_mean_filter_summary": BENCHMARK_LOW_MEAN_SUMMARY_PATH,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export the low-mean building filter used before benchmarking.")
    parser.add_argument(
        "--feature-path",
        default=FEATURES_BDG2_PATH,
        help="Path to the feature parquet used to compute per-building mean actual load.",
    )
    parser.add_argument(
        "--quantile",
        type=float,
        default=DEFAULT_LOW_MEAN_QUANTILE,
        help="Lower-tail quantile used to exclude ultra-low-load buildings.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    outputs = export_low_mean_building_filter(feature_path=args.feature_path, quantile=args.quantile)
    for name, path in outputs.items():
        print(f"{name}: {path}")
