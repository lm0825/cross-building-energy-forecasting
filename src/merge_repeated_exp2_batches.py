from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = ROOT / "tables"


FILE_STEMS = {
    "strategy_seed": "repeated_exp2_strategy_metrics",
    "strategy_summary": "repeated_exp2_strategy_metrics",
    "cold_seed": "repeated_exp2_cold_start_metrics",
    "cold_summary": "repeated_exp2_cold_start_metrics",
    "accuracy_seed": "repeated_exp2_grouping_accuracy",
    "accuracy_summary": "repeated_exp2_grouping_accuracy",
    "cluster_sizes": "repeated_exp2_cluster_sizes",
}


def _suffix(label: str | None) -> str:
    return f"_{label}" if label else ""


def _read_csv(stem: str, label: str, summary: bool = False) -> pd.DataFrame:
    suffix = _suffix(label)
    extra = "_summary" if summary else ""
    path = TABLES_DIR / f"{stem}{suffix}{extra}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _dedupe_columns(frame: pd.DataFrame) -> list[str]:
    preferred = [
        "model",
        "strategy",
        "history_days",
        "random_seed",
        "selected_k",
        "n_rows",
        "n_buildings",
        "mae",
        "rmse",
        "cv_rmse",
        "accuracy",
    ]
    return [col for col in preferred if col in frame.columns]


def _summarize(seed_frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
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
            values = group[metric].to_numpy(dtype=float)
            mean = float(values.mean())
            std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            half_width = 1.96 * std / (len(values) ** 0.5) if len(values) > 1 else 0.0
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            row[f"{metric}_ci95_low"] = mean - half_width
            row[f"{metric}_ci95_high"] = mean + half_width
            row[f"{metric}_min"] = float(values.min())
            row[f"{metric}_max"] = float(values.max())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def _merge_seed_files(stem: str, labels: list[str], output_label: str) -> pd.DataFrame:
    frames = [_read_csv(stem, label, summary=False) for label in labels]
    merged = pd.concat(frames, ignore_index=True)
    dedupe_cols = _dedupe_columns(merged)
    if dedupe_cols:
        merged = merged.drop_duplicates(subset=dedupe_cols).reset_index(drop=True)
    merged.to_csv(TABLES_DIR / f"{stem}{_suffix(output_label)}.csv", index=False)
    return merged


def merge_batches(labels: list[str], output_label: str) -> None:
    strategy_seed = _merge_seed_files(FILE_STEMS["strategy_seed"], labels, output_label)
    cold_seed = _merge_seed_files(FILE_STEMS["cold_seed"], labels, output_label)
    accuracy_seed = _merge_seed_files(FILE_STEMS["accuracy_seed"], labels, output_label)
    cluster_sizes = _merge_seed_files(FILE_STEMS["cluster_sizes"], labels, output_label)

    strategy_summary = _summarize(strategy_seed, ["model", "strategy"])
    strategy_summary.to_csv(TABLES_DIR / f"{FILE_STEMS['strategy_summary']}{_suffix(output_label)}_summary.csv", index=False)

    cold_summary = _summarize(cold_seed, ["model", "history_days", "strategy"])
    cold_summary.to_csv(TABLES_DIR / f"{FILE_STEMS['cold_summary']}{_suffix(output_label)}_summary.csv", index=False)

    accuracy_summary = _summarize(accuracy_seed, ["history_days"])
    accuracy_summary.to_csv(TABLES_DIR / f"{FILE_STEMS['accuracy_summary']}{_suffix(output_label)}_summary.csv", index=False)

    # Keep cluster size rows in deterministic order.
    cluster_sizes = cluster_sizes.sort_values(["random_seed", "cluster_label"]).reset_index(drop=True)
    cluster_sizes.to_csv(TABLES_DIR / f"{FILE_STEMS['cluster_sizes']}{_suffix(output_label)}.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge repeated Experiment 2 seed batches and recompute summaries.")
    parser.add_argument("--labels", nargs="+", required=True, help="Input output labels to merge, e.g. lgbm lgbm_add2.")
    parser.add_argument("--output-label", required=True, help="Merged output label.")
    args = parser.parse_args()
    merge_batches(labels=args.labels, output_label=args.output_label)


if __name__ == "__main__":
    main()
