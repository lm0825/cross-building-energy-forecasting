from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = ROOT / "tables"


def _suffix(label: str | None) -> str:
    if label in {None, "", "base"}:
        return ""
    return f"_{label}" if label else ""


def _read_seed_csv(dataset: str, label: str) -> pd.DataFrame:
    path = TABLES_DIR / f"repeated_main_metrics_{dataset}{_suffix(label)}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _summarize(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (dataset, split_name, model), group in seed_metrics.groupby(["dataset", "split_name", "model"], sort=True):
        row: dict[str, object] = {
            "dataset": dataset,
            "split_name": split_name,
            "model": model,
            "n_seeds": int(group["random_seed"].nunique()),
        }
        for metric in ["mae", "rmse", "cv_rmse"]:
            values = group[metric].to_numpy(dtype=float)
            mean = float(values.mean())
            std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            half_width = 1.96 * std / (len(values) ** 0.5) if len(values) > 1 else 0.0
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            row[f"{metric}_ci95_low"] = mean - half_width
            row[f"{metric}_ci95_high"] = mean + half_width
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["dataset", "split_name", "model"]).reset_index(drop=True)


def merge_batches(dataset: str, labels: list[str], output_label: str) -> None:
    merged = pd.concat([_read_seed_csv(dataset, label) for label in labels], ignore_index=True)
    merged = merged.drop_duplicates(subset=["dataset", "split_name", "model", "random_seed"]).reset_index(drop=True)
    merged = merged.sort_values(["split_name", "model", "random_seed"]).reset_index(drop=True)

    seed_path = TABLES_DIR / f"repeated_main_metrics_{dataset}{_suffix(output_label)}.csv"
    summary_path = TABLES_DIR / f"repeated_main_metrics_{dataset}{_suffix(output_label)}_summary.csv"

    merged.to_csv(seed_path, index=False)
    _summarize(merged).to_csv(summary_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge repeated main-metric seed batches and recompute summaries.")
    parser.add_argument("--dataset", required=True, choices=["bdg2", "gepiii"])
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--output-label", required=True)
    args = parser.parse_args()
    merge_batches(dataset=args.dataset, labels=args.labels, output_label=args.output_label)


if __name__ == "__main__":
    main()
