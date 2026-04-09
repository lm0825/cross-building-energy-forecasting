from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import binomtest, rankdata, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import EXP1_PREDICTIONS_DIR, TABLES_DIR, ensure_phase2_dirs


CHUNK_SIZE = 500_000
KEY_COLUMNS = ["split_name", "model", "building_id", "site_id", "building_type"]
USE_COLUMNS = KEY_COLUMNS + ["y_true", "y_pred"]

PER_BUILDING_PATH = TABLES_DIR / "exp1_per_building_metrics.csv"
SUMMARY_PATH = TABLES_DIR / "exp1_per_building_summary.csv"
PAIRWISE_PATH = TABLES_DIR / "exp1_per_building_pairwise.csv"

SPLIT_ORDER = ["t_split", "b_split", "s_split"]
MODEL_ORDER = ["lgbm", "lgbm_lag", "lstm", "patchtst"]


def _holm_adjust(p_values: pd.Series) -> pd.Series:
    valid = p_values.dropna().sort_values()
    adjusted = pd.Series(np.nan, index=p_values.index, dtype=np.float64)
    if valid.empty:
        return adjusted

    m = len(valid)
    running_max = 0.0
    for rank, (idx, value) in enumerate(valid.items(), start=1):
        candidate = float(value) * (m - rank + 1)
        running_max = max(running_max, candidate)
        adjusted.loc[idx] = min(running_max, 1.0)
    return adjusted


def _aggregate_single_prediction_file(path: Path) -> pd.DataFrame:
    accumulator: dict[tuple[str, str, str, str, str], list[float]] = defaultdict(
        lambda: [0.0, 0.0, 0.0, 0.0]
    )

    for chunk in pd.read_csv(path, usecols=USE_COLUMNS, chunksize=CHUNK_SIZE):
        errors = chunk["y_pred"].to_numpy(dtype=np.float64) - chunk["y_true"].to_numpy(dtype=np.float64)
        chunk["sum_true"] = chunk["y_true"].to_numpy(dtype=np.float64)
        chunk["sum_abs_err"] = np.abs(errors)
        chunk["sum_sq_err"] = errors ** 2
        partial = (
            chunk.groupby(KEY_COLUMNS, sort=False, as_index=False)
            .agg(
                sum_true=("sum_true", "sum"),
                sum_abs_err=("sum_abs_err", "sum"),
                sum_sq_err=("sum_sq_err", "sum"),
                n_rows=("y_true", "size"),
            )
        )
        for row in partial.itertuples(index=False):
            key = (
                str(row.split_name),
                str(row.model),
                str(row.building_id),
                str(row.site_id),
                str(row.building_type),
            )
            stats = accumulator[key]
            stats[0] += float(row.sum_true)
            stats[1] += float(row.sum_abs_err)
            stats[2] += float(row.sum_sq_err)
            stats[3] += float(row.n_rows)

    rows: list[dict[str, object]] = []
    for (split_name, model, building_id, site_id, building_type), values in accumulator.items():
        sum_true, sum_abs_err, sum_sq_err, n_rows = values
        mean_true = sum_true / n_rows if n_rows else np.nan
        mae = sum_abs_err / n_rows if n_rows else np.nan
        rmse = float(np.sqrt(sum_sq_err / n_rows)) if n_rows else np.nan
        cv_rmse = rmse / mean_true if mean_true else np.nan
        rows.append(
            {
                "split_name": split_name,
                "model": model,
                "building_id": building_id,
                "site_id": site_id,
                "building_type": building_type,
                "n_rows": int(n_rows),
                "mean_true": float(mean_true),
                "mae": float(mae),
                "rmse": float(rmse),
                "cv_rmse": float(cv_rmse),
            }
        )
    return pd.DataFrame(rows)


def _compute_summary(per_building: pd.DataFrame) -> pd.DataFrame:
    summary = (
        per_building.groupby(["split_name", "model"], as_index=False)["cv_rmse"]
        .agg(
            n_buildings="size",
            mean_cv_rmse="mean",
            median_cv_rmse="median",
            p25_cv_rmse=lambda s: float(s.quantile(0.25)),
            p75_cv_rmse=lambda s: float(s.quantile(0.75)),
            p90_cv_rmse=lambda s: float(s.quantile(0.90)),
        )
    )

    best_share_rows: list[dict[str, object]] = []
    for split_name, group in per_building.groupby("split_name", sort=False):
        pivot = group.pivot(index="building_id", columns="model", values="cv_rmse").dropna()
        if pivot.empty:
            continue
        best_model = pivot.idxmin(axis=1)
        shares = best_model.value_counts(normalize=True)
        for model in MODEL_ORDER:
            best_share_rows.append(
                {
                    "split_name": split_name,
                    "model": model,
                    "share_best_buildings": float(shares.get(model, 0.0)),
                }
            )
    best_share = pd.DataFrame(best_share_rows)
    summary = summary.merge(best_share, on=["split_name", "model"], how="left")
    return summary.sort_values(
        ["split_name", "model"],
        key=lambda col: col.map({v: i for i, v in enumerate(SPLIT_ORDER + MODEL_ORDER)}),
    ).reset_index(drop=True)


def _compute_pairwise(per_building: pd.DataFrame) -> pd.DataFrame:
    def _signed_rank_biserial(delta: pd.Series) -> float:
        nonzero = delta.loc[delta != 0.0].astype(np.float64)
        if nonzero.empty:
            return float("nan")
        ranks = rankdata(np.abs(nonzero.to_numpy(dtype=np.float64)), method="average")
        pos_rank_sum = float(ranks[nonzero.to_numpy(dtype=np.float64) > 0.0].sum())
        neg_rank_sum = float(ranks[nonzero.to_numpy(dtype=np.float64) < 0.0].sum())
        total_rank_sum = pos_rank_sum + neg_rank_sum
        if total_rank_sum == 0.0:
            return float("nan")
        return (pos_rank_sum - neg_rank_sum) / total_rank_sum

    rows: list[dict[str, object]] = []
    comparisons = [
        ("lgbm_lag", "lgbm"),
        ("lstm", "lgbm"),
        ("patchtst", "lgbm"),
        ("lstm", "lgbm_lag"),
        ("patchtst", "lgbm_lag"),
        ("lstm", "patchtst"),
    ]
    for split_name, group in per_building.groupby("split_name", sort=False):
        pivot = group.pivot(index="building_id", columns="model", values="cv_rmse").dropna()
        for left_model, right_model in comparisons:
            delta = pivot[left_model] - pivot[right_model]
            nonzero_delta = delta.loc[delta != 0.0]
            n_left_better = int((delta < 0.0).sum())
            n_right_better = int((delta > 0.0).sum())
            n_ties = int((delta == 0.0).sum())

            wilcoxon_statistic = np.nan
            wilcoxon_p_value = np.nan
            if len(nonzero_delta) > 0:
                wilcoxon_result = wilcoxon(
                    nonzero_delta.to_numpy(dtype=np.float64),
                    alternative="two-sided",
                    zero_method="wilcox",
                    method="auto",
                )
                wilcoxon_statistic = float(wilcoxon_result.statistic)
                wilcoxon_p_value = float(wilcoxon_result.pvalue)

            sign_test_p_value = np.nan
            if (n_left_better + n_right_better) > 0:
                sign_test_p_value = float(
                    binomtest(
                        k=min(n_left_better, n_right_better),
                        n=n_left_better + n_right_better,
                        p=0.5,
                        alternative="two-sided",
                    ).pvalue
                )

            rows.append(
                {
                    "split_name": split_name,
                    "left_model": left_model,
                    "right_model": right_model,
                    "n_buildings": int(len(delta)),
                    "n_left_better": n_left_better,
                    "n_right_better": n_right_better,
                    "n_ties": n_ties,
                    "share_left_better": float((delta < 0).mean()),
                    "median_delta_left_minus_right": float(delta.median()),
                    "mean_delta_left_minus_right": float(delta.mean()),
                    "rank_biserial_left_minus_right": float(_signed_rank_biserial(delta)),
                    "abs_rank_biserial": float(abs(_signed_rank_biserial(delta))),
                    "wilcoxon_statistic": wilcoxon_statistic,
                    "wilcoxon_p_value": wilcoxon_p_value,
                    "sign_test_p_value": sign_test_p_value,
                }
            )
    pairwise = pd.DataFrame(rows)
    pairwise["wilcoxon_p_value_holm"] = (
        pairwise.groupby("split_name", sort=False)["wilcoxon_p_value"].transform(_holm_adjust)
    )
    pairwise["sign_test_p_value_holm"] = (
        pairwise.groupby("split_name", sort=False)["sign_test_p_value"].transform(_holm_adjust)
    )
    return pairwise


def main() -> None:
    ensure_phase2_dirs()
    frames: list[pd.DataFrame] = []
    for path in sorted(EXP1_PREDICTIONS_DIR.glob("*.csv")):
        print(f"Processing {path.name} ...")
        frames.append(_aggregate_single_prediction_file(path))

    per_building = (
        pd.concat(frames, ignore_index=True)
        .sort_values(["split_name", "model", "building_id"])
        .reset_index(drop=True)
    )
    summary = _compute_summary(per_building)
    pairwise = _compute_pairwise(per_building)

    per_building.to_csv(PER_BUILDING_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)
    pairwise.to_csv(PAIRWISE_PATH, index=False)

    print(f"per_building: {PER_BUILDING_PATH}")
    print(f"summary: {SUMMARY_PATH}")
    print(f"pairwise: {PAIRWISE_PATH}")


if __name__ == "__main__":
    main()
