from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import EXP2_PREDICTIONS_DIR, TABLES_DIR, ensure_phase2_dirs


CANONICAL_METRICS_PATH = TABLES_DIR / "exp2_cold_start_metrics.csv"
FAIRNESS_CHECK_PATH = TABLES_DIR / "exp2_cold_start_fairness_check.csv"
WINDOW_SIZES_PATH = TABLES_DIR / "exp2_cold_start_window_sizes.csv"
REALISED_WINDOWS_PATH = TABLES_DIR / "exp2_cold_start_realised_windows.csv"

MODEL_ORDER = ["lgbm", "lgbm_lag", "lstm", "patchtst"]
HISTORY_DAYS = [3, 7, 14]


def _compute_cv_rmse(frame: pd.DataFrame) -> float:
    y_true = frame["y_true"].to_numpy(dtype=np.float64)
    y_pred = frame["y_pred"].to_numpy(dtype=np.float64)
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    mean_y = float(np.mean(y_true))
    return float(rmse / mean_y) if mean_y != 0 else np.nan


def main() -> None:
    ensure_phase2_dirs()
    canonical_metrics = pd.read_csv(CANONICAL_METRICS_PATH)

    fairness_rows: list[dict[str, object]] = []
    window_rows: list[dict[str, object]] = []

    for model in MODEL_ORDER:
        all_mix = pd.read_csv(
            EXP2_PREDICTIONS_DIR / f"all_mix_{model}.csv",
            parse_dates=["timestamp"],
        )
        full_all_mix_cv_rmse = _compute_cv_rmse(all_mix)
        full_all_mix_rows = int(len(all_mix))
        full_all_mix_min_ts = all_mix["timestamp"].min()

        for history_days in HISTORY_DAYS:
            cold = pd.read_csv(
                EXP2_PREDICTIONS_DIR / f"cold_start_cluster_group_{history_days}d_{model}.csv",
                parse_dates=["timestamp"],
            )
            cold_min_by_building = (
                cold.groupby("building_id", as_index=False)["timestamp"]
                .min()
                .rename(columns={"timestamp": "cold_min"})
            )
            common_window = all_mix.merge(cold_min_by_building, on="building_id", how="inner")
            common_window = common_window[common_window["timestamp"] >= common_window["cold_min"]].copy()
            common_window = common_window.drop(columns=["cold_min"])

            reported_all_mix_row = canonical_metrics[
                (canonical_metrics["model"] == model)
                & (canonical_metrics["history_days"] == history_days)
                & (canonical_metrics["strategy"] == f"all_mix_after_{history_days}d")
            ].iloc[0]
            grouped_row = canonical_metrics[
                (canonical_metrics["model"] == model)
                & (canonical_metrics["history_days"] == history_days)
                & (canonical_metrics["strategy"] == f"cold_start_cluster_group_{history_days}d")
            ].iloc[0]

            common_window_cv_rmse = _compute_cv_rmse(common_window)
            cold_cv_rmse = _compute_cv_rmse(cold)
            reported_all_mix_cv_rmse = float(reported_all_mix_row["cv_rmse"])
            start_shift_hours = int(
                (cold["timestamp"].min() - full_all_mix_min_ts).total_seconds() // 3600
            )

            fairness_rows.append(
                {
                    "model": model,
                    "history_days": int(history_days),
                    "reported_all_mix_after_nd_cv_rmse": reported_all_mix_cv_rmse,
                    "common_window_all_mix_cv_rmse": common_window_cv_rmse,
                    "cold_start_cluster_group_cv_rmse": cold_cv_rmse,
                    "reported_minus_common_window": reported_all_mix_cv_rmse - common_window_cv_rmse,
                    "delta_group_minus_reported_all_mix": cold_cv_rmse - reported_all_mix_cv_rmse,
                    "delta_group_minus_common_window": cold_cv_rmse - common_window_cv_rmse,
                }
            )

            window_rows.append(
                {
                    "model": model,
                    "history_days": int(history_days),
                    "full_all_mix_rows": full_all_mix_rows,
                    "reported_all_mix_after_nd_rows": int(reported_all_mix_row["n_rows"]),
                    "common_window_rows": int(len(common_window)),
                    "cold_start_cluster_group_rows": int(grouped_row["n_rows"]),
                    "full_all_mix_min_ts": full_all_mix_min_ts,
                    "common_window_min_ts": common_window["timestamp"].min(),
                    "cold_start_cluster_group_min_ts": cold["timestamp"].min(),
                    "start_shift_hours_vs_full_all_mix": start_shift_hours,
                    "row_gap_common_window_vs_full_all_mix": full_all_mix_rows - int(len(common_window)),
                    "full_all_mix_cv_rmse": full_all_mix_cv_rmse,
                    "reported_all_mix_after_nd_cv_rmse": reported_all_mix_cv_rmse,
                    "common_window_all_mix_cv_rmse": common_window_cv_rmse,
                    "cold_start_cluster_group_cv_rmse": cold_cv_rmse,
                }
            )

    fairness_frame = pd.DataFrame(fairness_rows).sort_values(["model", "history_days"]).reset_index(drop=True)
    window_frame = pd.DataFrame(window_rows).sort_values(["model", "history_days"]).reset_index(drop=True)

    fairness_frame.to_csv(FAIRNESS_CHECK_PATH, index=False)
    window_frame.to_csv(WINDOW_SIZES_PATH, index=False)
    window_frame.to_csv(REALISED_WINDOWS_PATH, index=False)

    print("Saved:", FAIRNESS_CHECK_PATH)
    print(fairness_frame.to_string(index=False))
    print()
    print("Saved:", WINDOW_SIZES_PATH)
    print(window_frame.to_string(index=False))


if __name__ == "__main__":
    main()
