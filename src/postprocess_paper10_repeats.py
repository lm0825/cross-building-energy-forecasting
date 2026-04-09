from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import FIGURES_DIR, TABLES_DIR
from src.repeated_exp3_sensitivity import _plot_seed_sensitivity_intervals


def _t_ci_half_width(values: np.ndarray) -> float:
    if len(values) <= 1:
        return 0.0
    std = float(values.std(ddof=1))
    critical = float(student_t.ppf(0.975, df=len(values) - 1))
    return critical * std / np.sqrt(len(values))


def _summarize_main(seed_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (dataset, split_name, model), group in seed_frame.groupby(["dataset", "split_name", "model"], sort=True):
        row: dict[str, object] = {
            "dataset": dataset,
            "split_name": split_name,
            "model": model,
            "n_seeds": int(len(group)),
        }
        for metric in ["mae", "rmse", "cv_rmse"]:
            values = group[metric].to_numpy(dtype=np.float64)
            mean = float(values.mean())
            std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            half_width = _t_ci_half_width(values)
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            row[f"{metric}_ci95_low"] = mean - half_width
            row[f"{metric}_ci95_high"] = mean + half_width
            row[f"{metric}_min"] = float(values.min())
            row[f"{metric}_max"] = float(values.max())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["dataset", "split_name", "model"]).reset_index(drop=True)


def _summarize_exp3(seed_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model_name, group in seed_frame.groupby("model_name", sort=True):
        values = group["overall_spearman_rho"].to_numpy(dtype=np.float64)
        mean = float(values.mean())
        std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        half_width = _t_ci_half_width(values)
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
    return pd.DataFrame(rows).sort_values("model_name").reset_index(drop=True)


def _combine_csvs(paths: list[Path], sort_cols: list[str]) -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in paths]
    return pd.concat(frames, ignore_index=True).sort_values(sort_cols).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge per-model 10-seed outputs and recompute t-based summaries.")
    parser.add_argument("--main-dataset", choices=["bdg2", "gepiii"], default="bdg2")
    parser.add_argument("--suffix", default="paper10")
    parser.add_argument("--promote-default", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    suffix = args.suffix

    main_seed_paths = [
        TABLES_DIR / f"repeated_main_metrics_{args.main_dataset}_{suffix}_lgbm.csv",
        TABLES_DIR / f"repeated_main_metrics_{args.main_dataset}_{suffix}_lstm.csv",
        TABLES_DIR / f"repeated_main_metrics_{args.main_dataset}_{suffix}_patchtst.csv",
    ]
    exp3_seed_paths = [
        TABLES_DIR / f"exp3_seed_sensitivity_{suffix}_lgbm.csv",
        TABLES_DIR / f"exp3_seed_sensitivity_{suffix}_lstm.csv",
        TABLES_DIR / f"exp3_seed_sensitivity_{suffix}_patchtst.csv",
    ]
    exp3_by_type_paths = [
        TABLES_DIR / f"exp3_seed_sensitivity_by_type_{suffix}_lgbm.csv",
        TABLES_DIR / f"exp3_seed_sensitivity_by_type_{suffix}_lstm.csv",
        TABLES_DIR / f"exp3_seed_sensitivity_by_type_{suffix}_patchtst.csv",
    ]

    main_seed = _combine_csvs(main_seed_paths, ["dataset", "split_name", "model", "random_seed"])
    main_summary = _summarize_main(main_seed)

    exp3_seed = _combine_csvs(exp3_seed_paths, ["model_name", "random_seed"])
    exp3_by_type = _combine_csvs(exp3_by_type_paths, ["model_name", "random_seed", "building_type"])
    exp3_summary = _summarize_exp3(exp3_seed)

    main_seed_out = TABLES_DIR / f"repeated_main_metrics_{args.main_dataset}_{suffix}.csv"
    main_summary_out = TABLES_DIR / f"repeated_main_metrics_{args.main_dataset}_{suffix}_summary.csv"
    exp3_seed_out = TABLES_DIR / f"exp3_seed_sensitivity_{suffix}.csv"
    exp3_summary_out = TABLES_DIR / f"exp3_seed_sensitivity_summary_{suffix}.csv"
    exp3_by_type_out = TABLES_DIR / f"exp3_seed_sensitivity_by_type_{suffix}.csv"
    exp3_fig_out = FIGURES_DIR / f"exp3_model_sensitivity_intervals_{suffix}.png"

    main_seed.to_csv(main_seed_out, index=False)
    main_summary.to_csv(main_summary_out, index=False)
    exp3_seed.to_csv(exp3_seed_out, index=False)
    exp3_summary.to_csv(exp3_summary_out, index=False)
    exp3_by_type.to_csv(exp3_by_type_out, index=False)
    _plot_seed_sensitivity_intervals(exp3_seed, exp3_summary, exp3_fig_out)

    print(f"main_seed: {main_seed_out}")
    print(f"main_summary: {main_summary_out}")
    print(f"exp3_seed: {exp3_seed_out}")
    print(f"exp3_summary: {exp3_summary_out}")
    print(f"exp3_by_type: {exp3_by_type_out}")
    print(f"exp3_figure: {exp3_fig_out}")

    if args.promote_default:
        main_seed.to_csv(TABLES_DIR / f"repeated_main_metrics_{args.main_dataset}.csv", index=False)
        main_summary.to_csv(TABLES_DIR / f"repeated_main_metrics_{args.main_dataset}_summary.csv", index=False)
        exp3_seed.to_csv(TABLES_DIR / "exp3_seed_sensitivity.csv", index=False)
        exp3_summary.to_csv(TABLES_DIR / "exp3_seed_sensitivity_summary.csv", index=False)
        exp3_by_type.to_csv(TABLES_DIR / "exp3_seed_sensitivity_by_type.csv", index=False)
        _plot_seed_sensitivity_intervals(
            exp3_seed,
            exp3_summary,
            FIGURES_DIR / "exp3_model_sensitivity_intervals.png",
        )
        print("promoted_default: true")


if __name__ == "__main__":
    main()
