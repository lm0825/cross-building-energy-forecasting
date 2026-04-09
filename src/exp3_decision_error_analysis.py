from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
TABLES_DIR = ROOT / "tables"


MODEL_FILES = {
    "LightGBM": RESULTS_DIR / "exp3_eui_vs_cps_ranking_lgbm.csv",
    "LightGBM+lag": RESULTS_DIR / "exp3_eui_vs_cps_ranking_lgbm_lag.csv",
    "LSTM": RESULTS_DIR / "exp3_eui_vs_cps_ranking_lstm.csv",
    "PatchTST": RESULTS_DIR / "exp3_eui_vs_cps_ranking_patchtst.csv",
}


def _load_rankings() -> pd.DataFrame:
    merged = None
    for model_name, path in MODEL_FILES.items():
        frame = pd.read_csv(path)[["building_id", "building_type", "cps_percentile"]].copy()
        frame = frame.rename(columns={"cps_percentile": f"cps_percentile_{model_name}"})
        if merged is None:
            merged = frame
        else:
            merged = merged.merge(frame, on=["building_id", "building_type"], how="inner")
    if merged is None:
        raise RuntimeError("No Experiment 3 ranking files were found.")
    return merged


def _summarise_against_reference(merged: pd.DataFrame, threshold: float, threshold_label: str) -> pd.DataFrame:
    reference_col = "cps_percentile_LightGBM+lag"
    reference_positive = merged[reference_col] >= threshold if threshold_label == "top_quartile" else merged[reference_col] <= threshold
    rows = []
    for model_name in ("LightGBM", "LSTM", "PatchTST"):
        model_positive = (
            merged[f"cps_percentile_{model_name}"] >= threshold
            if threshold_label == "top_quartile"
            else merged[f"cps_percentile_{model_name}"] <= threshold
        )
        tp = int((reference_positive & model_positive).sum())
        fn = int((reference_positive & ~model_positive).sum())
        fp = int((~reference_positive & model_positive).sum())
        tn = int((~reference_positive & ~model_positive).sum())
        ref_count = int(reference_positive.sum())
        total = int(len(merged))
        rows.append(
            {
                "reference_model": "LightGBM+lag",
                "threshold_label": threshold_label,
                "candidate_model": model_name,
                "n_buildings": total,
                "reference_positive_count": ref_count,
                "true_positive_count": tp,
                "false_negative_count": fn,
                "false_positive_count": fp,
                "true_negative_count": tn,
                "recall": tp / ref_count if ref_count else float("nan"),
                "precision": tp / (tp + fp) if (tp + fp) else float("nan"),
                "overall_mismatch_rate": (fn + fp) / total if total else float("nan"),
                "overlap_rate_vs_reference": tp / ref_count if ref_count else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    merged = _load_rankings()
    top = _summarise_against_reference(merged, threshold=0.75, threshold_label="top_quartile")
    bottom = _summarise_against_reference(merged, threshold=0.25, threshold_label="bottom_quartile")
    long_frame = pd.concat([top, bottom], ignore_index=True)

    summary = (
        top[
            [
                "candidate_model",
                "reference_positive_count",
                "true_positive_count",
                "false_negative_count",
                "false_positive_count",
                "recall",
                "overall_mismatch_rate",
            ]
        ]
        .rename(
            columns={
                "reference_positive_count": "parity_top_quartile_count",
                "true_positive_count": "correctly_retained_high_priority_count",
                "false_negative_count": "missed_high_priority_count",
                "false_positive_count": "wrongly_promoted_high_priority_count",
                "recall": "high_priority_recall",
                "overall_mismatch_rate": "overall_top_quartile_mismatch_rate",
            }
        )
    )

    long_path = TABLES_DIR / "exp3_decision_error_long.csv"
    summary_path = TABLES_DIR / "exp3_decision_error_summary.csv"
    long_frame.to_csv(long_path, index=False)
    summary.to_csv(summary_path, index=False)

    print(f"Wrote {long_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
