from __future__ import annotations

import logging
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import FEATURES_BDG2_PATH, EXP1_PREDICTIONS_DIR, TABLES_DIR
from src.data_splitting import B_SPLIT_PATH, load_pickle, unpack_mask
from src.metrics import summarize_metrics


LOGGER = logging.getLogger(__name__)

HISTORY_BUDGET_DAYS = [0, 1, 3, 7, 14]
MODEL_ORDER = ["lgbm", "lstm", "patchtst"]
SEQUENCE_MODELS = {"lstm", "patchtst"}

OUTPUT_PATH = TABLES_DIR / "exp1_history_budget_bsplit.csv"
PIVOT_PATH = TABLES_DIR / "exp1_history_budget_bsplit_pivot.csv"


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )


def _load_bsplit_test_starts() -> pd.DataFrame:
    frame = pd.read_parquet(FEATURES_BDG2_PATH)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame = frame.sort_values(["building_id", "timestamp"]).reset_index(drop=True)
    split_artifact = load_pickle(B_SPLIT_PATH)
    test_frame = frame.loc[unpack_mask(split_artifact["test_mask"])].reset_index(drop=True)
    return (
        test_frame.groupby("building_id", sort=True)["timestamp"]
        .min()
        .rename("test_start")
        .reset_index()
    )


def _model_context_days(model_name: str) -> int:
    return 7 if model_name in SEQUENCE_MODELS else 0


def _subset_for_budget(
    prediction_frame: pd.DataFrame,
    nominal_history_days: int,
    context_days: int,
) -> pd.DataFrame:
    threshold = prediction_frame["test_start"] + pd.to_timedelta(
        nominal_history_days + context_days,
        unit="D",
    )
    return prediction_frame.loc[prediction_frame["timestamp"] >= threshold].copy()


def run_history_budget_analysis() -> dict[str, Path]:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    test_starts = _load_bsplit_test_starts()

    rows: list[dict[str, object]] = []
    for model_name in MODEL_ORDER:
        prediction_path = EXP1_PREDICTIONS_DIR / f"{model_name}_b_split.csv"
        prediction_frame = pd.read_csv(prediction_path)
        prediction_frame["timestamp"] = pd.to_datetime(prediction_frame["timestamp"])
        prediction_frame = prediction_frame.merge(
            test_starts,
            on="building_id",
            how="left",
            validate="many_to_one",
        )
        canonical_rows = int(len(prediction_frame))
        context_days = _model_context_days(model_name)

        for nominal_history_days in HISTORY_BUDGET_DAYS:
            subset = _subset_for_budget(
                prediction_frame=prediction_frame,
                nominal_history_days=nominal_history_days,
                context_days=context_days,
            )
            metrics = summarize_metrics(subset[["y_true", "y_pred"]]).iloc[0].to_dict()
            rows.append(
                {
                    "model": model_name,
                    "nominal_history_days": int(nominal_history_days),
                    "required_context_days": int(context_days),
                    "realised_first_forecast_day": int(nominal_history_days + context_days),
                    "canonical_rows": canonical_rows,
                    "coverage_vs_canonical": float(len(subset) / canonical_rows) if canonical_rows else float("nan"),
                    **metrics,
                }
            )

    result = pd.DataFrame(rows).sort_values(["model", "nominal_history_days"]).reset_index(drop=True)
    pivot = (
        result.pivot(index="nominal_history_days", columns="model", values="cv_rmse")
        .reindex(HISTORY_BUDGET_DAYS)
        .reset_index()
    )

    result.to_csv(OUTPUT_PATH, index=False)
    pivot.to_csv(PIVOT_PATH, index=False)
    LOGGER.info("Saved %s and %s", OUTPUT_PATH, PIVOT_PATH)
    return {
        "exp1_history_budget_bsplit": OUTPUT_PATH,
        "exp1_history_budget_bsplit_pivot": PIVOT_PATH,
    }


if __name__ == "__main__":
    _configure_logging()
    run_history_budget_analysis()
