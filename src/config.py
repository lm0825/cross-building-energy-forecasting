from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
FEATURES_BDG2_PATH = DATA_DIR / "bdg2" / "features_all.parquet"
FEATURES_GEPIII_PATH = DATA_DIR / "gepiii" / "features_all.parquet"
FILTERED_META_BDG2_PATH = DATA_DIR / "bdg2" / "filtered_building_meta.csv"
ELIGIBLE_SITES_PATH = ROOT / "tables" / "eligible_sites_for_loso.csv"
SPLITS_DIR = DATA_DIR / "splits"
CLUSTERING_DIR = DATA_DIR / "clustering"

MODELS_DIR = ROOT / "models" / "saved"
RESULTS_DIR = ROOT / "results"
EXP1_PREDICTIONS_DIR = RESULTS_DIR / "exp1_predictions"
EXP2_PREDICTIONS_DIR = RESULTS_DIR / "exp2_predictions"
EXP4_PREDICTIONS_DIR = RESULTS_DIR / "exp4_predictions"
LOGS_DIR = ROOT / "logs"

TABLES_DIR = ROOT / "tables"
FIGURES_DIR = ROOT / "figures"

RANDOM_SEED = 42
DEFAULT_GPU_INDEX = 1
DEFAULT_TORCH_DEVICE = f"cuda:{DEFAULT_GPU_INDEX}"

TABULAR_FEATURE_COLUMNS = [
    "hour_of_day",
    "day_of_week",
    "month",
    "is_weekend",
    "airTemperature",
    "dewTemperature",
    "windSpeed",
    "cloudCoverage",
    "log_floor_area",
    "building_type",
    "site_id",
]
TABULAR_LAG_FEATURE_COLUMNS = [
    "lag_1h",
    "lag_24h",
    "lag_168h",
]

TABULAR_CATEGORICAL_COLUMNS = ["building_type", "site_id"]
SEQUENCE_DYNAMIC_COLUMNS = [
    "hour_of_day",
    "day_of_week",
    "month",
    "is_weekend",
    "airTemperature",
    "dewTemperature",
    "windSpeed",
    "cloudCoverage",
]
SEQUENCE_STATIC_COLUMNS = ["log_floor_area", "building_type_code", "site_id_code"]

CONTEXT_WINDOW = 168
PREDICTION_HORIZON = 24
TRAIN_WINDOW_STRIDE = 24
EVAL_WINDOW_STRIDE = 24


def ensure_phase2_dirs() -> None:
    for path in [
        SPLITS_DIR,
        CLUSTERING_DIR,
        MODELS_DIR,
        EXP1_PREDICTIONS_DIR,
        EXP2_PREDICTIONS_DIR,
        EXP4_PREDICTIONS_DIR,
        TABLES_DIR,
        FIGURES_DIR,
        LOGS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
