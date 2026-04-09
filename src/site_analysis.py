from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import FILTERED_META_BDG2_PATH, FIGURES_DIR, TABLES_DIR, ensure_phase2_dirs


SITE_ANALYSIS_FIG = FIGURES_DIR / "exp1_site_error_vs_features.png"
SITE_DIFFICULTY_PATH = TABLES_DIR / "site_generalization_difficulty.csv"
SITE_CORRELATIONS_PATH = TABLES_DIR / "exp1_site_feature_correlations.csv"
SITE_METRICS_PATH = TABLES_DIR / "exp1_site_metrics.csv"


def _shannon_entropy(series: pd.Series) -> float:
    probs = series.value_counts(normalize=True)
    return float(-(probs * np.log(probs)).sum()) if not probs.empty else 0.0


def build_site_feature_table() -> pd.DataFrame:
    meta = pd.read_csv(FILTERED_META_BDG2_PATH)

    site_base = (
        meta.groupby("site_id")
        .agg(
            site_building_count=("building_id", "nunique"),
            timezone=("timezone", lambda s: s.dropna().iloc[0] if not s.dropna().empty else "Unknown"),
        )
        .reset_index()
    )
    diversity = (
        meta.groupby("site_id")["building_type"]
        .apply(_shannon_entropy)
        .rename("building_type_shannon")
        .reset_index()
    )

    return (
        site_base.merge(diversity, on="site_id", how="left")
        .sort_values("site_id")
        .reset_index(drop=True)
    )


def run_site_analysis() -> dict[str, Path]:
    ensure_phase2_dirs()
    site_metrics = pd.read_csv(SITE_METRICS_PATH)
    site_features = build_site_feature_table()

    merged = site_metrics.merge(site_features, on="site_id", how="left")
    merged["difficulty_rank"] = (
        merged.groupby("model")["cv_rmse"].rank(ascending=False, method="dense")
    )
    merged = merged.sort_values(["model", "difficulty_rank", "site_id"]).reset_index(drop=True)
    merged.to_csv(SITE_DIFFICULTY_PATH, index=False)
    correlations = compute_site_feature_correlations(merged)
    correlations.to_csv(SITE_CORRELATIONS_PATH, index=False)

    _plot_site_feature_relationships(merged)
    return {
        "site_generalization_difficulty": SITE_DIFFICULTY_PATH,
        "site_feature_correlations": SITE_CORRELATIONS_PATH,
        "site_error_vs_features": SITE_ANALYSIS_FIG,
    }


def compute_site_feature_correlations(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    features = [
        ("site_building_count", "site_building_count"),
        ("building_type_shannon", "building_type_shannon"),
    ]
    for model, group in frame.groupby("model", sort=True):
        for feature, label in features:
            sample = group[[feature, "cv_rmse"]].dropna()
            if len(sample) < 2 or sample[feature].nunique() < 2 or sample["cv_rmse"].nunique() < 2:
                spearman_r = np.nan
                p_value = np.nan
            else:
                spearman_r, p_value = stats.spearmanr(sample[feature], sample["cv_rmse"])
            rows.append(
                {
                    "model": model,
                    "feature": label,
                    "n_sites": int(len(sample)),
                    "spearman_r": float(spearman_r) if pd.notna(spearman_r) else np.nan,
                    "p_value": float(p_value) if pd.notna(p_value) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _plot_site_feature_relationships(frame: pd.DataFrame) -> None:
    models = sorted(frame["model"].unique().tolist())
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x_cols = [
        "site_building_count",
        "building_type_shannon",
    ]
    titles = [
        "Error vs Site Building Count",
        "Error vs Building Type Diversity",
    ]

    # Climate-zone labels are intentionally excluded from the default analysis.
    # The current repository does not contain a verified site-to-climate-zone map,
    # and using a proxy variable would weaken the interpretation of site difficulty.
    for ax, column, title in zip(axes, x_cols, titles, strict=False):
        for model in models:
            group = frame[frame["model"] == model]
            ax.scatter(group[column], group["cv_rmse"], label=model, alpha=0.8)
        ax.set_xlabel(column)
        ax.set_ylabel("CV(RMSE)")
        ax.set_title(title)
    axes[0].legend()
    plt.tight_layout()
    plt.savefig(SITE_ANALYSIS_FIG, dpi=200)
    plt.close()
