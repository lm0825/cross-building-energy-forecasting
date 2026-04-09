from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.feature_engineering import (
    add_time_features,
    export_features,
    melt_bdg2_electricity_to_long,
    standardize_bdg2_metadata,
    standardize_bdg2_weather,
)
DATA_DIR = ROOT / "data" / "bdg2"
METADATA_PATH = DATA_DIR / "metadata" / "metadata.csv"
ELECTRICITY_PATH = DATA_DIR / "meters" / "cleaned" / "electricity_cleaned.csv"
WEATHER_PATH = DATA_DIR / "weather" / "weather.csv"
TABLES_DIR = ROOT / "tables"
FIGURES_DIR = ROOT / "figures"
FILTERED_METER_IDS_PATH = DATA_DIR / "filtered_meter_ids.csv"
FILTERED_META_PATH = DATA_DIR / "filtered_building_meta.csv"
FEATURES_PATH = DATA_DIR / "features_all.parquet"
SITE_SUMMARY_PATH = TABLES_DIR / "site_summary.csv"
FLOWCHART_PATH = TABLES_DIR / "filtering_flowchart_numbers.csv"
ELIGIBLE_SITES_PATH = TABLES_DIR / "eligible_sites_for_loso.csv"
SITE_BUILDING_COUNT_FIG = FIGURES_DIR / "site_building_count.png"
BUILDING_TYPE_FIG = FIGURES_DIR / "building_type_dist.png"
MISSING_RATE_CDF_FIG = FIGURES_DIR / "missing_rate_cdf.png"


@dataclass(frozen=True)
class FilterConfig:
    missing_rate_threshold: float = 0.25
    loso_min_buildings: int = 15


def ensure_output_dirs() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    FILTERED_METER_IDS_PATH.parent.mkdir(parents=True, exist_ok=True)


def load_bdg2_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata = pd.read_csv(METADATA_PATH)
    electricity = pd.read_csv(ELECTRICITY_PATH, parse_dates=["timestamp"])
    metadata["building_type"] = metadata["primaryspaceusage"]
    metadata["floor_area"] = metadata["sqm"]
    return metadata, electricity


def load_bdg2_weather() -> pd.DataFrame:
    return pd.read_csv(WEATHER_PATH, parse_dates=["timestamp"])


def compute_meter_stats(electricity: pd.DataFrame) -> tuple[pd.DataFrame, dict[int, int]]:
    timestamps = electricity["timestamp"]
    meter_frame = electricity.drop(columns=["timestamp"])

    missing_rate = meter_frame.isna().mean(axis=0).rename("missing_rate")
    non_null_count = meter_frame.notna().sum(axis=0).rename("non_null_hours")

    coverage_parts: list[pd.Series] = []
    hours_per_year: dict[int, int] = {}

    for year in sorted(timestamps.dt.year.unique()):
        year_mask = timestamps.dt.year == year
        year_values = meter_frame.loc[year_mask]
        hours_per_year[year] = int(year_mask.sum())

        spans: dict[str, int] = {}
        for building_id in year_values.columns:
            valid = year_values[building_id].notna().to_numpy()
            if valid.any():
                idx = np.flatnonzero(valid)
                spans[building_id] = int(idx[-1] - idx[0] + 1)
            else:
                spans[building_id] = 0

        coverage_parts.append(
            pd.Series(spans, name=f"coverage_hours_{year}", dtype="int64")
        )

    meter_stats = pd.concat([missing_rate, non_null_count, *coverage_parts], axis=1)
    meter_stats.index.name = "building_id"
    meter_stats = meter_stats.reset_index()

    full_year_flags: list[str] = []
    for year, required_hours in hours_per_year.items():
        column = f"coverage_hours_{year}"
        flag = f"coverage_full_year_{year}"
        meter_stats[flag] = meter_stats[column] >= required_hours
        full_year_flags.append(flag)

    meter_stats["coverage_full_year_any"] = meter_stats[full_year_flags].any(axis=1)
    return meter_stats, hours_per_year


def build_site_summary(metadata: pd.DataFrame, electricity_building_ids: list[str]) -> pd.DataFrame:
    electricity_meta = metadata[metadata["building_id"].isin(electricity_building_ids)].copy()
    site_summary = (
        electricity_meta.groupby("site_id", dropna=False)
        .agg(
            building_count=("building_id", "nunique"),
            building_type_count=("building_type", "nunique"),
            avg_floor_area_sqm=("floor_area", "mean"),
            median_floor_area_sqm=("floor_area", "median"),
            min_floor_area_sqm=("floor_area", "min"),
            max_floor_area_sqm=("floor_area", "max"),
        )
        .sort_values(["building_count", "site_id"], ascending=[False, True])
        .reset_index()
    )
    return site_summary


def build_building_type_summary(
    metadata: pd.DataFrame, electricity_building_ids: list[str]
) -> pd.DataFrame:
    electricity_meta = metadata[metadata["building_id"].isin(electricity_building_ids)].copy()
    electricity_meta["building_type"] = electricity_meta["building_type"].fillna("Unknown")
    return (
        electricity_meta.groupby("building_type", dropna=False)["building_id"]
        .nunique()
        .sort_values(ascending=False)
        .rename("building_count")
        .reset_index()
    )


def plot_site_building_count(site_summary: pd.DataFrame) -> None:
    ordered = site_summary.sort_values("building_count", ascending=False)
    plt.figure(figsize=(12, 6))
    plt.bar(ordered["site_id"], ordered["building_count"], color="#2f6690")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Building Count")
    plt.title("BDG2 Electricity Buildings by Site")
    plt.tight_layout()
    plt.savefig(SITE_BUILDING_COUNT_FIG, dpi=200)
    plt.close()


def plot_building_type_dist(building_type_summary: pd.DataFrame) -> None:
    plt.figure(figsize=(12, 6))
    plt.bar(
        building_type_summary["building_type"],
        building_type_summary["building_count"],
        color="#81b29a",
    )
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Building Count")
    plt.title("BDG2 Electricity Buildings by Building Type")
    plt.tight_layout()
    plt.savefig(BUILDING_TYPE_FIG, dpi=200)
    plt.close()


def plot_missing_rate_cdf(meter_stats: pd.DataFrame) -> None:
    values = np.sort(meter_stats["missing_rate"].to_numpy())
    cdf = np.arange(1, len(values) + 1) / len(values)

    plt.figure(figsize=(8, 6))
    plt.plot(values, cdf, color="#d1495b", linewidth=2)
    plt.xlabel("Missing Rate")
    plt.ylabel("CDF")
    plt.title("CDF of Missing Rates Across BDG2 Electricity Meters")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(MISSING_RATE_CDF_FIG, dpi=200)
    plt.close()


def apply_filter_rules(
    metadata: pd.DataFrame, meter_stats: pd.DataFrame, config: FilterConfig
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    steps: list[dict[str, object]] = []
    current = meter_stats.copy()

    def record_step(step_id: str, rule: str, previous_count: int, current_frame: pd.DataFrame) -> None:
        current_count = int(current_frame["building_id"].nunique())
        steps.append(
            {
                "step_id": step_id,
                "rule": rule,
                "buildings_remaining": current_count,
                "removed_in_step": previous_count - current_count,
            }
        )

    previous = int(current["building_id"].nunique())
    record_step("0_start", "All electricity meters in BDG2 cleaned electricity file", previous, current)

    electricity_ids = metadata.loc[metadata["electricity"] == "Yes", "building_id"]
    current = current[current["building_id"].isin(electricity_ids)].copy()
    record_step("1_electricity_only", "Keep electricity meters only", previous, current)
    previous = int(current["building_id"].nunique())

    current = current[current["coverage_full_year_any"]].copy()
    record_step(
        "2_coverage_ge_12_months",
        "Keep meters with full-year coverage in 2016 or 2017",
        previous,
        current,
    )
    previous = int(current["building_id"].nunique())

    current = current[current["missing_rate"] <= config.missing_rate_threshold].copy()
    record_step(
        "3_missing_rate_le_25pct",
        "Keep meters with missing rate <= 25%",
        previous,
        current,
    )
    previous = int(current["building_id"].nunique())

    # Filling missing values changes values, not building counts.
    record_step(
        "4_gap_filling",
        "Interpolate short gaps and forward-fill gaps longer than 6 hours",
        previous,
        current,
    )

    deduped = current.sort_values("missing_rate").drop_duplicates("building_id", keep="first")
    current = deduped.copy()
    record_step(
        "5_one_meter_per_building",
        "Keep one electricity meter per building; BDG2 electricity file is already unique by building",
        previous,
        current,
    )
    previous = int(current["building_id"].nunique())

    complete_meta = metadata[
        metadata["building_type"].notna()
        & metadata["floor_area"].notna()
        & metadata["site_id"].notna()
    ][["building_id", "site_id", "building_type", "floor_area"]]

    current = current[current["building_id"].isin(complete_meta["building_id"])].copy()
    record_step(
        "6_complete_core_metadata",
        "Keep buildings with non-null building type, floor area, and site_id",
        previous,
        current,
    )

    filtered_meta = (
        metadata[metadata["building_id"].isin(current["building_id"])]
        .copy()
        .sort_values(["site_id", "building_id"])
    )
    eligible_sites = (
        filtered_meta.groupby("site_id")["building_id"]
        .nunique()
        .rename("building_count")
        .reset_index()
        .sort_values(["building_count", "site_id"], ascending=[False, True])
    )
    eligible_sites = eligible_sites[eligible_sites["building_count"] >= config.loso_min_buildings].copy()

    flowchart = pd.DataFrame(steps)
    return current.sort_values("building_id").reset_index(drop=True), filtered_meta, flowchart, eligible_sites


def fill_series_gaps(series: pd.Series, short_gap_hours: int = 6) -> pd.Series:
    result = series.copy()
    is_missing = result.isna()
    if not is_missing.any():
        return result

    missing_groups = is_missing.ne(is_missing.shift(fill_value=False)).cumsum()
    gap_summary = (
        pd.DataFrame({"missing": is_missing, "group": missing_groups})
        .groupby("group")
        .agg(is_missing=("missing", "first"), gap_size=("missing", "size"))
    )

    for group_id, row in gap_summary.iterrows():
        if not row["is_missing"]:
            continue

        gap_idx = missing_groups[missing_groups == group_id].index
        start = gap_idx[0]
        end = gap_idx[-1]
        gap_size = int(row["gap_size"])

        start_pos = result.index.get_loc(start)
        left_pos = start_pos - 1
        right_pos = result.index.get_loc(end) + 1

        has_left = left_pos >= 0 and pd.notna(result.iloc[left_pos])
        has_right = right_pos < len(result) and pd.notna(result.iloc[right_pos])

        if gap_size <= short_gap_hours and has_left and has_right:
            x = np.array([left_pos, right_pos], dtype=float)
            y = np.array([result.iloc[left_pos], result.iloc[right_pos]], dtype=float)
            fill_positions = np.arange(left_pos + 1, right_pos, dtype=float)
            result.loc[gap_idx] = np.interp(fill_positions, x, y)
        elif has_left:
            result.loc[gap_idx] = result.iloc[left_pos]
        elif has_right:
            result.loc[gap_idx] = result.iloc[right_pos]

    return result


def fill_filtered_electricity(
    electricity: pd.DataFrame,
    building_ids: list[str],
) -> pd.DataFrame:
    filled = electricity.loc[:, ["timestamp", *building_ids]].copy()
    filled = filled.sort_values("timestamp").reset_index(drop=True)
    timestamp_index = filled["timestamp"]

    for building_id in building_ids:
        series = pd.Series(filled[building_id].to_numpy(), index=timestamp_index)
        filled[building_id] = fill_series_gaps(series).to_numpy()

    return filled


def prepare_weather(weather: pd.DataFrame) -> pd.DataFrame:
    standardized = standardize_bdg2_weather(weather)
    all_sites = sorted(standardized["site_id"].dropna().unique().tolist())
    full_timestamps = pd.date_range(
        standardized["timestamp"].min(),
        standardized["timestamp"].max(),
        freq="h",
    )
    full_index = pd.MultiIndex.from_product(
        [all_sites, full_timestamps],
        names=["site_id", "timestamp"],
    )
    standardized = (
        standardized.set_index(["site_id", "timestamp"])
        .sort_index()
        .reindex(full_index)
        .reset_index()
    )
    for column in ["airTemperature", "dewTemperature", "windSpeed", "cloudCoverage"]:
        standardized[column] = standardized.groupby("site_id")[column].transform(
            lambda s: s.interpolate(limit_direction="both")
        )
        if standardized[column].isna().any():
            standardized[column] = standardized.groupby("site_id")[column].transform(
                lambda s: s if not s.notna().any() else s.fillna(s.median())
            )
        if standardized[column].isna().any():
            global_fill = float(standardized[column].median()) if standardized[column].notna().any() else 0.0
            standardized[column] = standardized[column].fillna(global_fill)
    return standardized


def build_features(
    filled_electricity: pd.DataFrame,
    filtered_meta: pd.DataFrame,
    weather: pd.DataFrame,
) -> pd.DataFrame:
    building_ids = filtered_meta["building_id"].sort_values().tolist()
    long_frame = melt_bdg2_electricity_to_long(filled_electricity, building_ids=building_ids)
    meta_std = standardize_bdg2_metadata(filtered_meta)
    weather_std = prepare_weather(weather)

    features = long_frame.merge(meta_std, on="building_id", how="left")
    features = features.merge(weather_std, on=["site_id", "timestamp"], how="left")
    features = add_time_features(features, timestamp_col="timestamp")
    return features.sort_values(["building_id", "timestamp"]).reset_index(drop=True)


def run(config: FilterConfig | None = None) -> dict[str, Path]:
    ensure_output_dirs()
    config = config or FilterConfig()

    metadata, electricity = load_bdg2_inputs()
    weather = load_bdg2_weather()
    meter_stats, _ = compute_meter_stats(electricity)

    electricity_building_ids = meter_stats["building_id"].tolist()
    site_summary = build_site_summary(metadata, electricity_building_ids)
    building_type_summary = build_building_type_summary(metadata, electricity_building_ids)
    filtered_meter_ids, filtered_meta, flowchart, eligible_sites = apply_filter_rules(
        metadata, meter_stats, config
    )
    filled_electricity = fill_filtered_electricity(
        electricity,
        filtered_meter_ids["building_id"].sort_values().tolist(),
    )
    features = build_features(filled_electricity, filtered_meta, weather)

    site_summary.to_csv(SITE_SUMMARY_PATH, index=False)
    filtered_meter_ids.to_csv(FILTERED_METER_IDS_PATH, index=False)
    filtered_meta.to_csv(FILTERED_META_PATH, index=False)
    flowchart.to_csv(FLOWCHART_PATH, index=False)
    eligible_sites.to_csv(ELIGIBLE_SITES_PATH, index=False)
    export_features(features, FEATURES_PATH)

    plot_site_building_count(site_summary)
    plot_building_type_dist(building_type_summary)
    plot_missing_rate_cdf(meter_stats)

    return {
        "site_summary": SITE_SUMMARY_PATH,
        "filtered_meter_ids": FILTERED_METER_IDS_PATH,
        "filtered_building_meta": FILTERED_META_PATH,
        "features_all": FEATURES_PATH,
        "flowchart": FLOWCHART_PATH,
        "eligible_sites": ELIGIBLE_SITES_PATH,
        "site_building_count_figure": SITE_BUILDING_COUNT_FIG,
        "building_type_figure": BUILDING_TYPE_FIG,
        "missing_rate_cdf_figure": MISSING_RATE_CDF_FIG,
    }


if __name__ == "__main__":
    outputs = run()
    for name, path in outputs.items():
        print(f"{name}: {path}")
