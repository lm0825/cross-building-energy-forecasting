from __future__ import annotations

import pickle
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (
    ELIGIBLE_SITES_PATH,
    FEATURES_BDG2_PATH,
    RANDOM_SEED,
    SPLITS_DIR,
    ensure_phase2_dirs,
)


T_SPLIT_PATH = SPLITS_DIR / "t_split_index.pkl"
B_SPLIT_PATH = SPLITS_DIR / "b_split_index.pkl"
S_SPLIT_PATH = SPLITS_DIR / "s_split_indices.pkl"


def load_split_frame(feature_path: str | Path = FEATURES_BDG2_PATH) -> pd.DataFrame:
    frame = pd.read_parquet(
        feature_path,
        columns=["building_id", "site_id", "timestamp"],
    )
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame = frame.reset_index(drop=True)
    frame["row_id"] = np.arange(len(frame), dtype=np.int64)
    return frame


def pack_mask(mask: np.ndarray) -> dict[str, object]:
    mask = np.asarray(mask, dtype=np.bool_)
    return {
        "length": int(mask.size),
        "packed": np.packbits(mask.astype(np.uint8)),
    }


def unpack_mask(packed_mask: dict[str, object]) -> np.ndarray:
    length = int(packed_mask["length"])
    unpacked = np.unpackbits(np.asarray(packed_mask["packed"], dtype=np.uint8))
    return unpacked[:length].astype(bool)


def materialize_indices(packed_mask: dict[str, object]) -> np.ndarray:
    return np.flatnonzero(unpack_mask(packed_mask)).astype(np.int64)


def make_t_split(frame: pd.DataFrame) -> dict[str, object]:
    years = frame["timestamp"].dt.year
    train_mask = years == 2016
    test_mask = years == 2017
    return {
        "split_name": "t_split",
        "description": "2016 train / 2017 test with no temporal gap",
        "num_rows": int(len(frame)),
        "train_years": [2016],
        "test_years": [2017],
        "train_mask": pack_mask(train_mask.to_numpy()),
        "test_mask": pack_mask(test_mask.to_numpy()),
    }


def make_b_split(
    frame: pd.DataFrame,
    test_fraction: float = 0.2,
    random_seed: int = RANDOM_SEED,
) -> dict[str, object]:
    rng = np.random.default_rng(random_seed)
    building_site = frame[["building_id", "site_id"]].drop_duplicates()

    test_buildings_by_site: dict[str, list[str]] = {}
    for site_id, group in building_site.groupby("site_id", sort=True):
        buildings = np.sort(group["building_id"].to_numpy())
        sample_size = max(1, int(np.ceil(len(buildings) * test_fraction)))
        sampled = np.sort(rng.choice(buildings, size=sample_size, replace=False)).tolist()
        test_buildings_by_site[str(site_id)] = sampled

    test_buildings = {
        building_id
        for building_ids in test_buildings_by_site.values()
        for building_id in building_ids
    }
    test_mask = frame["building_id"].isin(test_buildings)
    train_mask = ~test_mask

    return {
        "split_name": "b_split",
        "description": "Site-stratified random building split; test buildings fully held out",
        "num_rows": int(len(frame)),
        "random_seed": int(random_seed),
        "test_fraction": float(test_fraction),
        "test_buildings_by_site": test_buildings_by_site,
        "train_mask": pack_mask(train_mask.to_numpy()),
        "test_mask": pack_mask(test_mask.to_numpy()),
    }


def make_s_splits(frame: pd.DataFrame, eligible_sites: list[str]) -> dict[str, object]:
    folds: list[dict[str, object]] = []
    for site_id in eligible_sites:
        test_mask = frame["site_id"].astype(str) == str(site_id)
        train_mask = ~test_mask
        site_building_count = int(
            frame.loc[test_mask, "building_id"].nunique()
        )
        folds.append(
            {
                "fold_id": str(site_id),
                "test_site": str(site_id),
                "site_building_count": site_building_count,
                "train_mask": pack_mask(train_mask.to_numpy()),
                "test_mask": pack_mask(test_mask.to_numpy()),
            }
        )

    return {
        "split_name": "s_split",
        "description": "Leave-one-site-out over sites eligible for LOSO",
        "num_rows": int(len(frame)),
        "eligible_sites": [str(site_id) for site_id in eligible_sites],
        "folds": folds,
    }


def save_pickle(obj: object, path: Path) -> None:
    with path.open("wb") as file:
        pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)


def _load_pickle_with_numpy_compat(file_obj) -> object:
    try:
        return pickle.load(file_obj)
    except ModuleNotFoundError as exc:
        if exc.name != "numpy._core.numeric":
            raise
        import numpy.core.numeric as numpy_core_numeric

        sys.modules.setdefault("numpy._core.numeric", numpy_core_numeric)
        file_obj.seek(0)
        return pickle.load(file_obj)


def load_pickle(path: str | Path) -> object:
    with Path(path).open("rb") as file:
        return _load_pickle_with_numpy_compat(file)


def save_split_artifacts(feature_path: str | Path = FEATURES_BDG2_PATH) -> dict[str, Path]:
    ensure_phase2_dirs()
    frame = load_split_frame(feature_path)
    eligible_sites = pd.read_csv(ELIGIBLE_SITES_PATH)["site_id"].astype(str).tolist()

    t_split = make_t_split(frame)
    b_split = make_b_split(frame)
    s_split = make_s_splits(frame, eligible_sites)

    save_pickle(t_split, T_SPLIT_PATH)
    save_pickle(b_split, B_SPLIT_PATH)
    save_pickle(s_split, S_SPLIT_PATH)

    return {
        "t_split": T_SPLIT_PATH,
        "b_split": B_SPLIT_PATH,
        "s_split": S_SPLIT_PATH,
    }


if __name__ == "__main__":
    outputs = save_split_artifacts()
    for name, path in outputs.items():
        print(f"{name}: {path}")
