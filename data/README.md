# Data Guide

This repository intentionally does **not** ship any raw datasets, processed feature files, prediction tables, or trained models.

Only the code and the expected directory skeleton are provided here. Please download the datasets directly from their original public sources and cite the original data owners in your paper or derivative work.

## Expected Layout

Place the files so the repository matches the following structure:

```text
data/
├── bdg2/
│   ├── metadata/metadata.csv
│   ├── meters/cleaned/electricity_cleaned.csv
│   └── weather/weather.csv
├── gepiii/
│   ├── building_metadata.csv
│   ├── train.csv
│   └── weather_train.csv
├── heww/
│   └── heew_daily_energy_weather.csv
├── splits/          # generated automatically
└── clustering/      # generated automatically
```

## Source Links

| Dataset | Role In This Repository | Source |
| --- | --- | --- |
| `BDG2` | Primary benchmark dataset | `https://doi.org/10.5281/zenodo.3887306` |
| `GEPIII` | Same-population boundary check | `https://www.kaggle.com/c/ashrae-energy-prediction/data` |
| `HEEW` | Narrow external portability check | `https://doi.org/10.6084/m9.figshare.28425647` |

## Minimal Required Files

| Script | Required files |
| --- | --- |
| `src/stage1_bdg2.py` | `data/bdg2/metadata/metadata.csv`, `data/bdg2/meters/cleaned/electricity_cleaned.csv`, `data/bdg2/weather/weather.csv` |
| `src/stage1_gepiii.py` | `data/gepiii/building_metadata.csv`, `data/gepiii/train.csv`, `data/gepiii/weather_train.csv` |
| `src/experiment5_heew.py` | `data/heww/heew_daily_energy_weather.csv` |

After stage-1 preprocessing, the code will generate additional local artifacts such as:

- `data/bdg2/features_all.parquet`
- `data/gepiii/features_all.parquet`
- `data/bdg2/filtered_meter_ids.csv`
- `data/gepiii/filtered_meter_ids.csv`
- `tables/eligible_sites_for_loso.csv`

These generated files should also stay out of version control.

## Citation Prompts

Use the original dataset citations rather than citing this repository alone.

### BDG2

Clayton Miller, Anjukan Kathirgamanathan, Bianca Picchetti, Pandarasamy Arjunan, June Young Park, Zoltan Nagy, Paul Raftery, Brodie W. Hobson, Zixiao Shi, and Forrest Meggers. *The Building Data Genome Project 2, energy meter data from the ASHRAE Great Energy Predictor III competition*. Scientific Data, 7:368, 2020. DOI: `10.1038/s41597-020-00712-x`

### GEPIII

Addison Howard, Chris Balbach, Clayton Miller, Jeff Haberl, Krishnan Gowri, and Sohier Dane. *ASHRAE - Great Energy Predictor III*. Kaggle competition dataset, 2019. URL: `https://kaggle.com/competitions/ashrae-energy-prediction`

### HEEW

Dong H, Zhu J, Chung CY. *HEEW, a Hierarchical Dataset on Multiple Energy Consumption, PV Generation, Emissions, and Weather Information*. figshare, 2025. DOI: `10.6084/m9.figshare.28425647`

## Notes

- `BDG2` and `GEPIII` are hourly workflows.
- `HEEW` is implemented here as a daily-resolution auxiliary experiment.
- The folder name is intentionally `heww/` because the current code expects `data/heww/heew_daily_energy_weather.csv`.
