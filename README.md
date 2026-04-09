<div align="center">
  <h1>Cross-Building Energy Forecasting Benchmark</h1>
  <p>Reproducible research code for parity-aware forecasting experiments on BDG2, GEPIII, and HEEW.</p>
</div>

## What This Repository Is

This repository is organized as a **paper reproduction repository**, not as a raw development dump.

It is designed so that another researcher can:

1. install the environment,
2. place the public datasets in the expected paths,
3. run a small number of standardized wrapper scripts,
4. reproduce the main benchmark outputs and paper figures.

The released code covers:

- `BDG2` as the primary benchmark
- `GEPIII` as the same-population boundary check
- `HEEW` as a narrow external portability check

The main model families are:

- `LightGBM`
- `LightGBM+lag`
- `LSTM`
- `PatchTST`

The main evaluation protocols are:

- `T-split`: future forecasting on seen buildings
- `B-split`: held-out building evaluation with rolling history
- `S-split`: held-out site evaluation with rolling history

## What Is Not Included

This release does **not** include:

- raw dataset files
- processed parquet features
- trained checkpoints
- prediction CSVs
- generated tables and figures

Dataset download links, required paths, and citation prompts are documented in [data/README.md](data/README.md).

## Repository Structure

```text
.
├── README.md                  # main entry point
├── Makefile                   # simple reproduction targets
├── requirements.txt           # fallback dependency list
├── pyproject.toml             # primary environment definition
├── data/                      # dataset placeholders and data guide
├── docs/                      # reproduction guide
├── scripts/
│   ├── main/                  # main paper reproduction entrypoints
│   ├── auxiliary/             # optional extra analyses
│   ├── README.md              # script categories and usage
│   └── common.sh              # shared shell helpers
├── src/                       # Python code, categorized in src/README.md
├── results/                   # generated outputs, gitignored
├── tables/                    # generated outputs, gitignored
├── figures/                   # generated outputs, gitignored
├── models/                    # generated outputs, gitignored
└── logs/                      # generated outputs, gitignored
```

## Quick Reproduction

### Option 1: Recommended

```bash
make setup
make check
make prepare-data
make main-results
make extended-results
make render-figures
```

### Option 2: Direct Wrapper Scripts

```bash
uv sync
source .venv/bin/activate

bash scripts/main/00_check_environment.sh
bash scripts/main/01_prepare_data.sh
bash scripts/main/02_reproduce_main_results.sh
bash scripts/main/03_reproduce_extended_results.sh
bash scripts/main/04_render_figures.sh
```

## Minimal Path For Most Readers

If you only want the core paper results rather than every auxiliary analysis:

```bash
uv sync
source .venv/bin/activate

bash scripts/main/01_prepare_data.sh
bash scripts/main/02_reproduce_main_results.sh
```

This runs:

- stage-1 preprocessing for `BDG2`
- stage-1 preprocessing for `GEPIII`
- main `BDG2` benchmark
- supplementary baselines
- `GEPIII` boundary check

## Environment

Primary environment manager:

- `uv`

Supported fallback:

- `pip` with [requirements.txt](requirements.txt)

Core software requirements:

- Python `>= 3.12`
- `PyTorch`
- `LightGBM`
- `scikit-learn`
- `SciPy`

## Data Setup

1. Download the public datasets listed in [data/README.md](data/README.md).
2. Place them under the exact paths documented there.
3. Run:

```bash
bash scripts/main/01_prepare_data.sh
```

## Reproducibility Guide

For the full step-by-step reproduction workflow, use:

- [docs/REPRODUCE.md](docs/REPRODUCE.md)
- [scripts/README.md](scripts/README.md)
- [src/README.md](src/README.md)

That document explains:

- minimal vs extended reproduction
- wrapper scripts and `make` targets
- output locations
- script-to-paper mapping
- source-file categories

## Main Output Locations

Generated files are written to:

- `data/splits/`
- `data/clustering/`
- `models/saved/`
- `results/`
- `tables/`
- `figures/`
- `logs/`

These directories are preserved in the repository only as placeholders and are excluded from version control for public release.

## Hardware And Runtime Notes

- `BDG2` and `GEPIII` are the heavier workflows.
- `HEEW` is a smaller daily-resolution auxiliary experiment.
- Wrapper scripts support environment-variable overrides such as `DEVICE`, `HEEW_DEVICE`, `CPU_FRACTION`, and `PYTHON_BIN`.

Example:

```bash
CPU_FRACTION=0.5 DEVICE=cuda:1 make main-results
```

Optional analyses that are not required for a first-pass paper reproduction are separated under `scripts/auxiliary/`.

## Citation

Please cite:

- the accompanying paper
- the original dataset sources

Dataset citations are collected in [data/README.md](data/README.md).

Before public release, you should still add:

- a final `CITATION.cff`
- a project `LICENSE`
