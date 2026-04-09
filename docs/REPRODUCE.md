# Reproduce The Paper

This document gives a practical reproduction path for users who want to run the released code with as little guesswork as possible.

## Reproduction Levels

Choose one of the following depending on how much of the paper you need to reproduce.

### Level 1: Main Results Only

This is the recommended starting point for most users.

1. Prepare the datasets in the directory structure described in [../data/README.md](../data/README.md).
2. Install the environment.
3. Run stage-1 preprocessing.
4. Run the main BDG2 benchmark.
5. Run the GEPIII boundary check.

Commands:

```bash
uv sync
source .venv/bin/activate

bash scripts/main/01_prepare_data.sh
bash scripts/main/02_reproduce_main_results.sh
```

## Level 2: Main Results + Key Robustness Checks

This level reproduces the main paper storyline more completely.

Commands:

```bash
uv sync
source .venv/bin/activate

bash scripts/main/01_prepare_data.sh
bash scripts/main/02_reproduce_main_results.sh
bash scripts/main/03_reproduce_extended_results.sh
```

This includes:

- `Experiment 1` on `BDG2`
- supplementary baselines for `Experiment 1`
- `Experiment 4` on `GEPIII`
- `Experiment 5` on `HEEW`
- `Experiment 7` information-budget analysis
- `Experiment 8` strict cold-start analysis

## Level 3: Full Local Reproduction Workspace

If you also want paper-ready figures from already generated tables:

```bash
bash scripts/main/04_render_figures.sh
```

## Simple One-Command Interface

If you prefer `make` targets:

```bash
make setup
make check
make prepare-data
make main-results
make extended-results
make render-figures
```

Or run the full recommended flow:

```bash
make reproduce
```

## Runtime Configuration

The wrapper scripts use environment variables so you do not need to edit commands inline every time.

Common variables:

- `PYTHON_BIN`: Python interpreter to use
- `CPU_FRACTION`: CPU fraction limit for supported scripts, default `0.7`
- `DEVICE`: torch device string, default `cuda:0`
- `HEEW_DEVICE`: device for the HEEW auxiliary experiment, default `cpu`
- `LOG_DIR`: output log directory, default `logs`

Example:

```bash
CPU_FRACTION=0.5 DEVICE=cuda:1 bash scripts/main/02_reproduce_main_results.sh
```

## Expected Output Directories

Generated artifacts are written to:

- `data/splits/`
- `data/clustering/`
- `models/saved/`
- `results/`
- `tables/`
- `figures/`
- `logs/`

These outputs are intentionally excluded from version control in this release.

## Script-to-Paper Mapping

| Paper component | Main script |
| --- | --- |
| Data preprocessing for `BDG2` | `src/stage1_bdg2.py` |
| Data preprocessing for `GEPIII` | `src/stage1_gepiii.py` |
| Main cross-building benchmark | `src/experiment1.py` |
| Supplementary baselines | `src/exp1_supplementary_baselines.py` |
| GEPIII boundary check | `src/experiment4_gepiii.py` |
| HEEW portability check | `src/experiment5_heew.py` |
| Information-budget analysis | `src/experiment7_information_budget.py` |
| Strict cold-start analysis | `src/experiment8_strict_cold_start.py` |
| Paper figures | `src/render_paper_figures.py` |

## Optional Extra Analyses

The following wrappers are intentionally kept outside the main reproduction path:

- `scripts/auxiliary/10_run_grouping_analysis.sh`
- `scripts/auxiliary/11_run_dynamic_benchmarking.sh`
- `scripts/auxiliary/12_run_lag_ablation.sh`
- `scripts/auxiliary/13_run_heew_deep_checks.sh`
- `scripts/auxiliary/14_run_repeated_seed_checks.sh`
- `scripts/auxiliary/15_run_target_transform_ablation.sh`

## Notes

- `BDG2` and `GEPIII` are hourly workflows.
- `HEEW` is implemented here as a daily-resolution auxiliary experiment.
- `B-split` and `S-split` are rolling-history evaluation protocols, not universal zero-history deployment tests.
