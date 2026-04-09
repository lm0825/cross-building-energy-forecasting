# Script Layout

Shell entrypoints are intentionally split into two groups.

## `scripts/main/`

These wrappers define the recommended reproduction path for most readers.

| Script | Purpose |
| --- | --- |
| `00_check_environment.sh` | Verify that the Python environment is usable |
| `01_prepare_data.sh` | Run stage-1 preprocessing for `BDG2` and `GEPIII` |
| `02_reproduce_main_results.sh` | Reproduce the main benchmark and GEPIII boundary check |
| `03_reproduce_extended_results.sh` | Run main supplementary analyses included in the recommended path |
| `04_render_figures.sh` | Render paper figures from generated outputs |

## `scripts/auxiliary/`

These wrappers are optional and are separated so they do not distract from the main paper workflow.

| Script | Purpose |
| --- | --- |
| `10_run_grouping_analysis.sh` | Run the grouping analysis (`Experiment 2`) |
| `11_run_dynamic_benchmarking.sh` | Run the dynamic benchmarking analysis (`Experiment 3`) |
| `12_run_lag_ablation.sh` | Run the BDG2 lag-combination ablation |
| `13_run_heew_deep_checks.sh` | Run deeper HEEW diagnostics |
| `14_run_repeated_seed_checks.sh` | Run repeated-seed auxiliary checks |
| `15_run_target_transform_ablation.sh` | Run the target-transform ablation |

## Shared Helper

- `common.sh` centralizes path resolution, logging, and common environment variables.
