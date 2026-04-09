# Source Layout

This repository keeps the Python implementation in one `src/` tree, but the files are conceptually split into main paper workflows, shared infrastructure, and auxiliary analyses.

## Main Paper Workflows

These are the scripts most readers should look at first.

| File | Role |
| --- | --- |
| `stage1_bdg2.py` | Stage-1 preprocessing for `BDG2` |
| `stage1_gepiii.py` | Stage-1 preprocessing for `GEPIII` |
| `experiment1.py` | Main `BDG2` benchmark |
| `exp1_supplementary_baselines.py` | Supplementary baselines for `Experiment 1` |
| `experiment4_gepiii.py` | `GEPIII` boundary check |
| `experiment5_heew.py` | `HEEW` portability check |
| `experiment7_information_budget.py` | Information-budget analysis |
| `experiment8_strict_cold_start.py` | Strict cold-start analysis |
| `render_paper_figures.py` | Final figure rendering |

## Shared Infrastructure

These files support both the main and auxiliary workflows.

| File or directory | Role |
| --- | --- |
| `config.py` | Shared repository paths and constants |
| `feature_engineering.py` | Feature processing |
| `data_splitting.py` | `T/B/S` split creation |
| `metrics.py` | Metrics and summary tables |
| `runtime.py` | Runtime limit helpers |
| `site_analysis.py` | Site-level summaries |
| `models/` | Forecasting model implementations |
| `clustering/` | Grouping feature utilities |
| `benchmarking/` | Residual and `CPS` utilities |

## Auxiliary Or Optional Analyses

These scripts are not required for a first-pass paper reproduction.

| File | Role |
| --- | --- |
| `experiment2.py` | Similarity-guided grouping analysis |
| `experiment3.py` | Dynamic benchmarking against static `EUI` |
| `experiment5_heew_pair_enumeration.py` | Exhaustive HEEW held-out-pair analysis |
| `experiment5_heew_lag_ablation.py` | HEEW lag ablation |
| `experiment6_lag_ablation.py` | BDG2 lag-combination ablation |
| `repeated_main_metrics.py` | Repeated-seed main reruns |
| `repeated_exp2_metrics.py` | Repeated-seed Experiment 2 reruns |
| `repeated_exp3_sensitivity.py` | Repeated-seed Experiment 3 reruns |
| `ablation_target_transforms.py` | Target-transform ablation |
| `aggregate_information_budget_results.py` | Information-budget aggregation helper |
| `merge_repeated_main_batches.py` | Repeated main batch merge helper |
| `merge_repeated_exp2_batches.py` | Repeated Experiment 2 batch merge helper |
| `postprocess_paper10_repeats.py` | Extra repeated-seed postprocessing |
| `benchmark_filtering.py` | Low-mean building filter export |
| `exp1_history_budget_analysis.py` | Extra Experiment 1 analysis |
| `exp1_per_building_analysis.py` | Extra per-building summaries |
| `exp2_cold_start_fairness_check.py` | Extra cold-start fairness check |
| `exp3_decision_error_analysis.py` | Extra decision-error analysis |
| `exp4_gepiii_bsplit_stratified_analysis.py` | Extra GEPIII stratified analysis |
| `verify_references_batch.py` | Manuscript reference verification helper |

## Reading Order

For a clean first pass, start with:

1. `stage1_bdg2.py`
2. `stage1_gepiii.py`
3. `experiment1.py`
4. `exp1_supplementary_baselines.py`
5. `experiment4_gepiii.py`
6. `experiment5_heew.py`
7. `experiment7_information_budget.py`
8. `experiment8_strict_cold_start.py`

For runnable shell wrappers matching that order, use:

- `scripts/main/` for the paper path
- `scripts/auxiliary/` for optional extra analyses
