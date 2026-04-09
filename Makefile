PYTHON_BIN ?= .venv/bin/python
CPU_FRACTION ?= 0.7
DEVICE ?= cuda:0
HEEW_DEVICE ?= cpu

.PHONY: setup check prepare-data main-results extended-results render-figures reproduce
.PHONY: aux-grouping aux-benchmarking aux-lag-ablation aux-heew-deep-checks aux-repeated-seeds aux-target-transform

setup:
	uv sync

check:
	PYTHON_BIN="$(PYTHON_BIN)" bash scripts/main/00_check_environment.sh

prepare-data:
	PYTHON_BIN="$(PYTHON_BIN)" bash scripts/main/01_prepare_data.sh

main-results:
	PYTHON_BIN="$(PYTHON_BIN)" CPU_FRACTION="$(CPU_FRACTION)" DEVICE="$(DEVICE)" bash scripts/main/02_reproduce_main_results.sh

extended-results:
	PYTHON_BIN="$(PYTHON_BIN)" CPU_FRACTION="$(CPU_FRACTION)" DEVICE="$(DEVICE)" HEEW_DEVICE="$(HEEW_DEVICE)" bash scripts/main/03_reproduce_extended_results.sh

render-figures:
	PYTHON_BIN="$(PYTHON_BIN)" bash scripts/main/04_render_figures.sh

aux-grouping:
	PYTHON_BIN="$(PYTHON_BIN)" CPU_FRACTION="$(CPU_FRACTION)" DEVICE="$(DEVICE)" bash scripts/auxiliary/10_run_grouping_analysis.sh

aux-benchmarking:
	PYTHON_BIN="$(PYTHON_BIN)" CPU_FRACTION="$(CPU_FRACTION)" bash scripts/auxiliary/11_run_dynamic_benchmarking.sh

aux-lag-ablation:
	PYTHON_BIN="$(PYTHON_BIN)" CPU_FRACTION="$(CPU_FRACTION)" bash scripts/auxiliary/12_run_lag_ablation.sh

aux-heew-deep-checks:
	PYTHON_BIN="$(PYTHON_BIN)" CPU_FRACTION="$(CPU_FRACTION)" HEEW_DEVICE="$(HEEW_DEVICE)" bash scripts/auxiliary/13_run_heew_deep_checks.sh

aux-repeated-seeds:
	PYTHON_BIN="$(PYTHON_BIN)" CPU_FRACTION="$(CPU_FRACTION)" DEVICE="$(DEVICE)" bash scripts/auxiliary/14_run_repeated_seed_checks.sh

aux-target-transform:
	PYTHON_BIN="$(PYTHON_BIN)" CPU_FRACTION="$(CPU_FRACTION)" DEVICE="$(DEVICE)" bash scripts/auxiliary/15_run_target_transform_ablation.sh

reproduce: prepare-data main-results extended-results render-figures
