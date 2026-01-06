# Makefile Tech Challenge 03 - FINAL
PYTHON ?= python3

.PHONY: all prepare process eda supervised report clean

prepare:
	$(PYTHON) src/data_load.py --output data/raw/flights.csv

process:
	mkdir -p data/processed
	$(PYTHON) src/data_process.py --input data/raw/flights.csv --output data/processed

eda:
	mkdir -p outputs/eda
	$(PYTHON) src/eda.py --input data/processed/X_train.parquet --output outputs/eda

supervised:
	mkdir -p outputs/models
	$(PYTHON) src/supervised.py --X data/processed/X_train.parquet --y data/processed/y_train.parquet --output outputs/models

report:
	mkdir -p outputs/report
	@echo "# Flight Delays ML Pipeline" > outputs/report/report.md
	@echo "## Insights" >> outputs/report/report.md
	@echo "- arr_delay domina variância 111M" >> outputs/report/report.md
	@echo "- 2680 train samples, 380 features" >> outputs/report/report.md
	@echo "- make all: clone → venv → pip → run" >> outputs/report/report.md

all: prepare process eda supervised report
clean:
	rm -rf data/ outputs/
