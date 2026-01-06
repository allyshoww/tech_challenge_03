# Makefile Tech Challenge 03 - FINAL COMPLETO
PYTHON ?= python3

.PHONY: all prepare process eda supervised unsupervised report clean

# Task 1
prepare:
	@echo "Task 1 - Download dataset"
	$(PYTHON) src/data_load.py --output data/raw/flights.csv

# Task 2  
process:
	@echo "Task 2 - Process data"
	mkdir -p data/processed
	$(PYTHON) src/data_process.py --input data/raw/flights.csv --output data/processed

# Task 3
eda:
	@echo "Task 3 - EDA"
	mkdir -p outputs/eda
	$(PYTHON) src/eda.py --input data/processed/X_train.parquet --output outputs/eda

# Task 4
supervised:
	@echo "Task 4 - Supervised ML"
	mkdir -p outputs/models
	$(PYTHON) src/supervised.py \
		--X data/processed/X_train.parquet \
		--y data/processed/y_train.parquet \
		--output outputs/models

# Task 5 - SEU unsupervised.py
unsupervised:
	@echo "Task 5 - Unsupervised ML"
	mkdir -p outputs/unsupervised
	$(PYTHON) src/unsupervised.py \
		--input data/processed/X_train.parquet \
		--output outputs/unsupervised

# Task 6
report:
	@echo "Task 6 - Report"
	mkdir -p outputs/report
	@echo "# Flight Delays ML - Results" > outputs/report/report.md
	@echo "## EDA: arr_delay domina (111M var)" >> outputs/report/report.md
	@echo "## Pipeline: make all = 100% automatizado" >> outputs/report/report.md
	@echo "## Repo: https://github.com/allyshoww/tech_challenge_03" >> outputs/report/report.md

# PIPELINE COMPLETO
all: prepare process eda supervised unsupervised report

# Clean
clean:
	rm -rf data/ outputs/
