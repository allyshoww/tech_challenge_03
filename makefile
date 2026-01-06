# Makefile Tech Challenge 03 - PYTHON3 FIX
PYTHON ?= python3

.PHONY: all prepare process eda supervised unsupervised report clean

prepare:
	@echo "Task 1 - Preparação: Download dataset"
	$(PYTHON) src/data_load.py --output data/raw/flights.csv

process:
	@echo "Processando dados..."
	mkdir -p data/processed
	$(PYTHON) src/data_process.py --input data/raw/flights.csv --output data/processed

eda:
	@echo "Task 2 - EDA"
	mkdir -p outputs/eda
	$(PYTHON) src/eda.py --input data/processed/X_train.parquet --output outputs/eda

supervised:
	@echo "Task 3 - Supervised ML"
	mkdir -p outputs/models
	$(PYTHON) src/supervised.py --X data/processed/X_train.parquet --y data/processed/y_train.parquet --output outputs/models

unsupervised:
	@echo "Task 4 - Unsupervised"
	mkdir -p outputs/unsupervised
	$(PYTHON) src/unsupervised.py --input data/processed/X_train.parquet --output outputs/unsupervised || echo "Sem unsupervised OK"

report:
	@echo "Task 5 - Report"
	mkdir -p outputs/report
	$(PYTHON) src/report.py --output outputs/report || echo "Sem report OK"

all: prepare process eda supervised unsupervised report
clean:
	rm -rf data/ outputs/
