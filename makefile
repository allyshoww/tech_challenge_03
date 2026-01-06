# Makefile Tech Challenge 03 - CORRIGIDO
# Rode: make all → Pipeline ML completo!

.PHONY: all prepare eda supervised unsupervised report clean

# Task 1 - Preparação: Download dataset
prepare:
	@echo "Task 1 - Preparação: Download dataset"
	python src/data_load.py --output data/raw/flights.csv

# Task 2 - Processamento dados
process:
	@echo "Processando dados..."
	@mkdir -p data/processed
	python src/data_process.py \
		--input data/raw/flights.csv \
		--output data/processed

# Task 3 - EDA + Visualizações
eda:
	@echo "Task 2 - EDA + Visualizações"
	@mkdir -p outputs/eda
	python src/eda.py \
		--input data/processed/X_train.parquet \
		--output outputs/eda

# Task 4 - Supervised ML (CORRIGIDO!)
supervised:
	@echo "Task 3 - Supervised ML"
	@mkdir -p outputs/models
	python src/supervised.py \
		--X data/processed/X_train.parquet \
		--y data/processed/y_train.parquet \
		--output outputs/models

# Task 5 - Unsupervised (se tiver)
unsupervised:
	@echo "Task 4 - Unsupervised ML"
	@mkdir -p outputs/unsupervised
	python src/unsupervised.py \
		--input data/processed/X_train.parquet \
		--output outputs/unsupervised

# Task 6 - Relatório final
report:
	@echo "Task 5 - Relatório"
	@mkdir -p outputs/report
	python src/report.py --output outputs/report

# Pipeline COMPLETO
all: prepare process eda supervised unsupervised report

# Clean (limpa tudo)
clean:
	rm -rf data/ outputs/ venv/
