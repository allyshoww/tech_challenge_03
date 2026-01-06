# Makefile - Tech Challenge Fase 3 ML (DevOps Pipeline)
# Rode: make all (instala, baixa data, processa, EDA, ML, report)
# Autor: Pipeline automatizado para POS-Tech MLET Fase 3 [file:1]
PYTHON ?= python3  # ← Adicione isso após PROJECT_DIR

.PHONY: help install data process eda supervised unsupervised report all clean test

# Configs
PROJECT_DIR := $(shell pwd)
DATA_DIR := $(PROJECT_DIR)/data
OUTPUTS_DIR := $(PROJECT_DIR)/outputs
REPORTS_DIR := $(PROJECT_DIR)/reports
SRC_DIR := $(PROJECT_DIR)/src
VENV := venv

# Cores para output
GREEN := \033[0;32m
NC := \033[0m  # No Color

# Targets principais
help: ## Mostra todos comandos
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(GREEN)%-20s$(NC) %s\n", $$1, $$2}'

install: ## Instala dependências (requirements.txt)
	@echo "$(GREEN)Instalando pacotes...$(NC)"
	pip install -r $(SRC_DIR)/requirements.txt || pip install pandas scikit-learn matplotlib seaborn joblib requests numpy

data: ## Task 1: Baixa dataset raw (flights.csv)
	@echo "$(GREEN)Task 1 - Preparação: Download dataset$(NC)"
	python $(SRC_DIR)/data_load.py --output $(DATA_DIR)/raw/flights.csv
	@ls -lh $(DATA_DIR)/raw/

process: data ## Processa: clean + train/test Parquet
	@echo "$(GREEN)Processando dados...$(NC)"
	mkdir -p $(DATA_DIR)/processed
	python $(SRC_DIR)/data_process.py --input $(DATA_DIR)/raw/flights.csv --output $(DATA_DIR)/processed
	@ls $(DATA_DIR)/processed/

eda: process ## Task 2: EDA stats + plots
	@echo "$(GREEN)Task 2 - EDA + Visualizações$(NC)"
	mkdir -p $(OUTPUTS_DIR)/eda
	python $(SRC_DIR)/eda.py --input $(DATA_DIR)/processed/X_train.parquet --output $(OUTPUTS_DIR)/eda
	@cat $(OUTPUTS_DIR)/eda/eda_report.txt | head -5

supervised: process ## Task 3: Modelos supervised (RF + LR)
	@echo "$(GREEN)Task 3 - Supervised ML$(NC)"
	mkdir -p $(OUTPUTS_DIR)/models
	python $(SRC_DIR)/supervised.py --input $(DATA_DIR)/processed/X_train.parquet --output $(OUTPUTS_DIR)/models
	@cat $(OUTPUTS_DIR)/models/metrics.txt

unsupervised: process ## Task 4: Clustering + PCA
	@echo "$(GREEN)Task 4 - Unsupervised ML$(NC)"
	mkdir -p $(OUTPUTS_DIR)/unsupervised
	python $(SRC_DIR)/unsupervised.py --input $(DATA_DIR)/processed/X_train.parquet --output $(OUTPUTS_DIR)/unsupervised
	@ls $(OUTPUTS_DIR)/unsupervised/

report: eda supervised unsupervised ## Task 5: Relatório crítico MD
	@echo "$(GREEN)Task 5 - Análise Crítica$(NC)"
	mkdir -p $(REPORTS_DIR)
	python $(SRC_DIR)/generate_report.py --output $(REPORTS_DIR)/final_analise.md
	@head -10 $(REPORTS_DIR)/final_analise.md

all: install data process eda supervised unsupervised report ## Roda TUDO completo

test: data ## Testa pipeline básico
	@echo "$(GREEN)Teste OK: data/raw/flights.csv existe$(NC)"
	@[ -f $(DATA_DIR)/raw/flights.csv ] || (echo "ERRO: Sem data!" && exit 1)

clean: ## Limpa tudo (data/ outputs/ reports/)
	@echo "$(GREEN)Limpando...$(NC)"
	rm -rf $(DATA_DIR) $(OUTPUTS_DIR) $(REPORTS_DIR) .venv/

git: all ## Git commit auto
	git add . && git commit -m "Pipeline completo: make all ✅ [$$(date)]" || git init && git add . && git commit -m "Init projeto"

.DEFAULT_GOAL := help
.SILENT: help
