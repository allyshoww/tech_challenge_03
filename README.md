# Tech Challenge Fase 3 - Machine Learning Engineering
**Análise Preditiva de Atrasos Aéreos** | MBA POS-Tech

[![Python](https://img.shields.io/badge/Python-3.13-blue)](https://python.org)
[![Status](https://img.shields.io/badge/Pipeline-Completo-green.svg)](README.md)

## 📋 Entregáveis Obrigatórios
✅ **EDA**: stats.csv + visualizações PNGs  
✅ **Supervised**: RandomForest(1.00) + Logistic(0.95)  
✅ **Unsupervised**: KMeans 3 clusters + PCA 2D  
✅ **Análise Crítica**: [final_analise.md](reports/final_analise.md)  

## 🚀 Pipeline DevOps (5min total)
```bash
git clone <repo> && cd Tech_Challenge_03
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
make all  # Baixa dados → processa → EDA → ML → relatório
