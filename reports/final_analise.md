
# Tech Challenge Fase 3 - Machine Learning Engineering ✅
**Allyson** - MBA POS-Tech | Dataset Atrasos Voos (3351 rotas)

## 📊 EDA Insights
- **Dataset**: 3351 registros mensais (carrier/aeroporto)
- **Target**: delay_rate >10% (73% positivos - balanceado)
- **Features**: 380 (dummies carrier/airport + numéricas)
- Top variância: [ver top_features.png]

![Target Dist](outputs/eda/target_dist.png)

## 🤖 Modelagem Supervisionada (2 algoritmos)
| Modelo | Accuracy |
|--------|----------|
| RandomForest | 1.000 |
| LogisticReg | 0.947 |

**Insight**: RF perfeito (overfit?); LR bom baseline. Escalar features melhora LR.

![Clusters](outputs/unsupervised/kmeans_pca.png)

## 🎯 Unsupervised (KMeans 3 + PCA)
- **3 Clusters rotas**:
  * Verde: Aeroportos OK (<5% atraso)
  * Amarelo: Médio risco
  * Roxo: **CRÍTICOS** (>25% atraso) - evitar!
- PCA explica ~25-35% var (dimensionalidade alta OK)

## ⚠️ Limitações & Melhorias
- **Dataset pequeno**: Agregado mensal (não granular/horário)
- **Overfit RF**: Cross-validation + prune
- **Melhorias**:
  * Dataset 10M+ rows (BTS.gov)
  * Feature eng: feriados, clima real-time
  * XGBoost + hiperparâmetros
  * Dashboard Streamlit

