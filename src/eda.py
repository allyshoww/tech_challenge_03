#!/usr/bin/env python3
"""
Task 2 EDA: Stats + Plots obrigatórios (FIX top_features)
Roda da RAIZ projeto!
"""
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

sns.set_style("whitegrid")

def eda(input_path, output_dir):
    # Paths da raiz projeto
    script_dir = os.path.dirname(os.path.abspath(__file__))  
    project_root = os.path.dirname(script_dir)
    
    X_path = os.path.join(project_root, input_path)
    y_path = os.path.join(project_root, 'data/processed/y_train.parquet')
    
    X_train = pd.read_parquet(X_path)
    y_train_df = pd.read_parquet(y_path)
    y_train = y_train_df['target']
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"🔍 EDA X_train: {X_train.shape}")
    
    # 1. Stats descritivas
    stats = X_train.describe().round(3)
    stats.to_csv(os.path.join(project_root, output_dir, 'stats.csv'))
    print("📊 stats.csv salvo")
    
    # 2. Target distribution
    plt.figure(figsize=(8, 5))
    y_counts = y_train.value_counts()
    colors = ['green' if i==0 else 'red' for i in y_counts.index]
    y_counts.plot(kind='bar', color=colors)
    plt.title('Distribuição: Normal (0) vs Atraso Alto (1)')
    plt.ylabel('Contagem'); plt.xlabel('Classe (0=OK, 1=Crítico)')
    plt.xticks(rotation=0)
    plt.savefig(os.path.join(project_root, output_dir, 'target_dist.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Top features VARIÂNCIA (FIX: remove duplicates)
    variances = X_train.var().sort_values(ascending=False)
    # Remove arr_flights duplicados, pega top únicos
    top_unique = variances[~variances.index.str.contains('arr_flights')].nlargest(10)
    if len(top_unique) < 5:
        top_unique = variances.nlargest(10)  # Fallback
    
    plt.figure(figsize=(12, 6))
    top_unique.head(10).plot(kind='bar', color='skyblue')
    plt.title('Top 10 Features por Variância (Mais Informativas)')
    plt.ylabel('Variância'); plt.xlabel('Features')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(project_root, output_dir, 'top_features.png'), dpi=150)
    plt.close()
    
    # Debug print
    print("Top 5 variância únicas:")
    print(top_unique.head().to_dict())
    
    print(f"✅ EDA completo: {os.listdir(os.path.join(project_root, output_dir))}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EDA Flights Dataset')
    parser.add_argument('--input', '-i', default='data/processed/X_train.parquet')
    parser.add_argument('--output', '-o', default='outputs/eda')
    args = parser.parse_args()
    eda(args.input, args.output)
