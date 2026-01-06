#!/usr/bin/env python3
"""
Task 2 EDA: Stats + Plots obrigatórios Tech Challenge Fase 3
Roda da RAIZ projeto!
"""
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

sns.set_style("whitegrid")

def eda(input_path, output_dir):
    # Paths ABSOLUTOS da raiz projeto
    script_dir = os.path.dirname(os.path.abspath(__file__))  # src/
    project_root = os.path.dirname(script_dir)  # raiz/
    
    X_train = pd.read_parquet(os.path.join(project_root, input_path))
    y_train_df = pd.read_parquet(os.path.join(project_root, 'data/processed/y_train.parquet'))
    y_train = y_train_df['target']  # Extrai Series
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"🔍 EDA X_train: {X_train.shape}")
    
    # Stats descritivas (obrigatório)
    stats = X_train.describe().round(3)
    stats.to_csv(os.path.join(project_root, output_dir, 'stats.csv'))
    
    # Plot 1: Target distribution
    plt.figure(figsize=(8, 5))
    y_train.value_counts().plot(kind='bar', color=['green', 'red'])
    plt.title('Voos: Normal (0) vs Atraso Alto (1)')
    plt.ylabel('Contagem')
    plt.savefig(os.path.join(project_root, output_dir, 'target_dist.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Top variância
    vars_top = X_train.var().nlargest(10)
    plt.figure(figsize=(12, 6))
    vars_top.plot(kind='bar', color='skyblue')
    plt.title('Top 10 Features (Variância)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(project_root, output_dir, 'top_features.png'), dpi=150)
    plt.close()
    
    print(f"✅ EDA: {os.listdir(os.path.join(project_root, output_dir))}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', default='data/processed/X_train.parquet')
    parser.add_argument('--output', '-o', default='outputs/eda')
    args = parser.parse_args()
    eda(args.input, args.output)
