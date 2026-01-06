#!/usr/bin/env python3
"""
Task 4: Unsupervised ML - KMeans + PCA (obrigatório)
Agrupa aeroportos/carriers por padrões atraso
"""
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sns.set_style("whitegrid")

def unsupervised(X_path, output_dir):
    X = pd.read_parquet(X_path)
    print(f"🔍 Unsupervised: {X.shape}")
    
    # Padroniza (obrigatório clustering)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 1. KMeans (3 clusters: baixo/médio/alto atraso)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    # 2. PCA 2D para visualizar
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Plot clusters PCA
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.7)
    plt.xlabel(f'PCA1 ({pca.explained_variance_ratio_[0]:.1%} var)')
    plt.ylabel(f'PCA2 ({pca.explained_variance_ratio_[1]:.1%} var)')
    plt.title('Clusters KMeans (3 grupos) + PCA 2D\nAeroportos/Carriers por padrão atraso')
    plt.colorbar(scatter, label='Cluster')
    plt.savefig(f'{output_dir}/kmeans_pca.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Salva clusters
    clusters_df = pd.DataFrame({'cluster': clusters})
    clusters_df.to_csv(f'{output_dir}/clusters.csv')
    
    # Explained variance
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, 11), [PCA(n_components=i).fit(X_scaled).explained_variance_ratio_.sum() for i in range(1, 11)])
    plt.title('PCA Explained Variance (Cumulative)')
    plt.xlabel('Componentes'); plt.ylabel('Variância Explicada')
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{output_dir}/pca_variance.png', dpi=150)
    plt.close()
    
    print(f"✅ Unsupervised: {os.listdir(output_dir)}")
    print("📊 Interpretação: 3 clusters rotas (verde=OK, roxo=crítico)")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', default='data/processed/X_train.parquet')
    parser.add_argument('--output', '-o', default='outputs/unsupervised')
    args = parser.parse_args()
    unsupervised(args.input, args.output)
