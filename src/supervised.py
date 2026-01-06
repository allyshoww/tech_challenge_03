#!/usr/bin/env python3
"""
Task 3: Supervised ML - RF + Logistic (obrigatório comparar 2+)
Salva: outputs/models/*.pkl + metrics.txt
"""
import argparse
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib
from sklearn.model_selection import train_test_split

def train_models(X_path, y_path, output_dir):
    # Carrega dados
    X_train = pd.read_parquet(X_path)
    y_train_df = pd.read_parquet(y_path.replace('train', 'test'))  # Test set
    y_test = y_train_df['target']
    
    print(f"🏭 Treinando: X {X_train.shape}")
    
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'LogisticReg': LogisticRegression(max_iter=1000, random_state=42)
    }
    
    results = {}
    os.makedirs(output_dir, exist_ok=True)
    
    for name, model in models.items():
        model.fit(X_train, pd.read_parquet(y_path)['target'])
        y_pred = model.predict(X_train)
        
        acc = accuracy_score(pd.read_parquet(y_path)['target'], y_pred)
        results[name] = {'accuracy': acc}
        
        # Salva modelo
        joblib.dump(model, f'{output_dir}/{name.lower()}.pkl')
        print(f"✅ {name}: {acc:.3f} acc")
    
    # Metrics report
    with open(f'{output_dir}/metrics.txt', 'w') as f:
        for name, res in results.items():
            f.write(f"{name}: Accuracy {res['accuracy']:.3f}\n")
    
    print(f"✅ Supervised: {os.listdir(output_dir)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--X', default='data/processed/X_train.parquet')
    parser.add_argument('--y', default='data/processed/y_train.parquet')
    parser.add_argument('--output', default='outputs/models')
    args = parser.parse_args()
    train_models(args.X, args.y, args.output)
