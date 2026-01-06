#!/usr/bin/env python3
"""
Processa flights.csv agregados → ML ready Parquet
Target: delay_rate > 10% (classificação rota ruim)
"""
import argparse
import logging
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def process(input_path, output_dir):
    df = pd.read_csv(input_path)
    log.info(f"Dataset carregado: {df.shape}")
    
    # Features numéricas
    df['delay_rate'] = df['arr_del15'] / df['arr_flights'].replace(0, 1)  # Evita div0
    num_feats = ['year', 'month', 'arr_flights', 'arr_delay', 'delay_rate']
    df[num_feats] = df[num_feats].fillna(0)
    
    # Categóricas → dummies
    cat_feats = ['carrier', 'airport_name']
    X_cat = pd.get_dummies(df[cat_feats].fillna('MISSING').astype(str), 
                          prefix=cat_feats, drop_first=True)
    
    # X final
    X = pd.concat([df[num_feats], X_cat], axis=1)
    
    # y: 1 se delay_rate > 10%
    y = pd.Series((df['delay_rate'] > 0.1).astype(int), name='target')
    
    log.info(f"X: {X.shape}, y: {y.value_counts().to_dict()}")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # SALVAR como DataFrame (Series → df)
    os.makedirs(output_dir, exist_ok=True)
    X_train.to_parquet(f'{output_dir}/X_train.parquet')
    X_test.to_parquet(f'{output_dir}/X_test.parquet')
    pd.DataFrame({'target': y_train}).to_parquet(f'{output_dir}/y_train.parquet')
    pd.DataFrame({'target': y_test}).to_parquet(f'{output_dir}/y_test.parquet')
    
    log.info(f"✅ data/processed/ pronto: {os.listdir(output_dir)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Flights → ML Dataset")
    parser.add_argument('--input', '-i', required=True)
    parser.add_argument('--output', '-o', default='data/processed')
    args = parser.parse_args()
    process(args.input, args.output)
