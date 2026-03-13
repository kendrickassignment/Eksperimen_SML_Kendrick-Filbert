"""
Automate Preprocessing - Kendrick Filbert
==========================================
Script untuk melakukan preprocessing data Breast Cancer secara otomatis.
Mengembalikan data yang siap dilatih (X_train, X_test, y_train, y_test).

Tahapan: Load -> Hapus Duplikasi -> Handle Missing -> Outlier Capping 
         -> Split -> Standarisasi -> Simpan
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import argparse


def load_data(save_raw=True, output_dir="."):
    cancer = load_breast_cancer()
    df = pd.DataFrame(data=cancer.data, columns=cancer.feature_names)
    df['target'] = cancer.target
    if save_raw:
        raw_path = os.path.join(output_dir, "breast_cancer_raw.csv")
        df.to_csv(raw_path, index=False)
        print(f"[INFO] Raw dataset disimpan ke: {raw_path}")
    print(f"[INFO] Dataset dimuat: {df.shape[0]} baris, {df.shape[1]} kolom")
    return df


def remove_duplicates(df):
    before = df.shape[0]
    df_clean = df.drop_duplicates()
    after = df_clean.shape[0]
    print(f"[INFO] Duplikasi dihapus: {before - after} baris ({before} -> {after})")
    return df_clean


def handle_missing_values(df):
    missing_total = df.isnull().sum().sum()
    if missing_total > 0:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)
        print(f"[INFO] Missing values ditangani: {missing_total} values diisi dengan median")
    else:
        print(f"[INFO] Tidak ada missing values ditemukan")
    return df


def cap_outliers(df, target_col='target'):
    df_capped = df.copy()
    feature_cols = [col for col in df.columns if col != target_col]
    total_capped = 0
    for col in feature_cols:
        Q1 = df_capped[col].quantile(0.25)
        Q3 = df_capped[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = ((df_capped[col] < lower_bound) | (df_capped[col] > upper_bound)).sum()
        total_capped += outliers
        df_capped[col] = df_capped[col].clip(lower=lower_bound, upper=upper_bound)
    print(f"[INFO] Total outlier yang di-cap: {total_capped}")
    return df_capped


def split_and_scale(df, target_col='target', test_size=0.2, random_state=42):
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"[INFO] Train-test split: Train={X_train.shape[0]}, Test={X_test.shape[0]}")
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
    print(f"[INFO] Standarisasi diterapkan (mean~0, std~1)")
    return X_train_scaled, X_test_scaled, y_train, y_test


def save_preprocessed_data(X_train, X_test, y_train, y_test, output_dir="."):
    os.makedirs(output_dir, exist_ok=True)
    train_data = X_train.copy()
    train_data['target'] = y_train.values
    test_data = X_test.copy()
    test_data['target'] = y_test.values
    train_path = os.path.join(output_dir, "breast_cancer_train_preprocessing.csv")
    test_path = os.path.join(output_dir, "breast_cancer_test_preprocessing.csv")
    train_data.to_csv(train_path, index=False)
    test_data.to_csv(test_path, index=False)
    print(f"[INFO] Train disimpan: {train_path} ({train_data.shape})")
    print(f"[INFO] Test disimpan: {test_path} ({test_data.shape})")
    return train_path, test_path


def run_preprocessing(output_dir="."):
    print("=" * 60)
    print("AUTOMATED PREPROCESSING PIPELINE")
    print("Dataset: Breast Cancer Wisconsin (Diagnostic)")
    print("Author: Kendrick Filbert")
    print("=" * 60)

    print("\n[STEP 1] Loading dataset...")
    df = load_data(save_raw=True, output_dir=output_dir)

    print("\n[STEP 2] Removing duplicates...")
    df = remove_duplicates(df)

    print("\n[STEP 3] Handling missing values...")
    df = handle_missing_values(df)

    print("\n[STEP 4] Capping outliers (IQR method)...")
    df = cap_outliers(df)

    print("\n[STEP 5] Splitting & scaling...")
    X_train, X_test, y_train, y_test = split_and_scale(df)

    print("\n[STEP 6] Saving preprocessed data...")
    save_preprocessed_data(X_train, X_test, y_train, y_test, output_dir)

    print("\n" + "=" * 60)
    print("PREPROCESSING SELESAI!")
    print(f"   Training set: {X_train.shape}")
    print(f"   Testing set: {X_test.shape}")
    print("=" * 60)
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automate Preprocessing")
    parser.add_argument("--output_dir", type=str, default=".", help="Output directory")
    args = parser.parse_args()
    run_preprocessing(output_dir=args.output_dir)
