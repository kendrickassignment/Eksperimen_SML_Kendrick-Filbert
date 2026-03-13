"""
Modelling - Kendrick Filbert
==============================
Model Random Forest Classifier + MLflow autolog (Basic level).
Dataset: Breast Cancer Wisconsin (Preprocessed)
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import mlflow
import mlflow.sklearn
import warnings
warnings.filterwarnings('ignore')


def load_preprocessed_data():
    train_df = pd.read_csv("breast_cancer_train_preprocessing.csv")
    test_df = pd.read_csv("breast_cancer_test_preprocessing.csv")
    X_train = train_df.drop('target', axis=1)
    y_train = train_df['target']
    X_test = test_df.drop('target', axis=1)
    y_test = test_df['target']
    print(f"[INFO] Data loaded - Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def train_model():
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Breast_Cancer_Classification")

    X_train, X_test, y_train, y_test = load_preprocessed_data()

    mlflow.sklearn.autolog()

    with mlflow.start_run(run_name="RandomForest_Autolog"):
        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        print("\n" + "=" * 50)
        print("MODEL EVALUATION RESULTS")
        print("=" * 50)
        print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
        print(f"Precision: {precision_score(y_test, y_pred):.4f}")
        print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
        print(f"F1-Score:  {f1_score(y_test, y_pred):.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Malignant', 'Benign']))
        print(f"\nRun ID: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    train_model()
