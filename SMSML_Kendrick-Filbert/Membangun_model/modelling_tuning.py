"""
Modelling Tuning - Kendrick Filbert
=====================================
- Manual logging (bukan autolog)
- Hyperparameter tuning (GridSearchCV)
- 5 artefak tambahan: confusion matrix, feature importance, ROC curve, classification report, dataset info
- DagsHub integration (Advanced)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, matthews_corrcoef,
    classification_report, confusion_matrix, roc_curve
)
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# DagsHub Integration (Advanced)
# ============================================================
import dagshub
dagshub.init(repo_owner='kendrickassignment', repo_name='Membangun_model', mlflow=True)


def load_preprocessed_data():
    train_df = pd.read_csv("breast_cancer_train_preprocessing.csv")
    test_df = pd.read_csv("breast_cancer_test_preprocessing.csv")
    X_train = train_df.drop('target', axis=1)
    y_train = train_df['target']
    X_test = test_df.drop('target', axis=1)
    y_test = test_df['target']
    print(f"[INFO] Data loaded - Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def create_confusion_matrix_plot(y_test, y_pred, save_path="confusion_matrix.png"):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Malignant', 'Benign'], yticklabels=['Malignant', 'Benign'])
    plt.title('Confusion Matrix - Random Forest (Tuned)', fontsize=14, fontweight='bold')
    plt.ylabel('Actual', fontsize=12)
    plt.xlabel('Predicted', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return save_path


def create_feature_importance_plot(model, feature_names, save_path="feature_importance.png"):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:15]
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(indices)), importances[indices], color='steelblue')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Feature Importance', fontsize=12)
    plt.title('Top 15 Feature Importance', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return save_path


def create_roc_curve_plot(y_test, y_prob, save_path="roc_curve.png"):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = roc_auc_score(y_test, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve - Random Forest (Tuned)', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return save_path


def train_model_with_tuning():
    # Tracking URI set oleh dagshub.init() -> artefak ke DagsHub online
    mlflow.set_experiment("Breast_Cancer_Classification_Tuning")

    X_train, X_test, y_train, y_test = load_preprocessed_data()

    # Hyperparameter Tuning
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }

    print("\n[INFO] Starting GridSearchCV...")
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_cv_score = grid_search.best_score_
    print(f"\n[INFO] Best Parameters: {best_params}")
    print(f"[INFO] Best CV F1-Score: {best_cv_score:.4f}")

    # Predictions & Metrics
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "log_loss": log_loss(y_test, y_prob),
        "matthews_corrcoef": matthews_corrcoef(y_test, y_pred),
        "specificity": recall_score(y_test, y_pred, pos_label=0),
        "best_cv_f1_score": best_cv_score,
    }

    # MLflow Manual Logging
    with mlflow.start_run(run_name="RandomForest_Tuned_Manual"):
        # Log Parameters
        for k, v in best_params.items():
            mlflow.log_param(k, v)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("cv_folds", 5)
        mlflow.log_param("scoring", "f1")
        mlflow.log_param("tuning_method", "GridSearchCV")

        # Log Metrics
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        # Log Model
        signature = infer_signature(X_train, best_model.predict(X_train))
        mlflow.sklearn.log_model(best_model, "random_forest_tuned_model", signature=signature)

        # Artefak 1: Confusion Matrix
        cm_path = create_confusion_matrix_plot(y_test, y_pred)
        mlflow.log_artifact(cm_path)

        # Artefak 2: Feature Importance
        fi_path = create_feature_importance_plot(best_model, X_train.columns.tolist())
        mlflow.log_artifact(fi_path)

        # Artefak 3: ROC Curve
        roc_path = create_roc_curve_plot(y_test, y_prob)
        mlflow.log_artifact(roc_path)

        # Artefak 4: Classification Report JSON
        report = classification_report(y_test, y_pred, target_names=['Malignant', 'Benign'], output_dict=True)
        with open("classification_report.json", 'w') as f:
            json.dump(report, f, indent=2)
        mlflow.log_artifact("classification_report.json")

        # Artefak 5: Dataset Info JSON
        dataset_info = {
            "train_shape": list(X_train.shape),
            "test_shape": list(X_test.shape),
            "features": X_train.columns.tolist(),
            "target_distribution_train": y_train.value_counts().to_dict(),
            "target_distribution_test": y_test.value_counts().to_dict()
        }
        with open("dataset_info.json", 'w') as f:
            json.dump(dataset_info, f, indent=2, default=str)
        mlflow.log_artifact("dataset_info.json")

        # Tags
        mlflow.set_tag("model_type", "RandomForestClassifier")
        mlflow.set_tag("tuning", "GridSearchCV")
        mlflow.set_tag("author", "Kendrick Filbert")
        mlflow.set_tag("dataset", "Breast Cancer Wisconsin")

        run_id = mlflow.active_run().info.run_id

        print("\n" + "=" * 60)
        print("MODEL EVALUATION RESULTS (TUNED)")
        print("=" * 60)
        for k, v in metrics.items():
            print(f"  {k:25s}: {v:.4f}")
        print(f"\n{classification_report(y_test, y_pred, target_names=['Malignant', 'Benign'])}")
        print(f"Run ID: {run_id}")
        print(f"Artefak: model, confusion_matrix, feature_importance, roc_curve, classification_report, dataset_info")

    # Cleanup temp files
    for f in [cm_path, fi_path, roc_path, "classification_report.json", "dataset_info.json"]:
        if os.path.exists(f):
            os.remove(f)


if __name__ == "__main__":
    train_model_with_tuning()
