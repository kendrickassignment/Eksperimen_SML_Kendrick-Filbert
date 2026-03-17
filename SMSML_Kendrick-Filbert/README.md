# 🧬 Breast Cancer Classification — End-to-End MLOps Pipeline

> **Submission Akhir — Membangun Sistem Machine Learning**
> 
> **Author:** Kendrick Filbert
> 
> **Dicoding Username:** kendrickfff
> 
> **Date:** March 2026

---

## 📋 Project Overview

Proyek ini mengimplementasikan **end-to-end MLOps pipeline** untuk klasifikasi kanker payudara (*Breast Cancer Classification*) menggunakan dataset Wisconsin Breast Cancer. Pipeline mencakup seluruh lifecycle machine learning mulai dari preprocessing data, eksperimen model, CI/CD workflow, hingga monitoring & alerting di production.

### 🎯 Objective
Membangun sistem klasifikasi yang dapat memprediksi apakah tumor bersifat **Benign** (jinak) atau **Malignant** (ganas) berdasarkan 30 fitur numerik dari hasil biopsi.

---

## 📁 Project Structure

```
SMSML_Kendrick-Filbert/
│   README.md
│   Eksperimen_SML_Kendrick-Filbert.txt
│   Workflow-CI.txt
│
├── Eksperimen_SML_Kendrick-Filbert/        # Kriteria 1 — Preprocessing & EDA
│   │   breast_cancer_raw.csv
│   └── preprocessing/
│       ├── Eksperimen_Kendrick-Filbert.ipynb
│       ├── automate_Kendrick-Filbert.py
│       ├── breast_cancer_train_preprocessing.csv
│       └── breast_cancer_test_preprocessing.csv
│
├── Membangun_model/                         # Kriteria 2 — Model Building & Tuning
│   ├── modelling.py
│   ├── modelling_tuning.py
│   ├── requirements.txt
│   ├── DagsHub.txt
│   ├── screenshoot_dashboard.jpg
│   ├── screenshoot_artifak.jpg
│   ├── breast_cancer_train_preprocessing.csv
│   ├── breast_cancer_test_preprocessing.csv
│   ├── mlartifacts/
│   └── mlruns/
│
├── Workflow-CI/                             # Kriteria 3 — CI/CD Pipeline
│   └── MLProject/
│       ├── modelling.py
│       ├── MLProject
│       ├── conda.yaml
│       ├── Dockerfile
│       ├── DockerHub.txt
│       ├── breast_cancer_train_preprocessing.csv
│       └── breast_cancer_test_preprocessing.csv
│
└── Monitoring dan Logging/                  # Kriteria 4 — Monitoring & Alerting
    ├── 7.inference.py
    ├── 2.prometheus.yml
    ├── 3.prometheus_exporter.py
    ├── 4.grafana_dashboard.json
    ├── 1.bukti_serving.png
    ├── 4.bukti monitoring Prometheus/       # 10 screenshots per metriks
    ├── 5.bukti monitoring Grafana/          # Screenshot dashboard Grafana
    └── 6.bukti alerting Grafana/            # 3 alert rules screenshots
```

---

## 🔬 Kriteria 1 — Preprocessing & Exploratory Data Analysis

### Deskripsi
Melakukan preprocessing dan EDA pada dataset Wisconsin Breast Cancer untuk mempersiapkan data sebelum modelling.

### Proses
- **Exploratory Data Analysis** — distribusi fitur, korelasi antar variabel, class imbalance check
- **Data Cleaning** — handling missing values, duplicate removal
- **Feature Engineering** — normalisasi/standardisasi fitur numerik
- **Train-Test Split** — pembagian data untuk training dan testing
- **Automation Script** — `automate_Kendrick-Filbert.py` untuk reproducible preprocessing
- **GitHub Actions** — workflow otomatis untuk menjalankan preprocessing setiap kali trigger terpantik

### Output
| File | Deskripsi |
|------|-----------|
| `Eksperimen_Kendrick-Filbert.ipynb` | Jupyter Notebook — full EDA & preprocessing |
| `automate_Kendrick-Filbert.py` | Script otomasi preprocessing |
| `breast_cancer_train_preprocessing.csv` | Dataset training (preprocessed) |
| `breast_cancer_test_preprocessing.csv` | Dataset testing (preprocessed) |

### GitHub Repository
[kendrickassignment/Eksperimen_SML_Kendrick-Filbert](https://github.com/kendrickassignment/Eksperimen_SML_Kendrick-Filbert)

---

## 🤖 Kriteria 2 — Membangun Model ML

### Deskripsi
Training dan hyperparameter tuning model klasifikasi menggunakan MLflow untuk experiment tracking dan DagsHub sebagai remote tracking server.

### Model & Tracking
- **Algorithm:** Random Forest Classifier
- **Hyperparameter Tuning:** Grid Search via `modelling_tuning.py`
- **Experiment Tracking:** MLflow + DagsHub (remote)
- **Manual Logging:** Metrics, parameters, dan artifacts secara manual (bukan autolog)

### Artifacts yang Di-log
| Artifact | Deskripsi |
|----------|-----------|
| `model.pkl` | Trained model (Random Forest) |
| `confusion_matrix.png` | Confusion matrix visualization |
| `classification_report.json` | Classification report |
| `feature_importance.png` | Feature importance chart |
| `roc_curve.png` | ROC-AUC curve |
| `dataset_info.json` | Dataset metadata |

### MLflow Metrics
- Accuracy, Precision, Recall, F1-Score
- Log Loss, ROC-AUC Score

### Output
| File | Deskripsi |
|------|-----------|
| `modelling.py` | Script training model (autolog) |
| `modelling_tuning.py` | Script hyperparameter tuning (manual log + DagsHub) |
| `mlruns/` | MLflow experiment runs (local) |
| `mlartifacts/` | Model artifacts & metrics |

### DagsHub
[kendrickassignment/Membangun_model](https://dagshub.com/kendrickassignment/Membangun_model)

---

## ⚙️ Kriteria 3 — Workflow CI/CD

### Deskripsi
Implementasi CI/CD pipeline menggunakan **GitHub Actions** yang secara otomatis melakukan training model, upload artifacts, dan build Docker image setiap kali ada push ke repository.

### Pipeline Flow
```
Push to main → Checkout → Setup Python 3.12 → Install Dependencies
    → Run ML Training → Upload Artifacts → Build Docker Image
    → Push to Docker Hub
```

### GitHub Actions Workflow
- **Trigger:** Push ke branch `main`, Pull Request, atau Manual Dispatch
- **Training:** Otomatis menjalankan `modelling.py` di runner
- **Artifacts:** ML model artifacts di-upload ke GitHub Artifacts (retention 90 hari)
- **Docker:** Build dan push image ke Docker Hub

### Links
| Resource | URL |
|----------|-----|
| GitHub Repository | [kendrickassignment/Workflow-CI](https://github.com/kendrickassignment/Workflow-CI) |
| Docker Hub Image | `kendrickfff/breast-cancer-ml:latest` |

---

## 📊 Kriteria 4 — Monitoring & Logging

### Deskripsi
Implementasi monitoring stack lengkap untuk ML inference API menggunakan **Prometheus** (metrics collection) dan **Grafana** (visualization & alerting).

### Architecture
```
┌─────────────────┐     ┌──────────────┐     ┌─────────────┐
│  Flask API       │────▶│  Prometheus   │────▶│   Grafana    │
│  (port 5001)     │     │  (port 9090)  │     │  (port 3000) │
│                  │     │               │     │              │
│  /predict (POST) │     │  Scrape /     │     │  Dashboard   │
│  /health  (GET)  │     │  metrics      │     │  Alerting    │
│  /metrics (GET)  │     │  every 15s    │     │              │
└─────────────────┘     └──────────────┘     └─────────────┘
```

### API Endpoints
| Endpoint | Method | Deskripsi |
|----------|--------|-----------|
| `/predict` | POST | Prediksi Benign/Malignant |
| `/health` | GET | Health check API |
| `/metrics` | GET | Prometheus metrics endpoint |

### Prometheus Metrics (12 Metriks)
| # | Metric | Type | Deskripsi |
|---|--------|------|-----------|
| 1 | `ml_request_total` | Counter | Total requests per endpoint/status |
| 2 | `ml_request_latency_seconds` | Histogram | Request latency distribution |
| 3 | `ml_prediction_class_total` | Counter | Predictions per class (benign/malignant) |
| 4 | `ml_inference_duration_seconds` | Histogram | Model inference duration |
| 5 | `ml_prediction_confidence` | Histogram | Prediction confidence distribution |
| 6 | `ml_active_requests` | Gauge | Currently active requests |
| 7 | `ml_error_total` | Counter | Total error count |
| 8 | `ml_model_loaded` | Gauge | Model loaded status (1=yes, 0=no) |
| 9 | `ml_predictions_total` | Counter | Total successful predictions |
| 10 | `ml_request_feature_count` | Histogram | Feature count per request |
| 11 | `ml_input_feature_mean` | Gauge | Mean input features per group |
| 12 | `ml_prediction_probability_summary` | Summary | Prediction probability summary |

### Grafana Dashboard — "ML Monitoring Dashboard — kendrickfff"
Dashboard terdiri dari **16 panels** dalam 4 section:

**Overview (6 stat panels)**
1. Total Requests (`ml_request_total`)
2. Predictions (`ml_predictions_total`)
3. API Health (`up`)
4. Model Status (`ml_model_loaded`)
5. Avg Latency (`ml_request_latency_seconds`)
6. Active Requests (`ml_active_requests`)

**Request & Latency Monitoring (2 timeseries)**
7. Request Rate — POST /predict & GET /health
8. Request Latency — Avg & p95

**Prediction & Inference Monitoring (2 timeseries)**
9. Prediction Class — Benign vs Malignant
10. Inference Duration — Avg & p95

**Confidence, Features & Errors (6 panels)**
11. Prediction Confidence — Avg & Median
12. Feature Count — Avg Features/Request
13. Input Feature Mean — Mean, SE, Worst Features
14. Error Rate — Success vs Errors
15. Probability Summary — Avg Probability
16. Predictions Over Time — Predictions/sec

### Alerting (3 Rules)
| # | Alert Name | Metric | Condition |
|---|-----------|--------|-----------|
| 1 | High Error Rate | `ml_request_total{status="200"}` | IS ABOVE 0.1 |
| 2 | High Latency | `ml_request_latency_seconds_count` | IS ABOVE 100 |
| 3 | Model Down | `ml_model_loaded` | IS BELOW 1 |

- **Evaluation Interval:** Every 1 minute
- **Evaluation Group:** ml-monitoring
- **Folder:** ML Alerts

---

## 🛠️ Tech Stack

| Category | Technology |
|----------|------------|
| Language | Python 3.12 |
| ML Framework | scikit-learn 1.5.2 |
| Experiment Tracking | MLflow 2.19.0 + DagsHub |
| Data Processing | Pandas 2.2.3, NumPy 1.26.4 |
| Visualization | Matplotlib 3.9.2, Seaborn 0.13.2 |
| API Framework | Flask |
| Metrics | Prometheus Client |
| Monitoring | Prometheus |
| Dashboard & Alerting | Grafana |
| CI/CD | GitHub Actions |
| Containerization | Docker |
| Container Registry | Docker Hub |
| Version Control | Git + GitHub |

---

## 🚀 Quick Start

### 1. Run Inference API
```bash
cd "Monitoring dan Logging"
pip install flask prometheus_client requests scikit-learn joblib
python 7.inference.py
```
API akan berjalan di `http://localhost:5001`

### 2. Test Prediction
```bash
curl -X POST http://localhost:5001/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [0.1, 0.2, ...]}'
```

### 3. Run Prometheus
```bash
# Download Prometheus, extract, lalu:
./prometheus --config.file=2.prometheus.yml
```
Prometheus UI di `http://localhost:9090`

### 4. Run Grafana
```bash
# Install Grafana, lalu buka:
# http://localhost:3000 (admin/admin)
# Import 4.grafana_dashboard.json
```

### 5. Generate Test Traffic
```bash
python 3.prometheus_exporter.py
```

---

## 📈 Results Summary

| Metric | Value |
|--------|-------|
| Model | Random Forest Classifier |
| API Response Time | ~17ms avg |
| p95 Latency | ~40ms |
| API Uptime | 100% (during testing) |
| Prediction Confidence | ~60% avg |
| CI/CD Pipeline | ✅ Passing (2m 52s) |
| Docker Image | ✅ Published |
| Prometheus Metrics | 12 metriks |
| Grafana Panels | 16 panels |
| Alerting Rules | 3 rules |

---

## 📝 License

This project is submitted as part of **Membangun Sistem Machine Learning** curriculum at Dicoding. For educational purposes only.

---

<p align="center">
  <b>Made with ❤️ by Kendrick Filbert (kendrickfff)</b>
</p>
