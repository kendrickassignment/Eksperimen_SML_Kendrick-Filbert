# 🚀 PANDUAN STEP-BY-STEP - Target Bintang 5
## Kendrick Filbert | GitHub: kendrickassignment | DagsHub: kendrickassignment | Docker: kendrickfff | Dicoding: kendrickfff

---

## STEP 1: Setup Environment (sekali saja)
```bash
python -m venv mlenv
# Windows: mlenv\Scripts\activate
# Mac/Linux: source mlenv/bin/activate

pip install pandas==2.2.3 numpy==1.26.4 scikit-learn==1.5.2 mlflow==2.19.0 matplotlib==3.9.2 seaborn==0.13.2 dagshub==0.3.37 flask prometheus_client requests
```

---

## STEP 2: Kriteria 1 — Eksperimen (4 pts)

### 2a. Buat repo GitHub
- Buka https://github.com/new
- Nama: `Eksperimen_SML_Kendrick-Filbert`
- Visibility: **Public**
- Jangan centang "Add README"

### 2b. Push files
```bash
# Extract ZIP, masuk folder Eksperimen
cd Eksperimen_SML_Kendrick-Filbert

git init
git add .
git commit -m "Eksperimen SML Kendrick Filbert"
git branch -M main
git remote add origin https://github.com/kendrickassignment/Eksperimen_SML_Kendrick-Filbert.git
git push -u origin main
```

### 2c. Jalankan Notebook
- Buka `preprocessing/Eksperimen_Kendrick-Filbert.ipynb` di Jupyter/Colab
- **Run ALL cells** dari atas ke bawah → pastikan tidak ada error
- Save notebook

### 2d. Push ulang setelah run notebook
```bash
git add .
git commit -m "Run all notebook cells"
git push
```

### 2e. Verifikasi GitHub Actions
- Buka repo di GitHub → tab **Actions**
- Workflow "Automated Preprocessing Pipeline" harus berjalan ✅
- Jika belum trigger: klik **Run workflow** manual

---

## STEP 3: Kriteria 2 — Model (4 pts)

### 3a. Buat repo DagsHub
- Buka https://dagshub.com → klik **Create +** → **New Repository**
- Nama: `Membangun_model`

### 3b. Jalankan modelling.py (Basic — autolog)
```bash
cd Membangun_model

# Terminal 1: Start MLflow UI
mlflow ui --host 127.0.0.1 --port 5000

# Terminal 2: Jalankan modelling
python modelling.py
```

### 3c. Jalankan modelling_tuning.py (Advanced — DagsHub)
```bash
python modelling_tuning.py
# Pertama kali akan minta login DagsHub → ikuti instruksi di terminal
```

### 3d. Screenshot (WAJIB!)
1. Buka http://127.0.0.1:5000
2. **screenshoot_dashboard.jpg** — screenshot halaman utama MLflow (daftar experiments & runs)
3. Klik run "RandomForest_Tuned_Manual" → **screenshoot_artifak.jpg** — screenshot halaman artifacts

Simpan kedua screenshot ke folder `Membangun_model/`

---

## STEP 4: Kriteria 3 — Workflow CI (4 pts)

### 4a. Buat repo GitHub
- https://github.com/new → Nama: `Workflow-CI` → **Public**

### 4b. Setup Docker Hub Token
- Buka https://hub.docker.com → Account Settings → Security → **New Access Token**
- Nama: `github-actions` → Generate → **Copy token**

### 4c. Setup GitHub Secrets
- Di repo `Workflow-CI` → Settings → Secrets and variables → Actions → **New repository secret**:
  - `DOCKERHUB_USERNAME` → `kendrickfff`
  - `DOCKERHUB_TOKEN` → (paste token dari step 4b)

### 4d. Push files
```bash
cd Workflow-CI

git init
git add .
git commit -m "Workflow CI"
git branch -M main
git remote add origin https://github.com/kendrickassignment/Workflow-CI.git
git push -u origin main
```

### 4e. Verifikasi
- Tab Actions → workflow harus berjalan ✅
- Cek Docker Hub: https://hub.docker.com/r/kendrickfff/breast-cancer-ml

---

## STEP 5: Kriteria 4 — Monitoring (4 pts)

### 5a. Jalankan API
```bash
cd "Monitoring dan Logging"
pip install flask prometheus_client requests

# Terminal 1
python 7.inference.py
```

### 5b. Test & Screenshot Serving
```bash
# Terminal 2
curl http://localhost:5001/health
curl -X POST http://localhost:5001/predict -H "Content-Type: application/json" -d "{\"features\": [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]}"
```
📸 Screenshot terminal → simpan sebagai `1.bukti_serving.png`

### 5c. Setup Prometheus
- Download dari https://prometheus.io/download/ (pilih OS kamu)
- Extract, copy `2.prometheus.yml` ke folder Prometheus sebagai `prometheus.yml`
```bash
./prometheus --config.file=prometheus.yml
```
Buka http://localhost:9090

### 5d. Generate traffic
```bash
# Terminal 3
python 3.prometheus_exporter.py
```

### 5e. Screenshot Prometheus (10 metriks)
Di Prometheus UI (http://localhost:9090), query satu per satu dan screenshot:

| No | Query | Filename |
|----|-------|----------|
| 1 | `ml_request_total` | `1.monitoring_request_total.png` |
| 2 | `ml_request_latency_seconds_bucket` | `2.monitoring_request_latency.png` |
| 3 | `ml_prediction_class_total` | `3.monitoring_prediction_class.png` |
| 4 | `ml_inference_duration_seconds_bucket` | `4.monitoring_inference_duration.png` |
| 5 | `ml_prediction_confidence_bucket` | `5.monitoring_prediction_confidence.png` |
| 6 | `ml_active_requests` | `6.monitoring_active_requests.png` |
| 7 | `ml_error_total` | `7.monitoring_error_total.png` |
| 8 | `ml_input_feature_mean` | `8.monitoring_input_feature_mean.png` |
| 9 | `ml_model_loaded` | `9.monitoring_model_loaded.png` |
| 10 | `ml_predictions_total` | `10.monitoring_predictions_total.png` |

Simpan ke folder `4.bukti monitoring Prometheus/`

### 5f. Setup Grafana
- Download dari https://grafana.com/grafana/download
- Jalankan → Buka http://localhost:3000 (login: admin/admin)
- **Add Data Source** → Prometheus → URL: `http://localhost:9090` → Save & Test

### 5g. Buat Dashboard (PENTING: nama = `kendrickfff`)
- Create → Dashboard → Settings → nama: **`kendrickfff`**
- Add 10 panels:

| Panel | Tipe | Query PromQL |
|-------|------|-------------|
| 1. Total Requests | Stat | `sum(ml_request_total)` |
| 2. Request Latency p95 | Graph | `histogram_quantile(0.95, rate(ml_request_latency_seconds_bucket[5m]))` |
| 3. Predictions/Class | Bar | `ml_prediction_class_total` |
| 4. Avg Inference Time | Graph | `rate(ml_inference_duration_seconds_sum[5m]) / rate(ml_inference_duration_seconds_count[5m])` |
| 5. Confidence Distribution | Histogram | `ml_prediction_confidence_bucket` |
| 6. Active Requests | Gauge | `ml_active_requests` |
| 7. Error Rate | Graph | `rate(ml_error_total[5m])` |
| 8. Input Feature Mean | Graph | `ml_input_feature_mean` |
| 9. Model Status | Stat | `ml_model_loaded` |
| 10. Prediction Rate | Graph | `rate(ml_predictions_total[5m])` |

📸 Screenshot setiap panel → simpan ke `5.bukti monitoring Grafana/` (nama sama seperti Prometheus)

### 5h. Setup 3 Alerting Rules
Di Grafana: Alerting → Alert rules → New alert rule:

**Alert 1: High Error Rate**
- Query: `rate(ml_error_total[5m]) > 0.1`
- Contact point: email/Discord
📸 Screenshot rule → `1.rules_high_error_rate.png`
📸 Screenshot notifikasi → `2.notifikasi_high_error_rate.png`

**Alert 2: High Latency**
- Query: `histogram_quantile(0.95, rate(ml_request_latency_seconds_bucket[5m])) > 1`
📸 `3.rules_high_latency.png` + `4.notifikasi_high_latency.png`

**Alert 3: Model Down**
- Query: `ml_model_loaded == 0`
📸 `5.rules_model_down.png` + `6.notifikasi_model_down.png`

Simpan semua ke `6.bukti alerting Grafana/`

---

## STEP 6: Susun & Submit

Pastikan `Membangun_model/` punya:
- `screenshoot_dashboard.jpg`
- `screenshoot_artifak.jpg`

Pastikan `Monitoring dan Logging/` punya semua screenshot.

```bash
# ZIP folder submission
zip -r SMSML_Kendrick-Filbert.zip SMSML_Kendrick-Filbert/
```

Upload ke Dicoding! 🎉

---

## ✅ FINAL CHECKLIST
- [ ] Notebook semua cell dirun tanpa error
- [ ] GitHub repo Eksperimen → PUBLIC, Actions ✅
- [ ] DagsHub repo Membangun_model ada artefak
- [ ] MLflow screenshots (dashboard + artifak)
- [ ] GitHub repo Workflow-CI → PUBLIC, Actions ✅
- [ ] Docker image di hub.docker.com/r/kendrickfff/breast-cancer-ml
- [ ] Inference API running + bukti serving
- [ ] Prometheus 10 metriks screenshots
- [ ] Grafana dashboard `kendrickfff` + 10 panels screenshots
- [ ] 3 alerting rules + 3 notifikasi screenshots
