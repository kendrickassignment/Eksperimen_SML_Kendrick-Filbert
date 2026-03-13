"""
Inference API - Kendrick Filbert
==================================
Flask API untuk serving model + Prometheus metrics (12 metriks).

Endpoints:
- POST /predict  : Prediksi
- GET  /health   : Health check
- GET  /metrics  : Prometheus metrics
"""

import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
import time
import os
import warnings
warnings.filterwarnings('ignore')

from prometheus_client import (
    Counter, Histogram, Gauge, Summary,
    generate_latest, CONTENT_TYPE_LATEST, REGISTRY
)

app = Flask(__name__)

# ============================================================
# 12 Prometheus Metrics
# ============================================================
REQUEST_COUNT = Counter('ml_request_total', 'Total prediction requests', ['method', 'endpoint', 'status'])
REQUEST_LATENCY = Histogram('ml_request_latency_seconds', 'Request latency', ['endpoint'],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0])
PREDICTION_CLASS = Counter('ml_prediction_class_total', 'Predictions per class', ['predicted_class'])
INFERENCE_TIME = Histogram('ml_inference_duration_seconds', 'Inference duration',
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25])
PREDICTION_CONFIDENCE = Histogram('ml_prediction_confidence', 'Prediction confidence',
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0])
ACTIVE_REQUESTS = Gauge('ml_active_requests', 'Active requests being processed')
ERROR_COUNT = Counter('ml_error_total', 'Total errors', ['error_type'])
INPUT_FEATURE_MEAN = Gauge('ml_input_feature_mean', 'Mean input features', ['feature_group'])
PREDICTION_SUMMARY = Summary('ml_prediction_probability_summary', 'Prediction probability summary')
MODEL_LOADED = Gauge('ml_model_loaded', 'Model loaded status (1=yes, 0=no)')
TOTAL_PREDICTIONS = Counter('ml_predictions_total', 'Total successful predictions')
REQUEST_SIZE = Histogram('ml_request_feature_count', 'Feature count per request',
    buckets=[5, 10, 15, 20, 25, 30, 35])

# ============================================================
# Model Loading
# ============================================================
model = None

def load_model():
    global model
    try:
        # Fallback: train a model on the spot
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.datasets import load_breast_cancer
        from sklearn.preprocessing import StandardScaler

        cancer = load_breast_cancer()
        X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
        y = cancer.target
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
        model.fit(X_scaled, y)
        MODEL_LOADED.set(1)
        print("[INFO] Model loaded successfully!")
    except Exception as e:
        MODEL_LOADED.set(0)
        print(f"[ERROR] Failed to load model: {e}")
        raise


@app.route('/health', methods=['GET'])
def health():
    REQUEST_COUNT.labels(method='GET', endpoint='/health', status='200').inc()
    return jsonify({"status": "healthy", "model_loaded": model is not None, "author": "Kendrick Filbert"})


@app.route('/predict', methods=['POST'])
def predict():
    ACTIVE_REQUESTS.inc()
    start_time = time.time()
    try:
        data = request.get_json(force=True)
        if 'features' not in data:
            ERROR_COUNT.labels(error_type='invalid_input').inc()
            REQUEST_COUNT.labels(method='POST', endpoint='/predict', status='400').inc()
            ACTIVE_REQUESTS.dec()
            return jsonify({"error": "Missing 'features'"}), 400

        features = np.array(data['features'])
        if features.ndim == 1:
            features = features.reshape(1, -1)

        REQUEST_SIZE.observe(features.shape[1])
        INPUT_FEATURE_MEAN.labels(feature_group='mean_features').set(float(np.mean(features[:, :10])))
        INPUT_FEATURE_MEAN.labels(feature_group='se_features').set(float(np.mean(features[:, 10:20])))
        INPUT_FEATURE_MEAN.labels(feature_group='worst_features').set(float(np.mean(features[:, 20:30])))

        inference_start = time.time()
        predictions = model.predict(features)
        probabilities = model.predict_proba(features)
        inference_duration = time.time() - inference_start
        INFERENCE_TIME.observe(inference_duration)

        for pred in predictions:
            PREDICTION_CLASS.labels(predicted_class='benign' if pred == 1 else 'malignant').inc()
        for prob in probabilities:
            max_prob = float(max(prob))
            PREDICTION_CONFIDENCE.observe(max_prob)
            PREDICTION_SUMMARY.observe(max_prob)

        TOTAL_PREDICTIONS.inc()

        results = []
        for i in range(len(predictions)):
            results.append({
                "prediction": int(predictions[i]),
                "class": "Benign" if predictions[i] == 1 else "Malignant",
                "probability_malignant": float(probabilities[i][0]),
                "probability_benign": float(probabilities[i][1]),
                "confidence": float(max(probabilities[i]))
            })

        total_time = time.time() - start_time
        REQUEST_LATENCY.labels(endpoint='/predict').observe(total_time)
        REQUEST_COUNT.labels(method='POST', endpoint='/predict', status='200').inc()
        ACTIVE_REQUESTS.dec()

        return jsonify({"predictions": results, "inference_time_ms": round(inference_duration * 1000, 2)})

    except Exception as e:
        ERROR_COUNT.labels(error_type='prediction_error').inc()
        REQUEST_COUNT.labels(method='POST', endpoint='/predict', status='500').inc()
        ACTIVE_REQUESTS.dec()
        return jsonify({"error": str(e)}), 500


@app.route('/metrics', methods=['GET'])
def metrics():
    return generate_latest(REGISTRY), 200, {'Content-Type': CONTENT_TYPE_LATEST}


if __name__ == '__main__':
    print("=" * 50)
    print("BREAST CANCER CLASSIFICATION API")
    print("Author: Kendrick Filbert")
    print("=" * 50)
    load_model()
    print("\nPOST /predict  |  GET /health  |  GET /metrics")
    print("Starting on http://0.0.0.0:5001")
    app.run(host='0.0.0.0', port=5001, debug=False)
