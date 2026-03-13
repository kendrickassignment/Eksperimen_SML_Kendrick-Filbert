"""
Prometheus Exporter Helper - Kendrick Filbert
Generates test traffic to populate metrics for Prometheus/Grafana monitoring.

12 Metriks: ml_request_total, ml_request_latency_seconds, ml_prediction_class_total,
ml_inference_duration_seconds, ml_prediction_confidence, ml_active_requests,
ml_error_total, ml_input_feature_mean, ml_prediction_probability_summary,
ml_model_loaded, ml_predictions_total, ml_request_feature_count
"""

import time
import requests
import numpy as np
import sys


def check_health(api_url="http://localhost:5001"):
    try:
        r = requests.get(f"{api_url}/health", timeout=5)
        if r.status_code == 200:
            print(f"[OK] API healthy: {r.json()}")
            return True
    except Exception as e:
        print(f"[WARN] API unreachable: {e}")
    return False


def send_prediction(api_url="http://localhost:5001"):
    features = np.random.randn(30).tolist()
    try:
        r = requests.post(f"{api_url}/predict", json={"features": features}, timeout=10)
        if r.status_code == 200:
            result = r.json()
            pred = result['predictions'][0]
            print(f"[OK] {pred['class']} (confidence: {pred['confidence']:.4f})")
            return True
    except Exception as e:
        print(f"[WARN] Prediction failed: {e}")
    return False


if __name__ == "__main__":
    api_url = "http://localhost:5001"
    print("=" * 50)
    print("PROMETHEUS METRICS EXPORTER")
    print("Author: Kendrick Filbert")
    print("=" * 50)
    print(f"Prometheus scrapes: {api_url}/metrics")

    if not check_health(api_url):
        print("[ERROR] Start inference.py first: python 7.inference.py")
        sys.exit(1)

    print("\n[INFO] Sending test predictions...")
    for i in range(5):
        send_prediction(api_url)
        time.sleep(0.5)

    print(f"\n[INFO] Metrics ready at {api_url}/metrics")
    print("[INFO] Running periodic test requests (Ctrl+C to stop)...")
    try:
        while True:
            send_prediction(api_url)
            time.sleep(10)
    except KeyboardInterrupt:
        print("\n[INFO] Stopped.")
