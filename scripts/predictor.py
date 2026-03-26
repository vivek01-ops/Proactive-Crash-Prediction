import requests
import numpy as np
import time
import joblib
from tensorflow.keras.models import load_model
from prometheus_client import start_http_server, Gauge

# ======================
# CONFIG
# ======================
PROM_URL = "http://192.168.59.106:30162/api/v1/query_range"

# ⚠️ MUST match training CSV columns EXACTLY (order matters)
METRICS = [
    "node_cpu_seconds_total",
    "node_memory_MemAvailable_bytes",
    "node_disk_io_time_seconds_total"
    # add ALL features used in training
]

SEQ_LENGTH = 30
PRED_STEPS = 10

# ======================
# LOAD MODEL
# ======================
model = load_model("lstm_model.h5", compile=False)
scaler = joblib.load("scaler.save")

# ======================
# PROMETHEUS METRICS (export)
# ======================
gauges = {}
for metric in METRICS:
    gauges[metric] = Gauge(f"predicted_{metric}", f"Predicted {metric}")

# ======================
# FETCH DATA
# ======================
def fetch_data():
    all_data = []

    for metric in METRICS:
        params = {
            "query": metric,
            "start": "now-5m",
            "end": "now",
            "step": "10s"
        }

        res = requests.get(PROM_URL, params=params).json()

        if len(res['data']['result']) == 0:
            raise Exception(f"No data for {metric}")

        values = res['data']['result'][0]['values']
        series = [float(v[1]) for v in values]

        # ensure enough data
        if len(series) < SEQ_LENGTH:
            raise Exception(f"Not enough data for {metric}")

        all_data.append(series[-SEQ_LENGTH:])

    # shape → (30, num_features)
    data = np.array(all_data).T

    return data

# ======================
# PREDICT
# ======================
def predict_next(data_window):
    data_scaled = scaler.transform(data_window)

    X = np.expand_dims(data_scaled, axis=0)

    pred = model.predict(X, verbose=0)

    pred = pred.reshape(PRED_STEPS, data_window.shape[1])

    return scaler.inverse_transform(pred)

# ======================
# RUN
# ======================
if __name__ == "__main__":
    start_http_server(8000)
    print("Prediction service running on port 8000...")

    while True:
        try:
            data_window = fetch_data()
            pred = predict_next(data_window)

            next_step = pred[0]  # only next timestep

            # push all metrics
            for i, metric in enumerate(METRICS):
                gauges[metric].set(next_step[i])

            print("Predicted:", next_step)

        except Exception as e:
            print("Error:", e)

        time.sleep(10)