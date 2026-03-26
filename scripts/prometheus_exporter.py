from prometheus_client import start_http_server, Gauge
import time
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load model + scaler
model = load_model("model/lstm_model.keras")
scaler = joblib.load("model/scaler.save")

SEQ_LENGTH = 30
PRED_STEPS = 10

# Example metric (for 1 feature — extend for all)
pred_cpu = Gauge('predicted_cpu_usage', 'Predicted CPU Usage')

def predict_next(data_window):
    data_scaled = scaler.transform(data_window)
    X = np.expand_dims(data_scaled, axis=0)
    pred = model.predict(X, verbose=0)
    pred = pred.reshape(PRED_STEPS, data_window.shape[1])
    return scaler.inverse_transform(pred)

def get_latest_data():
    """
    Replace this with Prometheus API call
    or your buffer storage
    """
    return np.random.rand(SEQ_LENGTH, 10)  # dummy data

if __name__ == "__main__":
    # Start metrics server
    start_http_server(8000)
    print("Exporter running on port 8000...")

    while True:
        data_window = get_latest_data()
        predictions = predict_next(data_window)

        # Push only next step (important for time series)
        next_step = predictions[0]

        # Example: first metric = CPU
        pred_cpu.set(next_step[0])

        time.sleep(5)  # match scrape interval