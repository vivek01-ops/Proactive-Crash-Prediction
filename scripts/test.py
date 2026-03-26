import numpy as np
import joblib
from tensorflow.keras.models import load_model

SEQ_LENGTH = 30
PRED_STEPS = 10

# Load model + scaler
model = load_model("lstm_model.h5")
scaler = joblib.load("scaler.save")

def predict_next(data_window):
    """
    data_window: last 30 timesteps from Prometheus
    shape = (30, num_features)
    """

    data_scaled = scaler.transform(data_window)

    X = np.expand_dims(data_scaled, axis=0)

    pred = model.predict(X)

    pred = pred.reshape(PRED_STEPS, data_window.shape[1])

    # inverse scaling
    pred_actual = scaler.inverse_transform(pred)

    return pred_actual