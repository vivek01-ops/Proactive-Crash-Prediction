import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

# =========================
# CONFIG
# =========================
SEQ_LENGTH = 30      # past 30 timesteps
PRED_STEPS = 10      # predict next 10
MODEL_PATH = "lstm_model.h5"

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("data/train_data.csv")

# Drop timestamp if present
if 'timestamp' in df.columns:
    df = df.drop(columns=['timestamp'])

data = df.values

# =========================
# NORMALIZATION
# =========================
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# =========================
# CREATE SEQUENCES
# =========================
def create_sequences(data, seq_len, pred_steps):
    X, y = [], []
    for i in range(len(data) - seq_len - pred_steps):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len:i+seq_len+pred_steps])
    return np.array(X), np.array(y)

X, y = create_sequences(data_scaled, SEQ_LENGTH, PRED_STEPS)

print("X shape:", X.shape)
print("y shape:", y.shape)

# =========================
# MODEL
# =========================
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(SEQ_LENGTH, X.shape[2])),
    LSTM(32),
    Dense(PRED_STEPS * X.shape[2])
])

model.compile(optimizer='adam', loss='mse')

# =========================
# TRAIN
# =========================
early_stop = EarlyStopping(patience=5, restore_best_weights=True)

history = model.fit(
    X, y.reshape(y.shape[0], -1),
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    callbacks=[early_stop]
)

# =========================
# SAVE MODEL + SCALER
# =========================
model.save(MODEL_PATH)

import joblib
joblib.dump(scaler, "scaler.save")

print("Model trained and saved.")