import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# -----------------------------
# Load dataset
# -----------------------------
data = pd.read_csv("household_power_consumption.txt", sep=";", low_memory=False)

# Use "Global_active_power" column
data['Global_active_power'] = pd.to_numeric(data['Global_active_power'], errors='coerce')
data = data.dropna(subset=['Global_active_power'])
values = data['Global_active_power'].values.reshape(-1, 1)

# -----------------------------
# Normalize data
# -----------------------------
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(values)

# -----------------------------
# Prepare sequences
# -----------------------------
def create_sequences(dataset, time_step=50):
    X, y = [], []
    for i in range(len(dataset) - time_step - 1):
        X.append(dataset[i:(i + time_step), 0])
        y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(y)

time_step = 50
X, y = create_sequences(scaled_data, time_step)

X = X.reshape(X.shape[0], X.shape[1], 1)

# Split into train/test
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# -----------------------------
# Build LSTM model
# -----------------------------
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(time_step, 1)),
    LSTM(50),
    Dense(1)
])

model.compile(optimizer="adam", loss="mean_squared_error")
model.fit(X_train, y_train, epochs=5, batch_size=64, verbose=1)

# -----------------------------
# Predictions
# -----------------------------
predictions = model.predict(X_test)

# Inverse transform
y_test_inverse = scaler.inverse_transform(y_test.reshape(-1, 1))
predictions_inverse = scaler.inverse_transform(predictions)

# -----------------------------
# Save Results to CSV
# -----------------------------
results_df = pd.DataFrame({
    "Actual": y_test_inverse.flatten(),
    "Predicted": predictions_inverse.flatten()
})
results_df.to_csv("results.csv", index=False)
print(" Results saved to results.csv")

# -----------------------------
# Plot Results
# -----------------------------
plt.figure(figsize=(12, 6))
plt.plot(y_test_inverse, label="Actual")
plt.plot(predictions_inverse, label="Predicted")
plt.xlabel("Time Step")
plt.ylabel("Global Active Power")
plt.title("Energy Consumption Prediction (LSTM)")
plt.legend()

# Save plot
plt.savefig("prediction_plot.png")
plt.close()
print("Prediction plot saved as prediction_plot.png")
