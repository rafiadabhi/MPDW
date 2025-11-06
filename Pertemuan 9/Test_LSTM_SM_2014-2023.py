# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras

# Atur environment untuk menekan warning TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- 1. PENGAMBILAN DAN INSPEKSI DATA BARU ---

# URL data kelembaban tanah mingguan
data_url = "https://github.com/rafiadabhi/MPDW/raw/main/Pertemuan%201/Data/SM_2019-2023.xlsx"

# Membaca data dari file Excel
# Catatan: Data yang Anda berikan (2019-2023) akan digunakan, bukan 2014-2023
data = pd.read_excel(data_url)

# Ganti nama kolom untuk konsistensi: 'SM' adalah Kelembaban Tanah
data = data.rename(columns={'SM': 'soil_moisture'})

print("--- Data Info ---")
print(data.head())
print(data.info())
print(data.describe())
print(f"Total data points: {len(data)}")


# --- 2. PREPARASI DATA UNTUK LSTM ---

# Hanya gunakan kolom 'soil_moisture' (variabel target)
target_data = data.filter(["soil_moisture"])
dataset = target_data.values # Konversi ke numpy array

# Split data: 95% Train, 5% Test
training_data_len = int(np.ceil(len(dataset) * 0.95))

# Scaling data menggunakan MinMaxScaler (karena rentang 0-1 lebih umum untuk data kelembaban/persentase)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

# Ambil data training yang sudah di-scale
training_data = scaled_data[:training_data_len]

# Tentukan ukuran jendela geser (window size)
# Kita gunakan 52 minggu (1 tahun) agar model bisa mempelajari musiman tahunan
WINDOW_SIZE = 52

X_train, y_train = [], []

# Membuat jendela geser
for i in range(WINDOW_SIZE, len(training_data)):
    # X_train: data 52 minggu sebelumnya
    X_train.append(training_data[i - WINDOW_SIZE:i, 0])
    # y_train: nilai kelembaban tanah di minggu ke-53
    y_train.append(training_data[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)

# Reshape data untuk LSTM [samples, timesteps, features]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


# --- 3. PEMBANGUNAN DAN PELATIHAN MODEL GRU (Alternatif LSTM) ---
# Menggunakan GRU sebagai alternatif LSTM yang lebih ringan dan cepat

model = keras.models.Sequential()

# Layer GRU Pertama
# Return sequences=True agar output diteruskan ke layer berikutnya
model.add(keras.layers.GRU(100, return_sequences=True, input_shape=(X_train.shape[1], 1)))

# Layer GRU Kedua (Output layer hanya 1 unit)
model.add(keras.layers.GRU(100, return_sequences=False))

# Layer Dropout untuk mencegah overfitting
model.add(keras.layers.Dropout(0.3))

# Final Output Layer
model.add(keras.layers.Dense(1))

print("\n--- Model Summary ---")
model.summary()

# Compile model
model.compile(optimizer="adam",
              loss="mse", # Menggunakan MSE, umum untuk regresi time series
              metrics=[keras.metrics.MeanAbsolutePercentageError()])

# Pelatihan model (Epochs diturunkan karena data lebih sedikit, batch size tetap)
# Anda bisa coba menaikkan Epochs jika ingin eksplorasi lebih lanjut
print("\n--- Model Training ---")
training = model.fit(X_train, y_train, epochs=10, batch_size=32)


# --- 4. PREDIKSI DATA UJI ---

# Persiapan data uji
# Data uji harus mencakup WINDOW_SIZE (52) langkah waktu sebelumnya
test_data = scaled_data[training_data_len - WINDOW_SIZE:]
X_test = []

for i in range(WINDOW_SIZE, len(test_data)):
    X_test.append(test_data[i - WINDOW_SIZE:i, 0])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Lakukan Prediksi
predictions = model.predict(X_test)
# Inverse transform agar prediksi kembali ke skala kelembaban tanah asli
predictions = scaler.inverse_transform(predictions)


# --- 5. EVALUASI DAN VISUALISASI HASIL ---

# Ambil data aktual untuk perbandingan
test_actual = dataset[training_data_len:]

# Hitung MAPE Test (secara manual atau dari metrics)
mape_test = np.mean(np.abs(predictions - test_actual) / test_actual) * 100
print(f"\nMean Absolute Percentage Error (MAPE) Test: {mape_test[0]:.2f}%")

# Plotting data
train = data[:training_data_len]
test = data[training_data_len:]

# Pastikan data test memiliki kolom 'Predictions'
test = test.copy()
test['Predictions'] = predictions

plt.figure(figsize=(14, 7))
plt.plot(train['Date'], train['soil_moisture'], label="Train (Aktual)", color='blue')
plt.plot(test['Date'], test['soil_moisture'], label="Test (Aktual)", color='orange')
plt.plot(test['Date'], test['Predictions'], label="Predictions (GRU)", color='red', linestyle='--')
plt.title(f"Prediksi Kelembaban Tanah (Soil Moisture) Indramayu - MAPE Test: {mape_test[0]:.2f}%")
plt.xlabel("Tanggal")
plt.ylabel("Kelembaban Tanah (SM)")
plt.legend()
plt.grid(True)
plt.show()

# Jika Anda ingin membandingkan dengan MAPE SARIMA (8.55%)
if mape_test[0] < 8.55:
    print(f"Hasil GRU ({mape_test[0]:.2f}%) lebih baik dari SARIMA (8.55%).")
else:
    print(f"Hasil SARIMA (8.55%) lebih baik, perlu tuning lebih lanjut pada model GRU.")