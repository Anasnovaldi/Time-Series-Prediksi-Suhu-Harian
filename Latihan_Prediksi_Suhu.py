# ============================================
# TIME SERIES FORECASTING SUHU DENGAN TENSORFLOW (LSTM)
# ============================================
# Alur besarnya:
# 1. Baca data CSV
# 2. Siapkan data (scaling + windowing)
# 3. Bangun model LSTM
# 4. Latih model
# 5. Prediksi N hari ke depan
# 6. Tampilkan ringkasan suhu pagi, siang, malam per hari

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

def main(namafile, jumlah_hari):

    # Biar hasil training selalu sama tiap dijalankan ulang (reproducible)
    np.random.seed(42)
    tf.random.set_seed(42)

    # ============================================
    # LANGKAH 1: BACA DATA DARI CSV
    # ============================================
    df = pd.read_csv(f'dataset/{namafile}')
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    print("5 baris pertama data suhu:")
    print(df.head())
    print(f"\nTotal data: {len(df)} baris (jam)")

    suhu = df['suhu_celsius'].values
    timestamp = df['timestamp'].values

    # ============================================
    # LANGKAH 2: VISUALISASI DATA ASLI (SEBELUM DIPREDIKSI)
    # ============================================
    plt.figure(figsize=(12, 5))
    plt.plot(timestamp, suhu, color='steelblue', label='Suhu Aktual (Data Historis)')
    plt.title('Data Suhu per Jam (Sebelum Prediksi)')
    plt.xlabel('Waktu')
    plt.ylabel('Suhu (°C)')
    plt.xticks(rotation=30)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('dataset/grafik_1_sebelum_prediksi.png', dpi=120)
    plt.show()
    print("\nGrafik data asli tersimpan.")

    # ============================================
    # LANGKAH 3: SIAPKAN DATA UNTUK LSTM
    # ============================================
    scaler = MinMaxScaler(feature_range=(0, 1))
    suhu_scaled = scaler.fit_transform(suhu.reshape(-1, 1))

    window = 24
    def buat_window(data, window):
        X, y = [], []
        for i in range(len(data) - window):
            X.append(data[i:i + window, 0])
            y.append(data[i + window, 0])
        return np.array(X), np.array(y)
    X_train, y_train = buat_window(suhu_scaled, window)

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    print(f"\nJumlah paket data untuk training: {X_train.shape[0]}")
    print(f"Tiap paket berisi {X_train.shape[1]} jam data sebagai input")

    # ============================================
    # LANGKAH 4: BANGUN & LATIH MODEL LSTM
    # ============================================
    model = Sequential([
        LSTM(32, activation='tanh', input_shape=(window, 1)),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    print("\nMulai training...")
    history = model.fit(X_train, y_train, epochs=50, batch_size=8, verbose=0)
    print(f"Training selesai. Loss akhir: {history.history['loss'][-1]:.5f}  (makin kecil makin bagus)")

    # ============================================
    # LANGKAH 5: PREDIKSI N HARI KE DEPAN
    # ============================================
    jumlah_jam_prediksi = jumlah_hari * 24
    window_berjalan = suhu_scaled[-window:].reshape(1, window, 1).copy()
    hasil_prediksi_scaled = []
    for _ in range(jumlah_jam_prediksi):
        prediksi_1_jam = model.predict(window_berjalan, verbose=0)
        hasil_prediksi_scaled.append(prediksi_1_jam[0, 0])
        window_berjalan = np.append(window_berjalan[:, 1:, :], [[prediksi_1_jam[0]]], axis=1)

    hasil_prediksi = scaler.inverse_transform(
        np.array(hasil_prediksi_scaled).reshape(-1, 1)
    ).flatten()

    # Buat deretan waktu untuk N hari ke depan, lanjutan dari data terakhir
    timestamp_prediksi = pd.date_range(
        start=pd.Timestamp(timestamp[-1]) + pd.Timedelta(hours=1),
        periods=jumlah_jam_prediksi,
        freq='h'
    )

    # Simpan hasil prediksi ke CSV baru (nama file mengikuti jumlah hari)
    df_prediksi = pd.DataFrame({
        'timestamp': timestamp_prediksi,
        'suhu_prediksi_celsius': np.round(hasil_prediksi, 2)
    })
    df_prediksi.to_csv(f'dataset/hasil_prediksi_{jumlah_hari}_hari.csv', index=False)
    print(f"\nHasil prediksi tersimpan ke CSV (hasil_prediksi_{jumlah_hari}_hari.csv).")

    # ============================================
    # VISUALISASI AKHIR: DATA ASLI + HASIL PREDIKSI
    # ============================================
    plt.figure(figsize=(13, 5))
    plt.plot(timestamp, suhu, color='steelblue', label='Suhu Aktual')
    plt.plot(timestamp_prediksi, hasil_prediksi, color='crimson', linestyle='--',
              label=f'Prediksi LSTM ({jumlah_hari} Hari ke Depan)')
    plt.axvline(x=pd.Timestamp(timestamp[-1]), color='gray', linestyle=':', label='Batas Data Aktual vs Prediksi')
    plt.title(f'Prediksi Suhu {jumlah_hari} Hari ke Depan Menggunakan TensorFlow LSTM')
    plt.xlabel('Waktu')
    plt.ylabel('Suhu (°C)')
    plt.xticks(rotation=30)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('dataset/grafik_2_sesudah_prediksi.png', dpi=120)
    plt.show()

    # ============================================
    # LANGKAH 6: RINGKASAN SUHU PAGI, SIANG, MALAM PER HARI
    # ============================================
    # Definisi rentang jam untuk tiap waktu (bisa disesuaikan selera)
    jam_pagi  = range(5, 11)    # 05:00 - 10:00
    jam_siang = range(11, 16)   # 11:00 - 15:00
    jam_malam = range(18, 24)   # 18:00 - 23:00

    df_prediksi['tanggal'] = df_prediksi['timestamp'].dt.date
    df_prediksi['jam']     = df_prediksi['timestamp'].dt.hour

    ordinal_id = ["pertama", "kedua", "ketiga", "keempat", "kelima",
                  "keenam", "ketujuh", "kedelapan", "kesembilan", "kesepuluh"]

    print(f"\n{"="*100}")
    print(f"{"==================== RINGKASAN PREDIKSI SUHU PER HARI ====================":^100}")
    print("="*100)

    daftar_tanggal = sorted(df_prediksi['tanggal'].unique())
    for i, tgl in enumerate(daftar_tanggal):
        data_hari_ini = df_prediksi[df_prediksi['tanggal'] == tgl]

        suhu_pagi  = data_hari_ini[data_hari_ini['jam'].isin(jam_pagi)]['suhu_prediksi_celsius'].mean()
        suhu_siang = data_hari_ini[data_hari_ini['jam'].isin(jam_siang)]['suhu_prediksi_celsius'].mean()
        suhu_malam = data_hari_ini[data_hari_ini['jam'].isin(jam_malam)]['suhu_prediksi_celsius'].mean()

        label_hari = f"hari {ordinal_id[i]}" if i < len(ordinal_id) else f"hari ke-{i + 1}"

        print(
            f"suhu {label_hari:<12}: "
            f"{suhu_pagi:.0f}C saat pagi hari, "
            f"{suhu_siang:.0f}C saat siang hari dan "
            f"{suhu_malam:.0f}C saat malam hari"
        )

    print(f"\nSELESAI. Cek folder dataset untuk CSV prediksi & 2 grafik PNG.")


if __name__ == "__main__":
    os.system('cls' if os.name == 'nt' else 'clear')
    print('=' * 100)
    print(f'{"============================== Prediksi Suhu ==============================":^100}')
    print('=' * 100)
    namafile    =     input('Masukkan Nama File Format CSV         : ')
    try:
        jumlah_hari = int(input('Masukkan Jumlah Hari Untuk Diprediksi : '))
        main(namafile, jumlah_hari)
    except ValueError:
        print('Input Hari Berupa Angka Seperti (1/2/3/4).....!!')