# Time-Series-Prediksi-Suhu-Harian

Project ini memprediksi suhu per jam untuk beberapa hari ke depan menggunakan model LSTM (TensorFlow/Keras), berdasarkan data historis suhu dalam format CSV. Alur kerjanya meliputi membaca data, normalisasi dan sliding window, pelatihan model LSTM, prediksi suhu secara iteratif (recursive forecasting), hingga menampilkan ringkasan suhu pagi, siang, dan malam untuk tiap hari yang diprediksi.

## Cara Menjalankan

```bash
pip install -r requirements.txt
python main.py
```

Program akan meminta input nama file CSV (harus ada kolom `timestamp` dan `suhu_celsius`) dan jumlah hari yang ingin diprediksi.