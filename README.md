# 📊 Klasifikasi Status Minum Obat Pasien dengan Algoritma KNN

Project ini menggunakan algoritma **K-Nearest Neighbors (KNN)** untuk memprediksi apakah pasien sudah minum obat (**Taken**) atau belum (**Not Taken**) berdasarkan data medis dan lingkungan yang dikumpulkan dari sistem Smart Medication Dispenser.

## 🧠 Algoritma
K-Nearest Neighbors (KNN) bekerja dengan membandingkan data baru terhadap data latih, lalu mengambil **k tetangga terdekat** untuk menentukan prediksi kelas berdasarkan mayoritas suara.

## 🗂️ Dataset
Dataset berisi:
- Umur pasien
- Jenis kelamin
- Kondisi medis
- Nama dan bentuk obat
- Dosis yang diresepkan
- Frekuensi konsumsi obat
- Suhu dan kelembaban sekitar
- Jarak pasien ke alat
- Status target: `Taken` / `Not Taken`

## ⚙️ Alur Program
1. Load dataset dari file CSV
2. Preprocessing: encoding & normalisasi
3. Penanganan data tidak seimbang dengan **SMOTE**
4. Pemilihan nilai **k terbaik**
5. Pelatihan dan pengujian model KNN
6. Evaluasi model (akurasi, confusion matrix, classification report)

## 🔎 Hasil
- Nilai k terbaik: 1
- Akurasi: 65%
- Model berhasil memprediksi status kepatuhan pasien berdasarkan fitur yang tersedia

## 💡 Tujuan
Membantu sistem monitoring pengobatan pasien secara otomatis, terutama dalam mengevaluasi apakah pasien sudah mengambil obat sesuai jadwal atau belum, berdasarkan pola data yang ada.

## 🛠️ Library yang Digunakan
- pandas
- numpy
- scikit-learn
- imbalanced-learn (SMOTE)

## 📁 File
- `KNN-classification.py` → kode utama
- `dataset_medicion.csv` → dataset yang digunakan

