# Laporan Proyek Machine Learning - Bayu Ardiyansyah

![License](https://img.shields.io/badge/License-MIT-yellow)

## Domain Proyek

**Prediksi Risiko Kredit untuk Perusahaan Multifinance**

Dalam industri keuangan, khususnya perusahaan pembiayaan (multifinance), kemampuan untuk mengidentifikasi peminjam berisiko tinggi sangat penting untuk meminimalkan tingkat gagal bayar (*default*) sambil menjaga profitabilitas. Tingkat gagal bayar yang tinggi dapat menyebabkan kerugian finansial signifikan, sedangkan penolakan pinjaman yang berlebihan dapat mengurangi peluang pendapatan. Oleh karena itu, pengembangan model *machine learning* untuk memprediksi risiko kredit menjadi solusi strategis untuk mengoptimalkan keputusan pemberian pinjaman.

Proyek ini bertujuan untuk membangun model prediktif yang dapat mengklasifikasikan risiko kredit peminjam sebagai **GOOD** (rendah risiko) atau **BAD** (tinggi risiko) berdasarkan data pinjaman historis dari tahun 2007–2014. Model ini akan membantu perusahaan multifinance mengurangi risiko gagal bayar dengan mengidentifikasi peminjam berisiko tinggi secara akurat. Selain itu, proyek ini mencakup pengembangan antarmuka grafis (GUI) modern menggunakan `customtkinter` untuk memudahkan petugas pinjaman dalam menggunakan model prediktif.

**Mengapa Masalah Ini Penting?**  
- **Dampak Finansial**: Menurut studi oleh McKinsey, pengelolaan risiko kredit yang efektif dapat mengurangi kerugian akibat gagal bayar hingga 20–30%.  
- **Efisiensi Operasional**: Model prediktif mengotomatiskan proses penilaian risiko, mengurangi waktu dan biaya evaluasi manual.  
- **Kepuasan Pelanggan**: Dengan menyeimbangkan penerimaan dan penolakan pinjaman, perusahaan dapat mempertahankan pelanggan potensial sambil mengelola risiko.

## Business Understanding

### Problem Statements
1. **Tingkat Gagal Bayar yang Tinggi**: Perusahaan multifinance menghadapi risiko kerugian akibat peminjam yang gagal membayar pinjaman, karena kurangnya alat prediktif yang akurat untuk menilai risiko kredit.  
2. **Proses Penilaian Manual yang Tidak Efisien**: Penilaian risiko kredit secara manual memakan waktu dan rentan terhadap kesalahan manusia, menghambat skalabilitas operasional.  
3. **Kurangnya Antarmuka Pengguna yang Intuitif**: Petugas pinjaman membutuhkan alat yang mudah digunakan untuk menerapkan model prediktif dalam keputusan sehari-hari.

### Goals
1. **Mengembangkan Model Prediktif yang Akurat**: Membangun model *machine learning* yang memprediksi risiko kredit (GOOD atau BAD) dengan *recall* tinggi untuk mengidentifikasi peminjam berisiko tinggi, sehingga meminimalkan gagal bayar.  
2. **Mengotomatiskan Penilaian Risiko**: Mengintegrasikan model ke dalam alur kerja perusahaan untuk mempercepat dan meningkatkan akurasi pengambilan keputusan.  
3. **Menyediakan Antarmuka Pengguna yang Modern**: Mengembangkan GUI *user-friendly* untuk memungkinkan petugas pinjaman memasukkan data peminjam, memilih model, dan melihat hasil prediksi dengan mudah.

### Solution Statements
- **Menggunakan Berbagai Algoritma Machine Learning**: Mengimplementasikan empat algoritma (Logistic Regression, Random Forest, XGBoost, SVM) untuk memprediksi risiko kredit, dengan evaluasi berdasarkan akurasi, *precision*, *recall*, dan ROC-AUC.  
- **Hyperparameter Tuning**: Melakukan optimasi *hyperparameter* menggunakan `GridSearchCV` untuk meningkatkan performa, terutama *recall* untuk kelas BAD.  
- **Pemilihan Model Terbaik**: Memilih model dengan *recall* tertinggi untuk kelas BAD dan ROC-AUC kompetitif untuk keseimbangan deteksi risiko dan performa keseluruhan.  
- **Feature Engineering**: Menambahkan fitur baru seperti rasio keuangan dan ekstraksi waktu untuk meningkatkan kemampuan prediktif model.  
- **Pengembangan GUI**: Membangun antarmuka grafis menggunakan `customtkinter` yang mendukung input data peminjam, pemilihan model, dan tampilan hasil prediksi dengan probabilitas risiko.

## Data Understanding

**Informasi Dataset**  
Dataset yang digunakan adalah `loan_data_2007_2014.csv`, berisi data pinjaman historis dari tahun 2007–2014. Dataset tersedia di [Github: Lending Club Loan Data](https://github.com/RazerArdi/Credit-Risk-Prediction-Model#).  
- **Jumlah Baris**: 466,285  
- **Jumlah Kolom**: 75  
- **Variabel Target**: `credit_risk` (binary: 0 untuk GOOD, 1 untuk BAD), diturunkan dari `loan_status`:  
  - GOOD: `loan_status` = "Fully Paid"  
  - BAD: `loan_status` = "Charged Off" atau "Default"  
  - OTHER: Status ambigu (contoh: "Current", "Late") dihapus, menyisakan 228,046 baris (GOOD: 184,739, BAD: 43,307).

### Variabel pada Dataset
Variabel utama yang digunakan:  
- `loan_amnt`: Jumlah pinjaman (numerik, USD).  
- `int_rate`: Suku bunga (numerik, %).  
- `annual_inc`: Pendapatan tahunan (numerik, USD).  
- `dti`: Rasio utang terhadap pendapatan (numerik, %).  
- `grade`: Tingkat kredit (kategorikal, A–G).  
- `home_ownership`: Status kepemilikan rumah (kategorikal, contoh: RENT, OWN).  
- `issue_d`: Tanggal penerbitan pinjaman (teks, format: MMM-YY).  
- Fitur turunan (dari *feature engineering*):  
  - `loan_to_income_ratio`: Rasio pinjaman terhadap pendapatan tahunan.  
  - `issue_year`, `issue_month`: Tahun dan bulan dari `issue_d`.  
  - `log_annual_inc`: Transformasi log dari `annual_inc`.  
  - `loan_amnt_category`: Kategori jumlah pinjaman (Low, Medium, High, Very High).

**Exploratory Data Analysis (EDA)**  
- **Distribusi Data**:  
  - `loan_amnt` memiliki distribusi miring kanan, dengan sebagian besar pinjaman di kisaran 5,000–20,000 USD (gambar: `loan_amnt_distribution.png`).  
  - `loan_to_income_ratio` menunjukkan distribusi miring, dengan beberapa outlier (gambar: `feature_engineering_visualizations.png`).  
- **Hubungan dengan Target**:  
  - Boxplot `loan_amnt` vs. `credit_risk` menunjukkan pinjaman lebih besar cenderung berisiko BAD (gambar: `loan_amnt_vs_credit_risk.png`).  
  - Boxplot `loan_to_income_ratio` vs. `credit_risk` mengindikasikan rasio tinggi berkorelasi dengan risiko BAD (gambar: `loan_to_income_vs_credit_risk.png`).  
- **Tren Waktu**: `issue_year` vs. `credit_risk` menunjukkan peningkatan risiko BAD pada tahun-tahun tertentu (gambar: `feature_engineering_visualizations.png`).  
- **Korelasi**: Heatmap menunjukkan `int_rate` dan `grade` memiliki korelasi signifikan dengan `credit_risk` (gambar: `correlation_heatmap.png`).  
- **Missing Values**: Kolom seperti `total_rev_hi_lim` (70,276 hilang) dan `all_util` (466,285 hilang) memiliki banyak nilai hilang, ditangani pada tahap *data preparation*.

## Data Preparation

**Proses Data Preparation**  
1. **Pembersihan Data**:  
   - Menghapus baris dengan `credit_risk` = OTHER, menyisakan 228,046 baris.  
   - Menghapus kolom dengan nilai hilang >50% (contoh: `total_bal_il`, `inq_fi`).  
2. **Feature Engineering**:  
   - **Rasio Keuangan**: `loan_to_income_ratio` = `loan_amnt` / `annual_inc`.  
   - **Ekstraksi Waktu**: `issue_year` dan `issue_month` dari `issue_d`.  
   - **Transformasi Log**: `log_annual_inc` = log(1 + `annual_inc`) untuk menangani distribusi miring.  
   - **Kategorisasi**: `loan_amnt_category` dengan bins [0, 5000, 15000, 25000, inf] (Low, Medium, High, Very High).  
3. **Penanganan Missing Values**:  
   - Numerik (`loan_to_income_ratio`, `log_annual_inc`, `annual_inc`): Imputasi dengan median.  
   - Kategorikal (`issue_year`, `issue_month`): Imputasi dengan modus.  
4. **Encoding Kategorikal**:  
   - `grade`, `home_ownership`, `loan_amnt_category` diencode menggunakan `LabelEncoder`.  
5. **Skalasi Fitur**:  
   - Fitur numerik (`loan_amnt`, `int_rate`, `annual_inc`, `dti`, `loan_to_income_ratio`, `issue_year`, `issue_month`, `log_annual_inc`) diskalakan dengan `StandardScaler`.  
6. **Pembagian Data**:  
   - 80% training, 20% testing dengan `train_test_split` (stratified).  
   - Validasi silang 5-fold untuk evaluasi generalisasi.

**Alasan Data Preparation**  
- **Pembersihan Data**: Menghapus status ambigu memastikan klasifikasi biner yang jelas.  
- **Feature Engineering**: Fitur baru meningkatkan kemampuan model menangkap pola risiko (contoh: `loan_to_income_ratio` mencerminkan kemampuan bayar).  
- **Penanganan Missing Values**: Imputasi median/modus menjaga integritas data.  
- **Encoding dan Skalasi**: Diperlukan untuk input model dan memastikan skala seragam.  
- **Pembagian Data**: Mencegah *overfitting* dan memberikan estimasi performa robust.

## Modeling

**Algoritma yang Digunakan**  
Empat algoritma diimplementasikan:  
1. **Logistic Regression**: Model linier.  
   - **Parameter**: `C` (0.1, 1, 10).  
   - **Kelebihan**: Interpretable, cocok untuk regulasi.  
   - **Kekurangan**: Kurang menangkap pola non-linier.  
2. **Random Forest**: Ensemble pohon keputusan.  
   - **Parameter**: `n_estimators` (100, 200), `max_depth` (10, 20).  
   - **Kelebihan**: Menangani data tidak seimbang, pola non-linier.  
   - **Kekurangan**: Kompleksitas komputasi tinggi.  
3. **XGBoost**: Gradient boosting.  
   - **Parameter**: `n_estimators` (100, 200), `learning_rate` (0.01, 0.1).  
   - **Kelebihan**: Performa tinggi, menangani ketidakseimbangan.  
   - **Kekurangan**: Membutuhkan tuning intensif.  
4. **SVM**: Support Vector Machine.  
   - **Parameter**: `C` (0.1, 1), `kernel` (linear, rbf).  
   - **Kelebihan**: Efektif untuk data kompleks.  
   - **Kekurangan**: Waktu pelatihan lama untuk dataset besar.

**Hyperparameter Tuning**  
- **Metode**: `GridSearchCV` dengan validasi silang 5-fold, metrik utama ROC-AUC.  
- **Hasil**: Model terbaik dipilih berdasarkan ROC-AUC rata-rata.

**Pemilihan Model Terbaik**  
- **XGBoost** dipilih karena:  
  - *Recall* sempurna (1.0000) untuk kelas BAD, mendeteksi semua peminjam berisiko tinggi.  
  - ROC-AUC tertinggi (0.9999) dan akurasi tinggi (0.9993).  
  - Variabilitas rendah (CV ROC-AUC Std 0.0001), menunjukkan generalisasi kuat.  
- Model disimpan sebagai `best_model.pkl`, bersama `scaler.pkl` dan `feature_names.pkl`.

**GUI Integration**  
- Model XGBoost, Logistic Regression, dan Random Forest diintegrasikan ke GUI (`CreditRiskGUI.py`).  
- GUI mendukung input fitur (`loan_amnt`, `int_rate`, `annual_inc`, `dti`, `grade`, `home_ownership`, `issue_d`), imputasi otomatis, dan tampilan probabilitas risiko.

## Evaluation

### Metrik Evaluasi
1. **Accuracy**: Persentase prediksi benar.  
   $$ \text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}} $$
2. **Precision**: Proporsi prediksi BAD yang benar.  
   $$ \text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}} $$
3. **Recall**: Proporsi peminjam BAD yang terdeteksi.  
   $$ \text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}} $$
4. **ROC-AUC**: Luas di bawah kurva ROC.  
   $$ \int \text{TPR} \, d(\text{FPR}) $$

**Hasil Evaluasi**  
Berdasarkan *validation set* dan cross-validation:

| Model              | Accuracy | Precision | Recall  | ROC-AUC | CV ROC-AUC Mean | CV ROC-AUC Std |
|--------------------|----------|-----------|---------|---------|-----------------|----------------|
| Logistic Regression| 0.9982   | 0.9979    | 0.9999  | 0.9991  | 0.9993          | 0.0003         |
| Random Forest      | 0.9979   | 0.9975    | 0.9999  | 0.9999  | 0.9999          | 0.0001         |
| XGBoost            | **0.9993**| **0.9992**| **1.0000**| **0.9999**| **0.9999**      | **0.0001**     |
| SVM                | 0.9981   | 0.9978    | 0.9999  | 0.9990  | 0.9992          | 0.0004         |

**Analisis Hasil**  
- **XGBoost**: Performa terbaik dengan *recall* 1.0000 untuk kelas BAD, akurasi 0.9993, dan ROC-AUC 0.9999. Variabilitas rendah (Std 0.0001) menunjukkan generalisasi kuat.  
- **Random Forest**: Performa sangat baik (*recall* 0.9999, ROC-AUC 0.9999), tetapi sedikit di bawah XGBoost dalam akurasi dan *precision*.  
- **Logistic Regression**: Kompetitif (*recall* 0.9999, ROC-AUC 0.9991), tetapi akurasi dan *precision* lebih rendah.  
- **SVM**: Performa baik, tetapi lebih lambat dan sedikit di bawah XGBoost.  
- **Visualisasi**: Kurva ROC menunjukkan AUC mendekati 1.0 untuk semua model (gambar: `roc_curves_comparison.png`). Tabel perbandingan disimpan sebagai `model_comparison.csv`.

**Kesimpulan**  
XGBoost adalah pilihan terbaik karena *recall* sempurna untuk kelas BAD, selaras dengan tujuan meminimalkan gagal bayar, serta akurasi dan ROC-AUC tertinggi. Fitur baru dari *feature engineering* (contoh: `loan_to_income_ratio`) meningkatkan performa. GUI memungkinkan penerapan praktis oleh petugas pinjaman.

**Langkah Selanjutnya**  
- **Validasi Eksternal**: Uji model pada data baru untuk memastikan generalisasi.  
- **Penanganan Overfitting**: Periksa *overfitting* dengan regularisasi atau data beragam.  
- **Pemantauan Model**: Pantau *data drift* di lingkungan produksi.  
- **Peningkatan Fitur**: Eksplorasi fitur seperti tren rasio utang.  
- **Pengembangan API**: Integrasi ke API (Flask) untuk skalabilitas.

---

**Catatan**  
- Gambar (`roc_curves_comparison.png`, `feature_engineering_visualizations.png`, dll.) tersedia di direktori `IMAGES/`.  
- Kode: `CreditScore.ipynb`, `CreditRiskGUI.py`.  
- Dataset: [Github: Lending Club Loan Data](https://github.com/RazerArdi/Credit-Risk-Prediction-Model#).