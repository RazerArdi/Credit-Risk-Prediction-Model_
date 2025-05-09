# Laporan Proyek Machine Learning - Bayu Ardiyansyah

## Domain Proyek

**Prediksi Risiko Kredit untuk Perusahaan Multifinance**

Dalam industri keuangan, khususnya perusahaan pembiayaan (multifinance), kemampuan untuk mengidentifikasi peminjam berisiko tinggi sangat penting untuk meminimalkan tingkat gagal bayar (default) sambil tetap menjaga profitabilitas. Tingkat gagal bayar yang tinggi dapat menyebabkan kerugian finansial signifikan, sedangkan penolakan pinjaman yang berlebihan dapat mengurangi peluang pendapatan. Oleh karena itu, pengembangan model machine learning untuk memprediksi risiko kredit menjadi solusi strategis untuk mengoptimalkan keputusan pemberian pinjaman.

Proyek ini bertujuan untuk membangun model prediktif yang dapat mengklasifikasikan risiko kredit peminjam sebagai **GOOD** (rendah risiko) atau **BAD** (tinggi risiko) berdasarkan data pinjaman historis dari tahun 2007–2014. Model ini akan membantu perusahaan multifinance dalam mengurangi risiko gagal bayar dengan mengidentifikasi peminjam berisiko tinggi secara akurat. Selain itu, proyek ini mencakup pengembangan antarmuka grafis (GUI) modern menggunakan `customtkinter` untuk memudahkan petugas pinjaman dalam menggunakan model prediktif.

**Mengapa Masalah Ini Penting?**  
- **Dampak Finansial**: Menurut studi oleh McKinsey, pengelolaan risiko kredit yang efektif dapat mengurangi kerugian akibat gagal bayar hingga 20–30% [1].  
- **Efisiensi Operasional**: Model prediktif mengotomatiskan proses penilaian risiko, mengurangi waktu dan biaya yang diperlukan untuk evaluasi manual.  
- **Kepuasan Pelanggan**: Dengan menyeimbangkan penerimaan dan penolakan pinjaman, perusahaan dapat mempertahankan pelanggan potensial sambil mengelola risiko.

## Business Understanding

### Problem Statements
1. **Tingkat Gagal Bayar yang Tinggi**: Perusahaan multifinance menghadapi risiko kerugian akibat peminjam yang gagal membayar pinjaman, terutama karena kurangnya alat prediktif yang akurat untuk menilai risiko kredit.  
2. **Proses Penilaian Manual yang Tidak Efisien**: Penilaian risiko kredit secara manual memakan waktu dan rentan terhadap kesalahan manusia, sehingga menghambat skalabilitas operasional.  
3. **Kurangnya Antarmuka Pengguna yang Intuitif**: Petugas pinjaman membutuhkan alat yang mudah digunakan untuk menerapkan model prediktif dalam keputusan sehari-hari.

### Goals
1. **Mengembangkan Model Prediktif yang Akurat**: Membangun model machine learning yang dapat memprediksi risiko kredit (GOOD atau BAD) dengan recall tinggi untuk mengidentifikasi peminjam berisiko tinggi, sehingga meminimalkan tingkat gagal bayar.  
2. **Mengotomatiskan Penilaian Risiko**: Mengintegrasikan model ke dalam alur kerja perusahaan untuk mempercepat dan meningkatkan akurasi proses pengambilan keputusan.  
3. **Menyediakan Antarmuka Pengguna yang Modern**: Mengembangkan GUI yang user-friendly untuk memungkinkan petugas pinjaman memasukkan data peminjam, memilih model, dan melihat hasil prediksi dengan mudah.

### Solution Statements
- **Menggunakan Berbagai Algoritma Machine Learning**: Mengimplementasikan tiga algoritma (Logistic Regression, Random Forest, XGBoost) untuk memprediksi risiko kredit, dengan evaluasi berdasarkan metrik akurasi, precision, recall, dan ROC-AUC.  
- **Hyperparameter Tuning**: Melakukan optimasi hyperparameter pada setiap model menggunakan GridSearchCV untuk meningkatkan performa, terutama recall untuk kelas BAD.  
- **Pemilihan Model Terbaik**: Memilih model dengan recall tertinggi untuk kelas BAD dan ROC-AUC yang kompetitif untuk memastikan keseimbangan antara deteksi risiko dan performa keseluruhan.  
- **Pengembangan GUI**: Membangun antarmuka grafis menggunakan `customtkinter` yang mendukung input data peminjam, pemilihan model, dan tampilan hasil prediksi dengan probabilitas risiko.

## Data Understanding

**Informasi Dataset**  
Dataset yang digunakan adalah `loan_data_2007_2014.csv`, yang berisi data pinjaman historis dari tahun 2007 hingga 2014. Dataset ini tersedia di [Github: Lending Club Loan Data](https://github.com/RazerArdi/Credit-Risk-Prediction-Model#).  
- **Jumlah Baris**: 466,285  
- **Jumlah Kolom**: 75  
- **Variabel Target**: `credit_risk` (binary: 0 untuk GOOD, 1 untuk BAD), diturunkan dari `loan_status` dengan aturan:  
  - GOOD: `loan_status` = "Fully Paid"  
  - BAD: `loan_status` = "Charged Off" atau "Default"  
  - OTHER: Status ambigu seperti "Current" atau "Late" dihapus.

### Variabel pada Dataset
Berikut adalah beberapa variabel utama yang digunakan dalam pemodelan:  
- `loan_amnt`: Jumlah pinjaman yang diajukan (numerik, USD).  
- `int_rate`: Suku bunga pinjaman (numerik, %).  
- `annual_inc`: Pendapatan tahunan peminjam (numerik, USD).  
- `dti`: Rasio utang terhadap pendapatan (numerik, %).  
- `grade`: Tingkat kredit yang diberikan (kategorikal, A–G).  
- `term`: Jangka waktu pinjaman (kategorikal, ' 36 months' atau ' 60 months').  
- `loan_status`: Status pinjaman (kategorikal, digunakan untuk membuat `credit_risk`).  
- Lainnya: Kolom seperti `funded_amnt`, `installment`, `total_pymnt`, dll., yang mendukung analisis risiko.

**Exploratory Data Analysis (EDA)**  
- **Distribusi Data**: Visualisasi distribusi `loan_amnt` menunjukkan sebagian besar pinjaman berada di kisaran 5,000–20,000 USD (gambar: `loan_amount_distribution.png`).  
- **Hubungan dengan Target**: Boxplot `loan_amnt` vs. `credit_risk` mengindikasikan bahwa pinjaman dengan jumlah lebih tinggi cenderung memiliki risiko BAD lebih besar (gambar: `loan_amount_vs_credit_risk.png`).  
- **Korelasi**: Heatmap korelasi menunjukkan bahwa fitur seperti `int_rate` dan `grade` memiliki korelasi signifikan dengan risiko kredit (gambar: `correlation_matrix.png`).  
- **Missing Values**: Banyak kolom memiliki nilai yang hilang (contoh: `total_rev_hi_lim` hilang 70,276 baris), yang ditangani pada tahap data preparation.

## Data Preparation

**Proses Data Preparation**  
1. **Pembersihan Data**:  
   - Menghapus baris dengan status pinjaman ambigu (OTHER), menyisakan 228,046 baris (GOOD: 184,739, BAD: 43,307).  
   - Menghapus kolom yang tidak relevan atau memiliki nilai hilang >50% (contoh: `total_bal_il`, `inq_fi`).  
2. **Penanganan Missing Values**:  
   - Nilai numerik diimputasi dengan median (contoh: `annual_inc`).  
   - Nilai kategorikal diimputasi dengan modus (contoh: `grade`).  
3. **Encoding Kategorikal**:  
   - Variabel `grade` dan `term` diencode menggunakan `LabelEncoder` (contoh: A=0, B=1, ..., G=6 untuk `grade`).  
4. **Skalasi Fitur**:  
   - Fitur numerik diskalakan menggunakan `StandardScaler` untuk menormalkan distribusi.  
5. **Pembagian Data**:  
   - Data dibagi menjadi 80% training dan 20% testing menggunakan `train_test_split`.  
   - Validasi silang 5-fold digunakan untuk mengevaluasi generalisasi model.

**Alasan Data Preparation**  
- **Pembersihan Data**: Menghapus status ambigu memastikan target variabel jelas dan relevan untuk klasifikasi biner.  
- **Penanganan Missing Values**: Imputasi dengan median/modus menjaga integritas data tanpa memperkenalkan bias signifikan.  
- **Encoding dan Skalasi**: Encoding kategorikal diperlukan untuk input model, sedangkan skalasi memastikan fitur numerik memiliki skala seragam, yang penting untuk algoritma seperti Logistic Regression.  
- **Pembagian Data**: Memisahkan data training dan testing mencegah overfitting, sedangkan validasi silang memberikan perkiraan performa yang lebih robust.

## Modeling

**Algoritma yang Digunakan**  
Tiga algoritma diimplementasikan untuk memprediksi risiko kredit:  
1. **Logistic Regression**: Model linier untuk klasifikasi biner.  
   - **Parameter**: `C=1.0`, `max_iter=1000`.  
   - **Kelebihan**: Interpretable, cocok untuk kepatuhan regulasi.  
   - **Kekurangan**: Kurang mampu menangkap pola non-linier kompleks.  
2. **Random Forest**: Ensemble berbasis pohon keputusan.  
   - **Parameter Awal**: `n_estimators=100`, `max_depth=None`.  
   - **Kelebihan**: Menangani data tidak seimbang dan pola non-linier.  
   - **Kekurangan**: Kompleksitas komputasi tinggi.  
3. **XGBoost**: Ensemble berbasis gradient boosting.  
   - **Parameter Awal**: `n_estimators=100`, `learning_rate=0.1`.  
   - **Kelebihan**: Performa tinggi, menangani data tidak seimbang dengan baik.  
   - **Kekurangan**: Membutuhkan tuning hyperparameter intensif.

**Hyperparameter Tuning**  
- **Metode**: GridSearchCV dengan validasi silang 5-fold.  
- **Contoh Tuning**:  
  - Random Forest: Mengoptimalkan `n_estimators` (50, 100, 200) dan `max_depth` (10, 20, None).  
  - XGBoost: Mengoptimalkan `learning_rate` (0.01, 0.1, 0.3) dan `max_depth` (3, 5, 7).  
- **Hasil**: Model terbaik dipilih berdasarkan ROC-AUC rata-rata dari validasi silang.

**Pemilihan Model Terbaik**  
- XGBoost dipilih sebagai model terbaik karena:  
  - Recall sempurna (1.0000) untuk kelas BAD, memastikan semua peminjam berisiko tinggi terdeteksi.  
  - ROC-AUC tertinggi (0.9999) dan akurasi tertinggi (0.9993) pada validation set, menunjukkan kemampuan diskriminasi yang sangat baik.  
  - Variabilitas rendah (CV ROC-AUC Std 0.0001), mengindikasikan generalisasi yang kuat.  

**GUI Integration**  
- Model terbaik (XGBoost) diintegrasikan ke dalam GUI (`CreditRiskGUI.py`) bersama Logistic Regression dan Random Forest untuk fleksibilitas.  
- GUI memungkinkan input fitur utama (`loan_amnt`, `int_rate`, `annual_inc`, `dti`, `grade`, `term`) dan menangani imputasi fitur lainnya menggunakan nilai median/modus dari data pelatihan.

## Evaluation

### **Metrik Evaluasi**  
Model dievaluasi menggunakan metrik berikut:

1. **Accuracy**: Persentase prediksi yang benar.  
   - **Formula**:  
     $$ \text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}} $$
   - **Penjelasan**: Mengukur performa keseluruhan, tetapi kurang relevan untuk data tidak seimbang.

2. **Precision**: Proporsi prediksi BAD yang benar.  
   - **Formula**:  
     $$ \text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}} $$
   - **Penjelasan**: Penting untuk meminimalkan false positives (menolak peminjam yang sebenarnya GOOD).

3. **Recall**: Proporsi peminjam BAD yang terdeteksi.  
   - **Formula**:  
     $$ \text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}} $$
   - **Penjelasan**: Kunci untuk mengidentifikasi peminjam berisiko tinggi, meminimalkan gagal bayar.

4. **ROC-AUC**: Luas di bawah kurva ROC, mengukur kemampuan model membedakan kelas.  
   - **Formula**:  
     $$ \int \text{TPR} \, d(\text{FPR}) $$
   - **Penjelasan**: Metrik komprehensif untuk performa klasifikasi, terutama pada data tidak seimbang.

**Hasil Evaluasi**  
Berdasarkan performa pada validation set dan cross-validation:  

| Model              | Accuracy | Precision | Recall  | ROC-AUC | CV ROC-AUC Mean | CV ROC-AUC Std |
|--------------------|----------|-----------|---------|---------|-----------------|----------------|
| Logistic Regression| 0.9982   | 0.9979    | 0.9999  | 0.9991  | 0.9993          | 0.0003         |
| Random Forest      | 0.9979   | 0.9975    | 0.9999  | 0.9999  | 0.9999          | 0.0001         |
| XGBoost            | **0.9993**| **0.9992**| **1.0000**| **0.9999**| **0.9999**      | **0.0001**     |

**Cross-Validation Results**:  
- **Logistic Regression**: CV ROC-AUC Mean = 0.9993, Std = 0.0003  
- **Random Forest**: CV ROC-AUC Mean = 0.9999, Std = 0.0001  
- **XGBoost**: CV ROC-AUC Mean = 0.9999, Std = 0.0001  

**Analisis Hasil**  
- **XGBoost**: Menunjukkan performa terbaik dengan recall sempurna (1.0000) untuk kelas BAD, memastikan semua peminjam berisiko tinggi terdeteksi, dan akurasi tertinggi (0.9993). ROC-AUC 0.9999 menunjukkan kemampuan diskriminasi yang hampir sempurna. Variabilitas rendah (CV ROC-AUC Std 0.0001) mengkonfirmasi generalisasi yang sangat baik.  
- **Random Forest**: Performa sangat baik (recall 0.9999, ROC-AUC 0.9999), tetapi sedikit di bawah XGBoost dalam akurasi (0.9979) dan precision (0.9975).  
- **Logistic Regression**: Performa sangat kompetitif (recall 0.9999, ROC-AUC 0.9991), tetapi memiliki akurasi (0.9982) dan precision (0.9979) lebih rendah dibandingkan XGBoost.  
- **Visualisasi**: Kurva ROC menunjukkan XGBoost dan Random Forest memiliki AUC mendekati 1.0 (gambar: `roc_curves_comparison.png`). Matriks konfusi (contoh: `xgboost_test_confusion_matrix.png`) mengkonfirmasi deteksi BAD yang hampir sempurna.

**Kesimpulan**  
XGBoost adalah pilihan terbaik karena recall sempurna untuk kelas BAD, yang selaras dengan tujuan meminimalkan gagal bayar, serta akurasi dan ROC-AUC tertinggi. Performa mendekati sempurna menunjukkan model sangat efektif untuk dataset ini. GUI memungkinkan penerapan model ini secara praktis oleh petugas pinjaman.

**Langkah Selanjutnya**  
- **Validasi Eksternal**: Menguji model pada dataset eksternal atau data baru untuk memastikan generalisasi di luar dataset Lending Club.  
- **Penanganan Overfitting**: Meskipun performa sangat tinggi, memeriksa potensi overfitting dengan regularisasi tambahan atau pengujian pada data yang lebih beragam.  
- **Pemantauan Model**: Memantau performa model di lingkungan produksi untuk mendeteksi data drift.  
- **Peningkatan Fitur**: Mengeksplorasi fitur baru seperti tren rasio utang terhadap pendapatan untuk meningkatkan akurasi lebih lanjut.  
- **Pengembangan API**: Mengintegrasikan model ke dalam API (contoh: Flask) untuk skalabilitas lebih besar.

---

_Catatan_: Gambar seperti `roc_curves_comparison.png`, `correlation_matrix.png`, dll., tersedia di direktori `IMAGES/` untuk referensi visual. Kode relevan dapat ditemukan di `CreditScore.ipynb` dan `CreditRiskGUI.py`.