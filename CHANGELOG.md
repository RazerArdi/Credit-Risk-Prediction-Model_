# CHANGELOG

## [1.0.0] - 2025-05-09

**Initial Release**

### Added
- **CreditScore.ipynb**: Notebook untuk analisis data, pemodelan, dan evaluasi risiko kredit.  
  - Data Understanding: Pemuatan dataset `loan_data_2007_2014.csv` dan pembuatan variabel target `credit_risk`.  
  - EDA: Visualisasi distribusi `loan_amnt`, hubungan dengan `credit_risk`, dan korelasi fitur.  
  - Data Preparation: Pembersihan data, imputasi missing values, encoding kategorikal, dan skalasi fitur.  
  - Modeling: Implementasi Logistic Regression, Random Forest, XGBoost, dan SVM dengan hyperparameter tuning.  
  - Evaluation: Perbandingan model menggunakan akurasi, precision, recall, dan ROC-AUC.  
  - Penyimpanan: Model, scaler, encoder, dan nilai imputasi disimpan di direktori `Model/`.  
- **CreditRiskGUI.py**: Antarmuka grafis modern menggunakan `customtkinter`.  
  - Fitur: Input data peminjam, pemilihan model, prediksi risiko kredit, probabilitas BAD, contoh penggunaan, dan toggle tema.  
  - Penanganan Error: Validasi input dan pemeriksaan file model/preprocessing.  
- **Direktori IMAGES/**: Berisi visualisasi seperti `roc_curves_comparison.png`, `correlation_matrix.png`, dll.  
- **Direktori Model/**: Berisi file `.pkl` untuk model, scaler, encoder, dan nilai imputasi.  
- **model_comparison.csv**: Tabel perbandingan performa model.  
- **Report.md**: Laporan proyek dengan struktur Domain, Business Understanding, Data Understanding, Data Preparation, Modeling, dan Evaluation.  
- **CHANGELOG.md**: Dokumentasi riwayat pengembangan proyek.

### Notes
- Dataset `loan_data_2007_2014.csv` diunduh dari [Kaggle: Lending Club Loan Data](https://www.kaggle.com/datasets/wordsforthewise/lending-club).  
- Pastikan dependensi (`customtkinter`, `pandas`, `numpy`, `joblib`, `scikit-learn`, `xgboost`) terinstal sebelum menjalankan GUI.  
- Jalankan `CreditScore.ipynb` terlebih dahulu untuk menghasilkan file di direktori `Model/` sebelum menggunakan `CreditRiskGUI.py`.