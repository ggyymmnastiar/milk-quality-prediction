# 🥛 Milk Quality Prediction

Memprediksi kualitas susu (**high**, **medium**, **low**) berdasarkan sifat fisik dan kimianya menggunakan **Random Forest Classifier**, dikemas dalam dashboard interaktif berbasis Streamlit.

🔗 **Live Dashboard:** [g-milkqualityprediction.streamlit.app](https://g-milkqualityprediction.streamlit.app)

---

## Tentang Project

Kualitas susu adalah faktor kritis dalam industri dairy. Penilaian manual memakan waktu dan rentan terhadap inkonsistensi. Project ini membangun model klasifikasi yang dapat memprediksi grade susu secara otomatis berdasarkan 7 parameter terukur, lalu menyajikannya dalam dashboard yang memungkinkan pengguna bereksperimen dengan input secara real-time.

### Alur Kerja

```
Dataset (1059 sampel)
    ↓
Data Cleaning (fix typo kolom, cek missing values)
    ↓
EDA & Visualisasi (distribusi, korelasi, boxplot)
    ↓
Preprocessing (StandardScaler untuk numerik, passthrough untuk biner)
    ↓
Random Forest Classifier + GridSearchCV tuning
    ↓
Dashboard Prediksi Interaktif (Streamlit)
```

---

## Dataset

Dataset berisi **1.059 sampel susu** dengan 7 fitur dan 1 variabel target:

| Fitur | Tipe | Deskripsi |
|-------|------|-----------|
| `pH` | Numerik (float) | Tingkat keasaman/kebasaan susu |
| `Temperature` | Numerik (int) | Suhu susu (°F) |
| `Taste` | Biner (0/1) | Kepuasan rasa |
| `Odor` | Biner (0/1) | Kepuasan aroma |
| `Fat` | Biner (0/1) | Kandungan lemak |
| `Turbidity` | Biner (0/1) | Tingkat kekeruhan |
| `Colour` | Numerik (int) | Nilai intensitas warna (240–255) |

**Target:** `Grade` → `high` · `medium` · `low`

> **Catatan:** Dataset asli memiliki typo pada nama kolom (`Temprature`, `Fat ` dengan trailing space) yang diperbaiki saat data cleaning.

---

## Model

**Algoritma:** Random Forest Classifier — metode ensemble yang membangun banyak decision tree dan menentukan kelas berdasarkan voting mayoritas.

**Preprocessing Pipeline:**
- Fitur numerik (`pH`, `Temperature`, `Colour`) → `StandardScaler`
- Fitur biner (`Taste`, `Odor`, `Fat`, `Turbidity`) → Passthrough (tanpa transformasi)

**Hyperparameter Tuning** dilakukan secara offline menggunakan `GridSearchCV` dengan `StratifiedKFold` (5 splits) dan scoring `f1_macro`. Parameter terbaik yang ditemukan:

```python
{
    'bootstrap': True,
    'class_weight': None,
    'max_features': 'sqrt',
    'min_samples_leaf': 1,
    'min_samples_split': 2,
    'n_estimators': 200
}
```

**Split data:** 80% train / 20% test (`random_state=101`)

---

## Fitur Dashboard

| Bagian | Apa yang ditampilkan |
|--------|---------------------|
| **Data Overview** | Raw data, input features (X), dan label (y) |
| **Data Preparation & EDA** | Proses cleaning, missing value check, tipe fitur |
| **Data Visualization** | Distribusi target, boxplot per fitur, correlation heatmap, statistik deskriptif |
| **Data Preprocessing** | Detail pipeline, train/test split, tombol training dengan classification report & confusion matrix |
| **Hyperparameter Tuning** | Parameter grid dan best parameters dari GridSearchCV |
| **Predicted Grade** | Prediksi real-time berdasarkan input sidebar, lengkap dengan probability bar untuk setiap kelas |

Pengguna dapat mengatur nilai setiap fitur melalui **sidebar** (slider & dropdown), dan prediksi akan langsung berubah.

---

## Cara Menjalankan

```bash
# Clone repository
git clone https://github.com/ggyymmnastiar/milk-quality-prediction.git
cd milk-quality-prediction

# Install dependencies
pip install -r requirements.txt

# Jalankan aplikasi
streamlit run streamlit_app.py
```

Buka browser di **http://localhost:8501**.

---

## Struktur Project

```
milk-quality-prediction/
├── data/
│   └── milk_dataset.csv        # Dataset susu (1059 sampel)
├── .devcontainer/
│   └── devcontainer.json       # Konfigurasi GitHub Codespaces
├── .streamlit/
│   └── config.toml             # Konfigurasi tema Streamlit
├── streamlit_app.py            # Aplikasi utama
├── requirements.txt            # Dependencies Python
└── README.md
```

---

## Tech Stack

**Python** · **Streamlit** · **scikit-learn** · **pandas** · **NumPy** · **Matplotlib** · **Seaborn**

---

## License

MIT
