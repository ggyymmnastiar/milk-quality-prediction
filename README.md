# 🥛 Milk Quality Prediction

A Machine Learning web application that predicts milk quality grades (**high**, **medium**, **low**) based on physical and chemical parameters using a **Random Forest Classifier**, deployed via an interactive Streamlit dashboard.

🔗 **Live Dashboard:** [g-milkqualityprediction.streamlit.app](https://g-milkqualityprediction.streamlit.app)

---

## 📌 Project Overview

Milk quality evaluation is a critical aspect of the dairy industry to safeguard consumer health and ensure quality standards. Traditional manual quality checks can be time-consuming and prone to human inconsistency. 

This project delivers an end-to-end classification system that automates milk grade evaluation based on 7 measurable attributes. Through an intuitive web interface, users can explore data distributions, inspect preprocessing and model training metrics, and experiment with parameter inputs to obtain instant quality assessments.

### 🔄 End-to-End Workflow

```
Raw Dataset (1,059 samples)
    ↓
Data Cleaning (column typo fixes, missing value inspection)
    ↓
Exploratory Data Analysis (distribution charts, correlation heatmap, feature boxplots)
    ↓
Preprocessing Pipeline (StandardScaler for numerical, passthrough for binary)
    ↓
Random Forest Classifier + Offline GridSearchCV Tuning
    ↓
Interactive Streamlit Dashboard (Real-time inference & probability scores)
```

---

## 📊 Dataset Description

The dataset contains **1,059 milk samples** with 7 input features and 1 target variable:

| Feature | Data Type | Range / Values | Description |
|---|---|---|---|
| `pH` | Numerical (float) | 0.0 – 14.0 | Acidity / alkalinity level of milk |
| `Temperature` | Numerical (int) | 10 – 90 | Milk temperature (°C / °F depending on sensor) |
| `Taste` | Binary | 0 or 1 | Taste evaluation (0: Not Satisfied, 1: Satisfied) |
| `Odor` | Binary | 0 or 1 | Odor evaluation (0: Not Satisfied, 1: Satisfied) |
| `Fat` | Binary | 0 or 1 | Fat content evaluation (0: Not Satisfied, 1: Satisfied) |
| `Turbidity` | Binary | 0 or 1 | Turbidity / clarity (0: Not Satisfied, 1: Satisfied) |
| `Colour` | Numerical (int) | 240 – 255 | Color intensity measurement |

- **Target Variable (`Grade`):** Categorical class with three levels:
  - `high` — High quality milk
  - `medium` — Moderate quality milk
  - `low` — Low quality milk

> **Data Cleaning Note:** Original column headers in the raw dataset (`Temprature` and `Fat ` with trailing whitespace) are standardized during initial data ingestion.

---

## 🤖 Model & Machine Learning Pipeline

### Algorithm
- **Random Forest Classifier:** An ensemble decision tree model capable of capturing complex non-linear feature interactions without prone overfitting.

### Preprocessing Pipeline
Built with `sklearn.compose.ColumnTransformer` and `sklearn.pipeline.Pipeline`:
- **Numerical Features** (`pH`, `Temperature`, `Colour`) → `StandardScaler()`
- **Binary Features** (`Taste`, `Odor`, `Fat`, `Turbidity`) → `'passthrough'`

### Hyperparameter Tuning
Hyperparameters were evaluated offline with `GridSearchCV` using 5-fold `StratifiedKFold` cross-validation with `f1_macro` scoring metric:

```python
{
    'model__bootstrap': True,
    'model__class_weight': None,
    'model__max_features': 'sqrt',
    'model__min_samples_leaf': 1,
    'model__min_samples_split': 2,
    'model__n_estimators': 200
}
```

- **Train/Test Split:** 80% training set, 20% testing set (`random_state=101`).

---

## 🖥️ Dashboard Capabilities

| Section | Content & Functionality |
|---|---|
| **Data Overview** | Raw data preview, input matrix ($X$), and label vector ($y$). |
| **Data Preparation & EDA** | Cleaning log, null value counts, and feature datatype breakdown. |
| **Data Visualization** | Class distribution charts (bar & pie), numerical boxplots, binary countplots, correlation heatmap, and descriptive statistics. |
| **Data Preprocessing** | Code architecture, train-test split statistics, interactive on-demand model training button with dynamic classification report and confusion matrix. |
| **Hyperparameter Tuning** | GridSearchCV setup and best parameter configuration reference. |
| **Interactive Prediction** | Sidebar input sliders and select boxes for real-time grade estimation with class confidence probabilities. |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+ (Python 3.11 recommended)
- `pip` or virtual environment manager

### Local Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ggyymmnastiar/milk-quality-prediction.git
   cd milk-quality-prediction
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate       # macOS / Linux
   # .venv\Scripts\activate        # Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the application:**
   ```bash
   streamlit run streamlit_app.py
   ```

5. Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📁 Repository Structure

```
milk-quality-prediction/
├── data/
│   └── milk_dataset.csv        # Milk quality dataset (1,059 samples)
├── .devcontainer/
│   └── devcontainer.json       # Devcontainer / Codespaces configuration
├── .streamlit/
│   └── config.toml             # Streamlit visual theme configuration
├── streamlit_app.py            # Streamlit dashboard application
├── requirements.txt            # Project dependencies
├── .gitignore                  # Git ignore rules
└── README.md                   # Project documentation
```

---

## 🛠️ Tech Stack

- **Application Framework:** Streamlit
- **Machine Learning:** scikit-learn
- **Data Manipulation:** pandas, NumPy
- **Data Visualization:** Matplotlib, Seaborn
- **Language:** Python 3.11

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
