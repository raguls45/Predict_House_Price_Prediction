# 🏠 House Price Prediction

A complete end-to-end **machine learning project** that predicts real estate prices per unit area using regression models. The project includes data analysis, model training, a command-line predictor, and a web application with a REST API.

---

## 📌 Table of Content

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Features](#features)
- [ML Pipeline](#ml-pipeline)
- [Model Performance](#model-performance)
- [Installation](#installation)
- [Usage](#usage)
- [Web Application](#web-application)
- [Tech Stack](#tech-stack)

---

## Overview

This project uses the **Real Estate Valuation dataset** to train machine learning models that estimate the **house price of unit area** based on property characteristics such as location, age, and proximity to amenities. Three regression models are trained and compared automatically — the best-performing model is saved and used for prediction.

---

## Project Structure

```
house-price-prediction/
│
├── data/
│   └── Real estate valuation data set.xlsx   # Source dataset
│
├── models/                                    # Saved model artifacts (generated after training)
│   ├── house_price_model.pkl
│   ├── scaler.pkl
│   ├── selector.pkl
│   └── selected_features.pkl
│
├── notebooks/                                 # Jupyter notebooks (exploratory work)
│
├── reports/                                   # Auto-generated analysis outputs
│   ├── model_comparison.csv
│   ├── model_comparison.png
│   ├── model_training_report.md
│   ├── house_price_distribution.png
│   ├── house_price_boxplot.png
│   ├── correlation_heatmap.png
│   ├── house_age_vs_price.png
│   ├── mrt_distance_vs_price.png
│   └── convenience_stores_vs_price.png
│
├── src/
│   ├── data_analysis.py    # EDA and visualisation
│   ├── train_model.py      # Model training pipeline
│   ├── predict.py          # CLI-based predictor
│   └── web_app.py          # HTTP server + REST API
│
├── web/
│   ├── index.html          # Web UI
│   ├── app.js              # Frontend logic
│   └── styles.css          # Stylesheet
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Dataset

**Source**: UCI Machine Learning Repository — [Real Estate Valuation Data Set](https://archive.ics.uci.edu/ml/datasets/Real+estate+valuation+data+set)

| Feature | Description |
|---|---|
| `X1 transaction date` | Date of transaction (e.g., 2013.5) |
| `X2 house age` | Age of the house in years |
| `X3 distance to the nearest MRT station` | Distance in metres |
| `X4 number of convenience stores` | Count of nearby convenience stores |
| `X5 latitude` | Geographic latitude |
| `X6 longitude` | Geographic longitude |
| `Y house price of unit area` | **Target** — price per unit area |

- **Total records**: 414
- **Train / Test split**: 80% / 20%

---

## Features

- ✅ Exploratory Data Analysis (EDA) with visualisations
- ✅ Outlier detection using IQR method
- ✅ Feature scaling with `StandardScaler`
- ✅ Feature selection with `SelectKBest` (top 5 features)
- ✅ Dimensionality reduction with PCA (95% variance retained)
- ✅ Three regression models trained and benchmarked
- ✅ Best model automatically selected and saved
- ✅ CLI-based interactive predictor
- ✅ Web app with REST API for browser-based predictions
- ✅ Auto-generated training reports (CSV + Markdown + charts)

---

## ML Pipeline

```
Raw Data → Remove ID Column → Train/Test Split (80/20)
       → StandardScaler → SelectKBest (k=5) → Model Training
       → Evaluation (MAE, RMSE, R²) → Best Model Saved
```

**Models compared:**
| Model | Notes |
|---|---|
| Linear Regression | Baseline model |
| Random Forest Regressor | 200 estimators |
| Gradient Boosting Regressor | 200 estimators |

The model with the **lowest RMSE** is automatically selected as the best model.

---

## Model Performance

> Performance metrics are saved to `reports/model_comparison.csv` and `reports/model_training_report.md` after running `train_model.py`.

Metrics used:
- **MAE** — Mean Absolute Error
- **RMSE** — Root Mean Squared Error  
- **R²** — Coefficient of Determination

---

## Installation

### Prerequisites
- Python 3.9+
- pip

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/raguls45/Predict_House_Price_Prediction.git
cd Predict_House_Price_Prediction

# 2. Create a virtual environment
python -m venv venv

# 3. Activate the virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Step 1 — Run Data Analysis
```bash
python src/data_analysis.py
```
Generates EDA visualisations saved in `reports/`.

### Step 2 — Train the Model
```bash
python src/train_model.py
```
- Trains and compares all three models
- Saves the best model to `models/`
- Generates comparison report in `reports/`

### Step 3 — Make a Prediction (CLI)
```bash
python src/predict.py
```
Interactive command-line tool — prompts for property details and outputs the predicted price per unit area.

**Example input:**
```
Enter transaction date (Example: 2013.5): 2013.5
Enter house age in years: 10
Enter distance to nearest MRT station: 500
Enter number of convenience stores: 5
Enter latitude: 24.97
Enter longitude: 121.54
```

---

## Web Application

The project includes a browser-based prediction interface powered by a pure-Python HTTP server.

### Start the server
```bash
python src/web_app.py
```

Then open your browser at: **http://127.0.0.1:8000**

### REST API

**Endpoint**: `POST /api/predict`

**Request body:**
```json
{
  "X1 transaction date": 2013.5,
  "X2 house age": 10,
  "X3 distance to the nearest MRT station": 500,
  "X4 number of convenience stores": 5,
  "X5 latitude": 24.97,
  "X6 longitude": 121.54
}
```

**Response:**
```json
{
  "predicted_price": 38.45,
  "unit": "dataset price units per unit area"
}
```

> ⚠️ **Note**: Train the model first (`python src/train_model.py`) before starting the web app. The server requires the `.pkl` model files in `models/`.

---

## Tech Stack

| Category | Library / Tool |
|---|---|
| Language | Python 3.9+ |
| Data Processing | pandas, numpy |
| Machine Learning | scikit-learn |
| Visualisation | matplotlib, seaborn |
| Model Serialisation | joblib |
| Web Server | Python `http.server` (stdlib) |
| Frontend | HTML, CSS, JavaScript |
| Dataset Format | Excel (`.xlsx`) via openpyxl |

---

## Author

**RAGULKRISHNAN S** — [GitHub](https://github.com/raguls45)

---

## License

This project is open-source and available under the [MIT License](LICENSE).
