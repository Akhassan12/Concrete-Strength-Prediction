# 🏗️ Concrete Compressive Strength Prediction

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Flask-Web%20App-green.svg?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![ML Engine](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Gradient Boosting](https://img.shields.io/badge/XGBoost-Ensemble-red.svg)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Author](https://img.shields.io/badge/Author-Ali%20Hassan%20Kadri-purple.svg)](https://github.com/Akhassan12)

A state-of-the-art civil engineering machine learning platform engineered to predict the **28-day (and custom age) compressive strength of concrete** ($\text{MPa}$) directly from raw constituent mix proportions and curing duration. 

Designed and maintained exclusively by **Ali Hassan Kadri**.

---

## 📌 Executive Summary

Compressive strength is the paramount mechanical indicator used in structural civil engineering to verify safety, load-bearing compliance, and durability. Traditional physical lab tests require standard compressive cylinders or cubes to undergo curing for 7 to 28 days before hydraulic press testing, which causes project delays and prevents rapid mix optimization.

This platform bridges computational intelligence and materials engineering:
- **Dual Predictive Engine**: Evaluates concrete mixtures through an advanced **Ensemble Voting Regressor** ($R^2 = 0.9125$) and a multi-layer **Deep Neural Network**.
- **Interactive Modern Interface**: Features a glassmorphic dashboard with live sliders, 1-click industry mix presets, real-time water-to-cement ($w/c$) ratio diagnostics, and automated ASTM / Eurocode grade classification.
- **Microservice-Ready REST API**: Provides seamless programmatic access (`/api/predict`) for laboratory information management systems (LIMS) and batching plants.

---

## 🏗️ System Architecture

```mermaid
graph TD
    A[User / Batching Plant] -->|Input Mix Proportions| B[Modern Glassmorphic Web UI]
    A -->|JSON Payload| C[REST API: /api/predict]
    B -->|Async Fetch / Form POST| D[Flask Backend Controller]
    C --> D
    D -->|Feature Vector: 8 Dimensions| E[Data Validation & Preprocessing]
    E --> F[Dual Inference Pipeline]
    F -->|Ensemble Model| G[Voting Regressor: RF + AdaBoost + GradBoost + XGBoost + Lasso]
    F -->|Deep Learning| H[Multi-Layer Artificial Neural Network]
    G --> I[Strength Prediction in MPa]
    H --> I
    I --> J[Concrete Grade Classifier: C20/25, C30/37, C40/50, High-Strength]
    J --> K[Real-time Visual Gauge & Diagnostics]
    K --> B
```

---

## 🔬 Dataset & Input Features

The predictive models were trained on empirical concrete compressive strength datasets encompassing diverse mix formulations:

| Parameter | Feature Name | Unit | Typical Range | Role in Concrete Mix |
| :--- | :--- | :---: | :---: | :--- |
| **Cement** | `cement` | $\text{kg/m}^3$ | $100 - 550$ | Primary hydraulic binder providing hydration products ($C\text{-}S\text{-}H$ gel) |
| **Blast Furnace Slag** | `slag` | $\text{kg/m}^3$ | $0 - 360$ | Supplementary cementitious material; enhances long-term durability |
| **Fly Ash** | `fly_ash` | $\text{kg/m}^3$ | $0 - 200$ | Pozzolanic pozzolan; improves workability and reduces thermal cracking |
| **Water** | `water` | $\text{kg/m}^3$ | $120 - 250$ | Hydrates cement; critically determines the water-to-binder ($w/b$) ratio |
| **Superplasticizer** | `superplasticizer`| $\text{kg/m}^3$ | $0 - 35$ | High-range water reducer ensuring high slump with minimal water |
| **Coarse Aggregate** | `coarse_agg` | $\text{kg/m}^3$ | $800 - 1150$ | Crushed stone/gravel skeleton resisting compressive loads |
| **Fine Aggregate** | `fine_agg` | $\text{kg/m}^3$ | $550 - 1000$ | Natural or manufactured sand filling voids between coarse particles |
| **Curing Age** | `age` | Days | $1 - 365$ | Curing duration in moist conditions prior to destructive test |

---

## 📊 Model Benchmarks & Comparison

Both models were cross-validated on unseen test specimens with the following performance metrics:

| Model Architecture | Mean Absolute Error (MAE) | Mean Squared Error (MSE) | Coefficient of Determination ($R^2$) | Primary Advantage |
| :--- | :---: | :---: | :---: | :--- |
| **Ensemble Voting Regressor** ⭐ | **$3.32\text{ MPa}$** | **$25.74\text{ MPa}^2$** | **$0.9125$ ($91.2\%$)** | Exceptional generalization, zero variance drift |
| **Deep Neural Network (ANN)** | $4.09\text{ MPa}$ | $47.43\text{ MPa}^2$ | $0.8159$ ($81.6\%$) | Non-linear feature representations |

> **Recommendation**: The **Ensemble Voting Regressor** is the primary production model, combining decision trees, Random Forest, AdaBoost, Gradient Boosting, XGBoost, Lasso, and Linear Regression to minimize residual error.

---

## 🚀 Quickstart Guide

### 1. Clone the Repository
```bash
git clone https://github.com/Akhassan12/Concrete-Strength-Prediction.git
cd Concrete-Strength-Prediction
```

### 2. Set Up a Virtual Environment (Recommended)
```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch the Web Application
```bash
python app.py
```
Open your browser and navigate to:
```
http://127.0.0.1:5000
```

---

## 🌐 REST API Documentation

The platform includes a programmatic REST API endpoint for integration into external pipelines.

### Endpoint: `POST /api/predict`

#### Request Headers:
```http
Content-Type: application/json
```

#### Request Body (JSON Example):
```json
{
  "cement": 380.0,
  "slag": 95.0,
  "fly_ash": 0.0,
  "water": 180.0,
  "superplasticizer": 8.0,
  "coarse_agg": 1050.0,
  "fine_agg": 760.0,
  "age": 28
}
```

#### Response (JSON):
```json
{
  "status": "success",
  "prediction": {
    "regression_strength_mpa": 52.48,
    "neural_network_strength_mpa": 50.12,
    "concrete_grade": "C40/50 - High Strength Concrete",
    "water_cement_ratio": 0.47,
    "total_mix_density_kg_m3": 2473.0
  }
}
```

---

## 📂 Project Directory Structure

```plaintext
Concrete-Strength-Prediction/
├── Model/
│   ├── Regression.pkl           # Trained Ensemble Voting Regressor
│   ├── NeuralNetwork.pkl        # Trained Deep Neural Network
│   └── trained_model.pkl        # Serialized pipeline artifact
├── static/
│   ├── css/
│   │   └── style.css            # Premium glassmorphic design system
│   └── js/
│       └── app.js               # Reactive sliders, presets, and AJAX controller
├── templates/
│   └── index.html               # Responsive HTML5 dashboard
├── app.py                       # Production Flask server & REST API
├── requirements.txt             # Verified project dependencies
├── LICENSE                      # MIT License (Ali Hassan Kadri)
└── README.md                    # Project documentation
```

---

## 📜 License

This project is open-source software licensed under the **[MIT License](LICENSE)**.

```
Copyright (c) 2026 Ali Hassan Kadri
```

---

## 👤 Author & Ownership

**Ali Hassan Kadri**  
- **GitHub**: [@Akhassan12](https://github.com/Akhassan12)  
- **Email**: [hasanqadri1990@gmail.com](mailto:hasanqadri1990@gmail.com)

*All rights, architecture designs, and maintenance are strictly owned by Ali Hassan Kadri.*
