# 🚀 Sales Forecasting API & Model Serving

This service provides a RESTful API for forecasting sales quantities using machine learning. It serves predictions via **XGBoost**, **LightGBM**, and **Bi-directional LSTM** models, including an **Ensemble** capability for robust decision-making.

## 📋 Table of Contents
- [System Status](#-system-status)
- [Model Information](#-model-information)
- [API Endpoints & Examples](#-api-endpoints--examples)
  - [Single Predictions](#single-predictions)
  - [Ensemble Predictions](#ensemble-predictions)
  - [Batch Predictions](#batch-predictions)
- [Error Handling](#-error-handling)
- [Usage Guide (cURL)](#-usage-guide)

---

## 🏥 System Status

**Endpoint:** `GET /`

Returns the health status of the API, loaded models, and feature metadata.

**Response (200 OK):**
```json
{
  "status": "healthy",
  "models_loaded": {
    "xgboost": true,
    "lightgbm": true,
    "lstm": true
  },
  "metadata": {
    "num_features": 49,
    "target_variable": "sales_qty",
    "lstm_sequence_length": 30,
    "created_at": "2025-12-04 18:20:24",
    "model_versions": {
      "xgboost": "3.1.2",
      "lightgbm": "4.6.0",
      "tensorflow": "2.19.0"
    }
  }
}
```

---

## ℹ️ Model Information

**Endpoint:** `GET /info` (or similar metadata endpoint)

Provides detailed information about the feature engineering, model types, and specific versions running.

<details>
<summary><strong>View Feature Columns (49 Features)</strong></summary>

*   **Core:** `stock_on_hand`, `intransit_qty`, `pending_po_qty`, `lead_time_days`, `stockout_flag`
*   **Temporal:** `year`, `month`, `day`, `dayofweek`, `quarter`, `weekofyear`, `is_weekend`
*   **Cyclical:** `month_sin/cos`, `day_sin/cos`, `dayofweek_sin/cos`
*   **Encoded:** `branchcode_encoded`, `materialcode_encoded`
*   **Lags:** `sales_qty_lag_[1,3,7,14,30]`, `stock_on_hand_lag_[1,3,7,14,30]`
*   **Rolling Stats:** Mean/Std for 7, 14, 30 days (Sales & Stock)
*   **Ratios:** `stock_to_sales_ratio`, `inventory_coverage_days`, `sales_velocity_7d`, `stock_turnover`, `days_since_stockout`

</details>

**Response (200 OK):**
```json
{
  "models": {
    "xgboost": {
      "loaded": true,
      "type": "Gradient Boosting"
    },
    "lightgbm": {
      "loaded": true,
      "type": "Gradient Boosting"
    },
    "lstm": {
      "loaded": true,
      "type": "Deep Learning (Bidirectional LSTM)"
    }
  }
}
```

---

## 🎯 API Endpoints & Examples

### Single Predictions

#### XGBoost Prediction
**Endpoint:** `POST /predict/xgboost`

```json
{
  "branchcode": "BR001",
  "materialcode": "SKU12345",
  "date": "2025-01-15",
  "predicted_sales": 10.33,
  "model_used": "XGBoost",
  "confidence_interval": null
}
```

#### LightGBM Prediction
**Endpoint:** `POST /predict/lightgbm`

```json
{
  "branchcode": "BR002",
  "materialcode": "SKU67890",
  "date": "2025-01-20",
  "predicted_sales": 0.0,
  "model_used": "LightGBM",
  "confidence_interval": null
}
```

#### 🧠 LSTM Prediction
**Endpoint:** `POST /predict/lstm`

```json
{
  "branchcode": "BR003",
  "materialcode": "SKU11111",
  "date": "2025-01-25",
  "predicted_sales": 15.49,
  "model_used": "LSTM",
  "confidence_interval": null
}
```

### 🤝 Ensemble Predictions
**Endpoint:** `POST /predict/ensemble`

Combines weighted predictions from tree-based models and deep learning models to provide a result with confidence intervals.

```json
{
  "branchcode": "BR004",
  "materialcode": "SKU22222",
  "date": "2025-02-01",
  "predicted_sales": 8.52,
  "model_used": "Ensemble (XGBoost + LightGBM)",
  "confidence_interval": {
    "lower": 5.16,
    "upper": 11.88
  }
}
```

### 📦 Batch Predictions
**Endpoint:** `POST /predict/batch?model=xgboost`

Process multiple SKU/Branch combinations in a single request.

```json
{
  "predictions": [
    {
      "branchcode": "BR001",
      "materialcode": "SKU001",
      "date": "2025-01-15",
      "predicted_sales": 7.45
    },
    {
      "branchcode": "BR002",
      "materialcode": "SKU002",
      "date": "2025-01-15",
      "predicted_sales": 3.25
    },
    {
      "branchcode": "BR003",
      "materialcode": "SKU003",
      "date": "2025-01-15",
      "predicted_sales": 7.39
    }
  ],
  "total_predictions": 3,
  "processing_time_seconds": 0.052
}
```

---

## ⚠️ Error Handling

The API validates input constraints. For example, deep learning models require historical context.

**Scenario:** LSTM request without sufficient history.
**Status:** `400 Bad Request`

```json
{
  "detail": "LSTM requires at least 30 days of historical sales data"
}
```

---

## 🔧 Usage Guide

Quick start examples using `curl`.

### 1. Check Health
```bash
curl -X GET http://localhost:8000/
```

### 2. Get XGBoost Prediction
*Requires historical sales array for feature generation.*
```bash
curl -X POST http://localhost:8000/predict/xgboost \
  -H "Content-Type: application/json" \
  -d '{
    "branchcode": "BR001",
    "materialcode": "SKU12345",
    "date": "2025-01-15",
    "stock_on_hand": 500,
    "stockout_flag": 0,
    "historical_sales": [45, 52, 48, 55, 60, 58, 62, 65, 70, 68, 72, 75, 78, 80, 82, 85, 88, 90, 92, 95, 98, 100, 102, 105, 108, 110, 112, 115, 118, 120]
  }'
```

### 3. Get Ensemble Prediction
```bash
curl -X POST http://localhost:8000/predict/ensemble \
  -H "Content-Type: application/json" \
  -d '{
    "branchcode": "BR001",
    "materialcode": "SKU12345",
    "date": "2025-01-15",
    "stock_on_hand": 500,
    "stockout_flag": 0
  }'
```

### 4. Run Batch Prediction
```bash
curl -X POST "http://localhost:8000/predict/batch?model=xgboost" \
  -H "Content-Type: application/json" \
  -d '{
    "data": [
      {
        "branchcode": "BR001",
        "materialcode": "SKU001",
        "date": "2025-01-15",
        "stock_on_hand": 500,
        "stockout_flag": 0
      }
    ]
  }'
```