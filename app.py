"""
Supply Chain Forecasting FastAPI Application
Endpoints for XGBoost, LightGBM, and LSTM models
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import lightgbm as lgb
import tensorflow as tf
from datetime import datetime, timedelta
import json
import os

# ====================================================================
# CONFIGURATION
# ====================================================================

MODEL_PATH = "models"  # Update this path as needed

app = FastAPI(
    title="Supply Chain Forecasting API",
    description="Advanced ML/DL forecasting for sales prediction",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ====================================================================
# GLOBAL MODEL VARIABLES
# ====================================================================

xgb_model = None
lgb_model = None
lstm_model = None
scaler = None
encoders = None
metadata = None


# ====================================================================
# PYDANTIC MODELS
# ====================================================================

class ForecastRequest(BaseModel):
    """Request body for single prediction"""
    branchcode: str = Field(..., description="Branch identifier")
    materialcode: str = Field(..., description="SKU/Material code")
    date: str = Field(..., description="Date in YYYY-MM-DD format")
    stock_on_hand: float = Field(..., ge=0, description="Current stock quantity")
    intransit_qty: float = Field(0, ge=0, description="Quantity in transit")
    pending_po_qty: float = Field(0, ge=0, description="Pending purchase order quantity")
    lead_time_days: float = Field(7, ge=0, description="Lead time in days")
    stockout_flag: int = Field(0, ge=0, le=1, description="Stockout indicator (0 or 1)")
    historical_sales: Optional[List[float]] = Field(None, description="Last 30 days sales for LSTM")


class BatchForecastRequest(BaseModel):
    """Request body for batch predictions"""
    data: List[ForecastRequest]


class ForecastResponse(BaseModel):
    """Response for single prediction"""
    branchcode: str
    materialcode: str
    date: str
    predicted_sales: float
    model_used: str
    confidence_interval: Optional[Dict[str, float]] = None


class BatchForecastResponse(BaseModel):
    """Response for batch predictions"""
    predictions: List[ForecastResponse]
    total_predictions: int
    processing_time_seconds: float


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    models_loaded: Dict[str, bool]
    metadata: Dict


# ====================================================================
# MODEL LOADING
# ====================================================================

@app.on_event("startup")
async def load_models():
    """Load all models on startup"""
    global xgb_model, lgb_model, lstm_model, scaler, encoders, metadata

    try:
        print(f"📂 Loading models from: {MODEL_PATH}")

        # Load XGBoost
        xgb_model = xgb.Booster()
        xgb_model.load_model(f"{MODEL_PATH}/xgboost_model.json")
        print("✅ XGBoost loaded")

        # Load LightGBM
        lgb_model = joblib.load(f"{MODEL_PATH}/lightgbm_model.pkl")
        print("✅ LightGBM loaded")

        # Load LSTM
        lstm_model = tf.keras.models.load_model(f"{MODEL_PATH}/lstm_model.keras")
        print("✅ LSTM loaded")

        # Load preprocessing objects
        scaler = joblib.load(f"{MODEL_PATH}/scaler.pkl")
        encoders = joblib.load(f"{MODEL_PATH}/encoders.pkl")
        print("✅ Preprocessors loaded")

        # Load metadata
        with open(f"{MODEL_PATH}/model_metadata.json", "r") as f:
            metadata = json.load(f)
        print("✅ Metadata loaded")

        print("🚀 All models loaded successfully!")

    except Exception as e:
        print(f"❌ Error loading models: {e}")
        raise


# ====================================================================
# FEATURE ENGINEERING
# ====================================================================

def create_features_from_input(data: ForecastRequest) -> pd.DataFrame:
    """
    Create features from single input request
    Mimics the feature engineering from training
    """

    # Parse date
    date = pd.to_datetime(data.date)

    # Create base dataframe with ALL required fields
    df = pd.DataFrame([{
        'date': date,
        'branchcode': data.branchcode,
        'materialcode': data.materialcode,
        'stock_on_hand': data.stock_on_hand,
        'intransit_qty': data.intransit_qty,
        'pending_po_qty': data.pending_po_qty,
        'lead_time_days': data.lead_time_days,
        'stockout_flag': data.stockout_flag,
        'sales_qty': 0  # Placeholder, will be predicted
    }])

    # Time-based features
    df['year'] = date.year
    df['month'] = date.month
    df['day'] = date.day
    df['dayofweek'] = date.dayofweek
    df['quarter'] = date.quarter
    df['weekofyear'] = date.isocalendar().week
    df['is_weekend'] = int(date.dayofweek >= 5)
    df['is_month_start'] = int(date.is_month_start)
    df['is_month_end'] = int(date.is_month_end)

    # Cyclical encoding
    df['month_sin'] = np.sin(2 * np.pi * date.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * date.month / 12)
    df['day_sin'] = np.sin(2 * np.pi * date.day / 31)
    df['day_cos'] = np.cos(2 * np.pi * date.day / 31)
    df['dayofweek_sin'] = np.sin(2 * np.pi * date.dayofweek / 7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * date.dayofweek / 7)

    # Encode categoricals
    try:
        df['branchcode_encoded'] = encoders['branch_encoder'].transform([data.branchcode])[0]
        df['materialcode_encoded'] = encoders['material_encoder'].transform([data.materialcode])[0]
    except:
        # Unknown category - use default encoding
        df['branchcode_encoded'] = 0
        df['materialcode_encoded'] = 0

    # Historical features (use provided or defaults)
    if data.historical_sales and len(data.historical_sales) >= 30:
        sales_hist = data.historical_sales[-30:]
        df['sales_qty_lag_1'] = sales_hist[-1]
        df['sales_qty_lag_3'] = sales_hist[-3]
        df['sales_qty_lag_7'] = sales_hist[-7]
        df['sales_qty_lag_14'] = sales_hist[-14]
        df['sales_qty_lag_30'] = sales_hist[-30]
        df['sales_qty_rolling_mean_7'] = np.mean(sales_hist[-7:])
        df['sales_qty_rolling_mean_14'] = np.mean(sales_hist[-14:])
        df['sales_qty_rolling_mean_30'] = np.mean(sales_hist)
        df['sales_qty_rolling_std_7'] = np.std(sales_hist[-7:])
        df['sales_qty_rolling_std_14'] = np.std(sales_hist[-14:])
        df['sales_qty_rolling_std_30'] = np.std(sales_hist)
    else:
        # Default values if no history
        for lag in [1, 3, 7, 14, 30]:
            df[f'sales_qty_lag_{lag}'] = 0
        for window in [7, 14, 30]:
            df[f'sales_qty_rolling_mean_{window}'] = 0
            df[f'sales_qty_rolling_std_{window}'] = 0

    # Stock features
    df['stock_on_hand_lag_1'] = data.stock_on_hand
    df['stock_on_hand_lag_3'] = data.stock_on_hand
    df['stock_on_hand_lag_7'] = data.stock_on_hand
    df['stock_on_hand_lag_14'] = data.stock_on_hand
    df['stock_on_hand_lag_30'] = data.stock_on_hand

    for window in [7, 14, 30]:
        df[f'stock_on_hand_rolling_mean_{window}'] = data.stock_on_hand
        df[f'stock_on_hand_rolling_std_{window}'] = 0

    # Interaction features
    df['stock_to_sales_ratio'] = data.stock_on_hand / (df['sales_qty_rolling_mean_7'] + 1)
    df['inventory_coverage_days'] = data.stock_on_hand / (df['sales_qty_rolling_mean_7'] + 1)
    df['sales_velocity_7d'] = 0
    df['stock_turnover'] = 0
    df['days_since_stockout'] = 0 if data.stockout_flag == 0 else 1

    return df


# ====================================================================
# PREDICTION FUNCTIONS
# ====================================================================

def predict_xgboost(features: pd.DataFrame) -> float:
    """Predict using XGBoost model"""
    feature_cols = metadata['feature_columns']

    # Ensure all required features are present
    for col in feature_cols:
        if col not in features.columns:
            features[col] = 0

    X = features[feature_cols]
    dmatrix = xgb.DMatrix(X)
    prediction = xgb_model.predict(dmatrix)[0]
    return max(0, float(prediction))  # Ensure non-negative


def predict_lightgbm(features: pd.DataFrame) -> float:
    """Predict using LightGBM model"""
    feature_cols = metadata['feature_columns']

    # Ensure all required features are present
    for col in feature_cols:
        if col not in features.columns:
            features[col] = 0

    X = features[feature_cols]
    prediction = lgb_model.predict(X)[0]
    return max(0, float(prediction))


def predict_lstm(sales_history: List[float]) -> float:
    """Predict using LSTM model"""
    if len(sales_history) < 30:
        raise ValueError("LSTM requires at least 30 days of historical sales data")

    # Prepare sequence
    sequence = np.array(sales_history[-30:]).reshape(-1, 1)
    sequence_scaled = scaler.transform(sequence)
    X = sequence_scaled.reshape(1, 30, 1)

    # Predict
    prediction_scaled = lstm_model.predict(X, verbose=0)[0][0]
    prediction = scaler.inverse_transform([[prediction_scaled]])[0][0]

    return max(0, float(prediction))


# ====================================================================
# API ENDPOINTS
# ====================================================================

@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": {
            "xgboost": xgb_model is not None,
            "lightgbm": lgb_model is not None,
            "lstm": lstm_model is not None
        },
        "metadata": metadata if metadata else {}
    }


@app.post("/predict/xgboost", response_model=ForecastResponse)
async def predict_sales_xgboost(request: ForecastRequest):
    """
    Predict sales using XGBoost model
    Fast and accurate for most use cases
    """
    try:
        features = create_features_from_input(request)
        prediction = predict_xgboost(features)

        return ForecastResponse(
            branchcode=request.branchcode,
            materialcode=request.materialcode,
            date=request.date,
            predicted_sales=round(prediction, 2),
            model_used="XGBoost"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/lightgbm", response_model=ForecastResponse)
async def predict_sales_lightgbm(request: ForecastRequest):
    """
    Predict sales using LightGBM model
    Alternative tree-based model
    """
    try:
        features = create_features_from_input(request)
        prediction = predict_lightgbm(features)

        return ForecastResponse(
            branchcode=request.branchcode,
            materialcode=request.materialcode,
            date=request.date,
            predicted_sales=round(prediction, 2),
            model_used="LightGBM"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/lstm", response_model=ForecastResponse)
async def predict_sales_lstm(request: ForecastRequest):
    """
    Predict sales using LSTM deep learning model
    Requires 30 days of historical sales data
    Best for capturing complex temporal patterns
    """
    try:
        if not request.historical_sales or len(request.historical_sales) < 30:
            raise HTTPException(
                status_code=400,
                detail="LSTM requires at least 30 days of historical sales data"
            )

        prediction = predict_lstm(request.historical_sales)

        return ForecastResponse(
            branchcode=request.branchcode,
            materialcode=request.materialcode,
            date=request.date,
            predicted_sales=round(prediction, 2),
            model_used="LSTM"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/ensemble", response_model=ForecastResponse)
async def predict_sales_ensemble(request: ForecastRequest):
    """
    Ensemble prediction combining XGBoost and LightGBM
    More robust than single model
    """
    try:
        features = create_features_from_input(request)

        xgb_pred = predict_xgboost(features)
        lgb_pred = predict_lightgbm(features)

        # Weighted average (can be tuned)
        ensemble_pred = 0.5 * xgb_pred + 0.5 * lgb_pred

        return ForecastResponse(
            branchcode=request.branchcode,
            materialcode=request.materialcode,
            date=request.date,
            predicted_sales=round(ensemble_pred, 2),
            model_used="Ensemble (XGBoost + LightGBM)",
            confidence_interval={
                "lower": round(min(xgb_pred, lgb_pred), 2),
                "upper": round(max(xgb_pred, lgb_pred), 2)
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/batch", response_model=BatchForecastResponse)
async def predict_sales_batch(
        request: BatchForecastRequest,
        model: str = Query("xgboost", regex="^(xgboost|lightgbm|ensemble)$")
):
    """
    Batch prediction for multiple records
    Specify model: xgboost, lightgbm, or ensemble
    """
    import time
    start_time = time.time()

    try:
        predictions = []

        for item in request.data:
            features = create_features_from_input(item)

            if model == "xgboost":
                pred = predict_xgboost(features)
                model_name = "XGBoost"
            elif model == "lightgbm":
                pred = predict_lightgbm(features)
                model_name = "LightGBM"
            else:  # ensemble
                xgb_pred = predict_xgboost(features)
                lgb_pred = predict_lightgbm(features)
                pred = 0.5 * xgb_pred + 0.5 * lgb_pred
                model_name = "Ensemble"

            predictions.append(ForecastResponse(
                branchcode=item.branchcode,
                materialcode=item.materialcode,
                date=item.date,
                predicted_sales=round(pred, 2),
                model_used=model_name
            ))

        processing_time = time.time() - start_time

        return BatchForecastResponse(
            predictions=predictions,
            total_predictions=len(predictions),
            processing_time_seconds=round(processing_time, 3)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


@app.get("/models/info")
async def get_models_info():
    """Get information about loaded models"""
    return {
        "metadata": metadata,
        "models": {
            "xgboost": {
                "loaded": xgb_model is not None,
                "type": "Gradient Boosting"
            },
            "lightgbm": {
                "loaded": lgb_model is not None,
                "type": "Gradient Boosting"
            },
            "lstm": {
                "loaded": lstm_model is not None,
                "type": "Deep Learning (Bidirectional LSTM)"
            }
        }
    }


# ====================================================================
# MAIN
# ====================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)