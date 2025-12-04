# 🚀 Supply Chain Forecasting FastAPI Application

## 📋 Overview

A production-ready FastAPI application for supply chain sales forecasting using advanced machine learning models (XGBoost, LightGBM, and LSTM). This API provides real-time predictions for inventory management and demand forecasting.

## 🎯 Features

- **Multiple ML Models**: XGBoost, LightGBM, and Bidirectional LSTM
- **Ensemble Predictions**: Combine multiple models for better accuracy
- **Batch Processing**: Process multiple predictions in a single request
- **Real-time Inference**: Fast predictions (<100ms per request)
- **Automatic Feature Engineering**: 49+ engineered features
- **RESTful API**: Standard HTTP endpoints with JSON payloads
- **Interactive Documentation**: Swagger UI and ReDoc
- **Health Monitoring**: Built-in health check endpoints
- **CORS Enabled**: Ready for web application integration

## 📦 Installation

### Prerequisites

```bash
Python 3.8+
pip or conda
```

### Install Dependencies

```bash
pip install fastapi uvicorn pandas numpy scikit-learn xgboost lightgbm tensorflow joblib
```

Or use requirements.txt:

```bash
pip install -r requirements.txt
```

### Requirements.txt Content

```
fastapi==0.104.1
uvicorn[standard]==0.24.0
pandas==2.1.3
numpy==1.24.3
scikit-learn==1.3.2
xgboost==2.0.3
lightgbm==4.1.0
tensorflow==2.15.0
joblib==1.3.2
pydantic==2.5.0
python-multipart==0.0.6
```

## 🗂️ Project Structure

```
project/
│
├── app.py                      # Main FastAPI application
├── test.py                     # API testing suite
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
└── models/                     # Model files directory
    ├── xgboost_model.json     # XGBoost model
    ├── lightgbm_model.pkl     # LightGBM model
    ├── lstm_model.keras       # LSTM model
    ├── scaler.pkl             # StandardScaler for LSTM
    ├── encoders.pkl           # Label encoders
    └── model_metadata.json    # Model configuration
```

## ⚙️ Configuration

Update the `MODEL_PATH` variable in `app.py`:

```python
MODEL_PATH = "/path/to/your/models"  # Update this
```

## 🚀 Running the Application

### Development Mode

```bash
# Option 1: Using uvicorn with auto-reload
uvicorn app:app --reload --host 0.0.0.0 --port 8000

# Option 2: Direct Python execution
python app.py
```

### Production Mode

```bash
# Using uvicorn with multiple workers
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4

# Using gunicorn with uvicorn workers
gunicorn app:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

## 📚 API Documentation

Once running, access interactive documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

## 🔌 API Endpoints

### 1. Health Check

**GET** `/`

Check API status and model availability.

```bash
curl http://localhost:8000/
```

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": {
    "xgboost": true,
    "lightgbm": true,
    "lstm": true
  },
  "metadata": {...}
}
```

### 2. XGBoost Prediction

**POST** `/predict/xgboost`

Fast gradient boosting predictions.

```bash
curl -X POST http://localhost:8000/predict/xgboost \
  -H "Content-Type: application/json" \
  -d '{
    "branchcode": "BR001",
    "materialcode": "SKU12345",
    "date": "2025-01-15",
    "stock_on_hand": 500,
    "intransit_qty": 100,
    "pending_po_qty": 200,
    "lead_time_days": 7,
    "stockout_flag": 0,
    "historical_sales": [45, 52, 48, ..., 120]
  }'
```

### 3. LightGBM Prediction

**POST** `/predict/lightgbm`

Alternative gradient boosting model.

```bash
curl -X POST http://localhost:8000/predict/lightgbm \
  -H "Content-Type: application/json" \
  -d '{...}'  # Same payload as XGBoost
```

### 4. LSTM Prediction

**POST** `/predict/lstm`

Deep learning model (requires 30 days of historical data).

```bash
curl -X POST http://localhost:8000/predict/lstm \
  -H "Content-Type: application/json" \
  -d '{
    "branchcode": "BR001",
    "materialcode": "SKU12345",
    "date": "2025-01-15",
    "stock_on_hand": 500,
    "intransit_qty": 100,
    "pending_po_qty": 200,
    "lead_time_days": 7,
    "stockout_flag": 0,
    "historical_sales": [45, 52, ..., 120]  # Must have 30+ values
  }'
```

### 5. Ensemble Prediction

**POST** `/predict/ensemble`

Combined XGBoost + LightGBM prediction with confidence interval.

```bash
curl -X POST http://localhost:8000/predict/ensemble \
  -H "Content-Type: application/json" \
  -d '{...}'  # Same payload as XGBoost
```

**Response:**
```json
{
  "branchcode": "BR001",
  "materialcode": "SKU12345",
  "date": "2025-01-15",
  "predicted_sales": 125.45,
  "model_used": "Ensemble (XGBoost + LightGBM)",
  "confidence_interval": {
    "lower": 120.30,
    "upper": 130.60
  }
}
```

### 6. Batch Prediction

**POST** `/predict/batch?model=xgboost`

Process multiple predictions at once.

**Query Parameters:**
- `model`: `xgboost`, `lightgbm`, or `ensemble`

```bash
curl -X POST "http://localhost:8000/predict/batch?model=xgboost" \
  -H "Content-Type: application/json" \
  -d '{
    "data": [
      {...},  # First prediction
      {...},  # Second prediction
      {...}   # Third prediction
    ]
  }'
```

**Response:**
```json
{
  "predictions": [...],
  "total_predictions": 3,
  "processing_time_seconds": 0.145
}
```

### 7. Model Information

**GET** `/models/info`

Get loaded model details and metadata.

```bash
curl http://localhost:8000/models/info
```

## 📝 Request Schema

### Required Fields

| Field | Type | Description | Default |
|-------|------|-------------|---------|
| `branchcode` | string | Branch identifier | - |
| `materialcode` | string | SKU/Material code | - |
| `date` | string | Date (YYYY-MM-DD) | - |
| `stock_on_hand` | float | Current stock | - |
| `intransit_qty` | float | Quantity in transit | 0 |
| `pending_po_qty` | float | Pending PO quantity | 0 |
| `lead_time_days` | float | Lead time (days) | 7 |
| `stockout_flag` | int | Stockout (0 or 1) | 0 |
| `historical_sales` | list[float] | Past sales (30+ for LSTM) | null |

### Example Payload

```json
{
  "branchcode": "BR001",
  "materialcode": "SKU12345",
  "date": "2025-01-15",
  "stock_on_hand": 500,
  "intransit_qty": 100,
  "pending_po_qty": 200,
  "lead_time_days": 7,
  "stockout_flag": 0,
  "historical_sales": [45, 52, 48, 55, 60, 58, 62, 65, 70, 68,
                      72, 75, 78, 80, 82, 85, 88, 90, 92, 95,
                      98, 100, 102, 105, 108, 110, 112, 115, 118, 120]
}
```

## 🧪 Testing

Run the complete test suite:

```bash
python test.py
```

Test individual endpoints:

```python
from test import test_xgboost_prediction
test_xgboost_prediction()
```

## 🐍 Python Client Example

```python
import requests

# Single prediction
response = requests.post(
    "http://localhost:8000/predict/ensemble",
    json={
        "branchcode": "BR001",
        "materialcode": "SKU12345",
        "date": "2025-01-15",
        "stock_on_hand": 500,
        "intransit_qty": 100,
        "pending_po_qty": 200,
        "lead_time_days": 7,
        "stockout_flag": 0,
        "historical_sales": list(range(50, 80))
    }
)

prediction = response.json()
print(f"Predicted Sales: {prediction['predicted_sales']}")
```

## 🎨 Feature Engineering

The API automatically creates **49 features** from your input:

### Time-Based Features (15)
- Year, month, day, day of week, quarter, week of year
- Weekend flag, month start/end flags
- Cyclical encodings (sin/cos for month, day, day of week)

### Lag Features (10)
- Sales lags: 1, 3, 7, 14, 30 days
- Stock lags: 1, 3, 7, 14, 30 days

### Rolling Statistics (12)
- Sales rolling mean/std: 7, 14, 30 day windows
- Stock rolling mean/std: 7, 14, 30 day windows

### Interaction Features (7)
- Stock-to-sales ratio
- Inventory coverage days
- Sales velocity
- Stock turnover
- Days since stockout

### Encoded Categoricals (2)
- Branch code encoding
- Material code encoding

### Base Features (4)
- Stock on hand
- In-transit quantity
- Pending PO quantity
- Lead time days

## ⚡ Performance

- **Single Prediction**: <100ms
- **Batch (100 records)**: 3-5 seconds
- **Throughput**: 200-500 requests/second (with 4 workers)

## 🔒 Error Handling

The API provides detailed error messages:

```json
{
  "detail": "LSTM requires at least 30 days of historical sales data"
}
```

Common HTTP status codes:
- `200`: Success
- `400`: Bad request (missing/invalid data)
- `500`: Server error (model failure)

## 📊 Model Selection Guide

| Use Case | Recommended Model | Reason |
|----------|------------------|---------|
| Real-time production | XGBoost | Fastest inference |
| High accuracy | Ensemble | Best overall performance |
| Complex patterns | LSTM | Captures temporal dependencies |
| Batch processing | LightGBM | Efficient for large batches |

## 🐳 Docker Deployment

### Dockerfile

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY models/ ./models/

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Build and Run

```bash
docker build -t supply-chain-api .
docker run -p 8000:8000 supply-chain-api
```

## 🔧 Troubleshooting

### Models Not Loading

```python
# Check MODEL_PATH is correct
MODEL_PATH = "/correct/path/to/models"

# Verify all model files exist
ls -la /path/to/models/
```

### Missing Features Error

Ensure your request includes all required fields:
- `intransit_qty`
- `pending_po_qty`
- `lead_time_days`

### LSTM Errors

LSTM requires exactly 30+ historical sales values:
```python
"historical_sales": [val1, val2, ..., val30]  # At least 30 values
```

## 📈 Monitoring

Monitor API health:

```bash
# Check status
curl http://localhost:8000/

# Check model info
curl http://localhost:8000/models/info
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details

## 👥 Support

For issues or questions:
- Open a GitHub issue
- Check documentation: http://localhost:8000/docs
- Review test examples in `test.py`

## 🎓 Model Training

To retrain models, see `amulModelTraining-3.ipynb` for the complete training pipeline.

## 📚 Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [TensorFlow Documentation](https://www.tensorflow.org/)

---
