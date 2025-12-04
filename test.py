"""
FastAPI Usage Examples and Testing Scripts
Run these after starting the API server
"""

import requests
import json
from datetime import datetime, timedelta

# ====================================================================
# CONFIGURATION
# ====================================================================

BASE_URL = "http://localhost:8000"  # Change if deployed elsewhere

# ====================================================================
# EXAMPLE 1: HEALTH CHECK
# ====================================================================

def test_health_check():
    """Test if API is running"""
    print("\n" + "="*70)
    print("🏥 HEALTH CHECK")
    print("="*70)

    response = requests.get(f"{BASE_URL}/")
    print(f"Status Code: {response.status_code}")
    print(json.dumps(response.json(), indent=2))

# ====================================================================
# EXAMPLE 2: SINGLE PREDICTION - XGBOOST
# ====================================================================

def test_xgboost_prediction():
    """Test XGBoost prediction"""
    print("\n" + "="*70)
    print("🎯 XGBOOST PREDICTION")
    print("="*70)

    payload = {
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

    response = requests.post(f"{BASE_URL}/predict/xgboost", json=payload)
    print(f"Status Code: {response.status_code}")
    print(json.dumps(response.json(), indent=2))

# ====================================================================
# EXAMPLE 3: SINGLE PREDICTION - LIGHTGBM
# ====================================================================

def test_lightgbm_prediction():
    """Test LightGBM prediction"""
    print("\n" + "="*70)
    print("🎯 LIGHTGBM PREDICTION")
    print("="*70)

    payload = {
        "branchcode": "BR002",
        "materialcode": "SKU67890",
        "date": "2025-01-20",
        "stock_on_hand": 300,
        "stockout_flag": 0,
        "historical_sales": [30, 35, 32, 38, 40, 42, 45, 48, 50, 52,
                            55, 58, 60, 62, 65, 68, 70, 72, 75, 78,
                            80, 82, 85, 88, 90, 92, 95, 98, 100, 102]
    }

    response = requests.post(f"{BASE_URL}/predict/lightgbm", json=payload)
    print(f"Status Code: {response.status_code}")
    print(json.dumps(response.json(), indent=2))

# ====================================================================
# EXAMPLE 4: LSTM PREDICTION
# ====================================================================

def test_lstm_prediction():
    """Test LSTM prediction with historical data"""
    print("\n" + "="*70)
    print("🧠 LSTM PREDICTION")
    print("="*70)

    payload = {
        "branchcode": "BR003",
        "materialcode": "SKU11111",
        "date": "2025-01-25",
        "stock_on_hand": 450,
        "stockout_flag": 0,
        "historical_sales": [50, 55, 52, 58, 60, 62, 65, 68, 70, 72,
                            75, 78, 80, 82, 85, 88, 90, 92, 95, 98,
                            100, 102, 105, 108, 110, 112, 115, 118, 120, 122]
    }

    response = requests.post(f"{BASE_URL}/predict/lstm", json=payload)
    print(f"Status Code: {response.status_code}")
    print(json.dumps(response.json(), indent=2))

# ====================================================================
# EXAMPLE 5: ENSEMBLE PREDICTION
# ====================================================================

def test_ensemble_prediction():
    """Test ensemble prediction"""
    print("\n" + "="*70)
    print("🤝 ENSEMBLE PREDICTION")
    print("="*70)

    payload = {
        "branchcode": "BR004",
        "materialcode": "SKU22222",
        "date": "2025-02-01",
        "stock_on_hand": 600,
        "stockout_flag": 0,
        "historical_sales": [60, 65, 62, 68, 70, 72, 75, 78, 80, 82,
                            85, 88, 90, 92, 95, 98, 100, 102, 105, 108,
                            110, 112, 115, 118, 120, 122, 125, 128, 130, 132]
    }

    response = requests.post(f"{BASE_URL}/predict/ensemble", json=payload)
    print(f"Status Code: {response.status_code}")
    print(json.dumps(response.json(), indent=2))

# ====================================================================
# EXAMPLE 6: BATCH PREDICTION
# ====================================================================

def test_batch_prediction():
    """Test batch prediction with multiple records"""
    print("\n" + "="*70)
    print("📦 BATCH PREDICTION")
    print("="*70)

    payload = {
        "data": [
            {
                "branchcode": "BR001",
                "materialcode": "SKU001",
                "date": "2025-01-15",
                "stock_on_hand": 500,
                "stockout_flag": 0,
                "historical_sales": [45]*30
            },
            {
                "branchcode": "BR002",
                "materialcode": "SKU002",
                "date": "2025-01-15",
                "stock_on_hand": 300,
                "stockout_flag": 0,
                "historical_sales": [30]*30
            },
            {
                "branchcode": "BR003",
                "materialcode": "SKU003",
                "date": "2025-01-15",
                "stock_on_hand": 450,
                "stockout_flag": 1,
                "historical_sales": [50]*30
            }
        ]
    }

    # Test with XGBoost
    response = requests.post(f"{BASE_URL}/predict/batch?model=xgboost", json=payload)
    print(f"Status Code: {response.status_code}")
    print(json.dumps(response.json(), indent=2))

# ====================================================================
# EXAMPLE 7: GET MODEL INFO
# ====================================================================

def test_model_info():
    """Get information about loaded models"""
    print("\n" + "="*70)
    print("ℹ️  MODEL INFORMATION")
    print("="*70)

    response = requests.get(f"{BASE_URL}/models/info")
    print(f"Status Code: {response.status_code}")
    print(json.dumps(response.json(), indent=2))

# ====================================================================
# EXAMPLE 8: ERROR HANDLING - MISSING LSTM DATA
# ====================================================================

def test_lstm_error():
    """Test LSTM error handling when historical data is missing"""
    print("\n" + "="*70)
    print("⚠️  ERROR HANDLING - LSTM WITHOUT HISTORY")
    print("="*70)

    payload = {
        "branchcode": "BR005",
        "materialcode": "SKU33333",
        "date": "2025-02-01",
        "stock_on_hand": 600,
        "stockout_flag": 0
        # No historical_sales provided
    }

    response = requests.post(f"{BASE_URL}/predict/lstm", json=payload)
    print(f"Status Code: {response.status_code}")
    print(json.dumps(response.json(), indent=2))

# ====================================================================
# PERFORMANCE TEST
# ====================================================================

def performance_test(num_requests=100):
    """Test API performance"""
    import time

    print("\n" + "="*70)
    print(f"⚡ PERFORMANCE TEST - {num_requests} Requests")
    print("="*70)

    payload = {
        "branchcode": "BR001",
        "materialcode": "SKU12345",
        "date": "2025-01-15",
        "stock_on_hand": 500,
        "stockout_flag": 0,
        "historical_sales": list(range(30, 60))
    }

    start_time = time.time()

    for i in range(num_requests):
        response = requests.post(f"{BASE_URL}/predict/xgboost", json=payload)
        if i % 10 == 0:
            print(f"  Completed: {i}/{num_requests}")

    end_time = time.time()
    total_time = end_time - start_time

    print(f"\n📊 Results:")
    print(f"  Total Time: {total_time:.2f} seconds")
    print(f"  Average Time per Request: {(total_time/num_requests)*1000:.2f} ms")
    print(f"  Requests per Second: {num_requests/total_time:.2f}")

# ====================================================================
# CURL EXAMPLES
# ====================================================================

def print_curl_examples():
    """Print curl command examples"""
    print("\n" + "="*70)
    print("🔧 CURL EXAMPLES")
    print("="*70)

    print("""
# Health Check
curl -X GET http://localhost:8000/

# XGBoost Prediction
curl -X POST http://localhost:8000/predict/xgboost \\
  -H "Content-Type: application/json" \\
  -d '{
    "branchcode": "BR001",
    "materialcode": "SKU12345",
    "date": "2025-01-15",
    "stock_on_hand": 500,
    "stockout_flag": 0,
    "historical_sales": [45, 52, 48, 55, 60, 58, 62, 65, 70, 68, 72, 75, 78, 80, 82, 85, 88, 90, 92, 95, 98, 100, 102, 105, 108, 110, 112, 115, 118, 120]
  }'

# Ensemble Prediction
curl -X POST http://localhost:8000/predict/ensemble \\
  -H "Content-Type: application/json" \\
  -d '{
    "branchcode": "BR001",
    "materialcode": "SKU12345",
    "date": "2025-01-15",
    "stock_on_hand": 500,
    "stockout_flag": 0
  }'

# Batch Prediction
curl -X POST "http://localhost:8000/predict/batch?model=xgboost" \\
  -H "Content-Type: application/json" \\
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
    """)

# ====================================================================
# RUN ALL TESTS
# ====================================================================

def run_all_tests():
    """Run all test functions"""
    print("\n" + "="*70)
    print("🚀 RUNNING ALL API TESTS")
    print("="*70)

    try:
        test_health_check()
        test_model_info()
        test_xgboost_prediction()
        test_lightgbm_prediction()
        test_lstm_prediction()
        test_ensemble_prediction()
        test_batch_prediction()
        test_lstm_error()
        print_curl_examples()

        # Uncomment to run performance test
        # performance_test(100)

        print("\n" + "="*70)
        print("✅ ALL TESTS COMPLETED")
        print("="*70)

    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Cannot connect to API")
        print("Make sure the FastAPI server is running:")
        print("  python fastapi_app.py")
        print("  or")
        print("  uvicorn fastapi_app:app --reload")

if __name__ == "__main__":
    run_all_tests()