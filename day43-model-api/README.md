# Day 43: Turning Your Customer Intelligence Model into an API

## Overview
This directory contains the deployment phase of the Customer Churn Prediction Model. We have wrapped our machine learning model (RandomForest) into a robust, real-time REST API using **FastAPI**.

## Architecture
1. **Model**: A Scikit-Learn `RandomForestClassifier` trained on customer churn data.
2. **API Framework**: FastAPI, chosen for its high performance and built-in validation.
3. **Data Validation**: Pydantic models ensure all incoming JSON payloads are correctly formatted and strictly typed before reaching the model.
4. **Server**: Uvicorn acts as the ASGI server to handle concurrent requests efficiently.

## Files
- `train.py`: Script to train the model and save the `model.pkl`, `scaler.pkl`, and `label_encoder.pkl`.
- `app.py`: The FastAPI application defining the API endpoints.
- `test_api.py`: A Python script to test the prediction endpoint with mock data.
- `requirements.txt`: Python dependencies for running the API.

## Usage Instructions

### 1. Installation
Ensure you have the required dependencies installed:
```bash
pip install -r requirements.txt
```

### 2. Train the Model
Generate the `.pkl` files by running the training script:
```bash
python train.py
```
*(This will create a `models/` directory containing the serialized model and preprocessors).*

### 3. Run the API Server
Start the FastAPI server via Uvicorn:
```bash
uvicorn app:app --reload
```
*(The API will be available at `http://127.0.0.1:8000`)*

### 4. Interactive Documentation
FastAPI automatically generates interactive API documentation. You can view and test the API directly from your browser:
- **Swagger UI**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

### 5. Test with Sample Data
You can run the included test script to simulate API requests:
```bash
python test_api.py
```

## API Endpoint: `/predict` (POST)
**Payload Example:**
```json
{
    "age": 34,
    "tenure_months": 12,
    "monthly_charges": 89.5,
    "total_charges": 1074.0,
    "contract_type": "Month-to-month",
    "support_tickets": 3,
    "days_since_last_login": 15,
    "satisfaction_score": 2.5
}
```

**Response Example:**
```json
{
    "churn_probability": 0.82,
    "risk_category": "Critical",
    "prediction": 1
}
```
