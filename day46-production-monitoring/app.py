from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import uvicorn
import os
import logging
import json
from datetime import datetime

# Setup Logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename='logs/api_monitoring.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize app
app = FastAPI(
    title="Customer Churn Prediction API - Monitored",
    description="API to predict the likelihood of customer churn with production monitoring.",
    version="1.1.0"
)

# Load Models
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
try:
    model = joblib.load(os.path.join(MODEL_DIR, "churn_rf_model.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
    logger.info("Models loaded successfully.")
except Exception as e:
    logger.error(f"Error loading models: {e}")
    print(f"Error loading models: {e}")
    print("Please ensure you have run train.py first.")

# Input Validation with Pydantic
class CustomerData(BaseModel):
    age: int = Field(..., ge=18, le=120, description="Customer age must be between 18 and 120")
    tenure_months: int = Field(..., ge=0, description="Tenure in months cannot be negative")
    monthly_charges: float = Field(..., ge=0, description="Monthly charges cannot be negative")
    total_charges: float = Field(..., ge=0, description="Total charges cannot be negative")
    contract_type: str = Field(..., description="Contract type (e.g., Month-to-month, One year, Two year)")
    support_tickets: int = Field(..., ge=0, description="Number of support tickets cannot be negative")
    days_since_last_login: int = Field(..., ge=0, description="Days since last login cannot be negative")
    satisfaction_score: float = Field(..., ge=1, le=5, description="Satisfaction score must be between 1 and 5")

class PredictionResponse(BaseModel):
    churn_probability: float
    risk_category: str
    prediction: int
    request_id: str

@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Incoming request: {request.method} {request.url}")
    try:
        response = await call_next(request)
        logger.info(f"Request completed with status code {response.status_code}")
        return response
    except Exception as e:
        logger.error(f"Request failed: {str(e)}", exc_info=True)
        raise

@app.get("/")
def home():
    logger.info("Home endpoint accessed")
    return {"message": "Customer Churn Prediction API with Monitoring is up and running!"}

@app.post("/predict", response_model=PredictionResponse)
def predict_churn(customer: CustomerData):
    request_id = datetime.now().strftime("%Y%m%d%H%M%S%f")
    logger.info(f"Prediction requested [ID: {request_id}] - Input: {customer.dict()}")
    
    try:
        # Create a DataFrame from input data
        data = pd.DataFrame([customer.dict()])
        
        # Preprocessing
        if data['contract_type'][0] not in label_encoder.classes_:
            logger.warning(f"Unknown contract type received: {data['contract_type'][0]}")
            raise ValueError(f"Invalid contract_type: {data['contract_type'][0]}. Allowed: {list(label_encoder.classes_)}")
        
        data['contract_type_encoded'] = label_encoder.transform(data['contract_type'])
        
        # Select and order features
        features = [
            'age', 'tenure_months', 'monthly_charges', 'total_charges', 
            'contract_type_encoded', 'support_tickets', 
            'days_since_last_login', 'satisfaction_score'
        ]
        X = data[features]
        
        # Scale features
        X_scaled = scaler.transform(X)
        
        # Predict
        churn_prob = model.predict_proba(X_scaled)[0, 1]
        churn_pred = int(model.predict(X_scaled)[0])
        
        # Categorize risk
        if churn_prob <= 0.25:
            risk = "Low"
        elif churn_prob <= 0.5:
            risk = "Medium"
        elif churn_prob <= 0.75:
            risk = "High"
        else:
            risk = "Critical"
            
        response_data = PredictionResponse(
            churn_probability=float(churn_prob),
            risk_category=risk,
            prediction=churn_pred,
            request_id=request_id
        )
        
        logger.info(f"Prediction successful [ID: {request_id}] - Output: {response_data.dict()}")
        
        # Log to a separate tracking file
        with open("logs/prediction_tracking.jsonl", "a") as f:
            tracking_record = {
                "timestamp": datetime.now().isoformat(),
                "request_id": request_id,
                "input": customer.dict(),
                "output": response_data.dict()
            }
            f.write(json.dumps(tracking_record) + "\n")
            
        return response_data
        
    except ValueError as ve:
        logger.error(f"Validation error during prediction [ID: {request_id}]: {str(ve)}")
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        logger.error(f"Prediction exception [ID: {request_id}]: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during prediction.")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
