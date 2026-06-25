from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import uvicorn
import os

# Initialize app
app = FastAPI(
    title="Customer Churn Prediction API",
    description="API to predict the likelihood of customer churn using a RandomForest model.",
    version="1.0.0"
)

# Load Models
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
try:
    model = joblib.load(os.path.join(MODEL_DIR, "churn_rf_model.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
except Exception as e:
    print(f"Error loading models: {e}")
    print("Please ensure you have run train.py first.")

class CustomerData(BaseModel):
    age: int
    tenure_months: int
    monthly_charges: float
    total_charges: float
    contract_type: str
    support_tickets: int
    days_since_last_login: int
    satisfaction_score: float

class PredictionResponse(BaseModel):
    churn_probability: float
    risk_category: str
    prediction: int

@app.get("/")
def home():
    return {"message": "Customer Churn Prediction API is up and running!"}

@app.post("/predict", response_model=PredictionResponse)
def predict_churn(customer: CustomerData):
    try:
        # Create a DataFrame from input data
        data = pd.DataFrame([customer.dict()])
        
        # Preprocessing
        if data['contract_type'][0] not in label_encoder.classes_:
            # Handle unknown classes if necessary, or let it throw an error
            pass
        
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
            
        return PredictionResponse(
            churn_probability=float(churn_prob),
            risk_category=risk,
            prediction=churn_pred
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
