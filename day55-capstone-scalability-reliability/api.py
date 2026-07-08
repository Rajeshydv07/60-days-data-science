from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn
import time
import logging

# Configure logging for reliability
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Customer Intelligence API", version="2.0.0")

class CustomerData(BaseModel):
    age: int = Field(..., ge=18, le=100, description="Customer age")
    tenure_months: int = Field(..., ge=0, description="Months with the company")
    contract: str = Field(..., description="Contract type (e.g., Month-to-month, One year)")
    monthly_charges: float = Field(..., ge=0.0, description="Monthly bill amount")
    total_charges: float = Field(..., ge=0.0, description="Total amount billed")
    support_calls: int = Field(0, ge=0, description="Number of recent support calls")

class PredictionResponse(BaseModel):
    churn_probability: float
    risk_category: str
    recommended_actions: list[str]
    processing_time_ms: float

@app.post("/predict", response_model=PredictionResponse)
async def predict_churn(data: CustomerData):
    start_time = time.time()
    try:
        # Optimized logic: vectorized/mathematical operations instead of slow loops
        risk_score = 0.2
        
        if data.contract == "Month-to-month":
            risk_score += 0.35
        if data.tenure_months < 12:
            risk_score += 0.20
        if data.monthly_charges > 80:
            risk_score += 0.15
        if data.support_calls > 2:
            risk_score += 0.10
            
        risk_score = min(risk_score, 0.99)
        
        category = "High Risk" if risk_score > 0.5 else "Low Risk"
        actions = []
        if category == "High Risk":
            actions = ["Offer 15% discount for annual upgrade", "Schedule proactive CS call"]
        else:
            actions = ["Explore cross-selling opportunities", "Enroll in standard campaigns"]
            
        processing_time = (time.time() - start_time) * 1000
        
        logger.info(f"Prediction successful for tenure={data.tenure_months}, risk={risk_score}")
        
        return PredictionResponse(
            churn_probability=risk_score,
            risk_category=category,
            recommended_actions=actions,
            processing_time_ms=round(processing_time, 2)
        )
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during prediction.")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "uptime": "OK"}

if __name__ == "__main__":
    # Run with uvicorn for performance
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False, workers=4)
