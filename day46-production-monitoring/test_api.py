import requests
import json

API_URL = "http://127.0.0.1:8000/predict"

def test_prediction():
    # Sample customer data that resembles a high-risk profile
    high_risk_customer = {
        "age": 45,
        "tenure_months": 2,
        "monthly_charges": 105.5,
        "total_charges": 211.0,
        "contract_type": "Month-to-month",
        "support_tickets": 4,
        "days_since_last_login": 20,
        "satisfaction_score": 1.5
    }

    # Sample customer data for a low-risk profile
    low_risk_customer = {
        "age": 32,
        "tenure_months": 48,
        "monthly_charges": 45.0,
        "total_charges": 2160.0,
        "contract_type": "Two year",
        "support_tickets": 0,
        "days_since_last_login": 2,
        "satisfaction_score": 4.8
    }

    print("Testing High Risk Customer Profile...")
    response = requests.post(API_URL, json=high_risk_customer)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")

    print("Testing Low Risk Customer Profile...")
    response = requests.post(API_URL, json=low_risk_customer)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

if __name__ == "__main__":
    try:
        test_prediction()
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the API. Make sure the FastAPI server is running with 'uvicorn app:app --reload'")
