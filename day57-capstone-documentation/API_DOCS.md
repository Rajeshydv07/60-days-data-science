# 🔌 Customer Intelligence API Reference

The Customer Intelligence API is a high-performance RESTful web service built with FastAPI. It provides the core machine learning inference capabilities, allowing frontend dashboards or external CRM systems (like Salesforce or HubSpot) to score customer churn risk in real-time.

## Base URL
When running locally:
`http://localhost:8000`

---

## 1. Health Check Endpoint

Used by load balancers and deployment orchestrators (e.g., Kubernetes) to verify the service is running.

**HTTP Request**
`GET /health`

**Response Example**
```json
{
  "status": "healthy",
  "uptime": "OK"
}
```

---

## 2. Real-Time Churn Prediction Endpoint

This is the primary endpoint for the machine learning inference engine. It accepts a JSON payload containing customer attributes, passes them through the inference logic, and returns a churn probability, risk category, and recommended business actions.

**HTTP Request**
`POST /predict`

### 2.1 Request Schema (JSON)

The endpoint expects a JSON body adhering to the following schema. Strict validation is enforced via Pydantic; missing or invalid types will result in a `422 Unprocessable Entity` response.

| Field | Type | Restrictions | Description |
| :--- | :--- | :--- | :--- |
| `age` | Integer | 18 to 100 | Customer's age in years. |
| `tenure_months` | Integer | >= 0 | Number of months the customer has been with the company. |
| `contract` | String | e.g. "Month-to-month", "One year", "Two year" | Current contract type. |
| `monthly_charges` | Float | >= 0.0 | The customer's current monthly bill. |
| `total_charges` | Float | >= 0.0 | The total amount billed to the customer over their lifetime. |
| `support_calls` | Integer | >= 0 | Number of recent calls made to tech support. |

**Example Request Body:**
```json
{
  "age": 45,
  "tenure_months": 3,
  "contract": "Month-to-month",
  "monthly_charges": 95.50,
  "total_charges": 286.50,
  "support_calls": 3
}
```

### 2.2 Response Schema (JSON)

| Field | Type | Description |
| :--- | :--- | :--- |
| `churn_probability` | Float | The calculated probability of the customer churning (0.0 to 1.0). |
| `risk_category` | String | Categorical severity based on the probability (e.g., "High Risk", "Low Risk"). |
| `recommended_actions` | List[String] | Dynamically generated business recommendations based on the risk profile. |
| `processing_time_ms` | Float | Server-side execution time in milliseconds. |

**Example Response Body (High Risk):**
```json
{
  "churn_probability": 0.80,
  "risk_category": "High Risk",
  "recommended_actions": [
    "Offer 15% discount for annual upgrade",
    "Schedule proactive CS call"
  ],
  "processing_time_ms": 1.25
}
```

---

## 3. Usage Examples

### cURL

```bash
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "age": 28,
  "tenure_months": 24,
  "contract": "One year",
  "monthly_charges": 45.00,
  "total_charges": 1080.00,
  "support_calls": 0
}'
```

### Python (Requests Library)

```python
import requests
import json

url = "http://localhost:8000/predict"

payload = {
    "age": 35,
    "tenure_months": 2,
    "contract": "Month-to-month",
    "monthly_charges": 110.00,
    "total_charges": 220.00,
    "support_calls": 4
}
headers = {
    'Content-Type': 'application/json'
}

response = requests.post(url, headers=headers, data=json.dumps(payload))

if response.status_code == 200:
    result = response.json()
    print(f"Risk Category: {result['risk_category']}")
    print(f"Probability: {result['churn_probability'] * 100:.1f}%")
    print("Actions:", result['recommended_actions'])
else:
    print(f"Error: {response.status_code}")
```

---

## 4. Error Handling

The API uses standard HTTP status codes:
* `200 OK`: Request processed successfully.
* `422 Unprocessable Entity`: The JSON body is missing required fields or contains invalid data types. The response body will detail exactly which fields failed validation.
* `500 Internal Server Error`: An unexpected failure occurred during inference execution.
