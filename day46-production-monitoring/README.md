# Day 46: Monitoring Customer Intelligence Systems in Production

## Overview
This phase focuses on improving production reliability for our Customer Intelligence systems. We have enhanced the Model API from Day 43 with production-grade monitoring, comprehensive logging, strict input validation, and exception handling. We also introduced a monitoring dashboard to observe API usage and prediction tracking.

## Features Implemented
1. **Input Validation**: Utilized Pydantic models to strictly validate incoming API payload (ensuring positive values, age limits, proper satisfaction scores).
2. **Comprehensive Logging**: Added logging to capture both the API requests (via middleware) and application-level events, outputting to `logs/api_monitoring.log`.
3. **Exception Handling**: Implemented try/except blocks in the `/predict` endpoint, ensuring API failures translate into meaningful HTTP status codes (`422 Unprocessable Entity` or `500 Internal Server Error`).
4. **Prediction Tracking**: Logged every input data and its resulting prediction payload to a JSON Lines file (`logs/prediction_tracking.jsonl`), assigned with a unique Request ID.
5. **Monitoring Dashboard**: A Streamlit application built to visualize the API traffic, analyze churn probability across incoming requests, and inspect raw API logs.

## Files
- `app.py`: The robust FastAPI application featuring model loading, input validation, and request logging.
- `dashboard.py`: A Streamlit UI to track API health and predictions in real-time.
- `train.py`: Script to train and save the Random Forest model and preprocessors.
- `monitoring_report.md`: A detailed report on the reliability improvements added.

## Instructions to Run

1. **Train the Models**
   Run the training script to generate the models directory and `.pkl` files:
   ```bash
   python train.py
   ```

2. **Start the API Server**
   Start the FastAPI application on port 8000:
   ```bash
   uvicorn app:app --reload
   ```
   *Test the API using the Swagger UI at `http://localhost:8000/docs`.*

3. **Start the Monitoring Dashboard**
   Open a new terminal and run the Streamlit dashboard:
   ```bash
   streamlit run dashboard.py
   ```
   *The dashboard will read from the `logs/prediction_tracking.jsonl` file to visualize the incoming prediction requests.*

## Screenshots
![Monitoring Dashboard](monitoring_dashboard_screenshot.png)

## LinkedIn Post

🚀 **Day 46 of #60DaysOfDataScience: Monitoring ML Systems in Production!** 📊

Building a machine learning model is only half the battle. Ensuring it runs reliably, handles bad data gracefully, and can be monitored in real-time is what makes it production-ready. 

For Day 46, I focused on bulletproofing my Customer Intelligence Model API by implementing comprehensive monitoring and reliability systems:

🛡️ **What I implemented today:**
*   **Strict Input Validation:** Integrated `Pydantic` to enforce data types, bounds (like ensuring non-negative charges and realistic age ranges), and required fields before they even reach the model.
*   **Exception Handling:** Added robust `try/except` workflows to prevent the API from crashing on unseen data, returning clean `422` or `500` HTTP status codes instead.
*   **Comprehensive Logging:** Added middleware to track API traffic and capture application-level events.
*   **Prediction Tracking:** Assigned unique request IDs and logged all incoming features alongside their resulting predictions into a JSONL file for future auditing and drift detection.
*   **Monitoring Dashboard:** Built a sleek Streamlit dashboard (pictured below!) that reads these logs in real-time to track API health, request volumes, and the distribution of churn risk predictions.

It's incredibly satisfying to see a raw predictive model evolve into a stable, observable software application!

#DataScience #MachineLearning #MLOps #FastAPI #Streamlit #ProductionML #Python #Analytics #DataScienceJourney
