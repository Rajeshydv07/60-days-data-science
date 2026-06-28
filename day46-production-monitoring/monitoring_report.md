# Production Monitoring & Reliability Report

## Overview
This document outlines the monitoring, validation, and reliability improvements added to the Customer Intelligence analytics application to ensure it operates stably in a production environment. 

## Improvements Implemented

### 1. Robust Input Validation
- **Framework**: Implemented validation using `Pydantic` models.
- **Constraints**: 
  - Age: Restricted between 18 and 120.
  - Non-negative constraints: Ensured `tenure_months`, `monthly_charges`, `total_charges`, `support_tickets`, and `days_since_last_login` cannot be negative.
  - Satisfaction Score: Enforced a strict range of 1 to 5.
- **Impact**: Prevents malformed or anomalous data from causing unhandled exceptions during inference and guarantees model predictions are based on valid inputs.

### 2. Comprehensive Logging
- **Log Levels & Formatting**: Added structured Python logging (`INFO`, `WARNING`, `ERROR`, etc.) capturing timestamps, module names, and message details.
- **Middleware Integration**: Logged all incoming HTTP requests and their completion status using a FastAPI middleware to track overall API usage and health.
- **Log Output**: Directed standard application logs to `logs/api_monitoring.log`.

### 3. Exception Handling & Invalid Predictions
- **Try/Except Blocks**: Wrapped the prediction workflow in try/except blocks to gracefully handle missing/corrupted models, unknown categorical variables, and generalized runtime exceptions.
- **Status Codes**: 
  - `422 Unprocessable Entity` is correctly propagated when input validation fails.
  - `500 Internal Server Error` is propagated for unexpected backend crashes, ensuring the client receives standardized HTTP responses.

### 4. Prediction Request and Output Tracking
- **Request IDs**: Each API request is assigned a unique `request_id` (timestamp-based) to trace individual prediction lifecycles.
- **Tracking Log**: Persisted both input features and the resulting predictions to a JSON Lines file (`logs/prediction_tracking.jsonl`).
- **Impact**: This enables model drift monitoring, auditing, and debugging by keeping a permanent record of what the model scored and how it responded.

## Monitoring Files
- `logs/api_monitoring.log` - Application health and standard logs.
- `logs/prediction_tracking.jsonl` - Structured prediction IO logging.
