# System Reliability Analysis
**Phase:** Capstone Scalability & Reliability (Day 55)

## Objective
To ensure that the Customer Intelligence Platform is robust against failures, handles unpredictable user inputs gracefully, and maintains high availability.

## 1. Error Handling & Graceful Degradation
- **Frontend (Streamlit):** Wrapped critical data loading paths in `try-except` blocks. If the backend CSV or database is unavailable, the system displays a user-friendly error message (`st.error`) instead of crashing with a Python traceback.
- **Backend (FastAPI):** Added global exception handlers. If the prediction model fails or encounters a division-by-zero error, it catches the exception, logs the event using the `logging` module, and returns a structured 500 HTTP response, preventing the API from hanging or exposing internal stack traces.

## 2. Input Validation
- Previously, invalid inputs (e.g., negative monthly charges, unrealistic ages) could pass through to the model, returning nonsensical predictions or breaking the app.
- Implemented frontend input constraints in Streamlit (`min_value`, `max_value`) to prevent bad inputs at the UI level.
- Integrated Pydantic in the API layer, defining strict schemas (e.g., `ge=0`, type hints) ensuring that if the API is called directly, data integrity is strictly maintained.

## 3. Observability and Logging
- Migrated from generic `print()` statements to Python's standard `logging` library.
- Configured log levels (`INFO` for successful requests, `ERROR` for exceptions). This allows for easier parsing by monitoring tools (like ELK stack or Datadog) to alert engineers of spikes in error rates.
- Introduced a mock "System Health" tab in the dashboard to represent proactive monitoring of API uptime and cache hit rates.

## 4. Bottleneck Assessment & Future Proofing
- **Current Bottleneck:** The dataset is loaded entirely into RAM. While cached, a dataset larger than 5GB could cause out-of-memory (OOM) errors.
- **Mitigation Strategy:** For the next iteration, data loading should be migrated to a proper SQL database (e.g., PostgreSQL) or cloud data warehouse (e.g., Snowflake) using chunked queries or an ORM like SQLAlchemy to fetch only aggregated views rather than the raw data.
