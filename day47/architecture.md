# System Architecture

The Customer Intelligence Platform follows a modular, scalable architecture designed for end-to-end data processing, model serving, and visualization.

## High-Level Architecture Diagram

```mermaid
flowchart TD
    %% Data Sources
    subgraph Data Sources
        DB[(Transactional DB)]
        Logs[Behavioral Logs]
        CRM[CRM Data]
    end

    %% Data Engineering / ETL
    subgraph Data Pipeline
        ETL[ETL Process / Airflow]
        DWH[(Data Warehouse / Snowflake/BigQuery)]
    end

    %% Machine Learning Modules
    subgraph Core ML Modules
        FeatureStore[Feature Store]
        Segment[Segmentation Engine (RFM + K-Means)]
        Churn[Churn Prediction Model (XGBoost)]
        Forecast[CLV Forecasting (Prophet)]
    end

    %% Serving & API
    subgraph Serving Layer
        API[FastAPI Model Endpoints]
        Cache[(Redis Cache)]
    end

    %% Presentation Layer
    subgraph Presentation
        Dash[Interactive Dashboard (Streamlit/React)]
        Alerts[Notification Service]
    end

    %% Data Flow
    DB --> ETL
    Logs --> ETL
    CRM --> ETL
    ETL --> DWH
    
    DWH --> FeatureStore
    FeatureStore --> Segment
    FeatureStore --> Churn
    FeatureStore --> Forecast

    Segment --> API
    Churn --> API
    Forecast --> API
    
    API <--> Cache
    API --> Dash
    API --> Alerts
```

## Component Details
1. **Data Ingestion**: Raw data is pulled from simulated production databases and CRM systems.
2. **Data Warehouse**: Acts as the single source of truth, storing cleaned and transformed data.
3. **Feature Store**: Centralized repository for engineered features (e.g., rolling averages of purchase frequency, days since last purchase).
4. **Model Training & Inference**: Batch predictions for segmentation and churn run periodically, while real-time scoring is exposed via REST APIs.
5. **Presentation**: A front-end application consumes the API to present insights to business users.
