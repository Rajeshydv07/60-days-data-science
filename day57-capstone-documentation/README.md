# 📊 Customer Intelligence Platform: Churn Prediction & Retention Analytics

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-00a393.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-FF4B4B.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E.svg)
![Status](https://img.shields.io/badge/Status-Production_Ready-success.svg)

Welcome to the **Customer Intelligence Platform**, an end-to-end Machine Learning ecosystem designed to shift customer retention strategies from reactive to proactive. By analyzing customer behavior, contract configurations, and support interactions, this platform predicts churn risk in real-time and prescribes actionable interventions to increase Customer Lifetime Value (CLV).

This project was built as the final Capstone for a rigorous 60-Day Data Science sprint, encompassing everything from raw data engineering and model training to API deployment and business storytelling.

---

## 🎯 Business Objectives

Customer acquisition costs are continually rising, making retention a critical driver of profitability. This platform addresses this challenge by providing:
1. **Predictive Foresight:** Identifying high-risk customers *before* they cancel their service.
2. **Actionable Recommendations:** Moving beyond just a probability score by automatically mapping risk profiles to specific business actions (e.g., targeted discounts, proactive support calls).
3. **Explainable AI (XAI):** Building trust with stakeholders by explaining *why* a customer is likely to churn (e.g., high monthly charges, month-to-month contracts).
4. **Operational Efficiency:** Empowering Customer Success teams with a real-time Streamlit dashboard and instantaneous API inferences.

---

## ✨ Key Features & Modules

* **Robust Data Pipeline:** Custom scikit-learn transformers that handle missing values, encode categorical variables, and scale numerical features automatically.
* **Optimized ML Models:** A hyperparameter-tuned ensemble (Random Forest) serving as our champion model, balancing precision and recall for imbalanced churn data.
* **FastAPI Microservice:** A highly scalable, asynchronous API providing real-time inference in sub-millisecond response times.
* **Interactive Streamlit Dashboard:** An executive-facing UI for both retrospective aggregate analytics and live single-customer predictions.
* **Explainable AI (SHAP):** Integrated model interpretability ensuring transparent decision-making.

---

## 🏗️ System Architecture & API

To understand how the data flows from raw input to executive dashboards, and how to interface with our predictive endpoints, please refer to our dedicated technical documentation:

* [**Architecture & Workflows (ARCHITECTURE.md)**](./ARCHITECTURE.md) - Contains system design, component diagrams, and data flow.
* [**API Reference Guide (API_DOCS.md)**](./API_DOCS.md) - Contains request/response schemas, cURL examples, and integration patterns.

---

## 🚀 Setup & Deployment Instructions

### Prerequisites
* Python 3.9+
* pip package manager

### Local Environment Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Rajeshydv07/ABtalksDS.git
   cd ABtalksDS
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   *(Assuming a requirements.txt exists at the project root or capstone folder)*
   ```bash
   pip install fastapi uvicorn streamlit scikit-learn pandas shap
   ```

### Running the Application

You need to run both the Backend API and the Frontend Dashboard.

**Terminal 1: Start the FastAPI Backend**
```bash
cd day55-capstone-scalability-reliability
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```
*The API will be available at `http://localhost:8000`. You can view the interactive Swagger documentation at `http://localhost:8000/docs`.*

**Terminal 2: Start the Streamlit Dashboard**
```bash
cd day52-capstone-frontend
streamlit run app.py
```
*The dashboard will automatically open in your default browser at `http://localhost:8501`.*

---

## 📸 Platform Screenshots

### Real-Time Prediction Dashboard
*(Placeholder for Streamlit real-time prediction form screenshot)*
> *The UI immediately flags high-risk customers and provides targeted retention strategies.*

### Executive Analytics
*(Placeholder for aggregate churn analytics and XAI feature importance visualization)*
> *Macro-level insights allow leadership to restructure pricing and service bundles.*

---

## 🤝 Contributing & License
This project was developed by Rajesh Yadav as a capstone project. Feel free to fork, explore the code, and submit Pull Requests for enhancements.

**License:** MIT
