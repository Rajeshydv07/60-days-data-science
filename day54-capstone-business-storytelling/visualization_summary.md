# 📈 Visualization Summary: Capstone Business Storytelling

Visual storytelling bridges the gap between raw data and executive decision-making. Below is a summary of the core visualizations designed for our business stakeholders, what they communicate, and why they matter.

## 1. The Churn vs. Contract Type Bar Chart
* **What it shows:** A side-by-side comparison of churn rates across Month-to-Month, 1-Year, and 2-Year contracts.
* **The Narrative:** Month-to-month contracts have an exponentially higher churn rate compared to fixed-term contracts.
* **Why it matters:** It provides empirical backing for the recommendation to launch the "Commit & Save" campaign. It visibly shifts the focus from "we have a churn problem" to "we have a contract structure problem."

## 2. The Tenure Survival Curve (Kaplan-Meier)
* **What it shows:** The probability of a customer remaining active over time (months).
* **The Narrative:** The curve drops steeply in the first 6 months and then flattens out. If a customer stays past 12 months, they are highly likely to remain long-term.
* **Why it matters:** It highlights the "Onboarding Vulnerability." Executives can clearly see that retention efforts need to be heavily front-loaded in the customer's lifecycle, justifying the "First 90 Days Success Program."

## 3. The Feature Importance Waterfall (SHAP)
* **What it shows:** A breakdown of how individual features (Monthly Charges, Tech Support, Online Security) positively or negatively influence a specific customer's churn prediction.
* **The Narrative:** Explains *why* the model flagged a customer. For example, it visualizes how the absence of Tech Support increases churn risk.
* **Why it matters:** It demystifies the "black box" machine learning model for non-technical stakeholders, building trust in the predictive API and supporting the case for bundling "sticky" services.

## 4. The Revenue at Risk Dashboard Gauge
* **What it shows:** A real-time gauge displaying the total Monthly Recurring Revenue (MRR) associated with customers currently flagged as "High Risk."
* **The Narrative:** Translates churn probabilities into direct financial impact (dollars).
* **Why it matters:** Creates immediate urgency. Executives respond to financial metrics much faster than abstract probabilities. It justifies the budget for retention campaigns by showing exactly how much revenue is on the line today.
