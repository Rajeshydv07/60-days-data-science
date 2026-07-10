# 🎬 Capstone Demo Walkthrough Script

**Project:** Customer Intelligence Platform (CIP)
**Target Video Length:** 3-5 Minutes
**Tools Needed:** Screen recording software (Loom, OBS, or Zoom), Microphone.

## Preparation Before Recording
1. Ensure the **FastAPI backend** is running locally (`uvicorn main:app --reload`).
2. Ensure the **Streamlit frontend** is running locally (`streamlit run app.py`).
3. Open the dashboard in a clean, full-screen browser window.
4. Have the slide deck open in another tab.

---

## Script & Scene Directions

### Scene 1: Introduction & The Problem (0:00 - 0:45)
**Visual:** Show the Title Slide of your Presentation Deck.
**Audio:** 
"Hello everyone, my name is [Your Name], and today I’m presenting my capstone project: The Customer Intelligence Platform. 
In today's competitive market, customer acquisition is expensive, but losing customers—churn—is even costlier. My goal with this project was to build an end-to-end machine learning platform that doesn't just analyze past data, but actively predicts churn in real-time, allowing business teams to proactively save at-risk revenue."

### Scene 2: High-Level Analytics & Segmentation (0:45 - 1:45)
**Visual:** Switch to the Streamlit Dashboard (Executive Overview tab).
**Audio:** 
"Let’s jump right into the platform. This is the Executive Overview, built with Streamlit. Here, stakeholders can immediately see our macro-level health metrics: Total Customers, Churn Rate, and Average Lifetime Value. 
As we scroll down to the segmentation module, you'll see our survival analysis. This Kaplan-Meier curve clearly demonstrates that our highest risk period is the first 90 days. Furthermore, our contract segmentation highlights that month-to-month contracts hold the highest volatility. This isn't just data; this directly informs our marketing strategy to push 1-year contract upgrades."

### Scene 3: The Prediction Module in Action (1:45 - 3:15)
**Visual:** Navigate to the "Real-Time Prediction" tab in the dashboard.
**Audio:** 
"While executive summaries are great, the true power of this platform lies in its predictive capability. Let's pretend I'm a Customer Success Agent on a call with a customer.
*(Action: Begin filling out the form on the screen)*. 
I'll input their details: Month-to-Month contract, 3 months tenure, paying $90/month, and they don't have tech support. 
*(Action: Click the 'Predict Churn Risk' button)*.
When I hit predict, the Streamlit app sends a payload to our FastAPI backend. The API processes this through our trained Random Forest model and instantly returns a score. As you can see, this customer is flagged as **High Risk** with an 82% probability of churning."

### Scene 4: Explainable AI & Actionable Outcomes (3:15 - 4:15)
**Visual:** Scroll down to the SHAP Waterfall plot and the "Recommended Actions" section on the dashboard.
**Audio:** 
"To trust a machine learning model, we need to know *why* it made its decision. Here, we use SHAP values for Explainable AI. This chart shows exactly what factors drove this customer's high-risk score—specifically, their short tenure and lack of online security add-ons.
Because the system knows *why* they are at risk, it automatically generates a recommended action: *'Offer a 15% discount on a 1-year contract upgrade including Tech Support.'* We've turned a raw prediction into a targeted business intervention."

### Scene 5: Architecture & Conclusion (4:15 - 5:00)
**Visual:** Switch back to the Presentation Deck (Architecture Diagram Slide).
**Audio:** 
"Under the hood, this platform is robust. The data engineering pipeline handles robust preprocessing, the ML model is optimized for precision-recall, and the FastAPI service layer ensures asynchronous, scalable inference.
By deploying this Customer Intelligence Platform, a business can transition from reactive firefighting to predictive retention, directly protecting their Monthly Recurring Revenue. 
Thank you for watching, and feel free to check out the complete code repository on my GitHub."

---
**Tips for a Great Recording:**
* Speak clearly and at a measured pace.
* Don't worry about minor mistakes; keep the flow going.
* Make sure your mouse movements are smooth and intentional to guide the viewer's eye.
