# 🗺️ Final Improvement Roadmap
**Phase:** Capstone Final Sprint (Days 57–60)

Based on the Mid-Stage Review and stakeholder feedback, this roadmap outlines the final engineering and design tasks required before the Capstone presentation on Day 60.

## Day 57: Feature Enrichment & Polish
**Goal:** Address immediate functional feedback from the mid-stage review.
* **Task 1:** Implement a "Download CSV" button in the Streamlit dashboard for the Marketing team to export the "At Risk" customer segment.
* **Task 2:** Refactor the API prediction logic to return granular risk tiers (Low, Medium, High, Critical) instead of a simple binary classification.
* **Task 3:** Add historical trend visualizations (line charts) for Churn Rate and LTV to the Overview dashboard.

## Day 58: Advanced Model Tuning & XAI Integration
**Goal:** Finalize the machine learning backend to ensure maximum accuracy and transparency.
* **Task 1:** Re-train the Random Forest model using the full dataset and tune hyperparameters one last time to minimize false negatives (failing to identify a churning customer).
* **Task 2:** Integrate SHAP (SHapley Additive exPlanations) values directly into the dashboard so Customer Success agents can see *why* a specific user is at risk (e.g., "Risk is high because: Tenure < 3 months").

## Day 59: Final UI/UX Overhaul & Cloud Deployment Prep
**Goal:** Ensure the platform looks professional and is ready for production hosting.
* **Task 1:** Overhaul the Streamlit CSS. Apply a cohesive brand color palette, improve typography, and ensure all charts are responsive and beautifully styled.
* **Task 2:** Containerize both the Streamlit app and FastAPI backend using Docker. 
* **Task 3:** Draft the final `docker-compose.yml` file to ensure a seamless one-click deployment for the evaluation team.

## Day 60: The Capstone Presentation 🚀
**Goal:** Successfully present the end-to-end Customer Intelligence Platform.
* **Task 1:** Finalize the Capstone Slide Deck, focusing on the business impact, technical architecture, and ROI of the platform.
* **Task 2:** Conduct a full end-to-end rehearsal of the live demo.
* **Task 3:** Publish the final project repository to GitHub, complete with a comprehensive `README.md`, setup instructions, and architecture diagrams.
* **Task 4:** Write the final 60-Days of Data Science summary reflection for LinkedIn.
