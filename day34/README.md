# Day 34: Designing Executive-Level Customer Dashboards

Today, I worked on translating customer data into business-ready executive insights. Instead of dumping raw numbers or complicated machine learning models onto decision-makers, I built a structured customer analytics dashboard that helps leaders track key indicators at a glance and zoom into problem areas.

---

## 📊 Business Insights Summary

After analyzing the datasets, here are the major findings I presented to the leadership team:

1.  **Month-to-Month Contracts are Churn Hot-Spots**:
    *   **Month-to-month** customers have a massive churn rate of **42.71%**.
    *   Compare this to **One-year** contracts (**11.27%**) and **Two-year** contracts (**2.83%**).
    *   *Strategic Takeaway:* Attrition isn't necessarily a failure of our service, but rather a contract configuration problem. Any campaign migrating month-to-month users to 1-year terms has massive cost-saving potential.
2.  **The Fiber Optic Churn Paradox**:
    *   **Fiber Optic** internet users have a churn rate of **41.89%**, which is double that of **DSL** users (**18.96%**).
    *   *Strategic Takeaway:* Even though Fiber Optic provides faster speeds and higher contract values, customers are highly dissatisfied. This points to a post-onboarding support issue, billing sticker shock, or regional connection instability. Proactive customer check-ins are crucial.
3.  **Targeting Elite Affluents**:
    *   By applying K-Means clustering (from Day 31), I mapped out distinct personas. **The Elite Affluents** (high income, high spending score) make up **23.2%** of the user base. They are prime targets for premium cross-selling and long-term retention.

---

## 🎨 Storytelling & Dashboard Design Decisions

To make the dashboard look professional and easy to read:
*   **Executive Layout Hierarchy**: I put 4 KPI cards at the very top (Total Customers, Churn Rate, Avg Monthly Bill, and MRR). Leaders can see the overall health of the business in 3 seconds. Below the cards, I added side-by-side charts for visual comparison, and detailed tables at the bottom for those who want to examine the exact numbers.
*   **Harmonious Color Palette**: I avoided generic primary colors. Instead, I used a premium corporate color system: Steel Blue (`#1e3d59`) for general categories, Coral Orange (`#ff6e40`) for warnings, Green (`#17b978`) for positive financial health, and Crimson Red (`#d9534f`) for highlighting high churn risks.
*   **Interactive Sidebar Filters**: Leaders can filter metrics by contract type, internet service tech, and gender to immediately see how specific customer segments perform.

---

## 🚀 How to Run the Dashboard App

To launch the interactive dashboard locally:

1.  **Install Streamlit & Requirements**:
    ```bash
    pip install streamlit pandas numpy matplotlib seaborn
    ```
2.  **Run the Streamlit App**:
    Make sure you are in the project root directory, then run:
    ```bash
    streamlit run day34/app.py
    ```
3.  **Access the Dashboard**:
    Open the URL (typically `http://localhost:8501`) shown in your terminal.

---

## 📝 Student Reflection & LinkedIn Post

### LinkedIn Post Draft
> 📊 **Day 34 of 60: Building Executive Dashboards that Actually Drive Action**
>
> Today, I focused on a critical data science skill: communicating insights to business leaders. It’s one thing to build high-accuracy machine learning models, but if executives can’t understand or interact with the findings, the work goes to waste.
>
> I built an interactive customer analytics dashboard combining customer churn metrics and behavioral persona segments.
>
> **Key takeaways from the analysis:**
> 1. Month-to-month contracts are our biggest churn leak (42% churn rate vs. just 2% for 2-year terms). Focus should be on contract migration campaigns!
> 2. Fast speed isn't enough: Fiber optic customers are churning at double the rate of DSL customers. Proactive health checks and onboarding audits are needed here.
> 3. Visuals matter. By grouping high-level KPIs at the top and designing interactive filters, leaders can find where revenue is leaking in seconds.
>
> Built using Python, Pandas, Matplotlib, and Streamlit. Ready for the next phase of the challenge! 🚀
>
> #DataScience #BusinessIntelligence #CustomerAnalytics #Streamlit #DataVisualisation #60DaysOfDataScience
