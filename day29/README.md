# 🚀 60 Days Data Science Challenge | Day 29/60

## Customer Segmentation with K-Means Clustering

Day 29 focuses on **Customer Segmentation** using unsupervised machine learning—specifically K-Means clustering—to identify hidden customer cohorts in the Telco Customer Churn dataset and define customized retention and marketing strategies.

---

## 🎯 Day 29 Goals
- **Load and preprocess** customer usage data (handling missing data in `TotalCharges`).
- **Standardize behavioral features** (`tenure`, `MonthlyCharges`, `TotalCharges`) to ensure scale-independent clustering.
- **Optimize the number of clusters (K)** using the **Elbow Method (WCSS)** and **Silhouette Score Analysis**.
- **Train a final K-Means model** with the optimal cluster size ($K=4$).
- **Analyze segment characteristics** and cross-reference them with contract types and customer churn rates.
- **Define concrete marketing and retention playbooks** for each customer segment.

---

## 🛠️ Segmentation Pipeline

K-Means uses Euclidean distance to measure similarity. Since features are on completely different scales (Monthly Charges max \$118, Total Charges max \$8,684, Tenure max 72), standard scaling is a critical first step to ensure each behavior has equal weight.

```
Raw Data (Telco Churn Dataset)
    │
    ├── Cleaning (Convert TotalCharges empty strings to median)
    │
    ├── Scaling (StandardScaler applied to tenure, MonthlyCharges, TotalCharges)
    │
    └── K-Means Optimization (WCSS and Silhouette evaluated for K in 1 to 10)
          │
          └── Final Fit (K=4 selected for optimal interpretability & separation)
                │
                └── Profiling (Aggregated tenure, monthly bill, and churn rate per cluster)
```

---

## 📊 Optimal Cluster Selection Curves

We evaluated $K$ from 1 to 10. WCSS (inertia) and average silhouette scores were checked side-by-side:
- **$K=2$ and $K=3$** yield high silhouette scores, but aggregate the data too broadly.
- **$K=4$** provides the cleanest elbow transition and the most actionable business segmentation. It separates customers cleanly along a $2 \times 2$ grid representing Tenure (Loyalty) vs. Monthly Charges (Spend).

```
 K | Inertia (WCSS) | Silhouette Score
---|---|---
 1 | 21129.00       | N/A
 2 |  6729.06       | 0.5824
 3 |  3122.92       | 0.5113
 4 |  1819.50       | 0.5064 (Selected)
 5 |  1297.02       | 0.4498
```

---

## 🔍 Detailed Segment Characteristics & Business Playbooks

Our K-Means model divided the customer base into four distinct behavioral cohorts:

### 1. Cluster 0: New / Budget-Conscious (33.3% of users)
*   **Averages:** Tenure: 15.2 Months | Monthly Bill: \$26.9 | Cumulative Billing: \$414.3
*   **Behavioral Characteristics:** Short tenure, low monthly bill, and low total billing. These are entry-level accounts subscribing to basic telephone services without high-margin fiber or premium digital add-ons.
*   **Churn Rate:** **11.2%** (Moderate)
*   **Tactical Action Plan:**
    - **Nurture and Upsell:** Gradually introduce high-margin digital add-ons (like basic streaming or device security bundles) in small monthly steps to avoid bill shock.
    - **Auto-Pay Incentives:** Offer a small discount (e.g., \$2/month) to set up auto-pay, anchoring them to the service.

### 2. Cluster 1: Loyal / Low-Spend (Basic) (26.4% of users)
*   **Averages:** Tenure: 56.8 Months | Monthly Bill: \$27.4 | Cumulative Billing: \$1,577.8
*   **Behavioral Characteristics:** High tenure, low monthly bill, and moderate total billing. These are long-term, extremely loyal customers subscribing to basic landline or DSL plans.
*   **Churn Rate:** **3.4%** (Lowest Churn)
*   **Tactical Action Plan:**
    - **Advocacy & Referrals:** Target them with referral marketing programs (e.g., "Refer a friend, get a free month").
    - **Automated Retainers:** Keep them happy with low-cost loyalty rewards (e.g., complimentary router upgrades) rather than expensive discounts.

### 3. Cluster 2: High-Value Loyalists (22.0% of users)
*   **Averages:** Tenure: 57.4 Months | Monthly Bill: \$98.1 | Cumulative Billing: \$5,657.4
*   **Behavioral Characteristics:** Long tenure, high monthly charges, and high total billing. These are our most profitable customers, subscribing to premium services (Fiber-optic, cloud backup, online security, TV streaming) under long-term contracts.
*   **Churn Rate:** **7.7%** (Stable)
*   **Tactical Action Plan:**
    - **VIP Care & Retention:** Provide dedicated priority support queues and route calls immediately to high-tier agents to prevent friction.
    - **Early Renewal Outreach:** Proactively contact them 3–6 months before their contracts expire to lock them into new multi-year agreements with customized hardware upgrades.

### 4. Cluster 3: High-Spend / High-Risk (18.3% of users)
*   **Averages:** Tenure: 20.2 Months | Monthly Bill: \$80.8 | Cumulative Billing: \$1,655.9
*   **Behavioral Characteristics:** Moderate tenure, high monthly bill, and moderate total billing. These are newer customers subscribing to expensive, high-speed fiber internet and multiple streaming services, but paying month-to-month.
*   **Churn Rate:** **43.1%** (CRITICAL RISK!)
*   **Tactical Action Plan:**
    - **Proactive Retention Campaigns:** This group is experiencing massive churn. They should be the primary targets of the Customer Success team.
    - **Contract Conversion Incentives:** Offer a contract incentive (e.g., "Switch to a 12-month contract and save \$15/month for the first 6 months"). High monthly bills on month-to-month plans are the primary cause of churn. Switching them to long-term contracts will drop their churn rate to match Cluster 2 (~7%).

---

## 🛠️ Day 29 Deliverables

📓 [day29_customer_segmentation.ipynb](day29_customer_segmentation.ipynb) — Fully executed notebook showing K-Means step-by-step (visualizations displayed inline).  
🖥 [run_customer_segmentation.py](run_customer_segmentation.py) — Clean runner script to build and execute the notebook.  
📓 [README.md](README.md) — This comprehensive ежедневный challenge report.  

---

## 🔗 LinkedIn Reflection

**Day 29 of 60: Unlocking Hidden Customer Groups with K-Means Clustering! 📊**

Today, I shifted from predicting user outcomes (supervised classification) to exploring hidden patterns (unsupervised learning) in our customer data. 

I applied K-Means clustering to segment customers based on three core behavioral metrics: **Tenure (Loyalty)**, **Monthly Charges (Spend)**, and **Total Charges**.

Here are the key takeaways from my analysis:

1. **Feature Scaling is Essential:** Because Total Charges goes up to \$8,000 while tenure caps at 72, standardizing features using `StandardScaler` is a mandatory first step to prevent distance calculations from being dominated by scale.
2. **Finding the "Right" K:** Using the Elbow Method (WCSS) and Silhouette Score side-by-side, I determined that $K=4$ was the optimal cluster count. It splits the customer base into four highly distinct and actionable quadrant cohorts.
3. **The High-Risk/High-Spend Conundrum:** The clustering revealed a critical cohort—**High-Spend / High-Risk customers (18.3% of base)**. These are new customers with expensive packages who have not signed long-term contracts. Their churn rate is an astronomical **43.1%**!
4. **Bridging Unsupervised & Supervised Learning:** Identifying this group allows us to proactively intervene with specific contract conversion offers (e.g. converting month-to-month users to 12-month agreements), dropping their churn rate to match the loyalist group (~7%).

Unsupervised clustering turns raw, tabular customer lists into highly targeted, revenue-saving business action.

On to Day 30! 🚀

#DataScience #MachineLearning #KMeans #Clustering #CustomerSegmentation #CustomerAnalytics #Python #ScikitLearn #60DayChallenge #ABtalksDS

---

*60 Days Data Science Challenge — Day 29/60*
