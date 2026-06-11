# Day 29: Customer Segmentation with K-Means

## Goals
- Apply standard scaling and K-Means clustering on `tenure`, `MonthlyCharges`, and `TotalCharges`.
- Evaluate optimal cluster count via WCSS and Silhouette scores.
- Characterize segments and cross-reference with churn rates.

## Optimal K Selection
Evaluated $K$ from 1 to 10. $K=4$ was selected as the optimal cluster count to divide the base cleanly along Spend vs. Loyalty.
- **WCSS (K=4):** 1819.50
- **Silhouette (K=4):** 0.5064

## Segment Summary (K=4)
- **New / Budget-Conscious (33.3%):** Tenure 15.2M | Spend \$26.9 | Churn 11.2%. *Strategy: Auto-pay, basic upsells.*
- **Loyal / Low-Spend (26.4%):** Tenure 56.8M | Spend \$27.4 | Churn 3.4%. *Strategy: Referral marketing.*
- **High-Value Loyalists (22.0%):** Tenure 57.4M | Spend \$98.1 | Churn 7.7%. *Strategy: Early renewals, VIP support.*
- **High-Spend / High-Risk (18.3%):** Tenure 20.2M | Spend \$80.8 | Churn **43.1%** (CRITICAL). *Strategy: Convert month-to-month contracts to 12/24 months.*

## Deliverables
- [run_customer_segmentation.py](run_customer_segmentation.py) - Clean runner script.
- [day29_customer_segmentation.ipynb](day29_customer_segmentation.ipynb) - Executed Jupyter notebook (inline plots).
- [README.md](README.md) - This daily report.

## LinkedIn Reflection

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
