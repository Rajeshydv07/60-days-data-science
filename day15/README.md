# Day 15: Predicting Customer Churn with Logistic Regression

Welcome to Day 15 of my **60-Day Data Science Challenge**! today marks the start of the **Classification Foundations** phase. After working on regression to predict continuous sales values in the last sprint, I transitioned today to predicting binary categorical outcomes—specifically, whether a customer is likely to cancel their subscription (**Churn**).

Subscription-based companies (like telecom, SaaS, and utilities) live and die by customer retention. Predicting customer churn allows marketing and customer support teams to target high-risk accounts with proactive incentives before they cancel, preventing massive revenue loss.

---

## Files in this Directory
*   [day15_churn_prediction.ipynb](day15_churn_prediction.ipynb): Jupyter notebook containing the complete data cleaning, OHE preprocessing, stratified train-test splits, scaling, training, predictions, and evaluation.
*   [telco_customer_churn.csv](telco_customer_churn.csv): The local customer churn dataset downloaded from the public IBM Telco customer churn sample.
*   [predictions_churn.csv](predictions_churn.csv): Test set observations paired with their actual churn labels, predictions, and model probabilities.
*   [confusion_matrix_churn.png](confusion_matrix_churn.png): Visual heatmap highlighting correct predictions and misclassification errors on our test set.
*   [roc_curve_churn.png](roc_curve_churn.png): ROC-AUC curve visualization demonstrating class separation ability.
*   [feature_coefficients_churn.png](feature_coefficients_churn.png): Coefficient bar chart showing which attributes drive retention vs. churn.

---

## Data Cleaning & Preprocessing Highlights

1.  **Handling Messy TotalCharges**:
    *   `TotalCharges` was loaded as a string (`object` type) because of empty white spaces.
    *   By investigating the data, I discovered that these blank spaces occurred exclusively for customers with `tenure = 0` (new signups who haven't completed their first billing cycle).
    *   Instead of dropping these rows (which would introduce bias), I replaced empty spaces with `NaN` and filled them with `0.0`.
2.  **Dummy Variable Trap Protection**:
    *   To prepare categorical features for Logistic Regression, I applied One-Hot Encoding with `drop_first=True`. This drops the redundant reference column for each categorical feature, preventing perfect multi-collinearity.
3.  **Preventing Data Leakage**:
    *   I explicitly separated our unique identifier `customerID` before splitting.
    *   I split our dataset using an **80/20 ratio** with `stratify=y` to preserve the 26.5% churn class balance perfectly.
    *   I fit the standard scaler **only** on the training partition and transformed the test partition to ensure no statistical properties leaked from the future.

---

## Model Evaluation Metrics

Here is the exact performance breakdown of our Logistic Regression baseline classifier on the 20% unseen test set (1,409 observations):

| Metric | Score / Result | Business & Analytical Meaning |
| :--- | :---: | :--- |
| **Accuracy** | **80.62%** | Overall fraction of correct classifications (retained and churned). |
| **Precision** | **65.93%** | Out of all customers flagged as churn-risks, 65.93% actually left. |
| **Recall** | **55.88%** | The model successfully identified 55.88% of all customers who actually churned. |
| **F1-Score** | **60.49%** | The harmonic mean of precision and recall (balancing the trade-off). |
| **ROC-AUC** | **0.8422** | Overall predictive capability to rank churn risk correctly (strong performance). |

### Detailed Classification Report:
*   **Retained (Class 0)**: 0.85 Precision, 0.90 Recall, 0.87 F1-Score (Support: 1,035)
*   **Churned (Class 1)**: 0.66 Precision, 0.56 Recall, 0.60 F1-Score (Support: 374)

---

## The Business Impact of Prediction Errors (FP vs. FN)

In customer churn modeling, looking at overall accuracy is a dangerous trap. We must analyze classification errors (False Positives and False Negatives) through a physical business lens to calculate financial impact:

### 1. False Negatives (FN = 165 cases)
*   **Definition**: The model predicts a customer will **stay**, but they actually **churn**.
*   **Business Impact**: We miss the opportunity to offer them a retention campaign. The customer cancels their subscription and leaves.
*   **Estimated Cost**: Extremely high! The company loses the customer's Customer Lifetime Value (CLV). Assuming an average telecom CLV of **$800**, this error costs the company **$800** in lost revenue per customer.
*   **Total Cost on Test Set**: 165 × $800 = **$132,000 in lost revenue**.

### 2. False Positives (FP = 108 cases)
*   **Definition**: The model predicts a customer will **churn**, but they actually planned to **stay**.
*   **Business Impact**: The marketing team sends a proactive retention incentive (e.g., a **$50** credit or voucher) to a customer who wasn't planning to leave anyway.
*   **Estimated Cost**: Low. The company incurs the cost of the **$50** promotion. While it causes slight margin erosion, it increases overall customer loyalty.
*   **Total Cost on Test Set**: 108 × $50 = **$5,400 in marketing waste**.

### Strategic Business Conclusion:
Because a False Negative (**$800** loss) is **16 times more expensive** than a False Positive (**$50** cost), **Recall is our most critical optimization metric**. 

In production, we should lower our probability threshold (e.g., from `0.50` to `0.30`). This will capture more False Positives (sending vouchers to safe customers), but it will dramatically reduce False Negatives (saving customers who are about to leave), maximizing the financial savings of the retention program.

---

## Key Drivers of Customer Retention vs. Churn

Extracting the model's coefficients gives us clear insights into what variables drive churn or loyalty:

### Strong Churn Predictors (Positive Coefficients)
1.  **`Contract_Month-to-month` (Coefficient: +0.76)**: Month-to-month contracts are the strongest indicator of churn. Without contract lock-in, customers leave easily.
2.  **`InternetService_Fiber optic` (Coefficient: +0.42)**: Fiber optic users are unexpectedly churn-prone. This points to potential service satisfaction or billing issues.
3.  **`PaymentMethod_Electronic check` (Coefficient: +0.13)**: Indicates electronic check payers have higher churn rates compared to automated credit card billing.

### Strong Retention Predictors (Negative Coefficients)
1.  **`tenure` (Coefficient: -0.84)**: The single strongest driver of retention. Customers who stay longer become increasingly loyal.
2.  **`Contract_Two year` (Coefficient: -0.66)**: Long term lock-ins are highly effective at suppressing churn.
3.  **`OnlineSecurity_Yes` (Coefficient: -0.22)**: Subscribed security add-ons act as "sticky" features, reducing churn probability.

---

## LinkedIn Reflection

Here is my daily learning summary post for LinkedIn:

**Post**:
> Day 15 of my 60-Day Data Science Challenge! 📉 Today, I officially kicked off **Classification Foundations** by building a **Customer Churn Prediction system using Logistic Regression**!
> 
> Subscription companies live and die by customer retention. Using the public IBM Telco Churn dataset, I designed a pipeline to predict whether a customer will cancel their subscription.
> 
> Here are my key student takeaways from today's work:
> 
> 🛠️ **1. The TotalCharges Cleaning Trap:**
> `TotalCharges` was loaded as a string. Investigating the dataset revealed that blank strings occurred ONLY for customers with a tenure of 0 (new signups). Replacing empty spaces with NaN and filling them with 0.0 saved our data without introducing bias or dropping rows.
> 
> 📊 **2. The Baseline Performance:**
> Using a stratified 80/20 split and scaling numerical features, our baseline Logistic Regression classifier achieved an overall accuracy of **80.62%** and a very strong ROC-AUC of **84.22%**.
> 
> 🧠 **3. Calculating the Cost of Errors (The Real Business Value):**
> In machine learning, not all errors are created equal. 
> - **False Negative**: We predict a customer will stay, but they churn. Cost = Losing their Lifetime Value (**$800** lost).
> - **False Positive**: We predict a customer will churn, but they stay. Cost = The retention voucher (**$50** discount spent).
> 
> Since a False Negative is **16x more costly** than a False Positive, optimizing for **Recall (55.88%)** is far more valuable than optimizing for Precision (65.93%)! Adjusting our classification threshold will help us save at-risk accounts.
> 
> 🔑 **4. Top Churn vs. Retention Drivers:**
> Month-to-month contracts and Fiber Optic internet are our strongest positive drivers of churn. In contrast, longer customer tenure and two-year contract lengths are powerful anchors of customer retention.
> 
> This is a solid classification baseline, and now I have a clean framework ready to try out more complex models! 🚀
> 
> #DataScience #MachineLearning #Python #ScikitLearn #CustomerRetention #LogisticRegression #60DayChallenge #ABtalksDS
