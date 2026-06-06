# 🚀 60 Days Data Science Challenge | Day 25/60

## Model Validation: Why a Single Train-Test Split Can Mislead You

Today, I explored one of the most important aspects of Machine Learning: **Model Validation**.

A model isn't valuable just because it performs well on one test set. The real question is:

> **Will it perform consistently on unseen data?**

To answer that, I compared traditional **Train-Test Split Validation** with **Stratified K-Fold Cross-Validation** using the **Telco Customer Churn Dataset**.

---

## 🔍 What I Implemented

### ✅ Train-Test Split Variability Analysis

I trained three models:

* Logistic Regression
* Decision Tree
* Random Forest

Then I evaluated them across **10 different random train-test splits** (Seeds 1–10) to observe how performance changes based solely on data partitioning.

### ✅ Stratified Cross-Validation

To obtain more reliable estimates, I performed:

* 5-Fold Stratified Cross-Validation
* 10-Fold Stratified Cross-Validation

and compared their:

* ROC-AUC Score
* Accuracy
* F1-Score
* Standard Deviation

---

## 📊 Key Findings

### 🎲 The Train-Test Split Lottery

Model performance varied significantly depending on the random split.

For example:

**Random Forest ROC-AUC**

* Lowest Score: **0.824**
* Highest Score: **0.860**

That's a difference of nearly **4 percentage points** without changing the model or features!

This demonstrates why relying on a single split can lead to misleading conclusions.

---

### 🛡️ Cross-Validation Provides Stability

Cross-validation produced highly consistent performance estimates:

| Model               | Mean ROC-AUC |
| ------------------- | ------------ |
| Logistic Regression | 0.845        |
| Random Forest       | 0.844        |
| Decision Tree       | 0.827        |

Instead of depending on one lucky or unlucky split, CV evaluates performance across multiple data partitions, giving a much more trustworthy estimate.

---

### 📈 Model Reliability Comparison

🏆 **Logistic Regression**

* Highest overall ROC-AUC
* Best F1-Score (~0.60)
* Most stable across folds

🌲 **Random Forest**

* Competitive ROC-AUC
* Slightly lower F1-Score
* More sensitive to data variations

🌳 **Decision Tree**

* Lowest overall performance
* Highest sensitivity to data partitioning

---

### 🤔 5-Fold vs 10-Fold Cross-Validation

An interesting observation:

10-Fold CV showed slightly higher fold-to-fold variance than 5-Fold CV.

Why?

Because each validation fold is smaller, making individual scores more sensitive to data fluctuations. However, the final averaged score remains highly reliable and representative.

---

## 💡 Key Takeaway

**Never trust a single train-test split.**

A great score on one split may simply be the result of favorable sampling.

Cross-validation helps reveal:

✅ True model performance
✅ Model stability
✅ Generalization capability
✅ Reliability before deployment

When building production-ready ML systems, validation strategy matters just as much as model selection.

---

## 🛠️ Deliverables

📓 `day25_cross_validation.ipynb` – Complete implementation

📊 `cv_results.csv` – Cross-validation metrics and statistics

📈 `cv_comparison.png` – Visualization of split variance vs CV stability



