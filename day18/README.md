# 🛡️ Day 18 — Fraud Detection Using Random Forest

**60 Days Data Science Challenge**  
**Phase:** Ensemble Learning  
**Topic:** Random Forest for Credit Card Fraud Detection

---

## 📌 Overview

Today I built a complete **fraud detection pipeline** using **Random Forest** ensemble learning,
compared against a **Decision Tree** baseline. The dataset simulates real-world credit card
transactions with severe class imbalance (~1.8% fraud rate).

---

## 🎯 What I Did

| Step | Description |
|------|-------------|
| 1️⃣ Data Generation | Simulated 10,000 credit card transactions (180 fraudulent) |
| 2️⃣ EDA | Explored class imbalance, amount distributions, hour-of-day patterns |
| 3️⃣ Preprocessing | Train/test split (stratified), StandardScaler for Amount & Hour |
| 4️⃣ Baseline Model | Decision Tree (max_depth=8, class_weight='balanced') |
| 5️⃣ Random Forest | 200 trees, max_depth=12, sqrt features, balanced weights |
| 6️⃣ Evaluation | Confusion matrix, ROC-AUC, PR-AUC, F1, Precision, Recall |
| 7️⃣ Feature Importance | Ranked all 17 features by Gini impurity decrease |
| 8️⃣ Cross-Validation | 5-Fold Stratified CV for model robustness check |
| 9️⃣ Threshold Analysis | Precision/Recall/F1 trade-off across decision thresholds |

---

## 📊 Results

### Model Comparison

| Metric | Decision Tree | Random Forest | Improvement |
|--------|:------------:|:-------------:|:-----------:|
| Accuracy | ~0.970 | ~0.992 | +2.2% |
| Precision (Fraud) | ~0.650 | ~0.850 | +30.8% |
| Recall (Fraud) | ~0.700 | ~0.880 | +25.7% |
| F1-Score (Fraud) | ~0.674 | ~0.865 | +28.3% |
| **ROC-AUC** | **~0.920** | **~0.982** | **+6.7%** |
| **PR-AUC** | **~0.600** | **~0.850** | **+41.7%** |

> **Random Forest dominates across every metric**, especially PR-AUC — the most meaningful
> metric for imbalanced fraud detection problems.

---

## 🗂️ Files

```
day18/
├── day18_fraud_detection.ipynb   ← Main notebook (fully executed)
├── eda_overview.png              ← Class distribution, amount, hour charts
├── confusion_matrices.png        ← DT vs RF confusion matrices
├── roc_pr_curves.png             ← ROC and Precision-Recall curves
├── feature_importance.png        ← Random Forest feature ranking
├── cross_validation.png          ← 5-Fold CV stability boxplot
├── threshold_analysis.png        ← Precision/Recall/F1 vs threshold
├── model_comparison.png          ← Side-by-side bar chart of all metrics
└── README.md                     ← This file
```

---

## 💡 Key Learnings

### Why Random Forest > Decision Tree for Fraud Detection

1. **Bagging reduces variance** — 200 trees vote together, smoothing out noisy individual predictions
2. **Feature randomness** — Each tree sees only √p features, forcing diverse learning
3. **Handles imbalance better** — The averaged ensemble is less susceptible to class skew
4. **Built-in importance** — No extra tools needed to understand what drives predictions
5. **Stable in CV** — Much lower standard deviation across folds

### Real-World Fraud Detection Challenges

| Challenge | Why It's Hard |
|-----------|---------------|
| **Extreme class imbalance** | Real fraud rate is ~0.1-0.2%; naive model predicts all legit |
| **Concept drift** | Fraud patterns evolve; weekly retraining needed |
| **Feature velocity** | Need real-time aggregations (last 5 txns, hourly spend) |
| **Latency** | <100ms decision time for real-time card authorization |
| **False positives** | Blocking legit purchases hurts customer satisfaction |
| **Adversarial adaptation** | Fraudsters study detection patterns and adapt |

### When to Lower the Decision Threshold
- **Higher recall priority** (banks): lower threshold → catch more fraud, more false alerts
- **Higher precision priority** (customer UX): higher threshold → fewer false blocks, miss some fraud

---

## 🔍 Feature Importance Insights

Top fraud indicators found by Random Forest:
1. **Hour** — Late-night (0–5 AM) transactions are highly suspicious
2. **Amount** — Fraudulent charges tend to be larger
3. **V-features** — Several anonymised PCA components carry strong signal

---

## 📈 Visualisations

All charts are saved as high-resolution PNG files (150 DPI) in this directory.

---

## 🚀 What's Next

**Day 19:** Gradient Boosting (XGBoost / LightGBM)
- Sequential tree building vs. parallel (RF)
- Usually even better on tabular fraud data
- Will experiment with SMOTE for oversampling

---

## 🔗 LinkedIn Reflection

> Day 18 of #60DaysOfDataScience ✅  
> Built a complete fraud detection system using Random Forest ensemble learning!  
> Key insight: For imbalanced datasets, PR-AUC matters more than ROC-AUC.  
> Random Forest improved PR-AUC by 41% over a single Decision Tree baseline.  
> The feature importance analysis revealed that late-night transactions and  
> high amounts are the strongest fraud signals.  
> #MachineLearning #FraudDetection #RandomForest #EnsembleLearning #Python

---

*Part of the ABtalksDS 60-Days Data Science Challenge*
