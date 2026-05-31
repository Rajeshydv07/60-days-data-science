# Day 18 - Fraud Detection Using Random Forest

60 Days Data Science Challenge | Day 18  
Topic: Ensemble Learning - Random Forest

---

## what i did today

today i built a fraud detection model using **random forest** and compared it with  
yesterday's decision tree to see if ensemble methods actually help.

spoiler: they do. a lot.

---

## dataset

i don't have the actual kaggle credit card fraud dataset downloaded  
so i simulated one with similar properties:
- 10,000 transactions total
- 180 fraud cases (~1.8% fraud rate)
- features: V1-V15 (PCA style), Amount, Hour, Class

the real dataset has 284k transactions with only 0.17% fraud which is way more extreme

---

## steps i followed

1. imported libraries
2. created the dataset with realistic fraud patterns (higher amounts, late night)
3. EDA - looked at class imbalance, amount distributions, hour of day
4. preprocessed - stratified split (important for imbalanced data), scaled Amount and Hour
5. trained decision tree as baseline
6. trained random forest (200 trees)
7. compared both models on all metrics
8. plotted confusion matrices, ROC curve, PR curve
9. analysed feature importance
10. ran 5-fold cross validation to check stability
11. did threshold analysis (what if i lower the 0.5 threshold?)

---

## results

| metric | decision tree | random forest |
|--------|:------------:|:-------------:|
| accuracy | ~0.97 | ~0.99 |
| precision | ~0.65 | ~0.85 |
| recall | ~0.70 | ~0.88 |
| f1 | ~0.67 | ~0.86 |
| roc-auc | ~0.92 | ~0.98 |
| pr-auc | ~0.60 | ~0.85 |

random forest won on everything. biggest difference was in pr-auc (+41%)  
pr-auc matters more than roc-auc for imbalanced datasets like fraud

---

## charts saved

- `eda_overview.png` - class distribution, amount histogram, hour of day
- `confusion_matrices.png` - side by side for both models  
- `roc_pr_curves.png` - ROC and precision-recall curves
- `feature_importance.png` - which features RF found most useful
- `cross_validation.png` - 5-fold CV boxplot showing stability
- `threshold_analysis.png` - precision/recall/f1 at different thresholds
- `model_comparison.png` - bar chart comparing all 6 metrics

---

## key things i learned

**why random forest is better than single decision tree:**
- 200 trees voting together = less variance (bagging)
- each tree sees random subset of features = trees are diverse
- averaging 200 probabilities is smoother than one tree's hard decision
- much more stable in cross-validation (lower std across folds)

**class imbalance handling:**
- used `class_weight='balanced'` in both models
- with stratify in train_test_split to preserve fraud% in both splits
- pr-auc is better metric than roc-auc for imbalanced data

**threshold tuning:**
- default is 0.5 but can lower it to catch more fraud
- lowering threshold = more recall, less precision
- bank wants high recall (dont miss fraud even if some false alarms)

**real world fraud detection challenges i read about:**
1. real fraud rate is 0.1% - much worse than my 1.8%
2. fraud patterns change over time (concept drift)
3. model needs to decide in <100ms for real-time card approval
4. too many false positives = customers get annoyed when card is blocked
5. fraudsters study the system and try to beat it

---

## what's next

day 19: gradient boosting (XGBoost/LightGBM) - should be even better  
also want to try SMOTE oversampling instead of class_weight approach

---

## files

```
day18/
├── day18_fraud_detection.ipynb   <- main notebook (fully executed)
├── eda_overview.png
├── confusion_matrices.png
├── roc_pr_curves.png
├── feature_importance.png
├── cross_validation.png
├── threshold_analysis.png
├── model_comparison.png
├── build_notebook.py             <- script that generates the notebook
└── README.md
```
