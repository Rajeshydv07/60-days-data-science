# Day 19 - Boosting Model Performance with XGBoost & LightGBM

60 Days Data Science Challenge | Day 19  
Topic: Advanced Ensemble Systems - Gradient Boosting

---

## what i did today

today i took the credit card fraud model from yesterday and upgraded it from **Random Forest** (bagging) to **Gradient Boosting** (XGBoost and LightGBM) to see if we can push model performance even further. 

gradient boosting is basically the king of tabular datasets, so i wanted to see how much of a difference it makes in a real-world imbalanced classification task.

---

## dataset

to make it a 100% fair scientific test, i generated the exact same synthetic credit card transaction dataset as yesterday:
- 10,000 transactions total
- 180 fraud cases (~1.8% fraud rate)
- features: V1-V15 (PCA style), Amount, Hour, Class

---

## steps i followed

1. installed and configured `xgboost` and `lightgbm` in the environment
2. recreated the synthetic dataset (using seed 42 to make sure it's identical)
3. preprocessed - scaled `Amount` and `Hour`, did a stratified train/test split (75/25)
4. trained the **Random Forest** baseline (200 trees, balanced weights)
5. trained **XGBoost Classifier** (200 trees, eta=0.05, max_depth=6, scale_pos_weight=54.55)
6. trained **LightGBM Classifier** (200 trees, num_leaves=31, learning_rate=0.05, scale_pos_weight=54.55)
7. analyzed training improvements using **learning curves** (tracking training/validation logloss over iterations)
8. compared all three models side-by-side on all performance metrics (precision, recall, f1, roc-auc, pr-auc) and training time
9. plotted confusion matrices, ROC and PR curves, and feature importances for all models
10. ran 5-fold stratified cross-validation to verify model stability

---

## results

here are the performance metrics on the test set:

| metric | Random Forest (baseline) | XGBoost | LightGBM |
|--------|:-----------------------:|:-------:|:--------:|
| accuracy | ~0.99 | ~0.99 | ~0.99 |
| precision | ~0.85 | ~0.89 | ~0.89 |
| recall | ~0.88 | ~0.96 | ~0.96 |
| f1 | ~0.86 | ~0.92 | ~0.92 |
| roc-auc | ~0.981 | ~0.998 | ~0.998 |
| pr-auc | ~0.852 | ~0.970 | ~0.970 |
| train time | ~0.45s | ~0.35s | ~0.08s |

### observations
- **recall** went from **0.88** (Random Forest) to **0.96** (XGBoost/LightGBM). in credit card fraud, this means we caught 3 additional fraud cases that Random Forest missed entirely.
- **pr-auc** improved significantly, jumping from **0.85** to **0.97** (+12%). this is huge because precision-recall is the gold standard for highly imbalanced data.
- **training speed:** LightGBM was the fastest, taking only **0.08 seconds** to train, compared to 0.45s for Random Forest and 0.35s for XGBoost. 

---

## charts saved

- `eda_overview.png` - class distribution, amount, and hour distributions
- `learning_curves.png` - logloss vs boosting iterations for train and validation sets, showing convergence
- `confusion_matrices.png` - side-by-side confusion matrices for Random Forest, XGBoost, and LightGBM
- `roc_pr_curves.png` - multi-model comparison of ROC and Precision-Recall curves
- `feature_importance.png` - top feature importances comparison for all three models
- `cross_validation.png` - boxplot of 5-fold CV ROC-AUC scores showing model stability
- `model_comparison.png` - summary bar chart of all metrics across all models

---

## key things i learned

**bagging vs boosting:**
- **bagging (random forest):** builds independent, deep trees in parallel. reduces model variance by averaging many diverse predictions.
- **boosting (xgboost/lgbm):** builds shallow trees sequentially. each tree is trained to predict the *residuals* (errors) of all previous trees. reduces bias and variance.

**why gradient boosting is superior for tabular data:**
- sequentially correcting residuals lets the model learn complex boundaries where linear models or simple bagging structures struggle.
- gradient boosting algorithms include regularization (L1/L2 in XGBoost) to prevent overfitting, making them very robust.

**xgboost vs lightgbm architectures:**
- **xgboost:** traditionally grows trees **level-wise** (depth-wise). it scans all features and splits level by level.
- **lightgbm:** grows trees **leaf-wise**. it finds the leaf with the maximum loss reduction and splits it first. this is why lgbm is incredibly fast and uses less memory, though it can overfit more easily on small datasets if not regularized.

**loss curves & training convergence:**
- by plotting the logloss of train and validation sets across boosting iterations, we can see exactly when the model stops learning useful patterns and starts overfitting.
- the loss converges rapidly within the first 40-50 trees, which tells us that adding hundreds of extra trees is unnecessary and might lead to overfitting without early stopping.

---

## linkedin reflection

*here is my reflection post for today's challenge:*

---

Day 19 of my 60-Day Data Science Challenge is complete. Today, I took my credit card fraud detection project to the next level by exploring Advanced Ensemble Systems. 

I upgraded my workflow from yesterday's Random Forest classifier to gradient boosting, comparing XGBoost and LightGBM against the Random Forest baseline.

Here is what I accomplished:
- Installed and configured XGBoost and LightGBM on a highly imbalanced credit card fraud dataset.
- Configured class imbalance parameters (scale_pos_weight) to handle the 98:2 class ratio.
- Evaluated models using Precision-Recall AUC (PR-AUC) as the primary metric, since traditional ROC-AUC can be overly optimistic on imbalanced data.
- Plotted loss convergence curves over training iterations to find the optimal early stopping point.
- Ran 5-fold stratified cross-validation to guarantee the statistical stability of the results.

The Results:
- Baseline Random Forest: PR-AUC of 0.852, Recall of 0.88.
- XGBoost: PR-AUC of 0.970, Recall of 0.96.
- LightGBM: PR-AUC of 0.970, Recall of 0.96.

Gradient boosting captured complex fraud patterns that Random Forest missed, boosting recall by 8% and catching nearly all fraud cases on the test set. In terms of training speed, LightGBM was the standout performer, completing training in just 0.08 seconds (5x faster than Random Forest).

Key Takeaway: While bagging reduces variance by voting, sequential boosting focuses on hard-to-predict instances by learning from previous tree residuals. However, regularizing max_depth and utilizing early stopping are critical to prevent boosting models from overfitting training noise.

Next, I will be looking into hyperparameter tuning (GridSearchCV/Optuna) to fine-tune these architectures even further.

---

## files

```
day19/
├── day19_boosting_models.ipynb   <- main executed notebook
├── eda_overview.png
├── learning_curves.png           <- training vs validation loss convergence
├── confusion_matrices.png        <- side-by-side errors
├── roc_pr_curves.png             <- evaluation curves
├── feature_importance.png        <- model feature importances
├── cross_validation.png          <- 5-fold CV stability boxplot
├── model_comparison.png          <- summary bar chart of all metrics
├── build_notebook.py             <- notebook generation script
└── README.md                     <- day 19 notes and reflection
```
