# Day 20 - Why Accuracy Alone Can Mislead You

60 Days Data Science Challenge | Day 20  
Topic: Model Evaluation - Beyond Accuracy

---

## what i did today

today is day 20, and i focused on **Model Evaluation**. 
over the last few weeks, i've built five different classification models (logistic regression, decision tree, random forest, xgboost, and lightgbm). 
but today, i put them all in a room together and ran a direct head-to-head scientific comparison on the highly imbalanced fraud dataset from day 18/19.

i also added a **Naive Classifier** (a dumb control model that just predicts "no fraud" for every single transaction) to prove a massive point: **why accuracy is a highly dangerous and misleading metric.**

here is the absolute mind-blowing result: the naive model that does absolutely zero learning gets **98.20% accuracy**, which is actually *better* than some of our trained machine learning models!

---

## dataset

to make it a perfectly fair scientific test, i recreated the exact same synthetic credit card transaction dataset from the last two days:
- 10,000 transactions total
- 180 fraud cases (~1.8% fraud rate)
- features: V1-V15, Amount, Hour, Class
- 75/25 stratified split (retaining the 1.8% fraud ratio in both train and test)

---

## steps i followed

1. simulated the imbalanced credit card transaction dataset (with seed 42 so it's 100% identical)
2. scaled the numerical features (`Amount` and `Hour`) using `StandardScaler`
3. implemented the **Naive Classifier** baseline (predicts `Class=0` for everything)
4. trained the **Logistic Regression** baseline (unweighted - Day 11/15 style)
5. trained the **Decision Tree** (unweighted, max_depth=5 - Day 17 style)
6. trained the **Random Forest** (weighted, class_weight='balanced' - Day 18 style)
7. trained **XGBoost** (weighted, scale_pos_weight - Day 19 style)
8. trained **LightGBM** (weighted, scale_pos_weight - Day 19 style)
9. compiled all performance metrics (accuracy, precision, recall, f1-score, roc-auc, and pr-auc) into a pandas comparison dataframe
10. plotted a 2x3 grid of confusion matrices to visually inspect false negatives (missed fraud) vs false positives (customer annoyance)
11. plotted ROC curves and Precision-Recall (PR) curves side-by-side to compare model performance on minority class detection
12. generated a bar chart comparing all primary metrics across all 6 models

---

## results

here are the exact metrics on the test set:

| model | accuracy | precision | recall | f1-score | roc-auc | pr-auc |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Naive (Predict 0)** | 0.9820 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.0180 |
| **Logistic Regression** | 0.9840 | 0.6316 | 0.2667 | 0.3750 | 0.9376 | 0.4087 |
| **Decision Tree** | 0.9800 | 0.3684 | 0.1556 | 0.2188 | 0.5280 | 0.0935 |
| **Random Forest** | 0.9832 | 0.8000 | 0.0889 | 0.1600 | 0.9795 | 0.4893 |
| **XGBoost** | 0.9832 | 0.5556 | 0.3333 | 0.4167 | 0.9848 | 0.5806 |
| **LightGBM** | 0.9840 | 0.5862 | 0.3778 | 0.4595 | 0.9796 | 0.5355 |

### key observations:
- **the accuracy paradox is real:** the naive model gets **98.20% accuracy**, which is higher than our trained Decision Tree (98.00%) and matches the Random Forest and XGBoost (98.32%)! if a business only checked accuracy, they would deploy the naive model, which catches **zero** actual fraud cases.
- **precision vs recall trade-offs:**
  - standard unweighted classifiers (logistic regression, decision tree, random forest) struggle heavily with **Recall**. for example, random forest got a high precision of 0.80 but a pathetic recall of 0.0889 (it missed 41 out of 45 fraud cases!).
  - sequential boosting models (xgboost and lightgbm) achieve the best overall balance. lightgbm caught 17 fraud cases (recall = 0.3778) while maintaining a solid precision (0.5862) and achieving the highest F1-score (0.4595).
- **roc-auc is highly optimistic:** all our trained models get ROC-AUC scores over 0.93 (with XGBoost at 0.9848). this is because ROC-AUC incorporates true negatives (TN), which are massive in our dataset (2,455 cases). 
- **pr-auc is the truth-teller:** PR-AUC focuses entirely on the minority class (recall and precision). here, we see the real difference: decision tree has a PR-AUC of only 0.0935, while XGBoost achieves 0.5806.

---

## charts saved

- `confusion_matrices.png` - a 2x3 grid showing confusion matrices for all 6 models side-by-side. you can visually see the false negatives (bottom-left) and false positives (top-right).
- `roc_pr_curves.png` - side-by-side ROC and Precision-Recall curves. it perfectly illustrates how ROC-AUC looks overly optimistic, while the PR curve shows the true performance differences.
- `model_comparison.png` - a bar chart comparing accuracy, precision, recall, and f1-score across all models.

---

## key things i learned

**why accuracy is misleading (the accuracy paradox):**
- accuracy treats all classes as equally important. if class imbalance is severe (e.g. 98.2% legit and 1.8% fraud), a model can just guess the majority class and be 98.2% accurate while being completely useless.

**the primary metrics explained in simple terms:**
- **accuracy:** out of all predictions, how many were correct? (useless for imbalanced data).
- **precision:** when the model flags a transaction as fraud, how often is it actually fraud? (important for avoiding customer annoyance and false alarms).
- **recall (sensitivity):** out of all actual fraud cases, how many did the model catch? (extremely important for stopping losses).
- **f1-score:** the harmonic mean of precision and recall. it gives a single, honest score of how well a model balances both.
- **roc-auc:** measures the model's ability to rank a random positive sample higher than a random negative sample across all thresholds.
- **pr-auc (average precision):** measures the area under the precision-recall curve. this is the absolute gold standard for imbalanced classification because it doesn't get distorted by a massive number of true negatives.

**business implications of metric choice:**
- if a bank prioritizes **Recall**, they want to catch every single fraud case. they will lower the probability threshold, which will catch more fraud but increase false positives (annoying customers whose cards get blocked).
- if they prioritize **Precision**, they want to make sure they only block cards when they are absolutely sure. this reduces false positives but increases false negatives (letting fraud slide).

---

## linkedin reflection

*here is my reflection post for today's challenge:*

---

Day 20 of my 60-Day Data Science Challenge is complete. today, I hit a massive milestone in Phase: Model Evaluation. 

I set up a head-to-head scientific comparison between 5 different machine learning classifiers I've built over the last few weeks: Logistic Regression, Decision Trees, Random Forests, XGBoost, and LightGBM.

I also added a "Naive Classifier" (a completely dumb control model that simply predicts "no fraud" for every single transaction) to test a classic concepts: the Accuracy Paradox.

The findings were a massive wake-up call on why accuracy is a dangerous metric:
1. The completely naive model achieved a stellar 98.20% accuracy. If you only look at accuracy, you would think this model is production-ready. In reality, it caught zero actual fraud cases.
2. Unweighted classifiers like standard Logistic Regression and Decision Trees struggled heavily with Recall (catching only 15% to 26% of fraud cases) because they were biased toward the majority class.
3. The PR-AUC (Precision-Recall Area Under Curve) proved to be the ultimate truth-teller. While ROC-AUC looked optimistic (all models scored >0.93 due to the huge number of True Negatives), PR-AUC revealed the true performance, with XGBoost leading at 0.5806 compared to Decision Tree's 0.0935.
4. Model tradeoffs are real: Boosting models (XGBoost/LightGBM) achieved the highest overall balance (F1-score of 0.4595), catching the most fraud cases while keeping false alarms low.

Key Takeaway: In production ML systems, high accuracy is often an illusion. When dealing with imbalanced real-world data (fraud, medical diagnosis, churn), we must evaluate systems using Precision, Recall, F1-score, and PR-AUC. Relying on accuracy alone will mislead you and hurt the business.

Next up, I'll be looking into threshold tuning and advanced sampling techniques like SMOTE to see if we can push these recall scores even higher!

---

## files

```
day20/
├── day20_model_evaluation.ipynb  <- main executed notebook comparing all models
├── confusion_matrices.png        <- 2x3 grid of confusion matrices
├── roc_pr_curves.png             <- ROC and PR curve comparisons
├── model_comparison.png          <- bar chart comparing primary metrics
├── build_notebook.py             <- script that programmatically generates and runs the notebook
└── README.md                     <- this notes and reflection file
```
