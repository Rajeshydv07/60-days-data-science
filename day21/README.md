Day 21 - Choosing the Best Model Like a Real Data Scientist

60 Days Data Science Challenge | Day 21  
Phase: Sprint Review & Model Selection

---

what i did today

today is day 21, the last day of week 3! 
i did a complete sprint review and model selection. i compared all the classification models i've built over the last week on our simulated credit card transaction fraud dataset. i compiled their metrics, analyzed their strengths and weaknesses, and selected the best model for a real-world business system.

---

best model analysis report

here are the final metrics computed on the test set:

| model | accuracy | precision | recall | f1-score | roc-auc | pr-auc |
|---|---|---|---|---|---|---|
| Naive | 0.982 | 0.0 | 0.0 | 0.0 | 0.5 | 0.018 |
| Logistic Regression | 0.984 | 0.631 | 0.266 | 0.375 | 0.937 | 0.408 |
| Decision Tree | 0.980 | 0.368 | 0.155 | 0.218 | 0.528 | 0.093 |
| Random Forest | 0.983 | 0.800 | 0.088 | 0.160 | 0.979 | 0.489 |
| XGBoost | 0.983 | 0.555 | 0.333 | 0.416 | 0.984 | 0.580 |
| LightGBM | 0.984 | 0.586 | 0.377 | 0.459 | 0.979 | 0.535 |

---

model strengths and weaknesses

1. Naive Baseline
- strengths: zero training time, super simple.
- weaknesses: completely useless. it gets 98.20% accuracy just by predicting "no fraud" for everything, but it catches exactly 0 cases of fraud. it shows why accuracy is a dangerous metric on imbalanced data.

2. Logistic Regression
- strengths: trains instantly, highly interpretable, coefficients show feature impact directly.
- weaknesses: struggles heavily with class imbalance and non-linear patterns, missing 73.33% of fraud cases.

3. Decision Tree
- strengths: highly interpretable structure, can capture non-linear relationships.
- weaknesses: splits are too simple and biased toward the majority class without class weight tuning. it got poor recall (15.56%) and precision (36.84%).

4. Random Forest
- strengths: ensemble bagging reduces variance. got high precision (80.00%), meaning very few false alarms.
- weaknesses: recall was extremely low (8.89%). despite setting class_weight='balanced', the trees struggled to find deep splits on the rare fraud cases in this simulated sample.

5. XGBoost
- strengths: sequential boosting reduces bias. got a very solid PR-AUC (0.5806) and recall (33.33%).
- weaknesses: harder to tune, training takes slightly longer than LightGBM.

6. LightGBM
- strengths: leaf-wise tree growth is very efficient. achieved the best overall balance with the highest F1-score (0.4595) and the highest recall (37.78%). it's also incredibly fast to train.
- weaknesses: harder to interpret than a single tree or logistic regression, complex hyperparameter tuning.

---

model selection and business justification

selected model: LightGBM

in a real bank's fraud detection system, we must balance two competing business costs:
1. cost of false negatives (missed fraud): if a transaction is fraud and we miss it, the bank loses the transaction amount and pays chargeback fees.
2. cost of false positives (customer friction): if a transaction is legit and we block it, we annoy the customer and increase support center costs.

why LightGBM is the best fit:
- highest recall (37.78%): it caught 17 out of 45 fraud cases, which is the highest among all tested models under default settings.
- best balance (F1-score of 45.95%): it provides the best harmonic mean of precision and recall.
- precision of 58.62%: out of 29 flagged transactions, 17 were actual fraud. this keeps the customer false-alarm rate low enough to avoid blocking too many legit accounts.
- scalability: it trains in under 0.05 seconds and has very low prediction latency, which is critical for real-time payment gateways that must decide in milliseconds.

---

week 3 engineering reflection

this week was all about building different classification architectures and learning how to evaluate them. here's a recap of the key technical takeaways:

1. recommender systems (day 16): built a movie recommender system using collaborative filtering. learned about calculating similarity scores (cosine similarity) and how user-item matrices are processed.
2. decision trees & hyperparameter tuning (day 17): learned how single decision trees split data using entropy or gini impurity. noticed that deep trees overfit easily and how constraint parameters like max_depth help keep them in check.
3. bagging vs boosting (day 18 & 19):
   - bagging (random forest): averages predictions from multiple independent deep trees. good for reducing variance.
   - boosting (xgboost/lgbm): trains trees sequentially, where each tree corrects the residual errors of the previous ones. good for capturing complex boundaries and reducing bias.
4. the accuracy paradox (day 20 & 21): this was the biggest eye-opener. accuracy is a useless metric for imbalanced datasets. a model can get 98.2% accuracy by doing nothing. we must use recall, precision, F1-score, and PR-AUC.
5. handling class imbalance: using parameters like class_weight='balanced' in sklearn or scale_pos_weight in boosting models is crucial. they tell the algorithm to penalize errors on the minority class more heavily, which is the only way the model learns to identify rare events like fraud.

---

linkedin reflection

*here is my week 3 reflection post:*

---

Week 3 of my 60-Day Data Science Challenge is in the books! This week was focused on classification architectures and the realities of model evaluation in production.

I built a series of classifiers (Logistic Regression, Decision Trees, Random Forests, XGBoost, and LightGBM) and evaluated them head-to-head on an imbalanced credit card fraud dataset.

Here are my main takeaways from this week:
- Accuracy is a dangerous metric: A completely naive model that always predicts "no fraud" gets 98.20% accuracy on our dataset. If you rely on accuracy alone, you will deploy a model that catches zero fraud and loses money.
- Precision-Recall AUC is the real truth-teller: Traditional ROC-AUC looks high (>0.93) because of the massive number of true negatives. PR-AUC focuses on precision and recall, exposing the actual performance differences.
- Boosting vs Bagging: XGBoost and LightGBM outperformed Random Forest on this task. By sequentially focusing on residual errors and using scale_pos_weight, they captured significantly more fraud cases.
- Business Tradeoffs: Choosing a model isn't just about the highest score. It’s about balancing the financial cost of missed fraud (recall) against the customer friction of blocking legitimate transactions (precision). 

LightGBM was my final pick—offering the best F1-score (45.95%) and recall (37.78%) while maintaining a tiny training and inference footprint. 

Next week, I'll be moving into unsupervised learning and clustering.

---

files

```
day21/
├── day21_best_model_selection.ipynb  <- executed comparison notebook
├── confusion_matrices.png            <- grid of confusion matrices
├── roc_pr_curves.png                 <- ROC and Precision-Recall curves
├── model_comparison.png              <- summary bar chart of all metrics
├── build_notebook.py                 <- script that generates and runs the notebook
└── README.md                         <- this report and reflection file
```
