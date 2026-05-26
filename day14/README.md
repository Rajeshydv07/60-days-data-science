# Day 14: Adapting Models to Changing Constraints (Sprint 2 Review)

In real-world data science, datasets are rarely static. A feature that is highly predictive today might disappear tomorrow due to data privacy regulations (like GDPR), hardware sensor failures, or a change in database tracking. 

Today's task is a "Sprint Review" challenge: we simulate a sudden constraint shift by removing our single most important feature from Day 13 (**`Is_Weekend`**), retraining our models (OLS, Ridge, and Lasso), and analyzing how the system adapts to this loss.

---

## Files in this Directory
*   [day14_adaptation.ipynb](day14_adaptation.ipynb): Jupyter notebook containing the full pipeline of dropping the feature, retraining, sweeps, and coefficient comparison.
*   [predictions_adaptation.csv](predictions_adaptation.csv): Test set predictions and residuals for the three retrained models (without `Is_Weekend`).
*   [coefficient_paths_adaptation.png](coefficient_paths_adaptation.png): Visual showing how the coefficients shrink on the restricted dataset.
*   [train_test_performance_adaptation.png](train_test_performance_adaptation.png): Performance curves showing the bias-variance tradeoff without `Is_Weekend`.

---

## Performance Comparison: Before vs. After Feature Removal

Here is the exact breakdown of model performance on the 20% test set (191 observations) before and after removing the primary feature `Is_Weekend` (which had a Lasso coefficient of `-38.42`):

| Model Configuration | Train MAE | Test MAE | Train RMSE | Test RMSE | Train $R^2$ | Test $R^2$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **OLS (With Is_Weekend)** | \$346.9100 | \$340.1244 | \$412.4665 | \$404.0560 | 0.015702 | -0.025654 |
| **OLS (Without Is_Weekend)** | \$347.2600 | \$341.5468 | \$413.3210 | \$404.5453 | 0.011619 | -0.028139 |
| **Ridge (With Is_Weekend)** | \$346.6294 | \$339.1086 | \$412.8848 | \$402.1872 | 0.013704 | -0.016188 |
| **Ridge (Without Is_Weekend)** | \$346.8661 | \$339.8677 | \$413.6377 | \$402.5551 | 0.010104 | -0.018048 |
| **Lasso (With Is_Weekend)** | \$346.7240 | \$337.9061 | \$413.3415 | \$401.0180 | 0.011521 | -0.010288 |
| **Lasso (Without Is_Weekend)** | \$346.9371 | \$338.6134 | \$414.1003 | \$401.3406 | 0.007889 | -0.011915 |

### Key Observations
*   **Systemic Drop in Performance**: Removing a valid predictive signal causes a slight but clear decrease in performance across all models. For our best model, Lasso ($\alpha=5$), the test $R^2$ dropped from `-0.0103` to `-0.0119`, and the test RMSE rose from `$401.02` to `$401.34`.
*   **Regularization Buffers the Blow**: Regularization continues to keep the model from exploding. Standard OLS without the feature degrades the most, with test $R^2$ dropping to `-0.0281`. Lasso and Ridge restrict the models from chasing noise, keeping the test RMSE very close to the baseline.

---

## How the Model Adapts: Coefficient Weight Redistribution

When a major feature is dropped, does the regularized model start utilizing noisy features to compensate? Or does it adjust the weight of the remaining true signals? Let's look at the Lasso ($\alpha=5$) coefficient shifts:

| Feature Name | Coefficient (Before) | Coefficient (After) | Weight Shift |
| :--- | :---: | :---: | :---: |
| `Is_Weekend` | **-38.4247** | *Removed* | N/A |
| `Category_Office Supplies` | **+22.7522** | **+23.7859** | **+1.0337** |
| `Zip_10008` | **-21.4478** | **-22.8934** | **-1.4456** |
| `Zip_02108` | **+18.6767** | **+19.7802** | **+1.1035** |
| `Segment_Corporate` | **+13.5603** | **+15.2552** | **+1.6949** |
| `Order_Month_scaled` | **-8.7808** | **-9.0320** | **-0.2512** |
| *All other 9 features* | **0.0000** | **0.0000** | **0.0000** |

### Key Insights on System Adaptation
1.  **Explosion of the Remaining Signals**: The model compensates for the missing explanatory power by expanding the coefficients of the remaining active features. `Segment_Corporate` took the biggest boost, shifting by `+1.69`, while the regional zip codes also gained about `+1.10` each in magnitude.
2.  **Zero-Boundary Resilience**: This is the coolest part of Lasso regression. Even though we dropped our strongest feature, Lasso **did not** resurrect any of the 9 features it had previously driven to zero (like scaled quantity, furniture category, or home office segments). The L1 penalty keeps the model's sparsity boundaries robust—it would rather expand the weight of existing valid signals than let noise back into the model.

---

## Sprint 2 Reflection (Days 8-14)

Sprint 2 has been a massive eye-opener for how data science works in practice, moving from clean toy problems to the messy reality of modeling. Here is a reflection of what I have learned over the last seven days:

### 1. The Value of Feature Engineering (Days 8-10)
In Sprint 1, we worked with raw columns. Sprint 2 taught me that models are only as good as the representations we feed them. Creating log transforms for skewed sales data, extracting datetime features like `Is_Weekend`, and scaling continuous variables are what actually unlock linear model capability. Without standardizing features, coefficients are meaningless; with scaling, we can compare their dollar-impact directly.

### 2. The Battle with Noise and Overfitting (Days 11-13)
The retail transaction dataset we worked with is highly noisy (which is common in real retail environments). Watching our baseline Ordinary Least Squares (OLS) regression get a positive $R^2$ on the training set but a negative $R^2$ on the test set was a classic lesson in overfitting. OLS is incredibly naive—it will fit whatever random fluctuations are in the training partition. 
Learning Ridge (L2) and Lasso (L1) regularization gave me the tools to combat this bias-variance tradeoff. Seeing Lasso actively zero-out 9 out of 15 features to simplify our model and lower the test RMSE was incredibly satisfying.

### 3. Graceful Degradation in Production (Day 14)
Today's challenge was a wake-up call about system robustness. In production, we will lose features. What matters is how our models degrade. A simple OLS model degrades poorly because it overfits on whatever is left. Regularized models degrade gracefully—they scale up the weights of the remaining true signals and maintain a strict barrier against noise.

---

## LinkedIn Reflection

Here is my daily summary post for LinkedIn:

**Post**:
> Day 14 of my 60-Day Data Science Challenge! 📉 Today was Sprint 2 Review day, and it came with a massive real-world twist: **Adapting to changing constraints**.
>
> In production, features disappear. A sensor fails, an API changes, or privacy laws require dropping certain trackers. To test the robustness of my regression pipeline, I dropped my single most important predictor—`Is_Weekend`—and retrained all my models.
>
> Here is how the system adapted:
>
> 1. The Performance Cost:
> Dropping our best feature caused a systemic dip. Lasso (alpha=5) test R² dropped from -0.0103 to -0.0119, showing the real cost of losing data.
>
> 2. Explaining the Adaptation:
> The regularized model adapted by scaling up the coefficients of the remaining active features (like Category_Office Supplies and Segment_Corporate) to try to compensate for the lost signal.
>
> 3. Robustness of Sparsity:
> Lasso did not let noise creep in. The 9 features that were previously driven to absolute zero remained exactly zero. Regularization kept the model simple and robust, preventing it from grabbing noisy features to make up for the loss.
>
> This marks the end of Sprint 2! Over the last week, I've gone from feature engineering and log-scaling to fighting overfitting with L1/L2 penalties and handling real-world data loss. 
>
> Moving on to Sprint 3! 🚀
>
> #DataScience #MachineLearning #Python #ScikitLearn #Statistics #60DayChallenge #ABtalksDS #SprintReview
