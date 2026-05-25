# Day 13: Overfitting and Regularization (Ridge & Lasso Regression)

Today's project focuses on model optimization—specifically, how to detect overfitting and use regularization (Ridge and Lasso regression) to make models generalize better instead of just memorizing training data.

Building on the retail transactions dataset from yesterday, I trained baseline, Ridge, and Lasso models to see how L1 and L2 penalties handle the noisy, synthetic sales data.

---

## Files in this Directory
*   [day13_regularization.ipynb](day13_regularization.ipynb): Jupyter notebook containing the full pipeline: data split, OLS baseline, Ridge/Lasso parameter sweeps, and diagnostic plots.
*   [predictions_regularization.csv](predictions_regularization.csv): Test set predictions and residuals for all three models (OLS, Ridge at $\alpha=100$, and Lasso at $\alpha=5$).
*   [coefficient_paths.png](coefficient_paths.png): Line plot showing how feature coefficients shrink as regularization strength ($\alpha$) increases.
*   [train_test_performance.png](train_test_performance.png): Performance curve showing train vs. test $R^2$ across different alpha values to visualize the bias-variance tradeoff.

---

## Workflow Summary

1.  **Data Split**: Kept the exact same 80/20 split (`random_state=42`) and dropped the same target leakage features as Day 12 to ensure a fair comparison.
2.  **Baseline OLS**: Fitted a standard `LinearRegression` model. As expected, it showed clear signs of overfitting: a positive training $R^2$ but a negative test $R^2$.
3.  **Ridge Regression (L2 Penalty)**: Swept $\alpha$ values from 0.1 to 1000. Under L2 regularization, coefficients are shrunk proportionally but none are driven to absolute zero.
4.  **Lasso Regression (L1 Penalty)**: Swept $\alpha$ values from 0.01 to 20. Lasso successfully drove non-essential coefficients to exactly zero, simplifying the model.
5.  **Evaluation**: Extracted coefficients and test set predictions to compare residuals and metrics.

---

## Model Performance

All models were evaluated on the 20% test set (191 observations):

| Model | Train MAE | Test MAE | Train RMSE | Test RMSE | Train $R^2$ | Test $R^2$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **OLS Baseline** | \$346.9100 | \$340.1244 | \$412.4665 | \$404.0560 | 0.015702 | -0.025654 |
| **Ridge ($\alpha=100$)** | \$346.6294 | \$339.1086 | \$412.8848 | \$402.1872 | 0.013704 | -0.016188 |
| **Lasso ($\alpha=5$)** | \$346.7240 | \$337.9061 | \$413.3415 | \$401.0180 | 0.011521 | -0.010288 |

### Key Observations
*   **The Overfitting Signature**: The OLS baseline has a positive train $R^2$ (0.0157) but a negative test $R^2$ (-0.0257). This is a classic indicator that the model is fitting random noise in the training set and generalizes poorly to unseen data.
*   **Improving Generalization**: Both regularization methods successfully improved test performance. Ridge ($\alpha=100$) brought the test $R^2$ up to -0.0162, and Lasso ($\alpha=5$) brought it up to -0.0103. 
*   **RMSE Reduction**: Lasso achieved the lowest test RMSE (\$401.02 vs \$404.06 for OLS), confirming that stripping out noise improves prediction accuracy on new data.

---

## Lasso Feature Selection ($\alpha=5$)

Lasso regression acts as an embedded feature selector by driving less useful weights to zero. Out of the 15 features, Lasso kept only 6 and eliminated the other 9:

### Features Kept:
*   `Is_Weekend` (-38.42): Sales are slightly lower on weekends.
*   `Category_Office Supplies` (+22.75): Office supplies category gets a positive sales boost.
*   `Zip_10008` (-21.45): Regional location drag.
*   `Zip_02108` (+18.68): Regional location boost.
*   `Segment_Corporate` (+13.56): Corporate customer segment boost.
*   `Order_Month_scaled` (-8.78): Minor negative coefficient for order month.

### Features Eliminated (Coefficient = 0.0000):
`Segment_Consumer`, `Segment_Home Office`, `Category_Furniture`, `Category_Technology`, `Zip_60610`, `Zip_77041`, `Zip_90036`, `Zip_Unknown`, `Quantity_scaled`

On this synthetic dataset, features like Quantity (when scaled) and specific categories have no true relationship with order value. Lasso correctly identified these as noise and stripped them from the model.

---

## Visual Insights

1.  **Coefficient Paths (`coefficient_paths.png`)**: 
    *   **Ridge**: The coefficients decay asymptotically towards zero as alpha increases, but they never quite hit it.
    *   **Lasso**: The coefficients drop linearly and hit exactly zero at different thresholds, showing the L1 feature elimination mechanism.
2.  **Performance Curves (`train_test_performance.png`)**: Shows the classic bias-variance tradeoff. As alpha increases, training performance drops (more bias), but testing performance increases (less variance), pulling the negative test $R^2$ closer to 0.0.

---

## LinkedIn Reflection

Here is my daily summary for today's challenge:

**Post**:
> Day 13 of my 60-Day Data Science Challenge! Today was all about overfitting and how to prevent machine learning models from memorizing training data.
>
> I tested Ridge (L2) and Lasso (L1) regularization on the retail sales dataset to see how they compare to a standard OLS regression:
>
> 1. The Overfitting Trap:
> A standard OLS model had a positive R² on the training data (0.0157) but a negative R² on test data (-0.0257). It was fitting random noise instead of real patterns.
>
> 2. Ridge (L2) Regularization:
> Adding an L2 penalty shrinks all coefficients. At alpha=100, it reduced the test gap, bringing test R² up to -0.0162.
>
> 3. Lasso (L1) Regularization:
> Lasso went a step further. At alpha=5, it drove 9 of the 15 features (like scaled quantity and generic customer segments) to exactly zero, keeping only the 6 strongest signals. This simplified model achieved the best test R² of -0.0103 and dropped test RMSE by $3.
>
> Key takeaway: Simple linear models will happily overfit when there is no real signal in the data. Regularization is essential to control variance, enforce sparsity, and make sure models actually generalize to unseen data.
>
> #DataScience #MachineLearning #Python #ScikitLearn #Statistics #60DayChallenge #ABtalksDS
