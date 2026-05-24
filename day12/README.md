# Day 12: Regression Modeling — Predicting Continuous Values

Today's focus was shifting from classification (Day 11) to **Regression Modeling** to predict continuous numerical values. Using the preprocessed retail transactions dataset, I built a baseline **Linear Regression** model to predict transaction **`Sales`** values directly, analyzed model coefficients, and checked standard regression assumptions through residual diagnostic plots.

---

## Project Files
*   [day12_regression.ipynb](day12_regression.ipynb): The Jupyter notebook documenting data ingestion, leakage prevention, splitting, model fitting, and diagnostics.
*   [predictions_regression.csv](predictions_regression.csv): Export of test set transactions along with actual sales, predictions, and calculated residual errors.
*   [actual_vs_predicted.png](actual_vs_predicted.png): Scatter plot comparing model predictions against ground truth sales values.
*   [residuals_plot.png](residuals_plot.png): Residual errors vs. predicted sales to check variance consistency.
*   [residuals_dist.png](residuals_dist.png): Histogram and density curve analyzing prediction error distributions.
*   [feature_coefficients.png](feature_coefficients.png): Visualization of feature slopes outlining which variables drive sales.

---

## 🛠️ Step-by-Step Workflow

1.  **Ingestion**: Loaded `engineered_store_transactions.csv` containing 954 observations.
2.  **Target Selection**: Set continuous `Sales` as the target variable.
3.  **Target Leakage Prevention**: Systematically dropped features derived from sales to prevent target leakage:
    *   `Sales_log` and `Sales_log_scaled`
    *   `Sales_per_Unit`, `Sales_per_Unit_log`, and `Sales_per_Unit_log_scaled`
    *   Identifiers and raw variants (`Row ID`, `Order ID`, `Order Date`, `Customer Name`, `Quantity`, `Order_Month`, `Order_Year`, `Order_DayOfWeek`)
4.  **Train-Test Split**: Performed an 80/20 train-test split (`random_state=42`). No stratification was applied since the target is continuous.
5.  **Model Fitting**: Trained a standard Ordinary Least Squares (OLS) `LinearRegression` model.
    *   **Model Intercept ($\beta_0$)**: **$755.88** (the baseline value when features are zero).

---

## 📈 Model Performance Metrics

The model was evaluated on the unseen 20% test partition:

| Metric | Value | Practical Interpretation |
| :--- | :---: | :--- |
| **Mean Absolute Error (MAE)** | **$340.12** | Average prediction error is ~$340. |
| **Mean Squared Error (MSE)** | **163,261.26** | Average of the squared errors (penalizes larger errors). |
| **Root Mean Squared Error (RMSE)** | **$404.06** | Standard deviation of prediction errors. |
| **Mean Absolute Percentage Error (MAPE)** | **281.56%** | High percentage error driven by low-value orders. |
| **Coefficient of Determination ($R^2$)** | **-0.0257** | The model explains none of the variance in the target. |

### Note on the Negative $R^2$:
A negative $R^2$ indicates the model performs worse than a simple horizontal line predicting the test set's mean. This is an expected result for this synthetic dataset, as the uniform distribution generator creates zero physical correlation between category/geography features and the final order value. This highlights that machine learning models cannot find predictive signals where none physically exist in the data.

---

## 🔍 Feature Coefficient Diagnostics

Analyzing the model slopes ($\beta_i$) reveals how specific variables impact predicted order sizes:

*   **Weekend Sales Decline (`Is_Weekend`: -$59.18)**: Weekend orders decrease expected sales by ~$59 compared to weekdays.
*   **Regional ZIP Code Drivers**:
    *   `Zip_02108`: **+$44.98** (strongest regional boost)
    *   `Zip_Unknown`: **-$44.85**
    *   `Zip_10008`: **-$39.65**
*   **Product Categories**:
    *   `Category_Office Supplies`: **+$30.82** (strongest categorical boost)
    *   `Category_Furniture`: **-$16.69**
    *   `Category_Technology`: **-$14.13**
*   **Buyer Segments**:
    *   `Segment_Corporate`: **+$24.72**
    *   `Segment_Consumer`: **-$17.29**

---

## 📊 Visual Diagnostics

Four diagnostic plots were saved to the workspace:

1.  **Actual vs. Predicted Plot (`actual_vs_predicted.png`)**: Shows predictions concentrated horizontally around the mean ($700–$800) because the model predicts close to the constant average in the absence of a correlation signal.
2.  **Residual Plot (`residuals_plot.png`)**: Checks homoscedasticity. Errors are uniformly spread around zero, mirroring the target's underlying distribution.
3.  **Residual Distribution (`residuals_dist.png`)**: Errors show a symmetric, zero-centered distribution, confirming that the predictions are unbiased.
4.  **Coefficient Impact (`feature_coefficients.png`)**: Summarizes the dollar-value impact of each variable.

---

## 🚀 LinkedIn Reflection (Draft)

**Post**:
> 📉 Day 12 of my 60-Day Data Science Challenge! Today's task was implementing a **Linear Regression** pipeline to predict continuous sales values.
>
> Designing the pipeline from split to coefficient interpretation taught me a massive lesson about real-world data constraints:
> 📐 **Evaluation Metrics**: MAE of $340.12 and an R² score of **-0.0257**.
>
> 💡 **Key Takeaway**: 
> A negative R² indicates the model performs worse than predicting the simple mean. Because the underlying synthetic dataset was generated uniformly, there is no physical correlation between categorical variables and transaction sizes. 
> 
> Real data science is about identifying these data constraints and establishing rigorous baselines, rather than forcing models to find patterns in random noise. Pipeline constructed and ready for higher-signal data!
>
> #DataScience #MachineLearning #LinearRegression #Python #ScikitLearn #Statistics #60DayChallenge #ABtalksDS
