# Capstone Project: Model Optimization & Comparison Report

## 1. Overview
In Phase 3 (Day 50) of the Customer Intelligence Platform, we shifted our focus from simply building a functioning end-to-end pipeline to extracting the maximum predictive power from our data. The baseline Random Forest model provided a solid starting point, but production systems demand rigor in model selection and tuning. 

This report documents our evaluation of multiple machine learning algorithms, our hyperparameter tuning process, and the engineering tradeoffs involved.

## 2. Baseline Model Evaluation
Our initial baseline model (Random Forest with default parameters) demonstrated the following characteristics:
* **Accuracy:** ~84-86%
* **F1 Score:** ~0.82
* **Behavior:** Slight tendency to overfit the training data due to unconstrained tree depth (`max_depth=None`).

We established this as our benchmark to beat.

## 3. Algorithm Comparison
To ensure we were using the right algorithmic approach, we compared three distinct models:

1. **Logistic Regression:**
   * **Pros:** Highly interpretable (coefficients directly show feature impact), fast training, less prone to severe overfitting on simple data.
   * **Cons:** Assumes linear relationships; struggles with complex, non-linear interactions in customer behavior.
   * **Result:** Lowest overall F1 score, serving as our linear baseline.

2. **Random Forest (Ensemble Bagging):**
   * **Pros:** Handles non-linear data well, robust to outliers, provides feature importance out-of-the-box.
   * **Cons:** Can become memory-intensive, harder to interpret than linear models.
   * **Result:** Strong performance out of the box, with high accuracy.

3. **Gradient Boosting (Ensemble Boosting):**
   * **Pros:** Often achieves the highest predictive accuracy by sequentially correcting errors of previous trees.
   * **Cons:** Prone to overfitting if not tuned carefully, slower sequential training time.
   * **Result:** Slightly outperformed the baseline Random Forest in precision, but required careful tuning.

## 4. Hyperparameter Tuning & Overfitting Analysis
We selected **Random Forest** for deep tuning due to its balance of high accuracy and parallelizable training speed.

### Tuning Strategy (GridSearchCV)
We optimized the following hyperparameters to combat overfitting:
* `n_estimators`: Increased from 100 to [50, 100, 200] to improve ensemble stability.
* `max_depth`: Constrained to [10, 20] (vs. default `None`) to prevent trees from memorizing the training set (reducing variance).
* `min_samples_split`: Adjusted to ensure leaf nodes generalize better.

### Underfitting vs. Overfitting
* **Overfitting (High Variance):** The default Random Forest was memorizing noise, evidenced by near 100% training accuracy but lower validation accuracy. Constraining `max_depth` reduced training accuracy slightly but improved the test F1 score, demonstrating better generalization.
* **Underfitting (High Bias):** Logistic Regression exhibited higher bias, failing to capture the complex patterns in customer churn, leading to lower overall metrics.

## 5. Engineering Tradeoffs & Final Decision
* **Performance vs. Interpretability:** We chose an ensemble model (Random Forest) over a linear model (Logistic Regression). While we lose direct mathematical interpretability, we gain significant predictive power. We offset the loss of interpretability by using Feature Importance charts.
* **Training Speed vs. Accuracy:** Gradient Boosting offered marginal gains in accuracy but at the cost of much slower, non-parallel training times. Random Forest allowed us to leverage `n_jobs=-1` for parallel training, which is crucial as our data scales.

### Conclusion
The tuned Random Forest model provides the best balance of generalization, accuracy, and training efficiency for our Customer Intelligence Platform. The optimized hyperparameters successfully mitigated overfitting, resulting in a more robust model for production deployment.
