# Day 17: Loan Approval Prediction with Decision Trees

Welcome to Day 17 of my **60-Day Data Science Challenge**! Today, I entered **Phase: Decision-Based Learning** by designing, training, evaluating, and visualizing a **Decision Tree Classifier** to automate bank loan approvals.

Decision-based machine learning systems are highly favored in the financial services sector. Because banks must comply with strict fair lending regulations (e.g., the Equal Credit Opportunity Act), models must not only be highly accurate but also fully auditable and explainable. Decision Trees provide a perfect balance, acting as transparent flowcharts that trace exactly why an applicant was approved or rejected.

In today's work, I built a complete loan approval predictive system, analyzed the mathematics of node splitting (Gini Impurity vs. Entropy), explored the dangers of unconstrained tree growth (overfitting), demonstrated the power of pre-pruning, and mapped out the visual decision pathways.

---

## Workspace Directory Structure

*   [day17_loan_prediction.ipynb](day17_loan_prediction.ipynb): The complete Jupyter notebook containing exploratory data analysis (EDA), data split diagnostics, unpruned baseline training, hyperparameter pre-pruning tuning, tree visualization, and Gini feature importance analysis.
*   [loan_data.csv](loan_data.csv): The custom-generated dataset containing 1,200 synthetic loan applications with features like Credit Score, DTI, Income, and historical prior defaults.
*   [overfitting_validation_curve.png](overfitting_validation_curve.png): A diagnostic line plot comparing training and testing accuracy across tree depths from 1 to 14, highlighting the overfitting threshold.
*   [decision_tree_structure.png](decision_tree_structure.png): The visual layout of the pruned decision tree with detailed splitting criteria, Gini indexes, sample distributions, and predicted classes.
*   [feature_importance.png](feature_importance.png): A horizontal bar chart illustrating the relative Gini importance of each feature in the credit decision process.

---

## Theoretical Framework: Node Splitting Mechanics

At their core, Decision Trees construct a classification flowchart by recursively partitioning the feature space into axis-aligned rectangular regions. The goal at each step is to find the single feature and splitting threshold that maximizes the **homogeneity (purity)** of the child nodes.

We measure node impurity using two primary mathematical metrics:

### 1. Gini Impurity
Gini Impurity measures the probability of a randomly chosen element from the dataset being incorrectly labeled if it were randomly labeled according to the distribution of classes in the subset.

$$Gini(D) = 1 - \sum_{k=1}^{K} p_k^2$$

Where $p_k$ is the proportion of observations in node $D$ belonging to class $k$.
*   **Minimum Value (0.0)**: Represents a perfectly pure node (all samples belong to a single class).
*   **Maximum Value ($1 - 1/K$)**: Represents an even distribution across all $K$ classes (maximum impurity). For binary classification, this is $0.5$.

### 2. Entropy
Entropy represents the average amount of information or uncertainty in the node.

$$H(D) = - \sum_{k=1}^{K} p_k \log_2(p_k)$$

*   **Minimum Value (0.0)**: Perfect purity.
*   **Maximum Value ($\log_2 K$)**: For binary classification, this is $1.0$, representing equal uncertainty.

### 3. Information Gain
To decide on the best split, the algorithm calculates the reduction in impurity, known as **Information Gain (IG)**. For a split on feature $F$ at threshold $t$:

$$IG(D, F) = I(D) - \left( \frac{|D_{left}|}{|D|} I(D_{left}) + \frac{|D_{right}|}{|D|} I(D_{right}) \right)$$

Where $I(D)$ represents either Gini Impurity or Entropy. The split that yields the highest Information Gain is selected.

---

## Data Profile & Preprocessing

The generated dataset contains **1,200 observations** representing a typical modern retail banking portfolio.

### The Scale-Invariance Advantage
Unlike distance-based models (such as K-Nearest Neighbors or Support Vector Machines) or gradient-based models (such as Linear and Logistic Regression), **Decision Trees do not require feature scaling**.
*   **Why?** Splitting conditions are formulated as axis-aligned inequality tests (e.g., $x_j \le t$). The Gini reduction is invariant to any monotonic transformation of the feature (e.g. taking the logarithm or standardizing). Standardizing `Annual_Income` to a Z-score would yield the exact same split point relative to the other observations.
*   **Impact**: Preprocessing is simpler and features retain their original, human-readable units (e.g., actual dollars or FICO score points) throughout the entire pipeline.

---

## The Overfitting Crisis & Pruning Diagnostics

Decision Trees are highly flexible. Left unconstrained, the algorithm will continuously split nodes until all leaf nodes are perfectly pure ($Gini = 0$) or contain fewer than two samples. This causes the tree to memorize the training set, including random noise, resulting in poor generalization on unseen data.

I compared two configurations of the `DecisionTreeClassifier`:
1.  **Unpruned Baseline (`max_depth=None`)**: The tree was allowed to grow until pure.
2.  **Pruned Model (`max_depth=3`)**: The tree was pre-pruned using structural constraints.

### The Bias-Variance Trade-off (`overfitting_validation_curve.png`)

To locate the optimal model complexity, I evaluated training and testing accuracy across tree depths from 1 to 14:

| Depth Constraint | Training Accuracy | Testing Accuracy | Generalization Assessment |
| :---: | :---: | :---: | :--- |
| **Depth = 1** | 80.21% | 80.42% | **High Bias / Underfitting**: The tree is too shallow to capture multi-variable underwriting conditions. |
| **Depth = 2** | 80.21% | 80.42% | **High Bias**: Standard underfitting. Only splits on the most dominant single feature. |
| **Depth = 3** | **84.58%** | **83.75%** | **Sweet Spot**: Optimal generalization. The gap between train and test is narrow, showing high robustness. |
| **Depth = 4** | 89.27% | 82.50% | **Beginning of Overfitting**: Training accuracy climbs, but testing accuracy declines. |
| **Depth = 10** | 98.65% | 76.67% | **Severe Overfitting**: The model is memorizing training noise. |
| **Depth = 26 (Unpruned)** | **100.00%** | **75.42%** | **Catastrophic Overfitting**: Total training memorization with high test-set error (variance). |

### Overfitting Rationale
An unpruned tree achieves **100% training accuracy** at a depth of 26. However, testing accuracy drops to **75.42%**. When we pre-pruned the tree by setting `max_depth=3`, the training accuracy adjusted to **84.58%** while testing accuracy rose to **83.75%**.

By stopping the splits early, we prevent the model from creating complex, highly specific sub-branches designed to isolate noisy, anomalous applications. This results in a much simpler model that generalizes well.

---

## Visualizing the Decision Tree Flowchart

The pruned decision tree structure (`decision_tree_structure.png`) maps out a highly intuitive underwriting flowchart:

```mermaid
graph TD
    Node0["Prior_Defaults <= 0.5\nGini: 0.48\nSamples: 960\nClass: Approved"]
    Node1["Credit_Score <= 667.5\nGini: 0.38\nSamples: 838\nClass: Approved"]
    Node2["Credit_Score <= 579.5\nGini: 0.41\nSamples: 388\nClass: Rejected"]
    Node3["Debt_to_Income_Ratio <= 0.42\nGini: 0.17\nSamples: 450\nClass: Approved"]
    
    Node0 -->|Yes (No Defaults)| Node1
    Node0 -->|No (Prior Defaults)| RejectBranch["Prior Defaults Branch\nGini: 0.20\nSamples: 122\nClass: Rejected"]
    
    Node1 -->|Yes (Credit Score <= 667.5)| Node2
    Node1 -->|No (Credit Score > 667.5)| Node3
```

### Trace Path Analysis
1.  **First Line of Defense**: The root split checks `Prior_Defaults <= 0.5`. If an applicant has a history of default (`Prior_Defaults > 0.5`), they are channeled to the right branch where the rejection rate is **89.3%**.
2.  **Prime Credit Tier**: If the applicant has no prior defaults and a `Credit_Score > 667.5`, the model evaluates their capacity to pay by checking `Debt_to_Income_Ratio <= 0.42`.
    *   If DTI is $\le 0.42$, they are **Approved** with high probability (**97.4% purity**).
    *   If DTI is $> 0.42$, they are **Rejected** due to high debt load relative to income.
3.  **Subprime/Near-Prime Credit Tier**: For applicants with no defaults and a `Credit_Score <= 667.5`:
    *   If FICO is $< 580$, they are immediately **Rejected** (**95.7% purity**).
    *   If FICO is in the mid-tier ($580 \le \text{FICO} \le 667$), the tree evaluates their `Annual_Income` and `Debt_to_Income_Ratio` to assess marginal approvals.

---

## Feature Importance Analysis

Rather than relying on black-box predictions, we extract the relative **Gini Importance** of each feature. This value represents the normalized total reduction of Gini impurity brought by that feature.

### Feature Importance Rankings (`feature_importance.png`):
1.  **Credit Score (54.8%)**: The primary driver of creditworthiness, representing long-term repayment history.
2.  **Prior Defaults (28.7%)**: The strongest single binary risk filter.
3.  **Debt-to-Income Ratio (12.4%)**: The key measure of current capacity to support new debt.
4.  **Annual Income (4.1%)**: Used as a secondary filter for mid-tier credit score applicants.
5.  **Age, Loan Amount, Employment Duration (0.0%)**: These features do not appear in the pruned tree's splits, indicating they do not provide sufficient global Gini reduction when structure is regularized.

---

## LinkedIn Reflection

Here is my professional reflection and learning summary for LinkedIn:

**Post**:
> Day 17 of my 60-Day Data Science Challenge! 📊 Today, I dived into **Phase: Decision-Based Learning** by building, debugging, and visualizing a **Decision Tree Classifier for Loan Approval Prediction**!
>
> In financial services, model explainability is a legal requirement. Under regulations like the Equal Credit Opportunity Act, banks must be able to explain exactly why a credit application was rejected. Decision Trees are highly favored here because they act as transparent, auditable flowcharts.
>
> Key technical and theoretical highlights from today's work:
>
> 📐 **1. Splitting Mechanics (Gini Impurity vs. Entropy):**
> I reviewed the mathematics of node splits. Decision Trees search for feature thresholds that maximize Gini Impurity reduction (Information Gain). A Gini of 0.0 represents perfect homogeneity, making the classification rule highly robust.
>
> ⚖️ **2. The Scale-Invariance Advantage:**
> Because Decision Trees partition features using axis-aligned thresholds ($x \le t$), they are completely scale-invariant! Unlike KNN or Logistic Regression, feature scaling (standardization or normalization) is unnecessary, meaning financial variables (like Annual Income or FICO score) retain their native units throughout training.
>
> 📉 **3. Visualizing the Overfitting Threshold:**
> Decision trees are highly flexible. Left unconstrained (`max_depth=None`), the tree grows infinitely (depth 26, 130+ leaves) to perfectly memorize training noise, leading to overfitting.
> - **Unpruned Model:** 100% Training Accuracy vs. 75.42% Testing Accuracy (High Variance).
> - **Pre-Pruned Model (Optimal Depth = 3):** 84.58% Training Accuracy vs. 83.75% Testing Accuracy. Early pruning restricts the tree from growing sub-branches around noisy exceptions, creating a highly generalized model.
>
> 🌿 **4. Visual Explainability & Auditability:**
> Plotting the pruned tree structure reveals a clean flowchart. The root split uses `Prior_Defaults` as the first filter. If no defaults, the model checks if FICO is above `667.5`, and then branches into `Debt-to-Income (DTI)` thresholds. This mirrors real-world underwriting logic!
>
> 📊 **5. Feature Importance Rankings:**
> Extracting Gini importance showed that Credit Score (54.8%) and Prior Defaults (28.7%) dominate the model's decision-making, while variables like Age and Loan Amount were pruned away as secondary noise.
>
> Understanding not just how a model predicts, but *why* it predicts, is what makes decision-based systems incredibly powerful.
>
> #DataScience #MachineLearning #DecisionTrees #CreditScoring #Python #ScikitLearn #ExplainableAI #BiasVariance #60DayChallenge #ABtalksDS
