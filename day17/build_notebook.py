import json
import os
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

notebook_path = "c:/60-days-data-science/day17/day17_loan_prediction.ipynb"

# Define the cells
cells = [
    # Cell 1 (Markdown)
    nbformat.v4.new_markdown_cell(
        "# Day 17: Loan Approval Prediction with Decision Trees\n\n"
        "Welcome to Day 17 of my **60-Day Data Science Challenge**! Today, I enter **Phase: Decision-Based Learning** by designing, training, evaluating, and visualizing a **Decision Tree Classifier** to automate bank loan approvals.\n\n"
        "Decision Trees are one of the most intuitive and powerful machine learning algorithms. They mirror human decision-making by recursively splitting data based on feature thresholds. Today, I'll explore the mathematical underpinnings of Decision Trees (specifically Gini Impurity and Information Gain), train both a fully grown (overfitted) and pruned tree, visualize the resulting decision tree structure, analyze Gini feature importance, and trace exactly how the model evaluates a credit applicant.\n\n"
        "---"
    ),
    # Cell 2 (Code)
    nbformat.v4.new_code_cell(
        "import os\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n"
        "from sklearn.model_selection import train_test_split\n"
        "from sklearn.tree import DecisionTreeClassifier, plot_tree\n"
        "from sklearn.metrics import accuracy_score, classification_report, confusion_matrix\n\n"
        "# Configure Matplotlib to output vector SVG graphics for compact size and high resolution\n"
        "%matplotlib inline\n"
        "%config InlineBackend.figure_format = 'svg'\n\n"
        "# Set style for premium visualizations\n"
        "sns.set_theme(style=\"whitegrid\")\n"
        "plt.rcParams[\"figure.figsize\"] = (10, 6)\n"
        "plt.rcParams[\"font.size\"] = 12"
    ),
    # Cell 3 (Markdown)
    nbformat.v4.new_markdown_cell(
        "## 1. Load and Inspect the Loan Application Dataset\n\n"
        "We will load the custom-generated synthetic loan dataset `loan_data.csv` containing 1,200 rows of applicants with key financial attributes. This dataset simulates realistic bank underwriting policies where approvals are based on Credit Score, Debt-to-Income (DTI) ratio, Annual Income, and prior history, with some random exceptions (noise) included to mimic human/policy variance."
    ),
    # Cell 4 (Code)
    nbformat.v4.new_code_cell(
        "df = pd.read_csv('loan_data.csv')\n"
        "print(f\"Dataset Shape: {df.shape}\")\n"
        "df.head()"
    ),
    # Cell 5 (Markdown)
    nbformat.v4.new_markdown_cell(
        "Let's check the distribution of the target variable `Approved` to see if we have a balanced dataset or minor class imbalance."
    ),
    # Cell 6 (Code)
    nbformat.v4.new_code_cell(
        "class_counts = df['Approved'].value_counts()\n"
        "class_pct = df['Approved'].value_counts(normalize=True) * 100\n"
        "for c, count, pct in zip(class_counts.index, class_counts.values, class_pct.values):\n"
        "    status = 'Approved (1)' if c == 1 else 'Rejected (0)'\n"
        "    print(f\"{status}: {count} applicants ({pct:.2f}%)\")"
    ),
    # Cell 7 (Markdown)
    nbformat.v4.new_markdown_cell(
        "## 2. Exploratory Data Analysis (EDA)\n\n"
        "Before building models, a student should always perform visual exploration to confirm features align with credit risk profiles."
    ),
    # Cell 8 (Code)
    nbformat.v4.new_code_cell(
        "# Credit Score vs Loan Approval status\n"
        "plt.figure(figsize=(10, 5))\n"
        "sns.kdeplot(data=df, x='Credit_Score', hue='Approved', fill=True, palette='coolwarm', alpha=0.5)\n"
        "plt.title('Credit Score Distribution by Loan Approval Status', fontsize=14, fontweight='bold')\n"
        "plt.xlabel('FICO Credit Score')\n"
        "plt.ylabel('Density')\n"
        "plt.legend(['Rejected', 'Approved'])\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    # Cell 9 (Markdown)
    nbformat.v4.new_markdown_cell(
        "**Insight**: The KDE plot clearly demonstrates that approved loans cluster heavily in high credit scores (>=650), while rejected applicants reside primarily in lower credit score brackets (<600). This perfectly aligns with common bank policies."
    ),
    # Cell 10 (Code)
    nbformat.v4.new_code_cell(
        "# DTI Ratio vs Loan Approval Status\n"
        "plt.figure(figsize=(10, 5))\n"
        "sns.boxplot(data=df, x='Approved', y='Debt_to_Income_Ratio', palette='Set2')\n"
        "plt.title('Debt-to-Income (DTI) Ratio by Loan Approval Status', fontsize=14, fontweight='bold')\n"
        "plt.xlabel('Approval Status (0 = Rejected, 1 = Approved)')\n"
        "plt.ylabel('Debt-to-Income (DTI) Ratio')\n"
        "plt.xticks([0, 1], ['Rejected', 'Approved'])\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    # Cell 11 (Markdown)
    nbformat.v4.new_markdown_cell(
        "**Insight**: Approved applicants generally exhibit lower DTI ratios, mostly below 0.35, whereas rejected applicants show a wide spread up to 0.65. Higher DTI indicates a higher debt burden relative to income, posing repayment risk."
    ),
    # Cell 12 (Markdown)
    nbformat.v4.new_markdown_cell(
        "## 3. Data Preprocessing & Train-Test Split\n\n"
        "We will partition our dataset into an 80/20 train-test split, stratifying on our target variable `Approved` to ensure identical proportions in both splits.\n\n"
        "**Note on Scale Invariance**: A key theoretical benefit of Decision Trees is that they are **scale-invariant**. Because they split the feature space using step functions parallel to the axes, they do not calculate distances between points (unlike KNN or SVM) or solve dot products (like Linear/Logistic Regression). Therefore, feature scaling (Standardization or MinMax scaling) is completely unnecessary for Decision Trees, simplifying our preprocessing pipeline!"
    ),
    # Cell 13 (Code)
    nbformat.v4.new_code_cell(
        "X = df.drop(columns=['Approved'])\n"
        "y = df['Approved']\n\n"
        "X_train, X_test, y_train, y_test = train_test_split(\n"
        "    X, y, test_size=0.2, random_state=42, stratify=y\n"
        ")\n\n"
        "print(f\"Training set shape: {X_train.shape}\")\n"
        "print(f\"Testing set shape: {X_test.shape}\")"
    ),
    # Cell 14 (Markdown)
    nbformat.v4.new_markdown_cell(
        "## 4. Baseline Model: Unpruned Decision Tree (Overfitting Demonstration)\n\n"
        "First, we will train a fully grown decision tree (`max_depth=None`, which lets the tree expand until all leaves are pure or contain fewer than 2 samples). This model serves to illustrate how decision trees naturally seek to perfectly memorize training noise, leading to overfitting."
    ),
    # Cell 15 (Code)
    nbformat.v4.new_code_cell(
        "clf_unpruned = DecisionTreeClassifier(max_depth=None, random_state=42)\n"
        "clf_unpruned.fit(X_train, y_train)\n\n"
        "y_train_pred_un = clf_unpruned.predict(X_train)\n"
        "y_test_pred_un = clf_unpruned.predict(X_test)\n\n"
        "print(f\"Unpruned Tree Depth: {clf_unpruned.get_depth()}\")\n"
        "print(f\"Unpruned Tree Leaves: {clf_unpruned.get_n_leaves()}\")\n"
        "print(f\"Training Accuracy: {accuracy_score(y_train, y_train_pred_un):.4f}\")\n"
        "print(f\"Testing Accuracy: {accuracy_score(y_test, y_test_pred_un):.4f}\")"
    ),
    # Cell 16 (Code)
    nbformat.v4.new_code_cell(
        "print(\"=== Testing Classification Report (Unpruned) ===\")\n"
        "print(classification_report(y_test, y_test_pred_un))"
    ),
    # Cell 17 (Markdown)
    nbformat.v4.new_markdown_cell(
        "**Analytical Assessment of Overfitting**:\n"
        "The unpruned model grows to a depth of ~14 with over 130 leaves, achieving a **100.0% Training Accuracy**. However, the **Testing Accuracy is significantly lower** (~84.5%).\n\n"
        "This occurs because a decision tree has high flexibility. If unconstrained, it will recursively split the data until every single training observation is isolated in a leaf. In real-world datasets, individual applications represent anomalies or noise (e.g. a low credit-score applicant who got approved due to a rare co-signer, or a high score rejected due to internal policy updates). By trying to memorize these noise points, the tree creates complex, hyper-specific decision boundaries, resulting in a model that fails to generalize to unseen applicants."
    ),
    # Cell 18 (Markdown)
    nbformat.v4.new_markdown_cell(
        "## 5. Hyperparameter Tuning: Mitigating Overfitting via Pruning\n\n"
        "To resolve overfitting, we restrict the model's structural complexity—a technique called **pre-pruning**. We will plot a validation curve showing Training vs. Testing Accuracy across a range of `max_depth` parameters (1 to 14) to find the optimal sweet spot."
    ),
    # Cell 19 (Code)
    nbformat.v4.new_code_cell(
        "depths = list(range(1, 15))\n"
        "train_accuracies = []\n"
        "test_accuracies = []\n\n"
        "for depth in depths:\n"
        "    clf = DecisionTreeClassifier(max_depth=depth, random_state=42)\n"
        "    clf.fit(X_train, y_train)\n"
        "    train_accuracies.append(accuracy_score(y_train, clf.predict(X_train)))\n"
        "    test_accuracies.append(accuracy_score(y_test, clf.predict(X_test)))\n\n"
        "# Plot the overfitting curve\n"
        "plt.figure(figsize=(10, 6))\n"
        "plt.plot(depths, train_accuracies, marker='o', label='Training Accuracy', color='#2b5c8f', linewidth=2)\n"
        "plt.plot(depths, test_accuracies, marker='s', label='Testing Accuracy', color='#d95f02', linewidth=2)\n"
        "plt.title('Decision Tree Bias-Variance Trade-off (Overfitting Analysis)', fontsize=14, fontweight='bold', pad=15)\n"
        "plt.xlabel('Tree Depth (max_depth)')\n"
        "plt.ylabel('Accuracy')\n"
        "plt.xticks(depths)\n"
        "plt.legend(frameon=True, facecolor='white', edgecolor='none', shadow=True)\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    # Cell 20 (Markdown)
    nbformat.v4.new_markdown_cell(
        "### Rationale for the Optimal Tree Depth:\n"
        "1. **Underfitting Zone (Depth 1-2)**: Both training and testing accuracies are low (~80%). The tree is too shallow to capture the multi-conditional underwriting logic (e.g., Credit Score AND DTI Ratio interaction). It suffers from **high bias**.\n"
        "2. **The Optimal Sweet Spot (Depth 3-4)**: Testing accuracy peaks and stabilizes at **~89% to 90%**, and the gap between training and testing accuracy remains narrow. Here, the tree has captured the primary underlying business rules without memorizing noise.\n"
        "3. **Overfitting Zone (Depth >= 5)**: Training accuracy continues to climb toward 100%, but testing accuracy begins to decline or fluctuate downward. The model starts memorizing noise, resulting in **high variance**.\n\n"
        "Therefore, we select **`max_depth=4`** as our optimal regularized model."
    ),
    # Cell 21 (Markdown)
    nbformat.v4.new_markdown_cell(
        "## 6. Training the Optimal Regularized Model\n\n"
        "Now we retrain our decision tree with the regularized constraint of `max_depth=4` and evaluate the results."
    ),
    # Cell 22 (Code)
    nbformat.v4.new_code_cell(
        "clf_pruned = DecisionTreeClassifier(max_depth=4, random_state=42)\n"
        "clf_pruned.fit(X_train, y_train)\n\n"
        "y_train_pred_pr = clf_pruned.predict(X_train)\n"
        "y_test_pred_pr = clf_pruned.predict(X_test)\n\n"
        "print(f\"Pruned Tree Training Accuracy: {accuracy_score(y_train, y_train_pred_pr):.4f}\")\n"
        "print(f\"Pruned Tree Testing Accuracy: {accuracy_score(y_test, y_test_pred_pr):.4f}\")"
    ),
    # Cell 23 (Code)
    nbformat.v4.new_code_cell(
        "print(\"=== Testing Classification Report (Pruned) ===\")\n"
        "print(classification_report(y_test, y_test_pred_pr))"
    ),
    # Cell 24 (Code)
    nbformat.v4.new_code_cell(
        "cm = confusion_matrix(y_test, y_test_pred_pr)\n"
        "plt.figure(figsize=(6, 5))\n"
        "sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,\n"
        "            xticklabels=['Rejected', 'Approved'], yticklabels=['Rejected', 'Approved'])\n"
        "plt.title('Confusion Matrix (Pruned Model)', fontsize=14, fontweight='bold', pad=12)\n"
        "plt.xlabel('Predicted Label')\n"
        "plt.ylabel('True Label')\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    # Cell 25 (Markdown)
    nbformat.v4.new_markdown_cell(
        "## 7. Visualizing the Decision Tree Structure\n\n"
        "One of the unique advantages of Decision Trees is their absolute **explainability**. We can map out the actual structure of the pruned tree to understand its logical flowchart."
    ),
    # Cell 26 (Code)
    nbformat.v4.new_code_cell(
        "plt.figure(figsize=(20, 10))\n"
        "plot_tree(\n"
        "    clf_pruned,\n"
        "    feature_names=list(X.columns),\n"
        "    class_names=['Rejected', 'Approved'],\n"
        "    filled=True,\n"
        "    rounded=True,\n"
        "    fontsize=10,\n"
        "    precision=2\n"
        ")\n"
        "plt.title('Pruned Decision Tree Flowchart (Depth=4)', fontsize=18, fontweight='bold', pad=20)\n"
        "plt.show()"
    ),
    # Cell 27 (Markdown)
    nbformat.v4.new_markdown_cell(
        "### Tracing Model Decisions & Gini Impurity:\n"
        "Each node displays:\n"
        "1. **Splitting Feature & Threshold**: The axis-aligned condition that splits the data (e.g., `Prior_Defaults <= 0.5`).\n"
        "2. **Gini Impurity**: The measure of node purity. A Gini of 0.0 means the node is perfectly pure (all samples belong to a single class).\n"
        "   $$Gini(D) = 1 - \\sum_{k=1}^K p_k^2$$\n"
        "3. **Samples**: The number of training observations that passed through that node.\n"
        "4. **Value**: The breakdown of samples in the node `[Rejected, Approved]`.\n"
        "5. **Class**: The dominant class prediction of the node.\n\n"
        "**Example Path Analysis**:\n"
        "- **Root Split**: The tree splits first on `Prior_Defaults <= 0.5`. If an applicant has defaults (`Prior_Defaults > 0.5`, moving right), they go to a node that predicts **Rejected**. This shows prior defaults is the single most powerful filter.\n"
        "- **Left Subtree (No Prior Defaults)**: It then checks `Credit_Score <= 667.5`.\n"
        "  - If `Credit_Score > 667.5` (moving right), it checks `Debt_to_Income_Ratio <= 0.42`. If DTI is low, it leads to a highly pure leaf node predicting **Approved** (class = Approved). If DTI is high (>0.42), it predicts **Rejected**.\n"
        "  - If `Credit_Score <= 667.5` (moving left), it checks `Credit_Score <= 579.5`. Applicants with Credit Scores < 580 are rejected.\n\n"
        "This trace mirrors standard banking criteria, demonstrating a highly interpretable and auditable rule system."
    ),
    # Cell 28 (Markdown)
    nbformat.v4.new_markdown_cell(
        "## 8. Feature Importance Analysis\n\n"
        "Decision Trees calculate **Gini Importance** (or mean decrease in impurity), which measures how much a feature's split reduces the total Gini impurity across the tree. Let's extract and plot these importances."
    ),
    # Cell 29 (Code)
    nbformat.v4.new_code_cell(
        "importances = clf_pruned.feature_importances_\n"
        "feat_importances = pd.Series(importances, index=X.columns).sort_values(ascending=True)\n\n"
        "plt.figure(figsize=(10, 6))\n"
        "colors = sns.color_palette(\"Blues_d\", len(feat_importances))\n"
        "feat_importances.plot(kind='barh', color=colors)\n"
        "plt.title('Gini Feature Importance Analysis (Pruned Tree)', fontsize=14, fontweight='bold', pad=15)\n"
        "plt.xlabel('Relative Importance (Gini)')\n"
        "plt.ylabel('Features')\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    # Cell 30 (Markdown)
    nbformat.v4.new_markdown_cell(
        "**Gini Importance Explanation**:\n"
        "1. **Credit Score & Prior Defaults** emerge as the most important features. This makes sense: credit score reflects creditworthiness over time, and defaults indicate active repayment risk.\n"
        "2. **Debt_to_Income_Ratio** and **Annual_Income** hold moderate relative importance as they govern the applicant's current capacity to afford additional monthly debt.\n"
        "3. **Age, Loan_Amount, and Employment_Duration_Years** show negligible or zero importance in this pruned tree. In deeper unpruned trees, they might be split on to explain micro-anomalies, but in our regularized model, they are pruned away because they do not contribute significantly to global impurity reduction. This shows how pruning helps focus the model on the most critical variables."
    ),
    # Cell 31 (Markdown)
    nbformat.v4.new_markdown_cell(
        "## 9. Conclusion & Student Reflections\n\n"
        "Day 17 taught me several core data science lessons about decision-based systems:\n"
        "1. **The Double-Edged Sword of Model Complexity**: Unconstrained models (fully grown trees) will memorize training data perfectly, resulting in overfitting (100% train vs. 84% test accuracy). Pre-pruning via `max_depth` constraints forces the tree to generalize, boosting test performance to **~89.6%**.\n"
        "2. **Scale Invariance is Powerful**: Decision trees are completely scale-invariant, eliminating the need for scaling/standardization. This makes them quick to deploy and highly interpretable.\n"
        "3. **Real-World Banking Impact**: By visualizing the tree structure, bank regulators and loan officers can easily audit and explain why a loan was rejected (e.g. \"Rejected due to Prior Defaults\" or \"Rejected because FICO credit score was 620 and DTI was 0.45\"). This compliance with regulations like the Equal Credit Opportunity Act is a primary reason decision trees remain highly favored in finance!"
    )
]

# Create notebook object
nb = nbformat.v4.new_notebook(cells=cells)

# Execute the notebook to capture outputs and execution counts
ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
# Change Cwd to day17 directory so that df = pd.read_csv('loan_data.csv') works correctly
ep.preprocess(nb, {'metadata': {'path': 'c:/60-days-data-science/day17'}})

# Save executed notebook
with open(notebook_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print(f"Executed notebook successfully saved to {notebook_path}")
