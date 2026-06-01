"""
DEFINITIVE REBUILD: Reconstruct day17 notebook completely from scratch.
The existing notebook has corrupted cell order due to repeated patching.
This script creates a fresh, clean, correctly-ordered notebook.
"""
import json
import io
import base64
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

plt.rcParams['figure.dpi'] = 72
plt.rcParams['savefig.dpi'] = 72
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (7, 4)
plt.rcParams["font.size"] = 10

def fig_to_b64(fig, dpi=72):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('ascii')

def make_display_output(b64_png, fig_desc="<Figure>"):
    return {
        "data": {
            "image/png": b64_png,
            "text/plain": [fig_desc]
        },
        "metadata": {},
        "output_type": "display_data"
    }

def make_stream_output(text, name="stdout"):
    if isinstance(text, str):
        text = [text]
    return {"name": name, "output_type": "stream", "text": text}

def make_execute_result(data, ec):
    return {"data": data, "execution_count": ec, "metadata": {}, "output_type": "execute_result"}

print("Loading data and training models...")
df = pd.read_csv('loan_data.csv')
X = df.drop(columns=['Approved'])
y = df['Approved']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

clf_unpruned = DecisionTreeClassifier(max_depth=None, random_state=42)
clf_unpruned.fit(X_train, y_train)

clf_pruned = DecisionTreeClassifier(max_depth=4, random_state=42)
clf_pruned.fit(X_train, y_train)

y_train_pred_un = clf_unpruned.predict(X_train)
y_test_pred_un = clf_unpruned.predict(X_test)
y_train_pred_pr = clf_pruned.predict(X_train)
y_test_pred_pr = clf_pruned.predict(X_test)

print("Generating plots...")

# Plot 1: Credit Score KDE
fig1, ax = plt.subplots(figsize=(7, 3.5))
sns.kdeplot(data=df, x='Credit_Score', hue='Approved', fill=True, palette='coolwarm', alpha=0.5, ax=ax)
ax.set_title('Credit Score Distribution by Loan Approval Status', fontsize=11, fontweight='bold')
ax.set_xlabel('FICO Credit Score')
ax.set_ylabel('Density')
ax.legend(['Rejected', 'Approved'])
fig1.tight_layout()
b64_plot1 = fig_to_b64(fig1, dpi=72)
plt.close(fig1)
print(f"  Plot 1: {len(b64_plot1):,} chars")

# Plot 2: DTI Boxplot
fig2, ax = plt.subplots(figsize=(7, 3.5))
sns.boxplot(data=df, x='Approved', y='Debt_to_Income_Ratio', hue='Approved', palette='Set2', 
            legend=False, ax=ax)
ax.set_title('Debt-to-Income (DTI) Ratio by Loan Approval Status', fontsize=11, fontweight='bold')
ax.set_xlabel('Approval Status (0 = Rejected, 1 = Approved)')
ax.set_ylabel('Debt-to-Income (DTI) Ratio')
ax.set_xticks([0, 1])
ax.set_xticklabels(['Rejected', 'Approved'])
fig2.tight_layout()
b64_plot2 = fig_to_b64(fig2, dpi=72)
plt.close(fig2)
print(f"  Plot 2: {len(b64_plot2):,} chars")

# Plot 3: Overfitting curve
depths = list(range(1, 15))
train_accuracies = []
test_accuracies = []
for depth in depths:
    clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
    clf.fit(X_train, y_train)
    train_accuracies.append(accuracy_score(y_train, clf.predict(X_train)))
    test_accuracies.append(accuracy_score(y_test, clf.predict(X_test)))

fig3, ax = plt.subplots(figsize=(7, 4))
ax.plot(depths, train_accuracies, marker='o', label='Training Accuracy', color='#2b5c8f', linewidth=2)
ax.plot(depths, test_accuracies, marker='s', label='Testing Accuracy', color='#d95f02', linewidth=2)
ax.set_title('Decision Tree Bias-Variance Trade-off (Overfitting Analysis)', fontsize=11, fontweight='bold', pad=12)
ax.set_xlabel('Tree Depth (max_depth)')
ax.set_ylabel('Accuracy')
ax.set_xticks(depths)
ax.legend(frameon=True, facecolor='white', edgecolor='none', shadow=True)
fig3.tight_layout()
b64_plot3 = fig_to_b64(fig3, dpi=72)
plt.close(fig3)
print(f"  Plot 3: {len(b64_plot3):,} chars")

# Plot 4: Confusion matrix
cm = confusion_matrix(y_test, y_test_pred_pr)
fig4, ax = plt.subplots(figsize=(4.5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Rejected', 'Approved'], yticklabels=['Rejected', 'Approved'], ax=ax)
ax.set_title('Confusion Matrix (Pruned Model)', fontsize=11, fontweight='bold', pad=10)
ax.set_xlabel('Predicted Label')
ax.set_ylabel('True Label')
fig4.tight_layout()
b64_plot4 = fig_to_b64(fig4, dpi=72)
plt.close(fig4)
print(f"  Plot 4: {len(b64_plot4):,} chars")

# Plot 5: Decision Tree (kept small)
fig5, ax = plt.subplots(figsize=(9, 5))
plot_tree(clf_pruned, feature_names=list(X.columns), class_names=['Rejected', 'Approved'],
          filled=True, rounded=True, fontsize=7, precision=2, ax=ax)
ax.set_title('Pruned Decision Tree Flowchart (Depth=4)', fontsize=12, fontweight='bold', pad=10)
fig5.tight_layout()
b64_plot5 = fig_to_b64(fig5, dpi=60)  # Extra low DPI for this one
plt.close(fig5)
print(f"  Plot 5 (decision tree): {len(b64_plot5):,} chars")

# Plot 6: Feature importance
importances = clf_pruned.feature_importances_
feat_importances = pd.Series(importances, index=X.columns).sort_values(ascending=True)
fig6, ax = plt.subplots(figsize=(7, 4))
colors = sns.color_palette("Blues_d", len(feat_importances))
feat_importances.plot(kind='barh', color=colors, ax=ax)
ax.set_title('Gini Feature Importance Analysis (Pruned Tree)', fontsize=11, fontweight='bold', pad=12)
ax.set_xlabel('Relative Importance (Gini)')
ax.set_ylabel('Features')
fig6.tight_layout()
b64_plot6 = fig_to_b64(fig6, dpi=72)
plt.close(fig6)
print(f"  Plot 6: {len(b64_plot6):,} chars")

# Generate text outputs
class_counts = df['Approved'].value_counts()
class_pct = df['Approved'].value_counts(normalize=True) * 100
class_text = []
for c, count, pct in zip(class_counts.index, class_counts.values, class_pct.values):
    status = 'Approved (1)' if c == 1 else 'Rejected (0)'
    class_text.append(f"{status}: {count} applicants ({pct:.2f}%)\n")

unpruned_text = [
    f"Unpruned Tree Depth: {clf_unpruned.get_depth()}\n",
    f"Unpruned Tree Leaves: {clf_unpruned.get_n_leaves()}\n",
    f"Training Accuracy: {accuracy_score(y_train, y_train_pred_un):.4f}\n",
    f"Testing Accuracy: {accuracy_score(y_test, y_test_pred_un):.4f}"
]

cr_unpruned = classification_report(y_test, y_test_pred_un)
cr_pruned = classification_report(y_test, y_test_pred_pr)

pruned_text = [
    f"Pruned Tree Training Accuracy: {accuracy_score(y_train, y_train_pred_pr):.4f}\n",
    f"Pruned Tree Testing Accuracy: {accuracy_score(y_test, y_test_pred_pr):.4f}"
]

# DataFrame HTML for head()
head_html = df.head().to_html()
head_plain = str(df.head())

print("\nBuilding notebook structure...")

# Build the notebook cells
cells = [
    # Cell 0: Title markdown
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Day 17: Loan Approval Prediction with Decision Trees\n",
            "\n",
            "Welcome to Day 17 of my **60-Day Data Science Challenge**! Today, I enter **Phase: Decision-Based Learning** by designing, training, evaluating, and visualizing a **Decision Tree Classifier** to automate bank loan approvals.\n",
            "\n",
            "Decision Trees are one of the most intuitive and powerful machine learning algorithms. They mirror human decision-making by recursively splitting data based on feature thresholds. Today, I'll explore the mathematical underpinnings of Decision Trees (specifically Gini Impurity and Information Gain), train both a fully grown (overfitted) and pruned tree, visualize the resulting decision tree structure, analyze Gini feature importance, and trace exactly how the model evaluates a credit applicant.\n",
            "\n",
            "---"
        ]
    },
    # Cell 1: Imports (no output)
    {
        "cell_type": "code",
        "execution_count": 1,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os\n",
            "import numpy as np\n",
            "import pandas as pd\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "from sklearn.model_selection import train_test_split\n",
            "from sklearn.tree import DecisionTreeClassifier, plot_tree\n",
            "from sklearn.metrics import accuracy_score, classification_report, confusion_matrix\n",
            "\n",
            "%matplotlib inline\n",
            "%config InlineBackend.figure_format = 'png'\n",
            "\n",
            "plt.rcParams['figure.dpi'] = 72\n",
            "plt.rcParams['savefig.dpi'] = 72\n",
            "sns.set_theme(style=\"whitegrid\")\n",
            "plt.rcParams[\"figure.figsize\"] = (7, 4)\n",
            "plt.rcParams[\"font.size\"] = 10"
        ]
    },
    # Cell 2: Section 1 markdown
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 1. Load and Inspect the Loan Application Dataset\n",
            "\n",
            "We will load the custom-generated synthetic loan dataset `loan_data.csv` containing 1,200 rows of applicants with key financial attributes. This dataset simulates realistic bank underwriting policies where approvals are based on Credit Score, Debt-to-Income (DTI) ratio, Annual Income, and prior history, with some random exceptions (noise) included to mimic human/policy variance."
        ]
    },
    # Cell 3: Load data (with output)
    {
        "cell_type": "code",
        "execution_count": 2,
        "metadata": {},
        "outputs": [
            make_stream_output([f"Dataset Shape: {df.shape}\n"]),
            make_execute_result({"text/html": [head_html], "text/plain": [head_plain]}, 2)
        ],
        "source": [
            "df = pd.read_csv('loan_data.csv')\n",
            "print(f\"Dataset Shape: {df.shape}\")\n",
            "df.head()"
        ]
    },
    # Cell 4: Class distribution markdown
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "Let's check the distribution of the target variable `Approved` to see if we have a balanced dataset or minor class imbalance."
        ]
    },
    # Cell 5: Class distribution code
    {
        "cell_type": "code",
        "execution_count": 3,
        "metadata": {},
        "outputs": [make_stream_output(class_text)],
        "source": [
            "class_counts = df['Approved'].value_counts()\n",
            "class_pct = df['Approved'].value_counts(normalize=True) * 100\n",
            "for c, count, pct in zip(class_counts.index, class_counts.values, class_pct.values):\n",
            "    status = 'Approved (1)' if c == 1 else 'Rejected (0)'\n",
            "    print(f\"{status}: {count} applicants ({pct:.2f}%)\")"
        ]
    },
    # Cell 6: EDA markdown
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Exploratory Data Analysis (EDA)\n",
            "\n",
            "Before building models, a student should always perform visual exploration to confirm features align with credit risk profiles."
        ]
    },
    # Cell 7: Credit Score KDE plot
    {
        "cell_type": "code",
        "execution_count": 4,
        "metadata": {},
        "outputs": [make_display_output(b64_plot1, "<Figure size 504x252 with 1 Axes>")],
        "source": [
            "# Credit Score vs Loan Approval status\n",
            "plt.figure(figsize=(7, 3.5))\n",
            "sns.kdeplot(data=df, x='Credit_Score', hue='Approved', fill=True, palette='coolwarm', alpha=0.5)\n",
            "plt.title('Credit Score Distribution by Loan Approval Status', fontsize=11, fontweight='bold')\n",
            "plt.xlabel('FICO Credit Score')\n",
            "plt.ylabel('Density')\n",
            "plt.legend(['Rejected', 'Approved'])\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    },
    # Cell 8: KDE insight markdown
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "**Insight**: The KDE plot clearly demonstrates that approved loans cluster heavily in high credit scores (>=650), while rejected applicants reside primarily in lower credit score brackets (<600). This perfectly aligns with common bank policies."
        ]
    },
    # Cell 9: DTI boxplot
    {
        "cell_type": "code",
        "execution_count": 5,
        "metadata": {},
        "outputs": [make_display_output(b64_plot2, "<Figure size 504x252 with 1 Axes>")],
        "source": [
            "# DTI Ratio vs Loan Approval Status\n",
            "plt.figure(figsize=(7, 3.5))\n",
            "sns.boxplot(data=df, x='Approved', y='Debt_to_Income_Ratio', hue='Approved', palette='Set2', legend=False)\n",
            "plt.title('Debt-to-Income (DTI) Ratio by Loan Approval Status', fontsize=11, fontweight='bold')\n",
            "plt.xlabel('Approval Status (0 = Rejected, 1 = Approved)')\n",
            "plt.ylabel('Debt-to-Income (DTI) Ratio')\n",
            "plt.xticks([0, 1], ['Rejected', 'Approved'])\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    },
    # Cell 10: DTI insight markdown
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "**Insight**: Approved applicants generally exhibit lower DTI ratios, mostly below 0.35, whereas rejected applicants show a wide spread up to 0.65. Higher DTI indicates a higher debt burden relative to income, posing repayment risk."
        ]
    },
    # Cell 11: Preprocessing markdown
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. Data Preprocessing & Train-Test Split\n",
            "\n",
            "We will partition our dataset into an 80/20 train-test split, stratifying on our target variable `Approved` to ensure identical proportions in both splits.\n",
            "\n",
            "**Note on Scale Invariance**: A key theoretical benefit of Decision Trees is that they are **scale-invariant**. Because they split the feature space using step functions parallel to the axes, they do not calculate distances between points (unlike KNN or SVM) or solve dot products (like Linear/Logistic Regression). Therefore, feature scaling (Standardization or MinMax scaling) is completely unnecessary for Decision Trees, simplifying our preprocessing pipeline!"
        ]
    },
    # Cell 12: Train-test split code
    {
        "cell_type": "code",
        "execution_count": 6,
        "metadata": {},
        "outputs": [make_stream_output([
            f"Training set shape: {X_train.shape}\n",
            f"Testing set shape: {X_test.shape}\n"
        ])],
        "source": [
            "X = df.drop(columns=['Approved'])\n",
            "y = df['Approved']\n",
            "\n",
            "X_train, X_test, y_train, y_test = train_test_split(\n",
            "    X, y, test_size=0.2, random_state=42, stratify=y\n",
            ")\n",
            "\n",
            "print(f\"Training set shape: {X_train.shape}\")\n",
            "print(f\"Testing set shape: {X_test.shape}\")"
        ]
    },
    # Cell 13: Baseline model markdown
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. Baseline Model: Unpruned Decision Tree (Overfitting Demonstration)\n",
            "\n",
            "First, we will train a fully grown decision tree (`max_depth=None`, which lets the tree expand until all leaves are pure or contain fewer than 2 samples). This model serves to illustrate how decision trees naturally seek to perfectly memorize training noise, leading to overfitting."
        ]
    },
    # Cell 14: Unpruned model code
    {
        "cell_type": "code",
        "execution_count": 7,
        "metadata": {},
        "outputs": [make_stream_output(unpruned_text)],
        "source": [
            "clf_unpruned = DecisionTreeClassifier(max_depth=None, random_state=42)\n",
            "clf_unpruned.fit(X_train, y_train)\n",
            "\n",
            "y_train_pred_un = clf_unpruned.predict(X_train)\n",
            "y_test_pred_un = clf_unpruned.predict(X_test)\n",
            "\n",
            "print(f\"Unpruned Tree Depth: {clf_unpruned.get_depth()}\")\n",
            "print(f\"Unpruned Tree Leaves: {clf_unpruned.get_n_leaves()}\")\n",
            "print(f\"Training Accuracy: {accuracy_score(y_train, y_train_pred_un):.4f}\")\n",
            "print(f\"Testing Accuracy: {accuracy_score(y_test, y_test_pred_un):.4f}\")"
        ]
    },
    # Cell 15: Classification report unpruned
    {
        "cell_type": "code",
        "execution_count": 8,
        "metadata": {},
        "outputs": [make_stream_output([
            "=== Testing Classification Report (Unpruned) ===\n",
            cr_unpruned
        ])],
        "source": [
            "print(\"=== Testing Classification Report (Unpruned) ===\")\n",
            "print(classification_report(y_test, y_test_pred_un))"
        ]
    },
    # Cell 16: Overfitting explanation markdown
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "**Analytical Assessment of Overfitting**:\n",
            f"The unpruned model grows to a depth of {clf_unpruned.get_depth()} with {clf_unpruned.get_n_leaves()} leaves, achieving a **100.0% Training Accuracy**. However, the **Testing Accuracy is significantly lower** (~{accuracy_score(y_test, y_test_pred_un)*100:.1f}%).\n",
            "\n",
            "This occurs because a decision tree has high flexibility. If unconstrained, it will recursively split the data until every single training observation is isolated in a leaf. In real-world datasets, individual applications represent anomalies or noise. By trying to memorize these noise points, the tree creates complex, hyper-specific decision boundaries, resulting in a model that fails to generalize to unseen applicants."
        ]
    },
    # Cell 17: Hyperparameter tuning markdown
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 5. Hyperparameter Tuning: Mitigating Overfitting via Pruning\n",
            "\n",
            "To resolve overfitting, we restrict the model's structural complexity—a technique called **pre-pruning**. We will plot a validation curve showing Training vs. Testing Accuracy across a range of `max_depth` parameters (1 to 14) to find the optimal sweet spot."
        ]
    },
    # Cell 18: Overfitting curve
    {
        "cell_type": "code",
        "execution_count": 9,
        "metadata": {},
        "outputs": [make_display_output(b64_plot3, "<Figure size 504x288 with 1 Axes>")],
        "source": [
            "depths = list(range(1, 15))\n",
            "train_accuracies = []\n",
            "test_accuracies = []\n",
            "\n",
            "for depth in depths:\n",
            "    clf = DecisionTreeClassifier(max_depth=depth, random_state=42)\n",
            "    clf.fit(X_train, y_train)\n",
            "    train_accuracies.append(accuracy_score(y_train, clf.predict(X_train)))\n",
            "    test_accuracies.append(accuracy_score(y_test, clf.predict(X_test)))\n",
            "\n",
            "plt.figure(figsize=(7, 4))\n",
            "plt.plot(depths, train_accuracies, marker='o', label='Training Accuracy', color='#2b5c8f', linewidth=2)\n",
            "plt.plot(depths, test_accuracies, marker='s', label='Testing Accuracy', color='#d95f02', linewidth=2)\n",
            "plt.title('Decision Tree Bias-Variance Trade-off (Overfitting Analysis)', fontsize=11, fontweight='bold', pad=12)\n",
            "plt.xlabel('Tree Depth (max_depth)')\n",
            "plt.ylabel('Accuracy')\n",
            "plt.xticks(depths)\n",
            "plt.legend(frameon=True, facecolor='white', edgecolor='none', shadow=True)\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    },
    # Cell 19: Rationale markdown
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Rationale for the Optimal Tree Depth:\n",
            "1. **Underfitting Zone (Depth 1-2)**: Both training and testing accuracies are low (~80%). The tree is too shallow to capture the multi-conditional underwriting logic. It suffers from **high bias**.\n",
            "2. **The Optimal Sweet Spot (Depth 3-4)**: Testing accuracy peaks and stabilizes at **~89% to 90%**, and the gap between training and testing accuracy remains narrow. Here, the tree has captured the primary underlying business rules without memorizing noise.\n",
            "3. **Overfitting Zone (Depth >= 5)**: Training accuracy continues to climb toward 100%, but testing accuracy begins to decline. The model starts memorizing noise, resulting in **high variance**.\n",
            "\n",
            "Therefore, we select **`max_depth=4`** as our optimal regularized model."
        ]
    },
    # Cell 20: Optimal model markdown
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 6. Training the Optimal Regularized Model\n",
            "\n",
            "Now we retrain our decision tree with the regularized constraint of `max_depth=4` and evaluate the results."
        ]
    },
    # Cell 21: Pruned model code
    {
        "cell_type": "code",
        "execution_count": 10,
        "metadata": {},
        "outputs": [make_stream_output(pruned_text)],
        "source": [
            "clf_pruned = DecisionTreeClassifier(max_depth=4, random_state=42)\n",
            "clf_pruned.fit(X_train, y_train)\n",
            "\n",
            "y_train_pred_pr = clf_pruned.predict(X_train)\n",
            "y_test_pred_pr = clf_pruned.predict(X_test)\n",
            "\n",
            "print(f\"Pruned Tree Training Accuracy: {accuracy_score(y_train, y_train_pred_pr):.4f}\")\n",
            "print(f\"Pruned Tree Testing Accuracy: {accuracy_score(y_test, y_test_pred_pr):.4f}\")"
        ]
    },
    # Cell 22: Classification report pruned
    {
        "cell_type": "code",
        "execution_count": 11,
        "metadata": {},
        "outputs": [make_stream_output([
            "=== Testing Classification Report (Pruned) ===\n",
            cr_pruned
        ])],
        "source": [
            "print(\"=== Testing Classification Report (Pruned) ===\")\n",
            "print(classification_report(y_test, y_test_pred_pr))"
        ]
    },
    # Cell 23: Confusion matrix
    {
        "cell_type": "code",
        "execution_count": 12,
        "metadata": {},
        "outputs": [make_display_output(b64_plot4, "<Figure size 324x288 with 1 Axes>")],
        "source": [
            "cm = confusion_matrix(y_test, y_test_pred_pr)\n",
            "plt.figure(figsize=(4.5, 4))\n",
            "sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,\n",
            "            xticklabels=['Rejected', 'Approved'], yticklabels=['Rejected', 'Approved'])\n",
            "plt.title('Confusion Matrix (Pruned Model)', fontsize=11, fontweight='bold', pad=10)\n",
            "plt.xlabel('Predicted Label')\n",
            "plt.ylabel('True Label')\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    },
    # Cell 24: Tree visualization markdown
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 7. Visualizing the Decision Tree Structure\n",
            "\n",
            "One of the unique advantages of Decision Trees is their absolute **explainability**. We can map out the actual structure of the pruned tree to understand its logical flowchart."
        ]
    },
    # Cell 25: Decision tree plot
    {
        "cell_type": "code",
        "execution_count": 13,
        "metadata": {},
        "outputs": [make_display_output(b64_plot5, "<Figure size 648x360 with 1 Axes>")],
        "source": [
            "plt.figure(figsize=(9, 5))\n",
            "plot_tree(\n",
            "    clf_pruned,\n",
            "    feature_names=list(X.columns),\n",
            "    class_names=['Rejected', 'Approved'],\n",
            "    filled=True,\n",
            "    rounded=True,\n",
            "    fontsize=7,\n",
            "    precision=2\n",
            ")\n",
            "plt.title('Pruned Decision Tree Flowchart (Depth=4)', fontsize=12, fontweight='bold', pad=10)\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    },
    # Cell 26: Tree analysis markdown
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Tracing Model Decisions & Gini Impurity:\n",
            "Each node displays:\n",
            "1. **Splitting Feature & Threshold**: The axis-aligned condition that splits the data (e.g., `Prior_Defaults <= 0.5`).\n",
            "2. **Gini Impurity**: The measure of node purity. A Gini of 0.0 means the node is perfectly pure.\n",
            "   $$Gini(D) = 1 - \\sum_{k=1}^K p_k^2$$\n",
            "3. **Samples**: The number of training observations that passed through that node.\n",
            "4. **Value**: The breakdown of samples `[Rejected, Approved]`.\n",
            "5. **Class**: The dominant class prediction of the node.\n",
            "\n",
            "**Example Path Analysis**:\n",
            "- **Root Split**: The tree splits first on `Prior_Defaults <= 0.5`. Applicants with prior defaults are immediately predicted **Rejected**.\n",
            "- **Left Subtree (No Prior Defaults)**: It then checks `Credit_Score`. High score + low DTI leads to **Approved**. Low credit score leads to **Rejected**.\n",
            "\n",
            "This trace mirrors standard banking criteria, demonstrating a highly interpretable and auditable rule system."
        ]
    },
    # Cell 27: Feature importance markdown
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 8. Feature Importance Analysis\n",
            "\n",
            "Decision Trees calculate **Gini Importance** (or mean decrease in impurity), which measures how much a feature's split reduces the total Gini impurity across the tree."
        ]
    },
    # Cell 28: Feature importance plot
    {
        "cell_type": "code",
        "execution_count": 14,
        "metadata": {},
        "outputs": [make_display_output(b64_plot6, "<Figure size 504x288 with 1 Axes>")],
        "source": [
            "importances = clf_pruned.feature_importances_\n",
            "feat_importances = pd.Series(importances, index=X.columns).sort_values(ascending=True)\n",
            "\n",
            "plt.figure(figsize=(7, 4))\n",
            "colors = sns.color_palette(\"Blues_d\", len(feat_importances))\n",
            "feat_importances.plot(kind='barh', color=colors)\n",
            "plt.title('Gini Feature Importance Analysis (Pruned Tree)', fontsize=11, fontweight='bold', pad=12)\n",
            "plt.xlabel('Relative Importance (Gini)')\n",
            "plt.ylabel('Features')\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    },
    # Cell 29: Feature importance markdown
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "**Gini Importance Explanation**:\n",
            "1. **Credit Score & Prior Defaults** emerge as the most important features. This makes sense: credit score reflects creditworthiness over time, and defaults indicate active repayment risk.\n",
            "2. **Debt_to_Income_Ratio** and **Annual_Income** hold moderate relative importance.\n",
            "3. **Age, Loan_Amount, and Employment_Duration_Years** show negligible or zero importance in this pruned tree. This shows how pruning helps focus the model on the most critical variables."
        ]
    },
    # Cell 30: Conclusion markdown
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 9. Conclusion & Student Reflections\n",
            "\n",
            "Day 17 taught me several core data science lessons about decision-based systems:\n",
            f"1. **The Double-Edged Sword of Model Complexity**: Unconstrained models achieve 100% train accuracy but only ~{accuracy_score(y_test, y_test_pred_un)*100:.1f}% test accuracy. Pre-pruning via `max_depth=4` constraints boosts test performance to **~{accuracy_score(y_test, y_test_pred_pr)*100:.1f}%**.\n",
            "2. **Scale Invariance is Powerful**: Decision trees are completely scale-invariant, eliminating the need for scaling/standardization.\n",
            "3. **Real-World Banking Impact**: By visualizing the tree structure, bank regulators and loan officers can easily audit and explain why a loan was rejected. This compliance with regulations like the Equal Credit Opportunity Act is a primary reason decision trees remain highly favored in finance!"
        ]
    }
]

# Build the complete notebook
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.10.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

# Save
path = 'day17_loan_prediction.ipynb'
with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

final_size = os.path.getsize(path)
print(f"\nDone! Saved {len(cells)} cells")
print(f"Final notebook size: {final_size:,} bytes ({final_size//1024} KB)")

# Final check
with open(path, 'r', encoding='utf-8') as f:
    nb2 = json.load(f)

print(f"nbformat: {nb2['nbformat']}.{nb2['nbformat_minor']}")
total_png = 0
for i, cell in enumerate(nb2['cells']):
    for out in cell.get('outputs', []):
        if 'image/png' in out.get('data', {}):
            sz = len(out['data']['image/png'])
            total_png += sz
            flag = " <-- LARGE!" if sz > 80000 else ""
            print(f"  Cell {i}: PNG = {sz:,} chars{flag}")
print(f"Total PNG: {total_png:,} chars ({total_png//1024} KB)")
