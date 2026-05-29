import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Create directory if it doesn't exist
os.makedirs("c:/60-days-data-science/day17", exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)

# Generate 1200 synthetic loan applications
n_samples = 1200

# 1. Feature Generation
age = np.random.randint(21, 66, size=n_samples)
annual_income = np.round(np.random.normal(75000, 25000, size=n_samples))
annual_income = np.clip(annual_income, 20000, 180000)

credit_score = np.round(np.random.normal(660, 85, size=n_samples))
credit_score = np.clip(credit_score, 300, 850).astype(int)

# DTI ratio: lower is generally better
dti_ratio = np.round(np.random.beta(2, 5, size=n_samples) * 0.7, 3)
dti_ratio = np.clip(dti_ratio, 0.05, 0.65)

# Loan Amount: generally proportional to income with some noise
loan_amount = np.round((annual_income * np.random.uniform(0.15, 0.45, size=n_samples)) / 1000) * 1000
loan_amount = np.clip(loan_amount, 5000, 60000)

employment_duration = np.round(np.random.exponential(scale=6.0, size=n_samples), 1)
employment_duration = np.clip(employment_duration, 0, age - 20)

# Prior defaults (binary)
prior_defaults = np.random.choice([0, 1], size=n_samples, p=[0.88, 0.12])

# 2. Define Ground Truth Rules for Approval
# Underwriting criteria:
# 1. If Prior Defaults == 1 -> Rejected (92% probability of rejection)
# 2. If Credit Score >= 670 AND DTI <= 0.40 -> Approved (90% probability)
# 3. If Credit Score >= 580 AND Credit Score < 670 AND Income >= 50000 AND DTI <= 0.35 -> Approved (75% probability)
# 4. If Credit Score < 580 -> Rejected (95% probability)
# 5. Otherwise -> Rejected

approved = []
for i in range(n_samples):
    p_def = prior_defaults[i]
    c_score = credit_score[i]
    dti = dti_ratio[i]
    inc = annual_income[i]
    
    # Base approval probability based on underwriting guidelines
    if p_def == 1:
        p_approve = 0.08
    elif c_score >= 670:
        if dti <= 0.42:
            p_approve = 0.92
        else:
            p_approve = 0.20
    elif c_score >= 580:
        if inc >= 45000 and dti <= 0.35:
            p_approve = 0.78
        else:
            p_approve = 0.15
    else:
        p_approve = 0.04
        
    # Generate binary label with calculated probability (introducing structured noise)
    approved.append(np.random.choice([1, 0], p=[p_approve, 1 - p_approve]))

approved = np.array(approved)

# Create DataFrame
df = pd.DataFrame({
    'Age': age,
    'Annual_Income': annual_income,
    'Credit_Score': credit_score,
    'Loan_Amount': loan_amount,
    'Debt_to_Income_Ratio': dti_ratio,
    'Employment_Duration_Years': employment_duration,
    'Prior_Defaults': prior_defaults,
    'Approved': approved
})

# Save to CSV
csv_path = "c:/60-days-data-science/day17/loan_data.csv"
df.to_csv(csv_path, index=False)
print(f"Dataset saved to {csv_path}. Shape: {df.shape}")
print(f"Class Balance:\n{df['Approved'].value_counts(normalize=True)}")

# 3. Train-Test Split
X = df.drop(columns=['Approved'])
y = df['Approved']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Overfitting Analysis - Tuning max_depth
depths = range(1, 15)
train_accs = []
test_accs = []

for depth in depths:
    clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
    clf.fit(X_train, y_train)
    train_accs.append(accuracy_score(y_train, clf.predict(X_train)))
    test_accs.append(accuracy_score(y_test, clf.predict(X_test)))

# Save Overfitting Curve
plt.figure(figsize=(10, 6))
sns.set_theme(style="whitegrid")
plt.plot(depths, train_accs, marker='o', label='Training Accuracy', color='#2b5c8f', linewidth=2)
plt.plot(depths, test_accs, marker='s', label='Testing Accuracy', color='#d95f02', linewidth=2)
plt.title('Decision Tree Bias-Variance Trade-off (Overfitting Analysis)', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Tree Depth (max_depth)', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.xticks(depths)
plt.legend(frameon=True, facecolor='white', edgecolor='none', shadow=True)
plt.tight_layout()
validation_curve_path = "c:/60-days-data-science/day17/overfitting_validation_curve.png"
plt.savefig(validation_curve_path, dpi=300)
plt.close()
print(f"Validation curve saved to {validation_curve_path}")

# 5. Train Pruned Model (Sweet Spot: max_depth=3 or 4)
# Let's check which depth has the highest test accuracy
optimal_depth = depths[np.argmax(test_accs)]
print(f"Optimal Depth found programmatically: {optimal_depth}")
# We'll use max_depth=3 or 4 for readability in the visual tree plot, let's stick to 3 or 4.
selected_depth = min(max(optimal_depth, 3), 4)
print(f"Selected Depth for visualization: {selected_depth}")

clf_pruned = DecisionTreeClassifier(max_depth=selected_depth, random_state=42)
clf_pruned.fit(X_train, y_train)

# 6. Train Fully Grown Model (Overfit)
clf_overfit = DecisionTreeClassifier(max_depth=None, random_state=42)
clf_overfit.fit(X_train, y_train)

print(f"\nUnpruned Tree (Depth={clf_overfit.get_depth()}):")
print(f"Train Accuracy: {accuracy_score(y_train, clf_overfit.predict(X_train)):.4f}")
print(f"Test Accuracy: {accuracy_score(y_test, clf_overfit.predict(X_test)):.4f}")

print(f"\nPruned Tree (Depth={selected_depth}):")
print(f"Train Accuracy: {accuracy_score(y_train, clf_pruned.predict(X_train)):.4f}")
print(f"Test Accuracy: {accuracy_score(y_test, clf_pruned.predict(X_test)):.4f}")

# 7. Save Tree Visualization
plt.figure(figsize=(20, 10))
plot_tree(
    clf_pruned,
    feature_names=list(X.columns),
    class_names=['Rejected', 'Approved'],
    filled=True,
    rounded=True,
    fontsize=10,
    precision=2
)
plt.title(f'Pruned Decision Tree Structure (max_depth={selected_depth})', fontsize=16, fontweight='bold', pad=20)
tree_path = "c:/60-days-data-science/day17/decision_tree_structure.png"
plt.savefig(tree_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Decision tree structure saved to {tree_path}")

# 8. Save Feature Importance Plot
importances = clf_pruned.feature_importances_
indices = np.argsort(importances)[::-1]
feat_importances = pd.Series(importances, index=X.columns).sort_values(ascending=True)

plt.figure(figsize=(10, 6))
# Create horizontal bar plot with premium color palette
colors = sns.color_palette("Blues_d", len(feat_importances))
feat_importances.plot(kind='barh', color=colors)
plt.title('Gini Feature Importance Analysis (Pruned Tree)', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Relative Importance (Gini)', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.tight_layout()
importance_path = "c:/60-days-data-science/day17/feature_importance.png"
plt.savefig(importance_path, dpi=300)
plt.close()
print(f"Feature importance analysis saved to {importance_path}")
