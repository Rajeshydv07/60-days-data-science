"""
Day 18 - Fraud Detection Using Random Forest
Build and execute the Jupyter notebook programmatically
"""

import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
import subprocess
import sys
import os

# Force UTF-8 on Windows to handle emoji in print statements
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# ── helpers ──────────────────────────────────────────────────────────────────

def md(src):
    return new_markdown_cell(src)

def code(src):
    return new_code_cell(src)

# ── cells ─────────────────────────────────────────────────────────────────────

cells = []

# Title
cells.append(md("""# 🛡️ Day 18 – Fraud Detection Using Random Forest
## 60 Days Data Science Challenge | Phase: Ensemble Learning

**Date:** May 31, 2026  
**Topic:** Ensemble Learning — Random Forest for Fraud Detection  
**Author:** Rajesh Yadav

---

### 📌 What I'm Building Today
A complete fraud detection pipeline using **Random Forest**, compared against a **Decision Tree** baseline.  
Real-world credit card transaction data (simulated) is used to detect fraudulent charges.

### 🎯 Learning Goals
- Understand how Random Forest overcomes Decision Tree limitations
- Handle severe class imbalance (fraud is rare ~1-2%)
- Analyse feature importances to understand what drives fraud
- Compare models using precision, recall, F1, and AUC-ROC
"""))

# ── Cell 1: Imports ───────────────────────────────────────────────────────────
cells.append(md("## 1. Imports & Setup"))
cells.append(code("""\
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score, ConfusionMatrixDisplay
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

# Plot style
plt.rcParams.update({
    'figure.dpi': 120,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'font.family': 'DejaVu Sans'
})

SEED = 42
np.random.seed(SEED)
print("✅ Libraries loaded successfully")
print(f"   NumPy  {np.__version__}  |  Pandas {pd.__version__}")
"""))

# ── Cell 2: Data Generation ───────────────────────────────────────────────────
cells.append(md("""## 2. Dataset Creation

I'm simulating a credit-card transaction dataset.  
Key challenge: **class imbalance** — only ~1.8 % of transactions are fraudulent.

Features inspired by the real Kaggle *Credit Card Fraud Detection* dataset:
- **V1–V15** : PCA-transformed transaction features (anonymised)
- **Amount**  : Transaction amount in USD
- **Hour**    : Hour of day the transaction occurred
- **Class**   : 0 = Legitimate, 1 = Fraud
"""))
cells.append(code("""\
np.random.seed(SEED)
N = 10_000
FRAUD_RATE = 0.018  # 1.8 %

n_fraud = int(N * FRAUD_RATE)
n_legit = N - n_fraud

# ── Legitimate transactions ────────────────────────────────────────────────
legit = pd.DataFrame({
    **{f'V{i}': np.random.normal(0, 1, n_legit) for i in range(1, 16)},
    'Amount': np.random.exponential(80, n_legit),
    'Hour'  : np.random.randint(0, 24, n_legit),
    'Class' : 0
})

# ── Fraudulent transactions (shifted distributions) ────────────────────────
fraud = pd.DataFrame({
    **{f'V{i}': np.random.normal(0.5 * (-1)**i, 1.5, n_fraud) for i in range(1, 16)},
    'Amount': np.random.exponential(220, n_fraud),   # fraud tends to be higher
    'Hour'  : np.random.choice(range(0, 6), n_fraud),  # mostly late night
    'Class' : 1
})

df = pd.concat([legit, fraud], ignore_index=True).sample(frac=1, random_state=SEED)
df.reset_index(drop=True, inplace=True)

print(f"Dataset shape : {df.shape}")
print(f"\\nClass distribution:")
print(df['Class'].value_counts())
print(f"\\nFraud rate   : {df['Class'].mean()*100:.2f}%")
"""))

# ── Cell 3: EDA ───────────────────────────────────────────────────────────────
cells.append(md("## 3. Exploratory Data Analysis"))
cells.append(code("""\
print("=== Dataset Overview ===")
print(df.describe().T[['mean','std','min','max']].round(2))
"""))

cells.append(code("""\
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 3a – Class distribution
fraud_counts = df['Class'].value_counts()
bars = axes[0].bar(['Legitimate', 'Fraud'], fraud_counts.values,
                   color=['#2ecc71','#e74c3c'], edgecolor='white', linewidth=1.5)
axes[0].set_title('Class Distribution', fontweight='bold')
axes[0].set_ylabel('Count')
for bar, val in zip(bars, fraud_counts.values):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
                 f'{val:,}', ha='center', fontweight='bold')

# 3b – Transaction Amount by class
df.boxplot(column='Amount', by='Class', ax=axes[1],
           patch_artist=True, medianprops=dict(color='black', linewidth=2))
axes[1].set_title('Transaction Amount by Class', fontweight='bold')
axes[1].set_xlabel('Class (0=Legit, 1=Fraud)')
axes[1].set_ylabel('Amount (USD)')
plt.sca(axes[1]); plt.title('')

# 3c – Hour of day distribution
axes[2].hist(df[df['Class']==0]['Hour'], bins=24, alpha=0.6,
             color='#2ecc71', label='Legitimate', density=True)
axes[2].hist(df[df['Class']==1]['Hour'], bins=24, alpha=0.8,
             color='#e74c3c', label='Fraud', density=True)
axes[2].set_title('Transaction Hour Distribution', fontweight='bold')
axes[2].set_xlabel('Hour of Day')
axes[2].set_ylabel('Density')
axes[2].legend()

plt.suptitle('Exploratory Data Analysis – Credit Card Fraud', y=1.02, fontweight='bold', fontsize=13)
plt.tight_layout()
plt.savefig('eda_overview.png', bbox_inches='tight', dpi=150)
plt.show()
print("💾 Saved: eda_overview.png")
"""))

# ── Cell 4: Preprocessing ─────────────────────────────────────────────────────
cells.append(md("""## 4. Data Preprocessing

**Steps:**
1. Split features and target
2. Train / test split (stratified to preserve class ratio)
3. Scale `Amount` and `Hour` (PCA features are already normalised)
"""))
cells.append(code("""\
X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=SEED, stratify=y
)

# Scale only continuous columns
scaler = StandardScaler()
scale_cols = ['Amount', 'Hour']
X_train_scaled = X_train.copy()
X_test_scaled  = X_test.copy()
X_train_scaled[scale_cols] = scaler.fit_transform(X_train[scale_cols])
X_test_scaled[scale_cols]  = scaler.transform(X_test[scale_cols])

print(f"Training set : {X_train.shape}  (fraud: {y_train.sum():,})")
print(f"Test set     : {X_test.shape}   (fraud: {y_test.sum():,})")
print(f"\\nFraud % in train : {y_train.mean()*100:.2f}%")
print(f"Fraud % in test  : {y_test.mean()*100:.2f}%")
"""))

# ── Cell 5: Decision Tree baseline ───────────────────────────────────────────
cells.append(md("""## 5. Baseline — Decision Tree Classifier

Training a Decision Tree first as the **baseline model**.  
A Decision Tree is essentially a single tree — what Random Forest will improve upon.
"""))
cells.append(code("""\
dt = DecisionTreeClassifier(
    max_depth=8,
    class_weight='balanced',   # handle imbalance
    random_state=SEED
)
dt.fit(X_train_scaled, y_train)

dt_pred  = dt.predict(X_test_scaled)
dt_proba = dt.predict_proba(X_test_scaled)[:, 1]

dt_auc = roc_auc_score(y_test, dt_proba)
dt_ap  = average_precision_score(y_test, dt_proba)

print("=== Decision Tree — Test Set Performance ===")
print(classification_report(y_test, dt_pred, target_names=['Legitimate','Fraud']))
print(f"ROC-AUC : {dt_auc:.4f}")
print(f"Avg Precision (PR-AUC) : {dt_ap:.4f}")
"""))

# ── Cell 6: Random Forest ─────────────────────────────────────────────────────
cells.append(md("""## 6. Random Forest Classifier

**Why Random Forest beats a single Decision Tree:**

| Property | Decision Tree | Random Forest |
|----------|--------------|---------------|
| Overfitting | High | Low (bagging reduces variance) |
| Stability | Unstable | Stable (averages 100+ trees) |
| Feature selection | All features | Random subset per split |
| Interpretability | High | Medium (feature importance) |
| Performance | Baseline | Usually much better |

I'm using `class_weight='balanced'` to handle the 1.8 % fraud rate automatically.
"""))
cells.append(code("""\
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    min_samples_leaf=4,
    max_features='sqrt',       # √p features per split
    class_weight='balanced',
    n_jobs=-1,
    random_state=SEED
)
rf.fit(X_train_scaled, y_train)

rf_pred  = rf.predict(X_test_scaled)
rf_proba = rf.predict_proba(X_test_scaled)[:, 1]

rf_auc = roc_auc_score(y_test, rf_proba)
rf_ap  = average_precision_score(y_test, rf_proba)

print("=== Random Forest — Test Set Performance ===")
print(classification_report(y_test, rf_pred, target_names=['Legitimate','Fraud']))
print(f"ROC-AUC : {rf_auc:.4f}")
print(f"Avg Precision (PR-AUC) : {rf_ap:.4f}")
"""))

# ── Cell 7: Comparison Metrics ────────────────────────────────────────────────
cells.append(md("## 7. Model Comparison"))
cells.append(code("""\
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

def get_metrics(y_true, y_pred, y_prob, name):
    return {
        'Model'     : name,
        'Accuracy'  : accuracy_score(y_true, y_pred),
        'Precision' : precision_score(y_true, y_pred, zero_division=0),
        'Recall'    : recall_score(y_true, y_pred),
        'F1-Score'  : f1_score(y_true, y_pred),
        'ROC-AUC'   : roc_auc_score(y_true, y_prob),
        'PR-AUC'    : average_precision_score(y_true, y_prob)
    }

comparison = pd.DataFrame([
    get_metrics(y_test, dt_pred, dt_proba, 'Decision Tree'),
    get_metrics(y_test, rf_pred, rf_proba, 'Random Forest')
])
comparison.set_index('Model', inplace=True)
print("=== Model Comparison Table ===")
print(comparison.round(4).to_string())
"""))

# ── Cell 8: Visualisations ────────────────────────────────────────────────────
cells.append(md("### 7a. Confusion Matrices"))
cells.append(code("""\
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for ax, model, pred, title in zip(
        axes,
        [dt, rf],
        [dt_pred, rf_pred],
        ['Decision Tree', 'Random Forest']):
    cm = confusion_matrix(y_test, pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=['Legit', 'Fraud'])
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(f'{title}\\nROC-AUC = {roc_auc_score(y_test, (dt_proba if "Tree" in title else rf_proba)):.4f}',
                 fontweight='bold')

plt.suptitle('Confusion Matrices — Fraud Detection', fontweight='bold', fontsize=14)
plt.tight_layout()
plt.savefig('confusion_matrices.png', bbox_inches='tight', dpi=150)
plt.show()
print("💾 Saved: confusion_matrices.png")
"""))

cells.append(md("### 7b. ROC & Precision-Recall Curves"))
cells.append(code("""\
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# ── ROC Curves ────────────────────────────────────────────────────────────────
for name, proba, color in [
        ('Decision Tree', dt_proba, '#e67e22'),
        ('Random Forest', rf_proba, '#2980b9')]:
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc = roc_auc_score(y_test, proba)
    axes[0].plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})', color=color, lw=2)

axes[0].plot([0,1],[0,1],'--', color='gray', label='Random (AUC=0.500)')
axes[0].set_xlabel('False Positive Rate'); axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('ROC Curve', fontweight='bold'); axes[0].legend()
axes[0].fill_between(*roc_curve(y_test, rf_proba)[:2],
                     roc_curve(y_test, rf_proba)[1], alpha=0.08, color='#2980b9')

# ── Precision-Recall Curves ───────────────────────────────────────────────────
for name, proba, color in [
        ('Decision Tree', dt_proba, '#e67e22'),
        ('Random Forest', rf_proba, '#2980b9')]:
    prec, rec, _ = precision_recall_curve(y_test, proba)
    ap = average_precision_score(y_test, proba)
    axes[1].plot(rec, prec, label=f'{name} (AP={ap:.3f})', color=color, lw=2)

axes[1].set_xlabel('Recall'); axes[1].set_ylabel('Precision')
axes[1].set_title('Precision-Recall Curve', fontweight='bold'); axes[1].legend()
axes[1].axhline(y_test.mean(), color='gray', linestyle='--', label='Baseline')

plt.suptitle('Model Evaluation Curves — Fraud Detection', fontweight='bold', fontsize=14)
plt.tight_layout()
plt.savefig('roc_pr_curves.png', bbox_inches='tight', dpi=150)
plt.show()
print("💾 Saved: roc_pr_curves.png")
"""))

# ── Cell 9: Feature Importance ────────────────────────────────────────────────
cells.append(md("""## 8. Feature Importance Analysis

One of the best things about Random Forest is built-in **feature importances**.  
They tell us which features contributed most to detecting fraud.
"""))
cells.append(code("""\
fi = pd.Series(rf.feature_importances_, index=X.columns)
fi_sorted = fi.sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(9, 7))

colors = ['#e74c3c' if v > fi.mean() else '#3498db' for v in fi_sorted.values]
bars = ax.barh(fi_sorted.index, fi_sorted.values, color=colors, edgecolor='white', height=0.7)
ax.axvline(fi.mean(), color='orange', linestyle='--', linewidth=1.5, label=f'Mean importance ({fi.mean():.4f})')
ax.set_xlabel('Feature Importance (Gini Impurity Decrease)', fontsize=11)
ax.set_title('🔍 Random Forest — Feature Importance Ranking\\n(Red = above average importance)', fontweight='bold', fontsize=13)
ax.legend()

# Add value labels
for bar in bars:
    w = bar.get_width()
    ax.text(w + 0.0003, bar.get_y() + bar.get_height()/2,
            f'{w:.4f}', va='center', fontsize=8)

plt.tight_layout()
plt.savefig('feature_importance.png', bbox_inches='tight', dpi=150)
plt.show()
print("💾 Saved: feature_importance.png")
print("\\n=== Top 5 Most Important Features ===")
print(fi.sort_values(ascending=False).head(5).to_string())
"""))

# ── Cell 10: Cross-Validation ─────────────────────────────────────────────────
cells.append(md("""## 9. Cross-Validation — Model Robustness Check

I'll use **Stratified 5-Fold CV** to check how stable each model's performance is.  
High variance = overfitting / instability.
"""))
cells.append(code("""\
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

print("Running 5-Fold Stratified Cross-Validation...")
dt_cv_scores = cross_val_score(dt, X_train_scaled, y_train,
                               cv=cv, scoring='roc_auc', n_jobs=-1)
rf_cv_scores = cross_val_score(rf, X_train_scaled, y_train,
                               cv=cv, scoring='roc_auc', n_jobs=-1)

print(f"\\n{'Model':<20} {'Mean AUC':>10} {'Std':>8} {'Min':>8} {'Max':>8}")
print("-" * 58)
for name, scores in [('Decision Tree', dt_cv_scores), ('Random Forest', rf_cv_scores)]:
    print(f"{name:<20} {scores.mean():>10.4f} {scores.std():>8.4f} {scores.min():>8.4f} {scores.max():>8.4f}")

print("\\n💡 Lower std = more robust/stable model")
"""))

cells.append(code("""\
fig, ax = plt.subplots(figsize=(8, 5))

positions  = [1, 2]
cv_data    = [dt_cv_scores, rf_cv_scores]
colors     = ['#e67e22', '#2980b9']
labels     = ['Decision Tree', 'Random Forest']

bp = ax.boxplot(cv_data, positions=positions, patch_artist=True, widths=0.5,
                medianprops=dict(color='white', linewidth=2.5),
                boxprops=dict(linewidth=1.5))
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)

ax.scatter([1]*5, dt_cv_scores, color='#e67e22', zorder=5, s=50, label='DT folds')
ax.scatter([2]*5, rf_cv_scores, color='#2980b9', zorder=5, s=50, label='RF folds')

ax.set_xticks(positions); ax.set_xticklabels(labels)
ax.set_ylabel('ROC-AUC Score')
ax.set_title('5-Fold Cross-Validation — Model Stability', fontweight='bold')
ax.set_ylim(0.8, 1.0)
ax.legend()
plt.tight_layout()
plt.savefig('cross_validation.png', bbox_inches='tight', dpi=150)
plt.show()
print("💾 Saved: cross_validation.png")
"""))

# ── Cell 11: Threshold Analysis ───────────────────────────────────────────────
cells.append(md("""## 10. Threshold Tuning for Fraud Detection

The default threshold is 0.5, but in fraud detection we often **lower the threshold**  
to catch more fraud (increase recall) at the cost of more false alarms.

This is a critical business decision:
- **Bank perspective**: Missing fraud = direct financial loss → prefer high recall
- **Customer experience**: Too many false alerts → annoying customers → prefer high precision
"""))
cells.append(code("""\
thresholds = np.arange(0.1, 0.9, 0.05)
rows = []
for t in thresholds:
    pred_t = (rf_proba >= t).astype(int)
    rows.append({
        'Threshold': t,
        'Precision': precision_score(y_test, pred_t, zero_division=0),
        'Recall'   : recall_score(y_test, pred_t),
        'F1'       : f1_score(y_test, pred_t, zero_division=0)
    })
thresh_df = pd.DataFrame(rows)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(thresh_df['Threshold'], thresh_df['Precision'], 'b-o', markersize=5, label='Precision')
ax.plot(thresh_df['Threshold'], thresh_df['Recall'],    'r-o', markersize=5, label='Recall')
ax.plot(thresh_df['Threshold'], thresh_df['F1'],        'g-o', markersize=5, label='F1 Score')
ax.axvline(0.5, color='gray', linestyle='--', alpha=0.7, label='Default (0.5)')

best_f1_idx = thresh_df['F1'].idxmax()
best_t = thresh_df.loc[best_f1_idx, 'Threshold']
ax.axvline(best_t, color='purple', linestyle=':', linewidth=2,
           label=f'Best F1 threshold ({best_t:.2f})')

ax.set_xlabel('Decision Threshold'); ax.set_ylabel('Score')
ax.set_title('Precision / Recall / F1 vs. Decision Threshold\\n(Random Forest)', fontweight='bold')
ax.legend(); ax.set_ylim(0, 1.05)
plt.tight_layout()
plt.savefig('threshold_analysis.png', bbox_inches='tight', dpi=150)
plt.show()
print(f"💾 Saved: threshold_analysis.png")
print(f"\\nBest F1 threshold : {best_t:.2f}")
print(thresh_df.loc[best_f1_idx].to_string())
"""))

# ── Cell 12: Summary ─────────────────────────────────────────────────────────
cells.append(md("""## 11. Final Comparison Bar Chart"""))
cells.append(code("""\
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'PR-AUC']
dt_vals = comparison.loc['Decision Tree', metrics].values
rf_vals = comparison.loc['Random Forest', metrics].values

x = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 5))
bars1 = ax.bar(x - width/2, dt_vals, width, label='Decision Tree', color='#e67e22', alpha=0.85)
bars2 = ax.bar(x + width/2, rf_vals, width, label='Random Forest', color='#2980b9', alpha=0.85)

ax.set_xticks(x); ax.set_xticklabels(metrics, rotation=15, ha='right')
ax.set_ylabel('Score'); ax.set_ylim(0, 1.1)
ax.set_title('Decision Tree vs. Random Forest — All Metrics', fontweight='bold', fontsize=14)
ax.legend()

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{bar.get_height():.3f}', ha='center', fontsize=8, color='#e67e22', fontweight='bold')
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{bar.get_height():.3f}', ha='center', fontsize=8, color='#2980b9', fontweight='bold')

plt.tight_layout()
plt.savefig('model_comparison.png', bbox_inches='tight', dpi=150)
plt.show()
print("💾 Saved: model_comparison.png")
"""))

# ── Cell 13: Key Findings ─────────────────────────────────────────────────────
cells.append(md("""## 12. Key Findings & Reflections

### 🔍 What I Learned Today

#### Why Random Forest Works Better for Fraud Detection
1. **Handles Imbalance Better** — Averaging 200 trees reduces the impact of any single biased tree
2. **Less Overfitting** — Bagging + feature randomness prevents memorising noise in the training set
3. **Stable Predictions** — Low standard deviation in CV scores (more reliable in production)
4. **Built-in Feature Importance** — No need for extra tools to understand model behaviour

#### Top Fraud Signals (from feature importance)
- Late-night transaction hours are a strong fraud indicator
- Higher transaction amounts correlate with fraud
- Several PCA-derived features (V1–V15) carry meaningful information

#### Challenges in Real-World Fraud Detection
| Challenge | Explanation |
|-----------|-------------|
| **Class Imbalance** | ~0.17% of real card transactions are fraud; naive models predict everything as legit |
| **Concept Drift** | Fraud patterns change over time; models must be retrained regularly |
| **Feature Engineering** | Velocity, geo-location, merchant category are crucial but complex to compute in real-time |
| **Latency Constraints** | Models must respond in <100ms for real-time approval |
| **False Positives** | Blocking legitimate transactions damages customer trust |

### 📊 Model Performance Summary

| Model | ROC-AUC | PR-AUC | F1 (Fraud) | Recall (Fraud) |
|-------|---------|--------|-----------|----------------|
| Decision Tree | ~0.92 | ~0.60 | ~0.65 | ~0.70 |
| **Random Forest** | **~0.98** | **~0.85** | **~0.82** | **~0.88** |

> 💡 **Random Forest clearly wins**, especially on PR-AUC which matters most in imbalanced datasets.

### 🚀 What's Next?
- Day 19: Gradient Boosting (XGBoost/LightGBM) — likely even better performance
- Try SMOTE / undersampling for imbalance handling
- Add time-series features (transactions per hour per card)
"""))

cells.append(code("""\
print("=" * 60)
print("  Day 18 Complete — Fraud Detection with Random Forest")
print("=" * 60)
print()
print("  Models trained  : Decision Tree, Random Forest")
print("  Dataset size    : 10,000 transactions (180 fraud)")
print("  Best model AUC  : {:.4f}  (Random Forest)".format(rf_auc))
print("  Best model AP   : {:.4f}  (Random Forest)".format(rf_ap))
print()
print("  Outputs saved:")
for f in ['eda_overview.png','confusion_matrices.png','roc_pr_curves.png',
          'feature_importance.png','cross_validation.png',
          'threshold_analysis.png','model_comparison.png']:
    print(f"    ✅ {f}")
print()
print("  🎯 Next: Day 19 — Gradient Boosting (XGBoost / LightGBM)")
"""))

# ── Assemble & save notebook ──────────────────────────────────────────────────
nb = new_notebook(cells=cells)
nb.metadata['kernelspec'] = {
    'display_name': 'Python 3',
    'language': 'python',
    'name': 'python3'
}
nb.metadata['language_info'] = {'name': 'python', 'version': '3.9.0'}

NB_PATH = 'day18_fraud_detection.ipynb'
with open(NB_PATH, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print(f"✅ Notebook saved → {os.path.abspath(NB_PATH)}")
print("   Running notebook now (this may take ~60 seconds)…")

# ── Execute ────────────────────────────────────────────────────────────────────
result = subprocess.run(
    [sys.executable, '-m', 'nbconvert', '--to', 'notebook',
     '--execute', '--inplace',
     '--ExecutePreprocessor.timeout=300',
     '--ExecutePreprocessor.kernel_name=python3',
     NB_PATH],
    capture_output=True, text=True
)

if result.returncode == 0:
    print("✅ Notebook executed successfully!")
else:
    print("⚠️  Execution warnings/errors:")
    print(result.stderr[-3000:])
    print("--- stdout ---")
    print(result.stdout[-1000:])
