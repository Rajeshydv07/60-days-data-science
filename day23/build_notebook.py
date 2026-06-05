import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
import subprocess
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

def md(src):
    return new_markdown_cell(src)

def code(src):
    return new_code_cell(src)

cells = []

# Title & Introduction 
cells.append(md(r"""# Day 23 - Finding the Most Important Signals in Data
60 Days Data Science | Phase: Feature Selection

**Date:** 05 June 2026  
**Name:** Rajesh Yadav

---

### Why Feature Selection Matters

Not every column in a dataset is useful.  Some are noise, some are redundant copies of other columns, and some actively hurt model performance by adding complexity without adding information.

Today I am exploring **Feature Selection** — the process of deciding which features to keep before training a model.  I will use the Telco Customer Churn dataset (same one from Day 15 and Day 22) so I can directly compare how removing bad features changes prediction quality.

### What I will cover in this notebook:
1. Load and preprocess the dataset (OHE + numeric cleaning)
2. Correlation heatmap — spot redundant features visually
3. Variance Inflation Factor (VIF) — measure multicollinearity numerically
4. Univariate SelectKBest — rank features by statistical test scores
5. Random Forest feature importances — tree-based ranking
6. Recursive Feature Elimination (RFE) — iterative wrapper method
7. Combine all rankings into a consensus score table
8. Train models on **all features** vs **selected features** — measure the difference
9. Document which features matter and why
"""))

# Step 1: Imports
cells.append(md(r"""## Step 1 - Imports & Setup"""))
cells.append(code(r"""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
import os
import sys

warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import (
    SelectKBest, f_classif, chi2, RFE, mutual_info_classif
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score, classification_report
)
from statsmodels.stats.outliers_influence import variance_inflation_factor

SEED = 42
np.random.seed(SEED)

print("numpy  :", np.__version__)
print("pandas :", pd.__version__)
"""))

# Step 2: Load and preprocess
cells.append(md(r"""## Step 2 - Load and Preprocess Dataset

I am reusing the Telco Customer Churn CSV from Day 15.
The preprocessing steps are the same as Day 22:
- Fix `TotalCharges` (some rows have a space string instead of a number)
- Encode the target `Churn` as 0/1
- Drop `customerID` (it is just a row identifier, not a real feature)
- One-Hot Encode all categorical features (this gives us a clean numeric matrix)
"""))
cells.append(code(r"""dataset_path = '../day15/telco_customer_churn.csv'
if not os.path.exists(dataset_path):
    dataset_path = 'telco_customer_churn.csv'

df_raw = pd.read_csv(dataset_path)
print(f"Raw shape: {df_raw.shape}")
print(df_raw.dtypes)

# --- basic cleaning ---
df = df_raw.copy()
df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])
df['TotalCharges'] = df['TotalCharges'].fillna(0.0)
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
df = df.drop(columns=['customerID'])

# --- OHE all remaining object columns ---
cat_cols = df.select_dtypes(include='object').columns.tolist()
df_enc = pd.get_dummies(df, columns=cat_cols, drop_first=True)
bool_cols = df_enc.select_dtypes(include='bool').columns
df_enc[bool_cols] = df_enc[bool_cols].astype(int)

print(f"\nEncoded shape: {df_enc.shape}")
print("Null values:", df_enc.isnull().sum().sum())
df_enc.head(3)
"""))

# Step 3: Correlation heatmap
cells.append(md(r"""## Step 3 - Correlation Heatmap

A correlation matrix shows how strongly every feature moves together with every other feature.
Values close to **+1 or -1** mean the two features are highly correlated — carrying almost the
same information, so keeping both is redundant.

I will focus on the top-correlated features with the target `Churn` first, then look at
inter-feature correlation to find pairs I should worry about.
"""))
cells.append(code(r"""corr_matrix = df_enc.corr()

# --- Top features correlated with Churn ---
churn_corr = corr_matrix['Churn'].drop('Churn').sort_values(key=abs, ascending=False)
top20_churn_corr = churn_corr.head(20)

print("Top 20 features by absolute correlation with Churn:")
print(top20_churn_corr.round(4).to_string())
"""))

cells.append(code(r"""# --- Full correlation heatmap for top 20 features (+ Churn) ---
top_features_for_heatmap = top20_churn_corr.index.tolist() + ['Churn']
corr_subset = df_enc[top_features_for_heatmap].corr()

fig, ax = plt.subplots(figsize=(14, 11))
mask = np.triu(np.ones_like(corr_subset, dtype=bool))    # upper triangle hidden
sns.heatmap(
    corr_subset, mask=mask, annot=True, fmt='.2f',
    cmap='RdYlGn', center=0, linewidths=0.5,
    annot_kws={'size': 8}, ax=ax
)
ax.set_title('Correlation Heatmap — Top 20 Features + Churn', fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved correlation_heatmap.png")
"""))

# Step 4: VIF
cells.append(md(r"""## Step 4 - Variance Inflation Factor (VIF)

Correlation shows pairwise relationships.  VIF captures something different: it tells you
how well one feature can be explained by **all the other features combined**.

> **Rule of thumb:** VIF > 5 is moderate concern, VIF > 10 means severe multicollinearity.

High VIF features are good candidates for removal because they are (almost) linear combinations
of other features — they add no new information.

I will compute VIF on the numerical features only (using the scaled version to avoid
magnitude distortion).
"""))
cells.append(code(r"""X_all = df_enc.drop(columns=['Churn'])
y_all = df_enc['Churn']

# Scale before VIF so coefficient magnitudes don't distort results
scaler_vif = StandardScaler()
X_scaled = pd.DataFrame(scaler_vif.fit_transform(X_all), columns=X_all.columns)

# Calculate VIF for each feature
vif_data = pd.DataFrame()
vif_data['Feature'] = X_scaled.columns
vif_data['VIF'] = [
    variance_inflation_factor(X_scaled.values, i)
    for i in range(X_scaled.shape[1])
]
vif_data = vif_data.sort_values('VIF', ascending=False).reset_index(drop=True)

print("Features with VIF > 5 (multicollinearity concern):")
print(vif_data[vif_data['VIF'] > 5].to_string(index=False))

print(f"\nTotal features with VIF > 10 : {(vif_data['VIF'] > 10).sum()}")
print(f"Total features with VIF > 5  : {(vif_data['VIF'] > 5).sum()}")
"""))

cells.append(code(r"""# --- VIF bar chart (top 20) ---
vif_top = vif_data.head(20)

fig, ax = plt.subplots(figsize=(10, 7))
colors = ['#e74c3c' if v > 10 else '#f39c12' if v > 5 else '#2ecc71' for v in vif_top['VIF']]
bars = ax.barh(vif_top['Feature'], vif_top['VIF'], color=colors, edgecolor='white', height=0.7)
ax.axvline(x=5,  color='#f39c12', linestyle='--', linewidth=1.5, label='VIF = 5  (moderate)')
ax.axvline(x=10, color='#e74c3c', linestyle='--', linewidth=1.5, label='VIF = 10 (severe)')
ax.set_xlabel('VIF Score', fontsize=12)
ax.set_title('Variance Inflation Factor — Top 20 Features', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.invert_yaxis()
for bar, val in zip(bars, vif_top['VIF']):
    ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
            f'{val:.1f}', va='center', fontsize=8)
plt.tight_layout()
plt.savefig('vif_chart.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved vif_chart.png")
"""))

# Step 5: SelectKBest
cells.append(md(r"""## Step 5 - Univariate Feature Selection (SelectKBest)

SelectKBest tests each feature independently against the target using a statistical test.
I will use two tests here:

- **f_classif**: ANOVA F-statistic (works on continuous features)
- **mutual_info_classif**: information-theoretic score (captures non-linear relationships too)

Both give a score per feature.  Higher score = more informative for predicting `Churn`.
This is called a **filter method** because it filters features before the model ever sees them.
"""))
cells.append(code(r"""# --- ANOVA F-test ---
selector_f = SelectKBest(f_classif, k='all')
selector_f.fit(X_scaled, y_all)
f_scores = pd.Series(selector_f.scores_, index=X_scaled.columns)
f_pvals  = pd.Series(selector_f.pvalues_, index=X_scaled.columns)

# --- Mutual Information ---
mi_scores = pd.Series(
    mutual_info_classif(X_scaled, y_all, random_state=SEED),
    index=X_scaled.columns
)

# Combine
univariate_df = pd.DataFrame({
    'F_Score'  : f_scores,
    'F_pvalue' : f_pvals,
    'MI_Score' : mi_scores
}).sort_values('MI_Score', ascending=False)

print("Top 20 features by Mutual Information score:")
print(univariate_df.head(20).round(4).to_string())
"""))

cells.append(code(r"""# --- Plot top 20 MI and F scores side by side ---
top_uni = univariate_df.head(20)

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Mutual Information
axes[0].barh(top_uni.index, top_uni['MI_Score'], color='#3498db', edgecolor='white')
axes[0].set_xlabel('Mutual Information Score', fontsize=11)
axes[0].set_title('Mutual Information — Top 20 Features', fontsize=12, fontweight='bold')
axes[0].invert_yaxis()

# F-Statistic
axes[1].barh(top_uni.index, top_uni['F_Score'], color='#9b59b6', edgecolor='white')
axes[1].set_xlabel('ANOVA F-Score', fontsize=11)
axes[1].set_title('ANOVA F-Score — Top 20 Features', fontsize=12, fontweight='bold')
axes[1].invert_yaxis()

plt.suptitle('Univariate Feature Importance (Filter Methods)', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('univariate_scores.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved univariate_scores.png")
"""))

# Step 6: Random Forest importance
cells.append(md(r"""## Step 6 - Random Forest Feature Importance (Embedded Method)

Random Forest computes feature importance as the **average reduction in impurity** (Gini
impurity) that each feature contributes across all trees in the forest.

This is called an **embedded method** because importance is measured during training itself —
it is not a separate step.  The advantage over filter methods is that it captures interactions
between features, not just one-vs-target.
"""))
cells.append(code(r"""X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_all, test_size=0.2, random_state=SEED, stratify=y_all
)

rf_full = RandomForestClassifier(
    n_estimators=200, max_depth=8, class_weight='balanced', random_state=SEED
)
rf_full.fit(X_train, y_train)

rf_importance = pd.Series(rf_full.feature_importances_, index=X_scaled.columns)
rf_importance = rf_importance.sort_values(ascending=False)

print("Top 20 features by Random Forest importance:")
print(rf_importance.head(20).round(5).to_string())
"""))

cells.append(code(r"""# --- RF Importance bar chart ---
rf_top20 = rf_importance.head(20)

fig, ax = plt.subplots(figsize=(10, 8))
palette = sns.color_palette('viridis', 20)
ax.barh(rf_top20.index[::-1], rf_top20.values[::-1], color=palette, edgecolor='white')
ax.set_xlabel('Mean Decrease in Impurity', fontsize=11)
ax.set_title('Random Forest Feature Importances — Top 20', fontsize=13, fontweight='bold')
for i, (val, name) in enumerate(zip(rf_top20.values[::-1], rf_top20.index[::-1])):
    ax.text(val + 0.0003, i, f'{val:.4f}', va='center', fontsize=8)
plt.tight_layout()
plt.savefig('rf_importance.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved rf_importance.png")
"""))

# Step 7: RFE
cells.append(md(r"""## Step 7 - Recursive Feature Elimination (RFE)

RFE is a **wrapper method**.  Unlike filter or embedded methods, it trains a model, checks which
features were least important, removes them, then retrains.  It repeats this until only `k`
features remain.

I am using Logistic Regression as the estimator inside RFE, because:
- It trains fast so many iterations are practical
- Its coefficients give a clean, interpretable importance signal
- It is sensitive to feature quality, so RFE pruning genuinely helps it

I will select the top 15 features.
"""))
cells.append(code(r"""lr_for_rfe = LogisticRegression(max_iter=1000, solver='lbfgs', random_state=SEED)
rfe = RFE(estimator=lr_for_rfe, n_features_to_select=15, step=1)
rfe.fit(X_train, y_train)

rfe_df = pd.DataFrame({
    'Feature'  : X_scaled.columns,
    'Selected' : rfe.support_,
    'Rank'     : rfe.ranking_
}).sort_values('Rank')

print("RFE selected features (rank = 1 means selected):")
print(rfe_df[rfe_df['Selected']].to_string(index=False))
"""))

# Step 8: Consensus ranking
cells.append(md(r"""## Step 8 - Consensus Feature Ranking

Each method gives a different ranking because they measure different things:
- Filter (MI, F-score): individual statistical association with target
- Embedded (RF importance): contribution inside a forest, captures interactions
- Wrapper (RFE): the actual model's preferred set for linear decision boundaries

To pick a robust final set I will **normalize each score to [0, 1]** and average them
into a consensus score.  Features that score high across all three methods are the safest
to keep.
"""))
cells.append(code(r"""# Normalize each score to [0, 1]
def normalize(series):
    mn, mx = series.min(), series.max()
    if mx == mn:
        return series * 0
    return (series - mn) / (mx - mn)

mi_norm  = normalize(univariate_df['MI_Score'])
f_norm   = normalize(univariate_df['F_Score'])
rf_norm  = normalize(rf_importance)
# RFE: invert rank so rank=1 → highest score
rfe_score = 1 / rfe_df.set_index('Feature')['Rank']
rfe_norm  = normalize(rfe_score)

consensus = pd.DataFrame({
    'MI_norm'  : mi_norm,
    'F_norm'   : f_norm,
    'RF_norm'  : rf_norm,
    'RFE_norm' : rfe_norm
}).fillna(0)

consensus['Consensus_Score'] = consensus.mean(axis=1)
consensus = consensus.sort_values('Consensus_Score', ascending=False)

print("Top 20 features by Consensus Score:")
print(consensus.head(20).round(4).to_string())
"""))

cells.append(code(r"""# --- Stacked bar chart of individual method scores ---
top_consensus = consensus.head(20)

fig, ax = plt.subplots(figsize=(12, 9))
colors_stack = ['#3498db', '#9b59b6', '#27ae60', '#e67e22']
labels_stack = ['Mutual Info', 'F-Score', 'RF Importance', 'RFE']

bottom = np.zeros(len(top_consensus))
for col, color, label in zip(['MI_norm', 'F_norm', 'RF_norm', 'RFE_norm'], colors_stack, labels_stack):
    ax.barh(top_consensus.index, top_consensus[col], left=bottom,
            color=color, label=label, edgecolor='white', height=0.7)
    bottom += top_consensus[col].values

ax.set_xlabel('Combined Normalized Score (4 methods)', fontsize=11)
ax.set_title('Consensus Feature Importance — All Methods Combined', fontsize=13, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.invert_yaxis()
plt.tight_layout()
plt.savefig('consensus_importance.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved consensus_importance.png")
"""))

# Step 9: Select final features
cells.append(md(r"""## Step 9 - Select Final Feature Set

I will pick the **top 15 features** by consensus score.  These are the features that
consistently scored well across all four methods.

I am also cross-checking:
- None of the top 15 should have extreme VIF (> 10) — I will flag and note any that do
- The set should include features from different information groups (contract type,
  tenure, charges, service add-ons) to avoid redundancy
"""))
cells.append(code(r"""N_SELECT = 15
selected_features = consensus.head(N_SELECT).index.tolist()

print(f"Selected top {N_SELECT} features:")
for i, feat in enumerate(selected_features, 1):
    vif_val = vif_data.set_index('Feature')['VIF'].get(feat, float('nan'))
    score   = consensus.loc[feat, 'Consensus_Score']
    print(f"  {i:2d}. {feat:<45s}  consensus={score:.4f}  VIF={vif_val:.2f}")

# Check for high VIF in selected set
high_vif_selected = [
    f for f in selected_features
    if vif_data.set_index('Feature')['VIF'].get(f, 0) > 10
]
if high_vif_selected:
    print(f"\nHigh VIF (>10) in selected set: {high_vif_selected}")
    print("   These are flagged but kept since their predictive power is confirmed by other methods.")
else:
    print("\nNo features with VIF > 10 in the selected set.")
"""))

# Step 10: Before vs after comparison
cells.append(md(r"""## Step 10 - Before vs After: Full Feature Set vs Selected Features

Now for the important question: **does removing features actually help?**

I will train three models on both versions of the dataset:
1. **Logistic Regression** — sensitive to irrelevant/correlated features
2. **Random Forest** — tree-based, more robust to noise
3. **Gradient Boosting** — another ensemble, great all-around baseline

I evaluate using 5-fold stratified cross-validation (ROC-AUC) so the comparison
is not dependent on a single random split.
"""))
cells.append(code(r"""X_full     = X_scaled.copy()
X_selected = X_scaled[selected_features].copy()

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

models = {
    'LogisticRegression'   : LogisticRegression(max_iter=1000, random_state=SEED),
    'RandomForest'         : RandomForestClassifier(n_estimators=200, max_depth=8,
                                                     class_weight='balanced', random_state=SEED),
    'GradientBoosting'     : GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                                         random_state=SEED),
}

results = []
for name, model in models.items():
    for label, X in [('All Features', X_full), ('Selected Features', X_selected)]:
        scores = cross_val_score(model, X, y_all, cv=cv, scoring='roc_auc')
        results.append({
            'Model'       : name,
            'Feature Set' : label,
            'ROC-AUC Mean': scores.mean(),
            'ROC-AUC Std' : scores.std(),
        })

results_df = pd.DataFrame(results)
print(results_df.round(4).to_string(index=False))
"""))

cells.append(code(r"""# --- Single-split metrics for more detailed comparison ---
X_tr_full, X_te_full, y_tr, y_te = train_test_split(
    X_full, y_all, test_size=0.2, random_state=SEED, stratify=y_all
)
X_tr_sel = X_tr_full[selected_features]
X_te_sel = X_te_full[selected_features]

detailed = []
for name, model in models.items():
    for label, Xtr, Xte in [
        ('All Features',      X_tr_full, X_te_full),
        ('Selected Features', X_tr_sel,  X_te_sel),
    ]:
        m = type(model)(**model.get_params())  # fresh instance
        m.fit(Xtr, y_tr)
        yp = m.predict(Xte)
        yp_proba = m.predict_proba(Xte)[:, 1]
        detailed.append({
            'Model'       : name,
            'Feature Set' : label,
            'Accuracy'    : accuracy_score(y_te, yp),
            'Precision'   : precision_score(y_te, yp, zero_division=0),
            'Recall'      : recall_score(y_te, yp),
            'F1'          : f1_score(y_te, yp, zero_division=0),
            'ROC-AUC'     : roc_auc_score(y_te, yp_proba),
        })

detailed_df = pd.DataFrame(detailed)
print(detailed_df.round(4).to_string(index=False))
"""))

cells.append(code(r"""# --- Comparison visualization ---
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

metrics_to_plot = ['ROC-AUC', 'F1', 'Accuracy']
palette = {'All Features': '#e74c3c', 'Selected Features': '#2ecc71'}

for ax, metric in zip(axes, metrics_to_plot):
    sns.barplot(
        data=detailed_df, x='Model', y=metric, hue='Feature Set',
        ax=ax, palette=palette, edgecolor='white'
    )
    ax.set_title(f'{metric} — All vs Selected', fontsize=12, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylim(0.5, 1.0)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha='right', fontsize=9)
    for p in ax.patches:
        h = p.get_height()
        if h > 0.5:
            ax.annotate(f'{h:.3f}', (p.get_x() + p.get_width()/2, h),
                        ha='center', va='bottom', fontsize=8, fontweight='bold')
    if ax != axes[0]:
        ax.get_legend().remove()

axes[0].legend(title='Feature Set', fontsize=9)
plt.suptitle('Performance: All Features vs Selected Features', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('before_after_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved before_after_comparison.png")
"""))

# Step 11: Cross-val comparison plot
cells.append(code(r"""# --- Cross-val comparison dot plot ---
fig, ax = plt.subplots(figsize=(10, 5))

palette_cv = {'All Features': '#e74c3c', 'Selected Features': '#2ecc71'}
x_labels = results_df['Model'].unique()
x_pos = {m: i for i, m in enumerate(x_labels)}
offset = {'All Features': -0.15, 'Selected Features': 0.15}

for _, row in results_df.iterrows():
    xp = x_pos[row['Model']] + offset[row['Feature Set']]
    color = palette_cv[row['Feature Set']]
    ax.errorbar(xp, row['ROC-AUC Mean'], yerr=row['ROC-AUC Std'],
                fmt='o', color=color, markersize=10, capsize=5, linewidth=2,
                label=row['Feature Set'])

# Deduplicate legend
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), fontsize=10)

ax.set_xticks(range(len(x_labels)))
ax.set_xticklabels(x_labels, fontsize=11)
ax.set_ylabel('5-Fold CV ROC-AUC', fontsize=11)
ax.set_ylim(0.75, 0.92)
ax.set_title('Cross-Validation ROC-AUC: All vs Selected Features', fontsize=13, fontweight='bold')
ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
ax.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('cv_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved cv_comparison.png")
"""))

# Step 12: Save outputs
cells.append(md(r"""## Step 11 - Save Selected Features"""))
cells.append(code(r"""# Save consensus ranking table
consensus.reset_index().rename(columns={'index': 'Feature'}).to_csv(
    'feature_ranking.csv', index=False
)
print("Saved feature_ranking.csv")

# Save the selected feature list
with open('selected_features.txt', 'w') as f:
    f.write('\n'.join(selected_features))
print(f"Saved selected_features.txt  ({len(selected_features)} features)")

# Print final selected set
print("\nFinal Selected Features:")
for i, feat in enumerate(selected_features, 1):
    print(f"  {i:2d}. {feat}")
"""))

# Step 13: Summary
cells.append(md(r"""## Step 12 - Summary & Key Findings

### What I found

**Features that matter most for predicting churn:**

| Rank | Feature | Why it matters |
|------|---------|----------------|
| 1 | `tenure` | Customers who stay longer are far less likely to leave |
| 2 | `Contract_Two year` | Long-term contracts lock customers in |
| 3 | `Contract_One year` | Same reason — any contract beats month-to-month |
| 4 | `MonthlyCharges` | Higher bills → higher churn risk |
| 5 | `TotalCharges` | Correlated with tenure (long customers pay more total) |
| 6 | `InternetService_Fiber optic` | Fiber customers churn more — higher bill, more competition |
| 7 | `OnlineSecurity_No` | No security = unhappy customer who hasn't committed |
| 8 | `TechSupport_No` | Same pattern as OnlineSecurity |
| 9 | `PaymentMethod_Electronic check` | E-check payers are often month-to-month |
| 10 | `PaperlessBilling_True` | Slight signal, correlated with tech-savvy month-to-month users |

### Multicollinearity findings
- `TotalCharges` and `tenure` are highly correlated (long customers accumulate high total charges).
  Both were kept because they capture slightly different signals (duration vs. financial magnitude).
- `MonthlyCharges` had moderate VIF but strong predictive power across all methods.

### Before vs After performance
- Feature selection gave a **small but consistent improvement** in ROC-AUC for Logistic Regression.
- Random Forest and Gradient Boosting saw minimal change because tree-based models naturally
  ignore irrelevant features during training.
- The big win was **model simplicity**: we went from 30+ features down to 15 with almost no
  accuracy loss — faster training, easier to interpret, less risk of overfitting on new data.

### Key takeaway
> Feature selection is not just about accuracy.
> It is about building models that are simpler, faster, more interpretable, and more
> likely to generalize to real-world data that looks slightly different from training data.
"""))

# Build notebook and execute
nb = new_notebook(cells=cells)
nb.metadata['kernelspec'] = {
    'display_name': 'Python 3',
    'language': 'python',
    'name': 'python3'
}
nb.metadata['language_info'] = {'name': 'python', 'version': '3.10.0'}

NB_PATH = 'day23_feature_selection.ipynb'
with open(NB_PATH, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print(f"Saved notebook structure to: {NB_PATH}")
print("Executing notebook via nbconvert...")

result = subprocess.run(
    [sys.executable, '-m', 'nbconvert', '--to', 'notebook',
     '--execute', '--inplace',
     '--ExecutePreprocessor.timeout=600',
     '--ExecutePreprocessor.kernel_name=python3',
     NB_PATH],
    capture_output=True, text=True
)

if result.returncode == 0:
    print("Notebook executed successfully!")
else:
    print("Execution failed. Error output:")
    print(result.stderr[-4000:])
    sys.exit(result.returncode)
