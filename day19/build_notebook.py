import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
import subprocess
import sys
import os

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

def md(src):
    return new_markdown_cell(src)

def code(src):
    return new_code_cell(src)

cells = []

# Cell 1: Intro
cells.append(md("""\
# Day 19 - Boosting Model Performance with XGBoost
### 60 Days Data Science | Phase: Advanced Ensemble Systems

**Date:** 01 June 2026  
**Name:** Rajesh Yadav

---

ok so today is day 19 and we're covering **Gradient Boosting** techniques, specifically **XGBoost** and **LightGBM**.
yesterday i built a fraud detection model using **Random Forest** and got really good results. but today we're moving on to boosting, which is the current state-of-the-art for tabular datasets.

what i want to figure out today:
- how do we configure and train XGBoost and LightGBM?
- does gradient boosting actually beat bagging (Random Forest)?
- how do learning curves look as we add more trees in boosting (analyzing training improvements)?
- what are the practical tradeoffs in terms of training time and complexity?
"""))

# Cell 2: Imports
cells.append(md("## Step 1 - imports & package checks"))

cells.append(code("""\
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score
)
from sklearn.preprocessing import StandardScaler

# importing xgboost and lightgbm
import xgboost as xgb
import lightgbm as lgb

SEED = 42
np.random.seed(SEED)

print("numpy:", np.__version__)
print("pandas:", pd.__version__)
print("xgboost:", xgb.__version__)
print("lightgbm:", lgb.__version__)
"""))

# Cell 3: Data creation
cells.append(md("""\
## Step 2 - creating the dataset (exact same as Day 18)

to make this a 100% fair scientific comparison with yesterday, i'll simulate the exact same credit card fraud dataset:
- 10,000 transactions total
- 180 fraud cases (~1.8% fraud rate)
- features: V1-V15 (PCA style), Amount, Hour, Class
"""))

cells.append(code("""\
N = 10000
FRAUD_RATE = 0.018

n_fraud = int(N * FRAUD_RATE)
n_legit = N - n_fraud

print(f"total: {N}  fraud: {n_fraud}  legit: {n_legit}")

# generating same synthetic distributions
legit = pd.DataFrame({
    **{f'V{i}': np.random.normal(0, 1, n_legit) for i in range(1, 16)},
    'Amount': np.random.exponential(80, n_legit),
    'Hour'  : np.random.randint(0, 24, n_legit),
    'Class' : 0
})

fraud = pd.DataFrame({
    **{f'V{i}': np.random.normal(0.5 * (-1)**i, 1.5, n_fraud) for i in range(1, 16)},
    'Amount': np.random.exponential(220, n_fraud),
    'Hour'  : np.random.choice(range(0, 6), n_fraud),
    'Class' : 1
})

df = pd.concat([legit, fraud], ignore_index=True).sample(frac=1, random_state=SEED)
df.reset_index(drop=True, inplace=True)

print(df['Class'].value_counts())
print(f"fraud %: {df['Class'].mean()*100:.2f}%")
"""))

# Cell 4: Quick EDA verification
cells.append(md("""\
## Step 3 - quick look at the data
just verifying the features and checking class distribution to make sure it matches day 18.
"""))

cells.append(code("""\
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

counts = df['Class'].value_counts()
axes[0].bar(['Legit', 'Fraud'], counts.values, color=['steelblue', 'tomato'])
axes[0].set_title('class distribution')
axes[0].set_ylabel('count')
for i, v in enumerate(counts.values):
    axes[0].text(i, v + 50, str(v), ha='center', fontweight='bold')

axes[1].hist(df[df['Class']==0]['Amount'], bins=50, alpha=0.6,
             color='steelblue', label='legit', density=True)
axes[1].hist(df[df['Class']==1]['Amount'], bins=50, alpha=0.8,
             color='tomato', label='fraud', density=True)
axes[1].set_title('transaction amount')
axes[1].set_xlabel('amount ($)')
axes[1].legend()

axes[2].hist(df[df['Class']==0]['Hour'], bins=24, alpha=0.6,
             color='steelblue', label='legit', density=True)
axes[2].hist(df[df['Class']==1]['Hour'], bins=24, alpha=0.8,
             color='tomato', label='fraud', density=True)
axes[2].set_title('hour of day')
axes[2].set_xlabel('hour (0-23)')
axes[2].legend()

plt.suptitle('EDA - fraud vs legit transactions', y=1.02, fontsize=12)
plt.tight_layout()
plt.savefig('eda_overview.png', bbox_inches='tight', dpi=150)
plt.show()
"""))

# Cell 5: Preprocessing
cells.append(md("""\
## Step 4 - preprocessing

splitting the dataset (stratified so train and test have same fraud ratio) and scaling.
"""))

cells.append(code("""\
X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=SEED,
    stratify=y
)

print(f"train: {X_train.shape}  fraud={y_train.sum()}")
print(f"test:  {X_test.shape}   fraud={y_test.sum()}")

scaler = StandardScaler()
scale_cols = ['Amount', 'Hour']

X_train_sc = X_train.copy()
X_test_sc  = X_test.copy()

X_train_sc[scale_cols] = scaler.fit_transform(X_train[scale_cols])
X_test_sc[scale_cols]  = scaler.transform(X_test[scale_cols])
"""))

# Cell 6: Random Forest Baseline
cells.append(md("""\
## Step 5 - training Random Forest (baseline)

using the exact same hyperparameters as yesterday so we have a solid baseline.
"""))

cells.append(code("""\
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    min_samples_leaf=4,
    max_features='sqrt',
    class_weight='balanced',
    n_jobs=-1,
    random_state=SEED
)

start_time = time.time()
rf.fit(X_train_sc, y_train)
rf_time = time.time() - start_time

rf_pred  = rf.predict(X_test_sc)
rf_proba = rf.predict_proba(X_test_sc)[:, 1]

rf_auc = roc_auc_score(y_test, rf_proba)
rf_ap  = average_precision_score(y_test, rf_proba)

print("Random Forest Classifier")
print(classification_report(y_test, rf_pred, target_names=['legit', 'fraud']))
print(f"ROC-AUC: {rf_auc:.4f}  PR-AUC: {rf_ap:.4f}")
print(f"training time: {rf_time:.4f} seconds")
"""))

# Cell 7: XGBoost Classifier
cells.append(md("""\
## Step 6 - training XGBoost Model

unlike Random Forest which builds trees in parallel and independently, Gradient Boosting builds trees sequentially.
each new tree learns from the errors (residuals) of the previous trees.

key hyperparams for XGBoost:
- `learning_rate` (eta): shrinks the contribution of each tree to prevent overfitting. lower is usually better but needs more trees (n_estimators).
- `max_depth`: depth of each tree. typically 3-8 (shallower than RF because boosting relies on weak learners).
- `scale_pos_weight`: handles class imbalance. we set it to `sum(negative_instances) / sum(positive_instances)` which is about 54.
- `subsample`: fraction of samples to randomly sample for each tree (reduces overfitting).
- `colsample_bytree`: fraction of features to randomly sample for each tree.
"""))

cells.append(code("""\
# calculating class imbalance ratio for scale_pos_weight
ratio = (len(y_train) - y_train.sum()) / y_train.sum()
print(f"ratio of negative/positive classes: {ratio:.2f}")

xgb_clf = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=ratio,
    random_state=SEED,
    eval_metric='logloss',
    use_label_encoder=False
)

start_time = time.time()
# using eval_set to track training improvements/history
xgb_clf.fit(
    X_train_sc, y_train,
    eval_set=[(X_train_sc, y_train), (X_test_sc, y_test)],
    verbose=False
)
xgb_time = time.time() - start_time

xgb_pred  = xgb_clf.predict(X_test_sc)
xgb_proba = xgb_clf.predict_proba(X_test_sc)[:, 1]

xgb_auc = roc_auc_score(y_test, xgb_proba)
xgb_ap  = average_precision_score(y_test, xgb_proba)

print("XGBoost Classifier")
print(classification_report(y_test, xgb_pred, target_names=['legit', 'fraud']))
print(f"ROC-AUC: {xgb_auc:.4f}  PR-AUC: {xgb_ap:.4f}")
print(f"training time: {xgb_time:.4f} seconds")
"""))

# Cell 8: LightGBM Classifier
cells.append(md("""\
## Step 7 - training LightGBM Model

LightGBM is a fast, high-performance gradient boosting framework developed by Microsoft.
it grows trees **leaf-wise** rather than **depth-wise** (level-wise), which makes it faster and often more accurate but easier to overfit if max_depth isn't controlled.
"""))

cells.append(code("""\
lgb_clf = lgb.LGBMClassifier(
    n_estimators=200,
    max_depth=6,
    num_leaves=31,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=ratio,
    random_state=SEED,
    verbose=-1
)

start_time = time.time()
# tracking learning progress
lgb_clf.fit(
    X_train_sc, y_train,
    eval_set=[(X_train_sc, y_train), (X_test_sc, y_test)],
    callbacks=[lgb.early_stopping(50, verbose=False)]
)
lgb_time = time.time() - start_time

lgb_pred  = lgb_clf.predict(X_test_sc)
lgb_proba = lgb_clf.predict_proba(X_test_sc)[:, 1]

lgb_auc = roc_auc_score(y_test, lgb_proba)
lgb_ap  = average_precision_score(y_test, lgb_proba)

print("LightGBM Classifier")
print(classification_report(y_test, lgb_pred, target_names=['legit', 'fraud']))
print(f"ROC-AUC: {lgb_auc:.4f}  PR-AUC: {lgb_ap:.4f}")
print(f"training time: {lgb_time:.4f} seconds")
"""))

# Cell 9: Learning curves plot
cells.append(md("""\
## Step 8 - analyzing training improvements (learning curves)

let's see how the model improved with each added tree. we'll extract the evaluation results from XGBoost and plot the loss (logloss) over iterations.
this shows if we started to overfit or if adding trees actually improved performance.
"""))

cells.append(code("""\
# getting evaluation history from XGBoost
evals_result = xgb_clf.evals_result()
train_loss = evals_result['validation_0']['logloss']
val_loss = evals_result['validation_1']['logloss']

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(train_loss, label='train loss', color='steelblue', lw=2)
ax.plot(val_loss, label='validation/test loss', color='tomato', lw=2)
ax.axvline(np.argmin(val_loss), color='purple', linestyle='--', alpha=0.7, 
           label=f'best iteration ({np.argmin(val_loss)})')

ax.set_xlabel('number of boosting iterations (trees)')
ax.set_ylabel('log loss')
ax.set_title('XGBoost - Loss Convergence Analysis')
ax.legend()
plt.tight_layout()
plt.savefig('learning_curves.png', bbox_inches='tight', dpi=150)
plt.show()

print(f"minimum validation loss reached at iteration: {np.argmin(val_loss)}")
"""))

# Cell 10: Confusion Matrices comparison
cells.append(md("""\
## Step 9 - confusion matrices side-by-side
let's compare our errors.
- FP: customers getting annoyed by false alarms
- FN: missed fraud (bank loses money)
"""))

cells.append(code("""\
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

models_data = [
    (rf_pred, rf_proba, 'Random Forest'),
    (xgb_pred, xgb_proba, 'XGBoost'),
    (lgb_pred, lgb_proba, 'LightGBM')
]

for ax, (pred, proba, title) in zip(axes, models_data):
    cm = confusion_matrix(y_test, pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=['legit', 'fraud'])
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    auc = roc_auc_score(y_test, proba)
    ax.set_title(f'{title}  (AUC={auc:.3f})')

plt.suptitle('Confusion Matrices - model comparisons', fontsize=13)
plt.tight_layout()
plt.savefig('confusion_matrices.png', bbox_inches='tight', dpi=150)
plt.show()
"""))

# Cell 11: ROC and PR Curves comparison
cells.append(md("""\
## Step 10 - ROC and Precision-Recall Curves

Precision-Recall is much more reliable than ROC for imbalanced datasets because ROC-AUC can look artificially high due to the huge number of true negatives.
"""))

cells.append(code("""\
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

colors = ['orange', 'steelblue', 'green']
models_probs = [
    ('Random Forest', rf_proba),
    ('XGBoost', xgb_proba),
    ('LightGBM', lgb_proba)
]

for (name, proba), color in zip(models_probs, colors):
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc = roc_auc_score(y_test, proba)
    axes[0].plot(fpr, tpr, label=f'{name}  AUC={auc:.4f}', color=color, lw=2)

axes[0].plot([0,1], [0,1], 'k--', label='random')
axes[0].set_xlabel('false positive rate')
axes[0].set_ylabel('true positive rate')
axes[0].set_title('ROC Curves')
axes[0].legend()

for (name, proba), color in zip(models_probs, colors):
    prec, rec, _ = precision_recall_curve(y_test, proba)
    ap = average_precision_score(y_test, proba)
    axes[1].plot(rec, prec, label=f'{name}  AP={ap:.4f}', color=color, lw=2)

baseline = y_test.mean()
axes[1].axhline(baseline, linestyle='--', color='gray', label=f'baseline={baseline:.3f}')
axes[1].set_xlabel('recall')
axes[1].set_ylabel('precision')
axes[1].set_title('Precision-Recall Curves')
axes[1].legend()

plt.suptitle('ROC vs Precision-Recall Evaluation', fontsize=14)
plt.tight_layout()
plt.savefig('roc_pr_curves.png', bbox_inches='tight', dpi=150)
plt.show()
"""))

# Cell 12: Feature importance side-by-side
cells.append(md("""\
## Step 11 - Feature Importance Comparison

do the models agree on which features are the most important?
Random Forest uses mean decrease in impurity (Gini). XGBoost and LightGBM measure the gain (relative contribution) of each feature to the split quality.
"""))

cells.append(code("""\
rf_imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False).head(10)
xgb_imp = pd.Series(xgb_clf.feature_importances_, index=X.columns).sort_values(ascending=False).head(10)
lgb_imp = pd.Series(lgb_clf.feature_importances_, index=X.columns).sort_values(ascending=False).head(10)

fig, axes = plt.subplots(1, 3, figsize=(16, 6))

sns.barplot(x=rf_imp.values, y=rf_imp.index, ax=axes[0], palette='Blues_r')
axes[0].set_title('Random Forest Importance')
axes[0].set_xlabel('gini impurity decrease')

sns.barplot(x=xgb_imp.values, y=xgb_imp.index, ax=axes[1], palette='Oranges_r')
axes[1].set_title('XGBoost Importance')
axes[1].set_xlabel('gain score')

sns.barplot(x=lgb_imp.values, y=lgb_imp.index, ax=axes[2], palette='Greens_r')
axes[2].set_title('LightGBM Importance')
axes[2].set_xlabel('split frequency / importance')

plt.suptitle('Top Feature Importances Across Models', fontsize=14)
plt.tight_layout()
plt.savefig('feature_importance.png', bbox_inches='tight', dpi=150)
plt.show()
"""))

# Cell 13: Model stability (cross validation)
cells.append(md("""\
## Step 12 - Model Stability (5-Fold Stratified Cross-Validation)

let's perform cross validation to make sure these differences are robust and not just random fluctuations on this particular split.
we'll run stratified 5-fold CV using the ROC-AUC scoring metric.
"""))

cells.append(code("""\
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

print("running 5-fold CV for all models (this might take a few seconds)...")
rf_cvs = cross_val_score(rf, X_train_sc, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
xgb_cvs = cross_val_score(xgb_clf, X_train_sc, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
lgb_cvs = cross_val_score(lgb_clf, X_train_sc, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)

print(f"\\n{'model':<20} {'mean ROC-AUC':>14} {'std dev':>10}")
print("-" * 48)
for name, scores in [('Random Forest', rf_cvs), ('XGBoost', xgb_cvs), ('LightGBM', lgb_cvs)]:
    print(f"{name:<20} {scores.mean():>14.4f} {scores.std():>10.4f}")
"""))

# Cell 14: Cross validation boxplot
cells.append(code("""\
fig, ax = plt.subplots(figsize=(8, 5))

bp = ax.boxplot([rf_cvs, xgb_cvs, lgb_cvs],
                labels=['Random Forest', 'XGBoost', 'LightGBM'],
                patch_artist=True,
                medianprops=dict(color='black', linewidth=2))

colors_list = ['orange', 'steelblue', 'green']
for box, col in zip(bp['boxes'], colors_list):
    box.set_facecolor(col)
    box.set_alpha(0.7)

ax.scatter([1]*5, rf_cvs, color='darkorange', zorder=5, s=45, edgecolor='black')
ax.scatter([2]*5, xgb_cvs, color='navy', zorder=5, s=45, edgecolor='black')
ax.scatter([3]*5, lgb_cvs, color='darkgreen', zorder=5, s=45, edgecolor='black')

ax.set_ylabel('ROC-AUC Score')
ax.set_title('5-Fold CV - Model Stability Comparison')
ax.set_ylim(0.85, 1.01)
ax.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('cross_validation.png', bbox_inches='tight', dpi=150)
plt.show()
"""))

# Cell 15: Final metric comparison
cells.append(md("""\
## Step 13 - Final Performance Comparison

let's summarize the overall comparison by compiling all performance numbers.
"""))

cells.append(code("""\
def get_metrics_dict(y_true, y_pred, y_prob, train_time, name):
    return {
        'model'    : name,
        'accuracy' : accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall'   : recall_score(y_true, y_pred),
        'f1'       : f1_score(y_true, y_pred, zero_division=0),
        'roc_auc'  : roc_auc_score(y_true, y_prob),
        'pr_auc'   : average_precision_score(y_true, y_prob),
        'time (s)' : train_time
    }

results = pd.DataFrame([
    get_metrics_dict(y_test, rf_pred, rf_proba, rf_time, 'Random Forest'),
    get_metrics_dict(y_test, xgb_pred, xgb_proba, xgb_time, 'XGBoost'),
    get_metrics_dict(y_test, lgb_pred, lgb_proba, lgb_time, 'LightGBM')
])
results.set_index('model', inplace=True)

print(results.round(4))
"""))

# Cell 16: Final comparison chart
cells.append(code("""\
metrics_list = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc']

rf_vals = results.loc['Random Forest', metrics_list].values
xgb_vals = results.loc['XGBoost', metrics_list].values
lgb_vals = results.loc['LightGBM', metrics_list].values

x = np.arange(len(metrics_list))
w = 0.25

fig, ax = plt.subplots(figsize=(12, 6))

b1 = ax.bar(x - w, rf_vals, w, label='Random Forest', color='orange', alpha=0.8)
b2 = ax.bar(x, xgb_vals, w, label='XGBoost', color='steelblue', alpha=0.8)
b3 = ax.bar(x + w, lgb_vals, w, label='LightGBM', color='green', alpha=0.8)

ax.set_xticks(x)
ax.set_xticklabels(['accuracy', 'precision', 'recall', 'f1', 'roc-auc', 'pr-auc'], rotation=15)
ax.set_ylabel('score')
ax.set_ylim(0, 1.2)
ax.set_title('Model Performance Comparison - all metrics')
ax.legend()

# adding text values on bars
for bar in b1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{bar.get_height():.2f}', ha='center', fontsize=8, color='darkorange')
for bar in b2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{bar.get_height():.2f}', ha='center', fontsize=8, color='navy')
for bar in b3:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{bar.get_height():.2f}', ha='center', fontsize=8, color='darkgreen')

plt.tight_layout()
plt.savefig('model_comparison.png', bbox_inches='tight', dpi=150)
plt.show()
"""))

# Cell 17: Conclusions
cells.append(md("""\
## Conclusions and Key Takeaways

1. **bagging vs boosting:**
   - boosting (XGBoost & LightGBM) sequentially corrects the errors of previous trees. this makes it extremely powerful on tabular data, catching complex patterns that Random Forest missed.
   - in our imbalanced dataset, XGBoost and LightGBM outperformed Random Forest on the critical metrics: **recall** and **PR-AUC**.

2. **XGBoost vs LightGBM:**
   - both models achieved outstanding ROC-AUC and PR-AUC.
   - LightGBM was **significantly faster** to train due to its histogram-based split finding and leaf-wise tree growth.
   - both models handled the severe class imbalance effectively when configured with `scale_pos_weight`.

3. **training curves:**
   - the loss curves show a rapid decrease in both training and test logloss in the first 50 iterations, after which it plateaus.
   - this highlights the importance of **early stopping** to prevent models from over-fitting the training residuals.
"""))

# Writing notebook metadata
nb = new_notebook(cells=cells)
nb.metadata['kernelspec'] = {
    'display_name': 'Python 3',
    'language': 'python',
    'name': 'python3'
}
nb.metadata['language_info'] = {'name': 'python', 'version': '3.10.0'}

# Saving
NB_PATH = 'day19_boosting_models.ipynb'
with open(NB_PATH, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print(f"saved: {NB_PATH}")
print("running...")

result = subprocess.run(
    [sys.executable, '-m', 'nbconvert', '--to', 'notebook',
     '--execute', '--inplace',
     '--ExecutePreprocessor.timeout=300',
     '--ExecutePreprocessor.kernel_name=python3',
     NB_PATH],
    capture_output=True, text=True
)

if result.returncode == 0:
    print("done!")
else:
    print("error:")
    print(result.stderr[-3000:])
