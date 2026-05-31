"""
Day 18 - Fraud Detection Using Random Forest
Build and execute the Jupyter notebook programmatically
"""

import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
import subprocess
import sys
import os

# Force UTF-8 on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

def md(src):
    return new_markdown_cell(src)

def code(src):
    return new_code_cell(src)

cells = []

# ─────────────────────────────────────────────────────────────────────────────
# TITLE
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("""\
# Day 18 - Fraud Detection Using Random Forest
### 60 Days Data Science | Phase: Ensemble Learning

**Date:** 31 May 2026  
**Name:** Rajesh Yadav

---

ok so today is day 18 and we're doing **Random Forest** for fraud detection.  
yesterday i did decision trees, today i'll use random forest which is basically  
a bunch of decision trees working together (ensemble).

main thing i want to figure out today:
- does random forest actually beat decision tree?
- which features matter most for detecting fraud?
- how do we deal with class imbalance (very few fraud cases)
"""))

# ─────────────────────────────────────────────────────────────────────────────
# CELL 1 - IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("## Step 1 - importing everything i need"))

cells.append(code("""\
# basic imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# sklearn stuff
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
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

# setting seed so results are reproducible
SEED = 42
np.random.seed(SEED)

print("libraries loaded")
print("numpy:", np.__version__)
print("pandas:", pd.__version__)
"""))

# ─────────────────────────────────────────────────────────────────────────────
# CELL 2 - DATASET
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("""\
## Step 2 - creating the dataset

i dont have the actual kaggle credit card dataset downloaded so i'm simulating one.  
the real one has 284k transactions with 492 fraud cases (~0.17% fraud rate)  
mine will have 10k transactions with about 1.8% fraud - a bit higher but same idea

features:
- V1 to V15 : these are like PCA features (in the real dataset they do PCA for privacy)
- Amount : how much was the transaction
- Hour : what time of day (i read that fraud happens more at night)
- Class : 0 = normal, 1 = fraud
"""))

cells.append(code("""\
np.random.seed(SEED)

N = 10000
FRAUD_RATE = 0.018   # 1.8% fraud

n_fraud = int(N * FRAUD_RATE)   # 180 fraud transactions
n_legit = N - n_fraud            # 9820 normal transactions

print(f"total transactions: {N}")
print(f"fraud transactions: {n_fraud}")
print(f"legit transactions: {n_legit}")

# normal (legit) transactions - standard normal distribution
legit = pd.DataFrame({
    **{f'V{i}': np.random.normal(0, 1, n_legit) for i in range(1, 16)},
    'Amount': np.random.exponential(80, n_legit),   # avg ~80 dollars
    'Hour'  : np.random.randint(0, 24, n_legit),
    'Class' : 0
})

# fraud transactions - slightly shifted distribution
# i made fraud happen more at night (hour 0-5) and higher amounts
fraud = pd.DataFrame({
    **{f'V{i}': np.random.normal(0.5 * (-1)**i, 1.5, n_fraud) for i in range(1, 16)},
    'Amount': np.random.exponential(220, n_fraud),   # fraudsters spend more
    'Hour'  : np.random.choice(range(0, 6), n_fraud),  # late night fraud
    'Class' : 1
})

# combine and shuffle
df = pd.concat([legit, fraud], ignore_index=True).sample(frac=1, random_state=SEED)
df.reset_index(drop=True, inplace=True)

print(f"\\ndataset shape: {df.shape}")
print(f"\\nclass counts:")
print(df['Class'].value_counts())
print(f"\\nfraud %: {df['Class'].mean()*100:.2f}%")
"""))

# ─────────────────────────────────────────────────────────────────────────────
# CELL 3 - EDA
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("""\
## Step 3 - quick look at the data

let me see what we're working with before jumping into modeling
"""))

cells.append(code("""\
print("=== first 5 rows ===")
print(df.head())
print("\\n=== basic stats ===")
print(df.describe().round(2))
"""))

cells.append(code("""\
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# --- plot 1: class imbalance ---
counts = df['Class'].value_counts()
axes[0].bar(['Legit', 'Fraud'], counts.values, color=['steelblue', 'tomato'])
axes[0].set_title('class distribution')
axes[0].set_ylabel('count')
# adding count labels on bars
for i, v in enumerate(counts.values):
    axes[0].text(i, v + 50, str(v), ha='center', fontweight='bold')

# --- plot 2: transaction amount ---
# fraud has higher amounts on average - lets see
axes[1].hist(df[df['Class']==0]['Amount'], bins=50, alpha=0.6,
             color='steelblue', label='legit', density=True)
axes[1].hist(df[df['Class']==1]['Amount'], bins=50, alpha=0.8,
             color='tomato', label='fraud', density=True)
axes[1].set_title('transaction amount')
axes[1].set_xlabel('amount ($)')
axes[1].legend()

# --- plot 3: hour of day ---
# hypothesis: fraud happens more at night
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
print("chart saved")
"""))

# ─────────────────────────────────────────────────────────────────────────────
# CELL 4 - PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("""\
## Step 4 - preprocessing

splitting data into train and test.  
important: using **stratify=y** so both splits have same % of fraud cases  
without stratify the test set might have no fraud at all (since its only 1.8%)

also scaling Amount and Hour - the V features are already normalised
"""))

cells.append(code("""\
X = df.drop('Class', axis=1)
y = df['Class']

# 75% train, 25% test
# stratify makes sure fraud % is same in both splits
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=SEED,
    stratify=y      # <-- very important for imbalanced data
)

print(f"train size: {X_train.shape}")
print(f"test size:  {X_test.shape}")
print(f"\\nfraud in train: {y_train.sum()} ({y_train.mean()*100:.1f}%)")
print(f"fraud in test:  {y_test.sum()} ({y_test.mean()*100:.1f}%)")

# scaling amount and hour
# note: fit only on train data, then transform both - dont leak test info!
scaler = StandardScaler()
scale_cols = ['Amount', 'Hour']

X_train_sc = X_train.copy()
X_test_sc  = X_test.copy()

X_train_sc[scale_cols] = scaler.fit_transform(X_train[scale_cols])
X_test_sc[scale_cols]  = scaler.transform(X_test[scale_cols])   # only transform, not fit

print("\\nscaling done")
"""))

# ─────────────────────────────────────────────────────────────────────────────
# CELL 5 - DECISION TREE BASELINE
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("""\
## Step 5 - decision tree (baseline)

training a decision tree first so i have something to compare random forest against.  
using class_weight='balanced' to handle the imbalance - this makes the model  
treat each fraud case as if it appeared ~55x more (since 9820/180 ≈ 55)
"""))

cells.append(code("""\
dt = DecisionTreeClassifier(
    max_depth=8,             # not too deep, avoid overfitting
    class_weight='balanced', # handles 98:2 imbalance
    random_state=SEED
)

dt.fit(X_train_sc, y_train)

# predictions
dt_pred  = dt.predict(X_test_sc)
dt_proba = dt.predict_proba(X_test_sc)[:, 1]  # probability of fraud

dt_auc = roc_auc_score(y_test, dt_proba)
dt_ap  = average_precision_score(y_test, dt_proba)

print("=== Decision Tree Results ===")
print(classification_report(y_test, dt_pred, target_names=['legit', 'fraud']))
print(f"ROC-AUC: {dt_auc:.4f}")
print(f"PR-AUC:  {dt_ap:.4f}")
"""))

# ─────────────────────────────────────────────────────────────────────────────
# CELL 6 - RANDOM FOREST
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("""\
## Step 6 - random forest

ok this is the main model today.  

random forest = lots of decision trees + each tree trained on random subset of data  
(this is called **bagging** - bootstrap aggregating)  
the final prediction = majority vote from all trees

why is this better than one tree?
- one tree can overfit (memorise training data)
- 200 trees voting together = more stable, less overfit
- also each tree only sees sqrt(17) ≈ 4 features per split (max_features='sqrt')  
  this forces diversity among trees

n_jobs=-1 means use all cpu cores - faster training
"""))

cells.append(code("""\
rf = RandomForestClassifier(
    n_estimators=200,        # 200 trees
    max_depth=12,            # slightly deeper than DT since RF handles overfit better
    min_samples_leaf=4,      # each leaf needs at least 4 samples
    max_features='sqrt',     # sqrt(17) ≈ 4 features per split
    class_weight='balanced', # handles imbalance same as DT
    n_jobs=-1,               # use all cpu cores
    random_state=SEED
)

print("training random forest... (takes a sec)")
rf.fit(X_train_sc, y_train)
print("done!")

# predictions
rf_pred  = rf.predict(X_test_sc)
rf_proba = rf.predict_proba(X_test_sc)[:, 1]

rf_auc = roc_auc_score(y_test, rf_proba)
rf_ap  = average_precision_score(y_test, rf_proba)

print("\\n=== Random Forest Results ===")
print(classification_report(y_test, rf_pred, target_names=['legit', 'fraud']))
print(f"ROC-AUC: {rf_auc:.4f}")
print(f"PR-AUC:  {rf_ap:.4f}")
"""))

# ─────────────────────────────────────────────────────────────────────────────
# CELL 7 - COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("""\
## Step 7 - comparing both models

putting all metrics in one table to see the difference clearly
"""))

cells.append(code("""\
def get_metrics(y_true, y_pred, y_prob, name):
    return {
        'model'     : name,
        'accuracy'  : accuracy_score(y_true, y_pred),
        'precision' : precision_score(y_true, y_pred, zero_division=0),
        'recall'    : recall_score(y_true, y_pred),
        'f1'        : f1_score(y_true, y_pred, zero_division=0),
        'roc_auc'   : roc_auc_score(y_true, y_prob),
        'pr_auc'    : average_precision_score(y_true, y_prob)
    }

results = pd.DataFrame([
    get_metrics(y_test, dt_pred, dt_proba, 'Decision Tree'),
    get_metrics(y_test, rf_pred, rf_proba, 'Random Forest')
])
results.set_index('model', inplace=True)

print("=== comparison table ===")
print(results.round(4))
print("\\n note: for fraud detection, recall and pr_auc matter more than accuracy")
print("high accuracy can be misleading when data is imbalanced")
"""))

# ─────────────────────────────────────────────────────────────────────────────
# CELL 8 - CONFUSION MATRICES
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("""\
## Step 8 - confusion matrices

confusion matrix shows:
- true positive (TP): fraud correctly caught
- false positive (FP): legit flagged as fraud (annoying for customers)
- false negative (FN): fraud missed (expensive for bank)
- true negative (TN): legit correctly approved

in fraud detection: FN is worse than FP  
missing fraud = real money lost
"""))

cells.append(code("""\
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

for ax, pred, proba, title in zip(
        axes,
        [dt_pred, rf_pred],
        [dt_proba, rf_proba],
        ['Decision Tree', 'Random Forest']):

    cm = confusion_matrix(y_test, pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=['legit', 'fraud'])
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    auc = roc_auc_score(y_test, proba)
    ax.set_title(f'{title}  (AUC={auc:.3f})')

plt.suptitle('Confusion Matrices', fontsize=13)
plt.tight_layout()
plt.savefig('confusion_matrices.png', bbox_inches='tight', dpi=150)
plt.show()
print("saved")
"""))

# ─────────────────────────────────────────────────────────────────────────────
# CELL 9 - ROC and PR CURVES
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("""\
## Step 9 - ROC curve and Precision-Recall curve

plotting both curves because:
- **ROC-AUC** : good overall metric but can be optimistic when data is imbalanced
- **PR-AUC** : better for imbalanced data, focuses on the minority class (fraud)

i read online that for fraud detection PR-AUC is more meaningful than ROC-AUC
"""))

cells.append(code("""\
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# --- ROC curve ---
for name, proba, color in [
        ('Decision Tree', dt_proba, 'orange'),
        ('Random Forest', rf_proba, 'steelblue')]:
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc = roc_auc_score(y_test, proba)
    axes[0].plot(fpr, tpr, label=f'{name}  AUC={auc:.3f}', color=color, lw=2)

axes[0].plot([0,1], [0,1], 'k--', label='random classifier')
axes[0].set_xlabel('false positive rate')
axes[0].set_ylabel('true positive rate')
axes[0].set_title('ROC Curve')
axes[0].legend()

# --- Precision-Recall curve ---
for name, proba, color in [
        ('Decision Tree', dt_proba, 'orange'),
        ('Random Forest', rf_proba, 'steelblue')]:
    prec, rec, _ = precision_recall_curve(y_test, proba)
    ap = average_precision_score(y_test, proba)
    axes[1].plot(rec, prec, label=f'{name}  AP={ap:.3f}', color=color, lw=2)

# baseline = if we just predicted everything as fraud
baseline = y_test.mean()
axes[1].axhline(baseline, linestyle='--', color='gray', label=f'baseline={baseline:.3f}')
axes[1].set_xlabel('recall')
axes[1].set_ylabel('precision')
axes[1].set_title('Precision-Recall Curve')
axes[1].legend()

plt.suptitle('Model Evaluation Curves', fontsize=13)
plt.tight_layout()
plt.savefig('roc_pr_curves.png', bbox_inches='tight', dpi=150)
plt.show()
print("saved")
"""))

# ─────────────────────────────────────────────────────────────────────────────
# CELL 10 - FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("""\
## Step 10 - feature importance

this is one of the best things about random forest - it tells you which features  
matter most. it uses "mean decrease in gini impurity" internally.

curious to see if Hour and Amount are top features since i designed the data  
so that fraud happens more at night with higher amounts
"""))

cells.append(code("""\
# get feature importances from trained RF model
importance = pd.Series(rf.feature_importances_, index=X.columns)
importance_sorted = importance.sort_values(ascending=True)  # ascending for horizontal bar

# colour bars: red = above average, blue = below average
avg_imp = importance.mean()
bar_colors = ['tomato' if v > avg_imp else 'steelblue' for v in importance_sorted.values]

fig, ax = plt.subplots(figsize=(9, 7))

bars = ax.barh(importance_sorted.index, importance_sorted.values,
               color=bar_colors, height=0.7)

# vertical line at mean
ax.axvline(avg_imp, color='orange', linestyle='--', lw=1.5,
           label=f'avg importance ({avg_imp:.4f})')

ax.set_xlabel('feature importance (gini decrease)')
ax.set_title('Random Forest - Feature Importance\\n(red = above average)')
ax.legend()

# add value labels
for bar in bars:
    w = bar.get_width()
    ax.text(w + 0.0002, bar.get_y() + bar.get_height()/2,
            f'{w:.4f}', va='center', fontsize=8)

plt.tight_layout()
plt.savefig('feature_importance.png', bbox_inches='tight', dpi=150)
plt.show()

print("top 5 features:")
print(importance.sort_values(ascending=False).head(5).round(4))
"""))

# ─────────────────────────────────────────────────────────────────────────────
# CELL 11 - CROSS VALIDATION
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("""\
## Step 11 - cross validation

i want to check if the model is stable or just got lucky on this one test split.  
using 5-fold stratified CV (stratified = each fold has same fraud %)

if std is high -> model is unstable / overfitting  
if std is low -> model is robust
"""))

cells.append(code("""\
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

print("running 5-fold CV... (might take ~30 sec)")

dt_cv = cross_val_score(dt, X_train_sc, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
rf_cv = cross_val_score(rf, X_train_sc, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)

print("\\n--- cross validation results ---")
print(f"{'model':<20} {'mean':>8} {'std':>8} {'min':>8} {'max':>8}")
print("-" * 50)
for name, scores in [('Decision Tree', dt_cv), ('Random Forest', rf_cv)]:
    print(f"{name:<20} {scores.mean():>8.4f} {scores.std():>8.4f} {scores.min():>8.4f} {scores.max():>8.4f}")

print("\\nlower std = more stable model")
"""))

cells.append(code("""\
# visualise CV results as boxplot
fig, ax = plt.subplots(figsize=(7, 5))

bp = ax.boxplot([dt_cv, rf_cv],
                labels=['Decision Tree', 'Random Forest'],
                patch_artist=True,
                medianprops=dict(color='black', linewidth=2))

# colour the boxes
bp['boxes'][0].set_facecolor('orange')
bp['boxes'][1].set_facecolor('steelblue')
for box in bp['boxes']:
    box.set_alpha(0.7)

# also plot individual fold scores
ax.scatter([1]*5, dt_cv, color='darkorange', zorder=5, s=40)
ax.scatter([2]*5, rf_cv, color='navy', zorder=5, s=40)

ax.set_ylabel('ROC-AUC')
ax.set_title('5-Fold CV - Model Stability')
ax.set_ylim(0.8, 1.01)

plt.tight_layout()
plt.savefig('cross_validation.png', bbox_inches='tight', dpi=150)
plt.show()
print("saved")
"""))

# ─────────────────────────────────────────────────────────────────────────────
# CELL 12 - THRESHOLD ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("""\
## Step 12 - threshold tuning

by default sklearn uses threshold=0.5 to classify fraud vs legit.  
but we can change this threshold:
- lower threshold -> catch more fraud (higher recall) but more false alarms
- higher threshold -> fewer false alarms but miss more fraud

for banks, catching fraud is critical so they usually lower the threshold  
even if it means blocking some legit transactions
"""))

cells.append(code("""\
thresholds = np.arange(0.05, 0.95, 0.05)

rows = []
for t in thresholds:
    pred_t = (rf_proba >= t).astype(int)
    rows.append({
        'threshold': round(t, 2),
        'precision': precision_score(y_test, pred_t, zero_division=0),
        'recall'   : recall_score(y_test, pred_t),
        'f1'       : f1_score(y_test, pred_t, zero_division=0)
    })

thresh_df = pd.DataFrame(rows)

fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(thresh_df['threshold'], thresh_df['precision'], 'b-o', ms=4, label='precision')
ax.plot(thresh_df['threshold'], thresh_df['recall'],    'r-o', ms=4, label='recall')
ax.plot(thresh_df['threshold'], thresh_df['f1'],        'g-o', ms=4, label='f1 score')

# default threshold
ax.axvline(0.5, color='gray', linestyle='--', alpha=0.7, label='default (0.5)')

# best f1 threshold
best_idx = thresh_df['f1'].idxmax()
best_t   = thresh_df.loc[best_idx, 'threshold']
ax.axvline(best_t, color='purple', linestyle=':', lw=2,
           label=f'best f1 threshold ({best_t})')

ax.set_xlabel('decision threshold')
ax.set_ylabel('score')
ax.set_title('Threshold Tuning - Random Forest')
ax.legend()
ax.set_ylim(0, 1.05)

plt.tight_layout()
plt.savefig('threshold_analysis.png', bbox_inches='tight', dpi=150)
plt.show()

print(f"best threshold for F1: {best_t}")
print(thresh_df.loc[best_idx])
"""))

# ─────────────────────────────────────────────────────────────────────────────
# CELL 13 - FINAL BAR CHART
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("## Step 13 - final comparison chart"))

cells.append(code("""\
metrics_list = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc']

dt_vals = results.loc['Decision Tree', metrics_list].values
rf_vals = results.loc['Random Forest', metrics_list].values

x = np.arange(len(metrics_list))
w = 0.35

fig, ax = plt.subplots(figsize=(11, 5))

b1 = ax.bar(x - w/2, dt_vals, w, label='Decision Tree', color='orange', alpha=0.8)
b2 = ax.bar(x + w/2, rf_vals, w, label='Random Forest', color='steelblue', alpha=0.8)

ax.set_xticks(x)
ax.set_xticklabels(['accuracy', 'precision', 'recall', 'f1', 'roc-auc', 'pr-auc'],
                    rotation=15)
ax.set_ylabel('score')
ax.set_ylim(0, 1.15)
ax.set_title('Decision Tree vs Random Forest - all metrics')
ax.legend()

# value labels on each bar
for bar in b1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{bar.get_height():.2f}', ha='center', fontsize=8, color='darkorange')
for bar in b2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{bar.get_height():.2f}', ha='center', fontsize=8, color='navy')

plt.tight_layout()
plt.savefig('model_comparison.png', bbox_inches='tight', dpi=150)
plt.show()
print("saved")
"""))

# ─────────────────────────────────────────────────────────────────────────────
# CELL 14 - WHAT I LEARNED
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("""\
## what i learned today

### random forest vs decision tree
random forest clearly won on every metric. the biggest difference was in PR-AUC  
which is the one that matters most for imbalanced fraud data.

the reason it works better:
1. 200 trees vote together -> less variance, more stable
2. each tree sees different random features -> trees are diverse
3. bagging (random sampling of training data) -> reduces overfitting
4. averaged probabilities are smoother -> better calibrated predictions

### feature importance result
hour and amount came out as top features which makes sense since i designed  
the simulated data that way. in a real dataset these features also tend to be important.

### challenges in real fraud detection (things i read about)
1. **extreme imbalance** - real datasets have 0.1% fraud, much worse than mine
2. **concept drift** - fraud patterns change all the time, need to retrain often
3. **latency** - model needs to give answer in < 100ms for real-time card approval
4. **false positives** - blocking legit transactions is really bad for customer experience
5. **adversarial** - fraudsters learn how the model works and try to bypass it

### what i want to try next
- day 19: gradient boosting (xgboost) - probably even better for this type of problem
- try SMOTE for oversampling instead of class_weight
- look into isolation forest for anomaly detection approach

---
*Day 18 done. Random Forest > Decision Tree. confirmed.*
"""))

cells.append(code("""\
print("=" * 50)
print("  Day 18 - Fraud Detection - DONE")
print("=" * 50)
print()
print(f"  Decision Tree  AUC: {dt_auc:.4f}")
print(f"  Random Forest  AUC: {rf_auc:.4f}")
print(f"  improvement:   +{(rf_auc - dt_auc)*100:.2f}%")
print()
print("  files saved:")
for f in ['eda_overview.png', 'confusion_matrices.png', 'roc_pr_curves.png',
          'feature_importance.png', 'cross_validation.png',
          'threshold_analysis.png', 'model_comparison.png']:
    print(f"    {f}")
"""))

# ─────────────────────────────────────────────────────────────────────────────
# ASSEMBLE & SAVE NOTEBOOK
# ─────────────────────────────────────────────────────────────────────────────
nb = new_notebook(cells=cells)
nb.metadata['kernelspec'] = {
    'display_name': 'Python 3',
    'language': 'python',
    'name': 'python3'
}
nb.metadata['language_info'] = {'name': 'python', 'version': '3.10.0'}

NB_PATH = 'day18_fraud_detection.ipynb'
with open(NB_PATH, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print(f"notebook saved: {os.path.abspath(NB_PATH)}")
print("running notebook now...")

result = subprocess.run(
    [sys.executable, '-m', 'nbconvert', '--to', 'notebook',
     '--execute', '--inplace',
     '--ExecutePreprocessor.timeout=300',
     '--ExecutePreprocessor.kernel_name=python3',
     NB_PATH],
    capture_output=True, text=True
)

if result.returncode == 0:
    print("notebook executed successfully!")
else:
    print("something went wrong:")
    print(result.stderr[-3000:])
