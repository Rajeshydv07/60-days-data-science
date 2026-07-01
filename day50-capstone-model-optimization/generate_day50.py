import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import nbformat as nbf
import os
import json

def run_optimization():
    print("Loading data...")
    df = pd.read_csv("../day49-capstone-baseline/customer_data.csv")
    
    # Preprocessing (basic, similar to day49)
    X = df.drop(columns=['customer_id', 'churn'])
    X = pd.get_dummies(X, drop_first=True)
    y = df['churn']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Evaluating models...")
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42)
    }
    
    results = {}
    
    # Train Baselines
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name] = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "F1 Score": f1_score(y_test, y_pred)
        }
        
    print("Baseline Results:", results)
    
    print("Hyperparameter Tuning for Random Forest...")
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }
    grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring='f1', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_rf = grid_search.best_estimator_
    y_pred_best = best_rf.predict(X_test)
    results["Tuned Random Forest"] = {
        "Accuracy": accuracy_score(y_test, y_pred_best),
        "F1 Score": f1_score(y_test, y_pred_best)
    }
    
    print("Best RF Params:", grid_search.best_params_)
    print("Tuned RF Results:", results["Tuned Random Forest"])
    
    # Visualizations
    print("Generating visualizations...")
    # 1. Model Comparison Bar Chart
    plt.figure(figsize=(10, 6))
    res_df = pd.DataFrame(results).T
    res_df.plot(kind='bar', figsize=(10,6), colormap='viridis')
    plt.title('Model Comparison: Accuracy & F1 Score')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    
    # 2. ROC Curves
    plt.figure(figsize=(8, 6))
    for name, model in models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.2f})')
        
    y_prob_best = best_rf.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob_best)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, linestyle='--', label=f'Tuned RF (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig('roc_curves.png')
    
    # Generate Notebook
    print("Generating Jupyter Notebook...")
    nb = nbf.v4.new_notebook()
    
    cells = [
        nbf.v4.new_markdown_cell("# Day 50: Capstone Model Optimization\nEvaluating and tuning our models for the Customer Intelligence Platform."),
        nbf.v4.new_code_cell("import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nfrom sklearn.model_selection import train_test_split, GridSearchCV\nfrom sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.metrics import accuracy_score, f1_score\n\n# Load Data\ndf = pd.read_csv('../day49-capstone-baseline/customer_data.csv')\nprint(df.head())"),
        nbf.v4.new_markdown_cell("## Baseline Evaluation"),
        nbf.v4.new_code_cell("X = pd.get_dummies(df.drop(columns=['customer_id', 'churn']), drop_first=True)\ny = df['churn']\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n\nbaseline_model = RandomForestClassifier(random_state=42)\nbaseline_model.fit(X_train, y_train)\ny_pred = baseline_model.predict(X_test)\nprint(f'Baseline RF Accuracy: {accuracy_score(y_test, y_pred):.4f}')\nprint(f'Baseline RF F1: {f1_score(y_test, y_pred):.4f}')"),
        nbf.v4.new_markdown_cell("## Hyperparameter Tuning\nUsing GridSearchCV to find the optimal parameters and handle overfitting."),
        nbf.v4.new_code_cell("param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}\ngrid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring='f1')\ngrid.fit(X_train, y_train)\nprint(f'Best params: {grid.best_params_}')\nbest_model = grid.best_estimator_\nprint(f'Tuned F1 Score: {f1_score(y_test, best_model.predict(X_test)):.4f}')")
    ]
    nb.cells.extend(cells)
    
    with open('optimization.ipynb', 'w') as f:
        nbf.write(nb, f)
        
    print("Day 50 script completed successfully.")

if __name__ == '__main__':
    run_optimization()
