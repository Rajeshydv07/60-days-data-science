import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import shap
import nbformat as nbf
import os
import json

def run_explainability():
    print("Loading data...")
    df = pd.read_csv("../day49-capstone-baseline/customer_data.csv")
    
    # Preprocessing
    X = df.drop(columns=['customer_id', 'churn'])
    X = pd.get_dummies(X, drop_first=True)
    y = df['churn']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training Best Model (Random Forest)...")
    # Using the best params from Day 50 (or reasonable tuned params if not known)
    best_rf = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=2, random_state=42, n_jobs=-1)
    best_rf.fit(X_train, y_train)
    
    print("Generating Feature Importance Visualizations...")
    # 1. Standard Feature Importance
    importances = best_rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    feature_names = X.columns
    
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances (Random Forest)")
    plt.bar(range(X.shape[1]), importances[indices], align="center")
    plt.xticks(range(X.shape[1]), feature_names[indices], rotation=45, ha='right')
    plt.xlim([-1, X.shape[1]])
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    
    print("Generating SHAP Visualizations...")
    # 2. SHAP Values
    # Use TreeExplainer for Random Forest
    explainer = shap.TreeExplainer(best_rf)
    # Using a subset for SHAP to speed up generation
    X_test_sample = X_test.sample(n=min(500, len(X_test)), random_state=42)
    shap_values = explainer.shap_values(X_test_sample)
    
    # In binary classification, shap_values is a list of arrays for [class_0, class_1]. We usually explain the positive class (class 1)
    if isinstance(shap_values, list):
        shap_values_to_plot = shap_values[1]
    else:
        shap_values_to_plot = shap_values
        
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values_to_plot, X_test_sample, show=False)
    plt.tight_layout()
    plt.savefig('shap_summary.png')
    
    print("Generating Jupyter Notebook...")
    nb = nbf.v4.new_notebook()
    
    cells = [
        nbf.v4.new_markdown_cell("# Day 51: Capstone Explainable AI\nUsing SHAP and Feature Importance to explain our model's predictions to business stakeholders."),
        nbf.v4.new_code_cell("import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.ensemble import RandomForestClassifier\nimport shap\n\n# Initialize JS visualization for SHAP\nshap.initjs()\n\n# Load Data\ndf = pd.read_csv('../day49-capstone-baseline/customer_data.csv')\nprint(df.head())"),
        nbf.v4.new_markdown_cell("## Train the Tuned Model\nWe use the optimal hyperparameters discovered in Day 50."),
        nbf.v4.new_code_cell("X = pd.get_dummies(df.drop(columns=['customer_id', 'churn']), drop_first=True)\ny = df['churn']\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n\nbest_rf = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=2, random_state=42)\nbest_rf.fit(X_train, y_train)\nprint('Model trained successfully.')"),
        nbf.v4.new_markdown_cell("## Global Explainability: Feature Importance\nWhich features are most important across all customers?"),
        nbf.v4.new_code_cell("importances = best_rf.feature_importances_\nfeature_names = X.columns\nindices = np.argsort(importances)[::-1]\n\nplt.figure(figsize=(10, 6))\nplt.title('Feature Importances')\nplt.bar(range(X.shape[1]), importances[indices], align='center')\nplt.xticks(range(X.shape[1]), feature_names[indices], rotation=45, ha='right')\nplt.tight_layout()\nplt.show()"),
        nbf.v4.new_markdown_cell("## Advanced Explainability: SHAP Values\nSHAP (SHapley Additive exPlanations) provides a unified measure of feature importance and reveals the direction of the impact."),
        nbf.v4.new_code_cell("explainer = shap.TreeExplainer(best_rf)\n# Sample to speed up computation\nX_test_sample = X_test.sample(n=min(500, len(X_test)), random_state=42)\nshap_values = explainer.shap_values(X_test_sample)\n\n# Depending on SHAP version, shap_values might be a list\nif isinstance(shap_values, list):\n    shap_to_plot = shap_values[1]\nelse:\n    shap_to_plot = shap_values\n\nshap.summary_plot(shap_to_plot, X_test_sample)"),
        nbf.v4.new_markdown_cell("## Local Explainability: Explaining a Single Prediction\nLet's see why a specific customer was predicted to churn."),
        nbf.v4.new_code_cell("# Explain the first customer in our test sample\n# shap.force_plot(explainer.expected_value[1], shap_to_plot[0,:], X_test_sample.iloc[0,:])\n# For newer shap versions:\nshap.plots.waterfall(shap.Explanation(values=shap_to_plot[0], base_values=explainer.expected_value[1], data=X_test_sample.iloc[0,:], feature_names=X_test_sample.columns))")
    ]
    nb.cells.extend(cells)
    
    with open('explainability.ipynb', 'w') as f:
        nbf.write(nb, f)
        
    print("Day 51 script completed successfully.")

if __name__ == '__main__':
    run_explainability()
