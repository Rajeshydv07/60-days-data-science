import nbformat as nbf
import os

def create_notebook():
    nb = nbf.v4.new_notebook()

    nb.cells = [
        nbf.v4.new_markdown_cell("# Day 41: Predicting High-Risk Customers Before They Churn\n\nIn this notebook, we combine customer behavior, segmentation, and churn signals to build a predictive risk scoring model. We will rank customers by churn probability and visualize high-risk groups."),
        
        nbf.v4.new_markdown_cell("## 1. Import Libraries and Load Data"),
        nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

# Set plot style
sns.set_theme(style='whitegrid')

# Load the generated dataset
df = pd.read_csv('../data/churn_data.csv')
display(df.head())
"""),
        
        nbf.v4.new_markdown_cell("## 2. Exploratory Data Analysis (EDA)"),
        nbf.v4.new_code_cell("""# Check basic stats
display(df.describe())

# Check churn distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='churn', data=df)
plt.title('Churn Distribution')
plt.show()

# Churn vs Contract Type
plt.figure(figsize=(8, 5))
sns.countplot(x='contract_type', hue='churn', data=df)
plt.title('Churn by Contract Type')
plt.show()

# Satisfaction Score vs Churn
plt.figure(figsize=(8, 5))
sns.countplot(x='satisfaction_score', hue='churn', data=df)
plt.title('Churn by Satisfaction Score')
plt.show()
"""),
        
        nbf.v4.new_markdown_cell("## 3. Data Preprocessing & Feature Engineering"),
        nbf.v4.new_code_cell("""# Encode categorical features
le = LabelEncoder()
df['contract_type_encoded'] = le.fit_transform(df['contract_type'])

# Select features and target
features = [
    'age', 'tenure_months', 'monthly_charges', 'total_charges', 
    'contract_type_encoded', 'support_tickets', 
    'days_since_last_login', 'satisfaction_score'
]
X = df[features]
y = df['churn']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
"""),
        
        nbf.v4.new_markdown_cell("## 4. Model Training (Random Forest)"),
        nbf.v4.new_code_cell("""# Initialize and train Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_model.fit(X_train_scaled, y_train)

# Predictions
y_pred = rf_model.predict(X_test_scaled)
y_prob = rf_model.predict_proba(X_test_scaled)[:, 1]

# Evaluate
print("Classification Report:")
print(classification_report(y_test, y_pred))

print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")
"""),
        
        nbf.v4.new_markdown_cell("## 5. Feature Importance"),
        nbf.v4.new_code_cell("""feature_importances = pd.DataFrame({
    'Feature': features,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importances)
plt.title('Feature Importance for Churn Prediction')
plt.show()
"""),
        
        nbf.v4.new_markdown_cell("## 6. Risk Scoring & Ranking"),
        nbf.v4.new_code_cell("""# Apply model to entire dataset to generate risk scores
X_scaled_all = scaler.transform(X)
df['churn_risk_score'] = rf_model.predict_proba(X_scaled_all)[:, 1]

# Segment into risk groups
df['risk_group'] = pd.cut(df['churn_risk_score'], bins=[-0.01, 0.25, 0.5, 0.75, 1.01], labels=['Low', 'Medium', 'High', 'Critical'])

# View top 10 most critical customers
high_risk_customers = df.sort_values(by='churn_risk_score', ascending=False)
display(high_risk_customers[['customer_id', 'contract_type', 'satisfaction_score', 'churn_risk_score', 'risk_group']].head(10))

# Save the augmented dataset
df.to_csv('../data/churn_scored.csv', index=False)
print("Saved scored customers to ../data/churn_scored.csv")
""")
    ]

    os.makedirs('../notebooks', exist_ok=True)
    with open('../notebooks/churn_prediction.ipynb', 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print("Notebook created successfully.")

if __name__ == "__main__":
    create_notebook()
