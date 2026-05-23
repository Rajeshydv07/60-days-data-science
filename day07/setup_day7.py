import os
import pandas as pd
import numpy as np
import nbformat as nbf

# 1. Generate a synthetic dataset for EDA
np.random.seed(42)
n_rows = 500
data = {
    'Transaction_ID': range(1, n_rows + 1),
    'Age': np.random.normal(loc=35, scale=12, size=n_rows).astype(int),
    'Income': np.random.normal(loc=60000, scale=15000, size=n_rows),
    'Spending_Score': np.random.normal(loc=50, scale=20, size=n_rows),
    'Category': np.random.choice(['Electronics', 'Clothing', 'Groceries', 'Home'], size=n_rows)
}

# Inject some outliers
data['Income'][10] = 250000
data['Income'][100] = 300000
data['Age'][50] = 110
data['Spending_Score'][200] = 150

df = pd.DataFrame(data)
# Clip realistic boundaries
df['Age'] = df['Age'].clip(lower=18)
df['Income'] = df['Income'].clip(lower=20000)
df['Spending_Score'] = df['Spending_Score'].clip(lower=1)
df.to_csv('day7_dataset.csv', index=False)

# 2. Create Jupyter Notebook for EDA
nb = nbf.v4.new_notebook()

cells = [
    nbf.v4.new_markdown_cell("# Day 7: Exploratory Data Analysis (EDA)\nUnderstanding patterns and outliers before modeling."),
    
    nbf.v4.new_code_cell("import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\n\n# Set plot style\nsns.set_theme(style='whitegrid')"),
    
    nbf.v4.new_markdown_cell("### 1. Load Data"),
    nbf.v4.new_code_cell("df = pd.read_csv('day7_dataset.csv')\ndf.head()"),
    
    nbf.v4.new_code_cell("df.describe()"),
    
    nbf.v4.new_markdown_cell("### 2. Plot Distributions\nVisualizing distributions to understand the spread and central tendency of our data."),
    nbf.v4.new_code_cell("plt.figure(figsize=(15, 5))\n\nplt.subplot(1, 3, 1)\nsns.histplot(df['Age'], bins=20, kde=True)\nplt.title('Age Distribution')\n\nplt.subplot(1, 3, 2)\nsns.histplot(df['Income'], bins=20, kde=True)\nplt.title('Income Distribution')\n\nplt.subplot(1, 3, 3)\nsns.histplot(df['Spending_Score'], bins=20, kde=True)\nplt.title('Spending Score Distribution')\n\nplt.tight_layout()\nplt.show()"),
    
    nbf.v4.new_markdown_cell("### 3. Identify Patterns and Outliers\nUsing boxplots to spot outliers and scatter plots for relationships."),
    nbf.v4.new_code_cell("plt.figure(figsize=(15, 5))\n\nplt.subplot(1, 3, 1)\nsns.boxplot(y=df['Age'])\nplt.title('Age Boxplot')\n\nplt.subplot(1, 3, 2)\nsns.boxplot(y=df['Income'])\nplt.title('Income Boxplot')\n\nplt.subplot(1, 3, 3)\nsns.boxplot(y=df['Spending_Score'])\nplt.title('Spending Score Boxplot')\n\nplt.tight_layout()\nplt.show()"),
    
    nbf.v4.new_code_cell("plt.figure(figsize=(8, 6))\nsns.scatterplot(data=df, x='Income', y='Spending_Score', hue='Category')\nplt.title('Income vs Spending Score by Category')\nplt.show()"),
    
    nbf.v4.new_markdown_cell("### 4. Top 5 Insights\n\n1. **Age Distribution:** The age of customers is normally distributed, primarily centered around 35 years old, though there is one extreme outlier at 110 years old that requires cleaning.\n2. **Income Outliers:** Income is clustered around $60,000, but boxplots clearly show a few high-income outliers (>$200,000) that could skew mean-based models.\n3. **Spending Score Range:** The spending score has a wide spread, mostly between 30 and 70, with at least one anomalous score exceeding 140.\n4. **Category Popularity:** The scatter plot indicates an even distribution of customer preferences across Electronics, Clothing, Groceries, and Home categories.\n5. **Lack of Strong Linear Correlation:** There is no immediate visual linear correlation between Income and Spending Score, suggesting customer spending is driven by factors other than just their income level.")
]

nb['cells'] = cells

with open('day7_eda.ipynb', 'w') as f:
    nbf.write(nb, f)

print('Setup complete.')
