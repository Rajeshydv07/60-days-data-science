import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
import subprocess
import sys
import os

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

os.makedirs('day29', exist_ok=True)

def md(src):
    return new_markdown_cell(src)

def code(src):
    return new_code_cell(src)

cells = []

cells.append(md("""# Day 29: Customer Segmentation with K-Means

## Objective
Use K-Means Clustering to group customers based on purchasing behavior and identify meaningful customer segments.
"""))

cells.append(md("""### Step 1: Import Libraries"""))

cells.append(code("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
"""))

cells.append(md("""### Step 2: Load Dataset
We use the popular Mall Customers dataset.
"""))

cells.append(code("""df = pd.read_csv("Mall_Customers.csv")
print(df.head())
"""))

cells.append(md("""### Step 3: Select Features"""))

cells.append(code("""X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
"""))

cells.append(md("""### Step 4: Scale Data"""))

cells.append(code("""scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
"""))

cells.append(md("""### Step 5: Find Optimal Clusters (Elbow Method)"""))

cells.append(code("""wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()
"""))

cells.append(md("""### Step 6: Apply K-Means"""))

cells.append(code("""kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

df['Cluster'] = clusters
"""))

cells.append(md("""### Step 7: Visualize Customer Segments"""))

cells.append(code("""plt.figure(figsize=(10, 6))

plt.scatter(
    df['Annual Income (k$)'],
    df['Spending Score (1-100)'],
    c=df['Cluster'],
    cmap='viridis',
    s=80
)

plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.title('Customer Segmentation using K-Means')
plt.colorbar(label='Cluster')
plt.show()
"""))

cells.append(md("""### Step 8: Analyze Clusters"""))

cells.append(code("""print(df.groupby('Cluster').mean(numeric_only=True))
"""))

nb = new_notebook(cells=cells)
nb.metadata['kernelspec'] = {
    'display_name': 'Python 3',
    'language': 'python',
    'name': 'python3'
}
nb.metadata['language_info'] = {'name': 'python', 'version': '3.10.0'}

NB_PATH = 'day29/day29_customer_segmentation.ipynb'
with open(NB_PATH, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print(f"Notebook saved: {NB_PATH}")
print("Executing notebook via nbconvert...")

result = subprocess.run(
    [sys.executable, '-m', 'nbconvert', '--to', 'notebook',
     '--execute', '--inplace',
     '--ExecutePreprocessor.timeout=900',
     '--ExecutePreprocessor.kernel_name=python3',
     NB_PATH],
    capture_output=True, text=True
)

if result.returncode == 0:
    print("Success! Notebook executed successfully.")
else:
    print("Notebook execution failed:")
    print(result.stderr[-4000:])
    sys.exit(result.returncode)
