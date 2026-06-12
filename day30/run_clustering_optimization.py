import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
import subprocess
import sys
import os

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

os.makedirs('day30', exist_ok=True)

def md(src):
    return new_markdown_cell(src)

def code(src):
    return new_code_cell(src)

cells = []

# Introduction
cells.append(md("""# Day 30: Finding the Ideal Number of Customer Segments

## Objective
Apply the Elbow Method and Silhouette Score to determine the optimal number of clusters for customer segmentation.
"""))

# Step 1: Imports
cells.append(md("## step 1 - imports"))
cells.append(code("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

print("pandas:", pd.__version__)
print("numpy:", np.__version__)
"""))

# Step 2: Load Dataset
cells.append(md("## step 2 - load dataset"))
cells.append(code("""df = pd.read_csv("Mall_Customers.csv")
print("Dataset shape:", df.shape)
print(df.head())
"""))

# Step 3: Select Features
cells.append(md("## step 3 - select features"))
cells.append(code("""# we use annual income and spending score for clustering
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
"""))

# Step 4: Scale Data
cells.append(md("## step 4 - standard scaling"))
cells.append(code("""scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
"""))

# Step 5: Run K-Means for various K
cells.append(md("## step 5 - run k-means & calculate metrics for multiple K values"))
cells.append(code("""wcss = []
silhouette_scores = []
k_values = range(2, 11)

# we run k=1 to 10 for elbow, but silhouette score needs at least 2 clusters
wcss_k1_10 = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss_k1_10.append(kmeans.inertia_)
    if k >= 2:
        wcss.append(kmeans.inertia_)
        labels = kmeans.labels_
        score = silhouette_score(X_scaled, labels)
        silhouette_scores.append(score)

print("Metrics calculated successfully!")
"""))

# Step 6: Plot Elbow and Silhouette curves side-by-side
cells.append(md("## step 6 - visualize elbow and silhouette curves"))
cells.append(code("""fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# elbow method plot
axes[0].plot(range(1, 11), wcss_k1_10, marker='o', color='royalblue', lw=2)
axes[0].axvline(5, color='red', linestyle='--', label='Elbow Point (K=5)')
axes[0].set_title('Elbow Method (WCSS vs K)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Number of Clusters (K)')
axes[0].set_ylabel('Within-Cluster Sum of Squares (WCSS)')
axes[0].legend()
axes[0].grid(alpha=0.3)

# silhouette score plot
axes[1].plot(range(2, 11), silhouette_scores, marker='s', color='forestgreen', lw=2)
axes[1].axvline(5, color='red', linestyle='--', label='Optimal Peak (K=5)')
axes[1].set_title('Silhouette Scores vs K', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Number of Clusters (K)')
axes[1].set_ylabel('Silhouette Score')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.suptitle('Determining Optimal Clusters (K)', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('clustering_optimization_curves.png', bbox_inches='tight', dpi=150)
plt.show()
"""))

# Step 7: Compare clustering quality across K=3, 5, 7
cells.append(md("## step 7 - compare clustering quality across different K values"))
cells.append(code("""compare_k = [3, 5, 7]
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, k in enumerate(compare_k):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    
    # plot
    scatter = axes[i].scatter(
        df['Annual Income (k$)'],
        df['Spending Score (1-100)'],
        c=labels,
        cmap='viridis',
        s=60,
        alpha=0.8
    )
    axes[i].set_title(f'K = {k} Clusters (Silhouette: {silhouette_score(X_scaled, labels):.3f})', 
                      fontsize=12, fontweight='bold')
    axes[i].set_xlabel('Annual Income')
    axes[i].set_ylabel('Spending Score')
    axes[i].grid(alpha=0.2)

plt.suptitle('Visual Comparison of Clustering Quality', fontsize=16, fontweight='bold', y=1.05)
plt.tight_layout()
plt.savefig('clustering_comparison_k.png', bbox_inches='tight', dpi=150)
plt.show()
"""))

# Step 8: Run optimal K-Means
cells.append(md("## step 8 - apply optimal K-Means (K=5)"))
cells.append(code("""optimal_k = 5
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

print("Distribution of customers across clusters:")
print(df['Cluster'].value_counts())
"""))

# Step 9: Analyze Cluster Profiles
cells.append(md("## step 9 - analyze cluster averages and document strategy"))
cells.append(code("""cluster_profile = df.groupby('Cluster')[['Annual Income (k$)', 'Spending Score (1-100)']].mean()
cluster_profile['Customer Count'] = df['Cluster'].value_counts()
print("Cluster Profiles (Mean values):")
print(cluster_profile.round(2))
"""))

cells.append(md("""### business segmentation strategy based on optimal K=5:
- **Cluster 0:** Standard customer. Mid income, mid spend. Good baseline.
- **Cluster 1:** VIP/Target customer. High income, high spend. Focus area for premium loyalty programs.
- **Cluster 2:** Impetuous customer. Low income, high spend. Target with trendy, budget-friendly impulse offers.
- **Cluster 3:** Careful customer. High income, low spend. Target with value-driven campaigns or exclusive product features.
- **Cluster 4:** Sensible customer. Low income, low spend. Keep marketing costs low, focus on value/necessities.
"""))

# Save and run
nb = new_notebook(cells=cells)
nb.metadata['kernelspec'] = {
    'display_name': 'Python 3',
    'language': 'python',
    'name': 'python3'
}
nb.metadata['language_info'] = {'name': 'python', 'version': '3.10.0'}

NB_PATH = 'day30/day30_clustering_optimization.ipynb'
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
