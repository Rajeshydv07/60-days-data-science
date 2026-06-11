# Day 29: Customer Segmentation with K-Means

## Objective
Use K-Means Clustering to group mall customers based on purchasing behavior (Annual Income and Spending Score) to identify meaningful customer segments.

## Dataset
We use the popular **Mall Customers dataset** (`Mall_Customers.csv`):
- **Features selected:** `Annual Income (k$)`, `Spending Score (1-100)`

## Step-by-Step Summary

### Step 1-4: Prep & Scale
Libraries were imported, the dataset loaded, and features standardized using `StandardScaler` to ensure scale independence.

### Step 5: Optimal Clusters (Elbow Method)
Running K-Means for $K \in [1, 10]$ showed a clear elbow at **$K=5$** (representing the inflection point of WCSS/inertia).

### Step 6-7: Apply & Visualize K-Means
Fitted K-Means with 5 clusters and visualized the segments in a 2D scatter plot.

### Step 8: Analyze Clusters (Mean Averages)
Aggregated averages for each of the 5 segments:
- **Cluster 0 (Standard):** Middle income (~$55.3k), middle spend (~49.5 score).
- **Cluster 1 (VIP/Target):** High income (~$86.5k), high spend (~82.1 score).
- **Cluster 2 (Impetuous):** Low income (~$25.7k), high spend (~79.4 score).
- **Cluster 3 (Careful):** High income (~$88.2k), low spend (~17.1 score).
- **Cluster 4 (Sensible):** Low income (~$26.3k), low spend (~20.9 score).

## Deliverables
- [Mall_Customers.csv](Mall_Customers.csv) - The dataset file.
- [run_customer_segmentation.py](run_customer_segmentation.py) - Python runner script.
- [day29_customer_segmentation.ipynb](day29_customer_segmentation.ipynb) - Jupyter notebook showing code and output plots inline.
- [README.md](README.md) - This daily report.

## LinkedIn Reflection

**Day 29 of 60: Customer Segmentation on Mall Customers Dataset! 📊**

Today, I implemented K-Means clustering on the classic Mall Customers dataset to group customers based on two metrics: **Annual Income** and **Spending Score**.

Key takeaways:
1. **The Elbow Method works:** Plotting WCSS across different K values showed a sharp elbow at $K=5$, which is the optimal cluster size for this dataset.
2. **5 Distinct Customer Cohorts:**
   - *Standard:* Mid income, mid spend.
   - *VIP:* High income, high spend (prime marketing target).
   - *Impetuous:* Low income, high spend (often younger customers).
   - *Careful:* High income, low spend.
   - *Sensible:* Low income, low spend.
3. **Data Scaling is Vital:** Standardizing features using `StandardScaler` ensures both income (in thousands) and spending score (1-100) contribute equally to cluster boundaries.

Unsupervised segmentation allows retail businesses to personalize promotional campaigns and target high-value cohorts precisely.

On to Day 30! 🚀

#DataScience #MachineLearning #KMeans #Clustering #CustomerSegmentation #Python #ScikitLearn #60DayChallenge #ABtalksDS

---

*60 Days Data Science Challenge — Day 29/60*
