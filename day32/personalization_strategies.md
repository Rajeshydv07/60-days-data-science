# Day 32 Study Notes: Real-World Personalization & Recommendation Strategies

Today I spent time researching how companies like Amazon, Netflix, and Spotify scale recommendation systems beyond the simple User-User Collaborative Filtering model I built. In practice, calculating similarity between millions of users is too slow, and ratings data is extremely sparse. Here is my breakdown of how these issues are tackled in production.

---

## 1. Collaborative Filtering: User-User vs. Item-Item

In my code, I implemented **User-User Collaborative Filtering**. This is great for small databases, but it has distinct drawbacks compared to **Item-Item Collaborative Filtering**:

| Feature | User-User Collaborative Filtering | Item-Item Collaborative Filtering |
| :--- | :--- | :--- |
| **Basic Concept** | "Find customers similar to you and recommend what they bought." | "Find items similar to what you've interacted with in the past." |
| **Stability** | **Low**. Customer tastes and profiles change frequently as they browse different categories. | **High**. Item characteristics (genre, brand, utility) are stable over time. |
| **Computation** | Scales with the number of users ($O(N^2)$). High compute cost if customer base grows. | Scales with the number of items ($O(M^2)$). In e-commerce, customers >> items. |
| **Cold Start** | Very sensitive to new users. | Sensitive to new items. |

**Key Takeaway:** For large e-commerce platforms, **Item-Item Collaborative Filtering** is generally preferred because the item similarity matrix can be computed offline once and queried instantly.

---

## 2. Matrix Factorization & Latent Factor Models

To handle extreme sparsity (often $>99\%$ in real-world retail), we use dimensionality reduction to project customers and products into a shared **Latent Factor Space**.

### Singular Value Decomposition (SVD)
SVD decomposes the sparse user-item matrix $R$ into two lower-rank matrices:
- **User Matrix ($U$)**: Represents how much each user aligns with different latent themes (e.g., "enjoys action movies", "interested in tech items").
- **Item Matrix ($V$)**: Represents how much each item belongs to those same themes.

The predicted rating is calculated by taking the dot product of the user's vector and the item's vector:
$$\hat{R}_{u, i} = u_u \cdot v_i^T$$

### Alternating Least Squares (ALS)
SVD doesn't handle missing values well (we have to fill them or use iterative SVD). **ALS** is specifically designed for sparse matrices. It alternates between:
1. Holding item vectors constant and optimizing user vectors.
2. Holding user vectors constant and optimizing item vectors.

ALS is highly parallelizable and is the default algorithm used in Apache Spark for large-scale recommendations.

---

## 3. The Cold Start Problem & How to Fix It

A major limitation of similarity-based systems is the **Cold Start Problem**—what happens when we don't have enough rating history?

### For New Users:
1. **Onboarding Survey**: Ask users to select 3-5 categories or products they like during registration (e.g., Pinterest's topic selection).
2. **Popularity-Based Recommendations**: Show the top-trending items globally or in their geographic location.
3. **Demographic / Contextual Rules**: Recommend items popular among users of similar age, gender, or device type.

### For New Products:
1. **Content-Based Fallback**: Use product metadata (text descriptions, brand, category tags) to calculate item similarity rather than relying on purchase logs.
2. **Explore vs. Exploit (Bandits)**: Dedicate a small fraction of homepage real estate (e.g., 5%) to show new items to random users to quickly collect interaction logs.

---

## 4. Implicit Feedback vs. Explicit Feedback

In my notebook, I used **explicit feedback** (1 to 5 star ratings). In the real world, customers rarely rate things. Most data is **implicit**:
- Clicks / Views
- Add-to-cart actions
- Search queries
- Purchase history
- Scroll depth or hover time

### Handling Implicit Feedback:
Instead of predicting a star rating, the goal is to predict **probability of engagement**. We assign binary preferences ($p_{ui} = 1$ if user interacted, $0$ otherwise) paired with a **confidence score** ($c_{ui}$) based on interaction frequency:
$$c_{ui} = 1 + \alpha r_{ui}$$
where $r_{ui}$ is the interaction count (e.g., number of times they viewed a product) and $\alpha$ is a scaling factor.

---

## 5. Evaluation Metrics in Production

Offline RMSE is great for checking rating accuracy, but it doesn't tell us if recommendations actually increase sales. Real personalization engines use:

### Offline Ranking Metrics:
1. **MAP@K (Mean Average Precision at K)**: Measures the precision of recommendations, penalizing the system if relevant items are pushed down the list.
2. **NDCG (Normalized Discounted Cumulative Gain)**: Evaluates recommendations based on their positions. A hit at rank 1 is worth much more than a hit at rank 5.
3. **Coverage**: The percentage of catalog items that the recommender actually suggests. (High accuracy is bad if we only recommend the same 5 blockbusters).
4. **Novelty & Serendipity**: Recommending items that the user wouldn't easily discover themselves (avoiding bubbles).

### Online Evaluation (A/B Testing):
Ultimately, recommendation systems are tested via A/B testing, measuring business KPIs:
- **Click-Through Rate (CTR)**
- **Conversion Rate (CVR)**
- **Average Order Value (AOV)**
- **User Retention / LTV**
