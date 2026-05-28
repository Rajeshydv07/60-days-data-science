# Day 16: Movie Recommendation using K-Nearest Neighbors

Welcome to Day 16 of my **60-Day Data Science Challenge**! today marks my entry into **Phase: Distance-Based Learning**, where I transitioned from classification algorithms to designing and evaluating a **Similarity-Based Movie Recommendation System** using the **K-Nearest Neighbors (KNN)** algorithm.

Recommendation systems are among the most impactful real-world machine learning systems, driving engagement on platforms like Netflix, Spotify, and Amazon. Today, I set out to construct an item-based collaborative filtering engine using the MovieLens dataset, implement a similarity-weighted rating predictor from scratch, and evaluate how changing the hyperparameter $K$ affects prediction error on unseen test data.

---

## Workspace Directory Structure

*   [day16_movie_recommendation.ipynb](day16_movie_recommendation.ipynb): The complete Jupyter notebook containing exploratory data analysis (EDA), sparse matrix construction, Scikit-Learn `NearestNeighbors` implementation, custom rating prediction, and the $K$-value hyperparameter tuning experiment.
*   [movies.csv](movies.csv): Movie metadata containing 9,742 movies mapped to their titles and genres.
*   [ratings.csv](ratings.csv): 100,836 ratings on a 5-star scale across 610 users and 9,724 unique movies.
*   [predictions_sample.csv](predictions_sample.csv): A test partition sample (2,500 observations) showing actual user ratings paired with our similarity-weighted KNN predictions (using the optimal $K=15$).
*   [movie_rating_distribution.png](movie_rating_distribution.png): A visualization of the rating frequencies.
*   [ratings_per_user_movie.png](ratings_per_user_movie.png): A visualization showcasing the long-tail interaction rate and activity distributions.
*   [knn_rmse_k_comparison.png](knn_rmse_k_comparison.png): An evaluation curve displaying Root Mean Squared Error (RMSE) against neighborhood sizes ($K$).

---

## Theoretical Framework: Item-Based Collaborative Filtering

In similarity-based systems, we have two primary paradigms:
1.  **User-Based Collaborative Filtering**: Predicts a user's preference by finding similar users. However, user preferences are dynamic, complex, and scale poorly as the user base grows.
2.  **Item-Based Collaborative Filtering**: Predicts preferences by calculating similarities between items based on user rating patterns. This approach is significantly more stable (a movie's inherent genre and characteristics do not change, and collective tastes evolve slowly) and computationally efficient.

### Mathematical Formulation
To represent movies mathematically, we pivot our ratings log into a **User-Item Matrix** of shape $M \times N$ (where $M = 610$ users and $N = 9,724$ movies). Each movie is represented as an $M$-dimensional vector of user ratings:

$$\vec{i} = [r_{1,i}, r_{2,i}, \dots, r_{M,i}]^T$$

Where $r_{u,i} = 0$ if user $u$ has not rated movie $i$. To compute the similarity between two movies, we calculate their **Cosine Similarity**:

$$\text{Cosine Similarity}(\vec{a}, \vec{b}) = \frac{\vec{a} \cdot \vec{b}}{\|\vec{a}\| \|\vec{b}\|} = \frac{\sum_{u=1}^{M} r_{u,a} r_{u,b}}{\sqrt{\sum_{u=1}^{M} r_{u,a}^2} \sqrt{\sum_{u=1}^{M} r_{u,b}^2}}$$

This metric measures the cosine of the angle between two movie vectors in the $M$-dimensional space. It naturally normalizes for differences in rating scales across users, focusing entirely on the direction of taste.

---

## Data Sparsity & The "Long-Tail" Distribution

Before building the model, I performed exploratory analysis on the interaction patterns:
1.  **Distribution Skew (`movie_rating_distribution.png`)**:
    *   The distribution is heavily left-skewed, peaking at 4.0. Users are naturally self-selecting; they rarely rate movies they expect to dislike, creating a dataset dominated by positive interactions.
2.  **The Long-Tail Effect (`ratings_per_user_movie.png`)**:
    *   Plotting the interaction counts per user and per movie revealed an extreme power-law (long-tail) distribution.
    *   A minute fraction of blockbusters and active users account for the vast majority of ratings, while thousands of niche movies receive fewer than 5 ratings.
    *   Pivoting this interaction log into a User-Item Matrix resulted in a matrix that is **98.30% sparse**. Managing this high sparsity is the primary challenge in modern recommendation systems.

---

## Model Implementation & Recommendations Demo

I trained Scikit-Learn's unsupervised `NearestNeighbors` model on the movie-user vectors (the transpose of the pivoted user-item matrix) utilizing a brute-force search with the cosine metric:

```python
from sklearn.neighbors import NearestNeighbors
knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=15)
knn.fit(user_item_matrix.T)
```

Running queries for iconic titles yielded highly intuitive recommendations:

*   **Toy Story (1995)** $\rightarrow$ *Toy Story 2 (1999)* (Similarity: 0.5726), *Jurassic Park (1993)* (Similarity: 0.5656), *Independence Day (1996)* (Similarity: 0.5643).
*   **The Matrix (1999)** $\rightarrow$ *Fight Club (1999)* (Similarity: 0.7139), *Star Wars: Episode V - The Empire Strikes Back (1980)* (Similarity: 0.7009), *Saving Private Ryan (1998)* (Similarity: 0.6796).
*   **Fight Club (1999)** $\rightarrow$ *The Matrix (1999)* (Similarity: 0.7139), *Memento (2000)* (Similarity: 0.6696), *American History X (1998)* (Similarity: 0.6491).

---

## Hyperparameter Tuning: K-Value Comparison Analysis

To rigorously evaluate how changing the hyperparameter $K$ (number of neighbors) impacts recommendation quality, I split the 100k ratings into an **80/20 train-test split**.

For a test set partition of user-movie pairs, I predicted the rating $\hat{R}_{u,i}$ that user $u$ would give to movie $i$. The prediction was formulated as a **similarity-weighted average** of the user's ratings for the $K$ most similar items that the user had rated in the training partition:

$$\hat{R}_{u,i} = \frac{\sum_{j \in \mathcal{N}_K(i; I_u)} \text{Sim}(i, j) \cdot R_{u,j}}{\sum_{j \in \mathcal{N}_K(i; I_u)} |\text{Sim}(i, j)|}$$

Where $\mathcal{N}_K(i; I_u)$ represents the set of $K$ nearest movies to item $i$ that user $u$ has rated, and $R_{u,j}$ is their actual rating. If no similar items are available, the algorithm falls back to the item's average training rating, the user's average rating, or the global training mean.

The performance across different values of $K$ on a test sample partition of 2,500 observations yielded the following results:

| Neighbors ($K$) | Test Set RMSE | Analytical Assessment |
| :---: | :---: | :--- |
| $K = 5$ | **0.8808** | **High Variance / Under-smoothed**: Highly sensitive to individual rating anomalies. Predictions are volatile. |
| $K = 10$ | **0.8560** | **Improving Accuracy**: Neighborhood aggregation reduces local noise. |
| $K = 15$ | **0.8539** | **Optimal Sweet Spot**: Minimizes generalization error (RMSE) on unseen test data. |
| $K = 20$ | **0.8590** | **Mild Neighborhood Dilution**: Starts incorporating items with weaker similarity. |
| $K = 30$ | **0.8632** | **Increasing Bias**: Prediction starts to smooth out individual user tastes. |
| $K = 40$ | **0.8688** | **Over-smoothed**: Pulls predictions heavily toward the user or global means. |
| $K = 50$ | **0.8722** | **High Bias**: Recommendations lose their distinct item-specific characteristics. |

### Rationale for Optimal $K=15$
This curve (`knn_rmse_k_comparison.png`) demonstrates the classical **Bias-Variance Trade-off**:
*   **At low $K$ (e.g., $K = 5$)**: The prediction relies on a very small neighborhood. If a user rated one of those few neighbors anomalously, the prediction is heavily distorted. The model suffers from **high variance**.
*   **At high $K$ (e.g., $K = 50$)**: Because the User-Item matrix is 98.30% sparse, users have rated very few items. To satisfy a large neighborhood constraint, the model must include items that have extremely low similarity to the target movie. This dilutes the prediction, pulling it toward the global average and causing **high bias**.
*   **The Sweet Spot ($K = 15$)**: Large enough to filter out random rating noise and individual bias, yet small enough to restrict calculations to genuinely related movies.

---

## LinkedIn Reflection

Here is my professional reflection and daily learning summary for LinkedIn:

**Post**:
> Day 16 of my 60-Day Data Science Challenge! 📈 Today, I entered the **Distance-Based Learning** phase by building and evaluating an end-to-end **Similarity-Based Movie Recommendation System using K-Nearest Neighbors (KNN)**!
> 
> Recommendation engines are the silent drivers of conversion and retention on platforms like Netflix, Spotify, and Amazon. Moving past standard classification, I wanted to understand the mathematical mechanics behind these systems.
> 
> Key technical takeaways from today's work:
> 
> 🕸️ **1. Sparsity & The Long-Tail Trap:**
> Working with the MovieLens 100k dataset (9,742 movies, 610 users), pivoting the interaction logs into a User-Item Matrix reveals that it is **98.30% sparse**. The data also exhibits an extreme power-law (long-tail) distribution. Modern collaborative filtering relies on robust similarity metrics like Cosine Similarity to project movies into an M-dimensional space and find meaningful relationships despite this sparsity.
> 
> 🍿 **2. Item-Based vs. User-Based Collaborative Filtering:**
> I implemented an **Item-Based** system. Because a movie's features (genres, themes) are static, collective user rating patterns for movies are far more stable over time than individual user behaviors. This makes Item-Based filtering computationally efficient and highly resilient.
> 
> 📐 **3. Quantitative Evaluation (Hyperparameter Tuning):**
> Instead of treating recommendation as an unsupervised task, I formulated it as a rating prediction problem on an 80/20 train-test split. Using a custom-built **Similarity-Weighted KNN Rating Predictor** in Python, I tracked Root Mean Squared Error (RMSE) across various values of $K$:
> - **Under-smoothing (Small K = 5, RMSE: 0.8808):** High Variance. The neighborhood is too small, making predictions overly sensitive to outlier ratings.
> - **Over-smoothing (Large K = 50, RMSE: 0.8722):** High Bias. The model is forced to pull in unrelated, distant movies to fill the neighborhood, diluting personalization and pulling predictions toward the global mean.
> - **The Sweet Spot (Optimal K = 15, RMSE: 0.8539):** Achieved the perfect balance—large enough to cancel out random noise, small enough to remain highly personalized.
> 
> 💡 **4. Intuitive Recommendations:**
> Validating the Nearest Neighbors model on iconic titles returned highly logical recommendations:
> - Querying *Toy Story* suggested *Toy Story 2*, *Jurassic Park*, and *Independence Day*.
> - Querying *The Matrix* suggested *Fight Club*, *The Empire Strikes Back*, and *Saving Private Ryan*.
> 
> Transitioning mathematical concepts like high-dimensional vectors and cosine angles into real-world predictions that make intuitive sense is what makes data science so rewarding!
> 
> #DataScience #MachineLearning #Python #ScikitLearn #RecommendationSystems #CollaborativeFiltering #KNN #60DayChallenge #ABtalksDS
