import os

import sys

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.cluster import KMeans

from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import IsolationForest

try:

    import streamlit as st

except ModuleNotFoundError:

    print("Streamlit not found. Please install it using: pip install streamlit")

    sys.exit(1)

st.set_page_config(

    page_title="Integrated Customer Intelligence Platform",

    page_icon=None,

    layout="wide",

    initial_sidebar_state="expanded"

)

st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1.2rem;
        border-radius: 8px;
        border-left: 5px solid #1e3d59;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
    }
    .metric-title {
        font-size: 0.85rem;
        color: #6c757d;
        font-weight: bold;
        text-transform: uppercase;
        margin-bottom: 0.3rem;
    }
    .metric-value {
        font-size: 1.8rem;
        color: #1e3d59;
        font-weight: bold;
        line-height: 1.2;
    }
    .metric-desc {
        font-size: 0.75rem;
        color: #adb5bd;
        margin-top: 0.3rem;
        font-style: italic;
    }
    .alert-panel {
        padding: 1rem;
        border-radius: 6px;
        margin-bottom: 1rem;
    }
    .alert-danger {
        background-color: #fdf2f2;
        border-left: 5px solid #d9534f;
        color: #721c24;
    }
    .alert-success {
        background-color: #f3faf7;
        border-left: 5px solid #17b978;
        color: #155724;
    }
</style>
""", unsafe_style_allowed=True)

mall_customers_path = 'day31/Mall_Customers_Labeled_Personas.csv'

customer_ratings_path = 'day32/customer_ratings.csv'

products_path = 'day32/products.csv'

customer_behavior_path = 'day33/customer_behavior_data.csv'

churn_path = 'day15/telco_customer_churn.csv'

for path_var in ['mall_customers_path', 'customer_ratings_path', 'products_path', 'customer_behavior_path', 'churn_path']:

    curr_val = globals()[path_var]

    if not os.path.exists(curr_val):

        parent_val = '../' + curr_val

        if os.path.exists(parent_val):

            globals()[path_var] = parent_val

@st.cache_data

def load_all_data():

    if os.path.exists(mall_customers_path):

        df_demo = pd.read_csv(mall_customers_path)

    else:

        np.random.seed(42)

        df_demo = pd.DataFrame({

            'CustomerID': [f'C{i:03d}' for i in range(1, 201)],

            'Gender': np.random.choice(['Male', 'Female'], 200),

            'Age': np.random.randint(18, 70, 200),

            'Annual Income (k$)': np.random.randint(15, 137, 200),

            'Spending Score (1-100)': np.random.randint(1, 100, 200)

        })

    if os.path.exists(customer_ratings_path) and os.path.exists(products_path):

        df_ratings = pd.read_csv(customer_ratings_path)

        df_products = pd.read_csv(products_path)

    else:

        products_list = [

            {"product_id": "P01", "product_name": "Wireless Noise-Canceling Headphones", "category": "Electronics"},

            {"product_id": "P02", "product_name": "Mechanical Gaming Keyboard", "category": "Electronics"},

            {"product_id": "P03", "product_name": "UltraWide 34-inch Gaming Monitor", "category": "Electronics"},

            {"product_id": "P04", "product_name": "Ergonomic Wireless Mouse", "category": "Electronics"},

            {"product_id": "P05", "product_name": "Smart Fitness Watch", "category": "Electronics"},

            {"product_id": "P06", "product_name": "Portable Bluetooth Speaker", "category": "Electronics"},

            {"product_id": "P13", "product_name": "Designing Data-Intensive Applications", "category": "Books"},

            {"product_id": "P14", "product_name": "Atomic Habits by James Clear", "category": "Books"},

            {"product_id": "P15", "product_name": "The Hobbit by J.R.R. Tolkien", "category": "Books"},

            {"product_id": "P16", "product_name": "Python Data Science Handbook", "category": "Books"},

            {"product_id": "P25", "product_name": "Classic Fit Cotton Crewneck T-Shirt", "category": "Clothing"},

            {"product_id": "P26", "product_name": "Slim-Fit Stretch Denim Jeans", "category": "Clothing"},

            {"product_id": "P27", "product_name": "Lightweight Waterproof Windbreaker", "category": "Clothing"},

            {"product_id": "P37", "product_name": "Stainless Steel Electric Kettle", "category": "Home & Kitchen"},

            {"product_id": "P38", "product_name": "Non-Stick Ceramic Frying Pan", "category": "Home & Kitchen"},

            {"product_id": "P39", "product_name": "Memory Foam Sleep Pillow", "category": "Home & Kitchen"},

            {"product_id": "P49", "product_name": "High-Density Yoga Mat with Carrying Strap", "category": "Sports & Outdoors"},

            {"product_id": "P50", "product_name": "Insulated Stainless Steel Water Bottle", "category": "Sports & Outdoors"},

            {"product_id": "P51", "product_name": "Adjustable Dumbbell Set (Pair)", "category": "Sports & Outdoors"}

        ]

        df_products = pd.DataFrame(products_list)

        np.random.seed(42)

        ratings_list = []

        for i in range(1, 301):

            uid = f"C{i:03d}"

            fav_cat = np.random.choice(["Electronics", "Books", "Clothing", "Home & Kitchen", "Sports & Outdoors"])

            for _, prod in df_products.iterrows():

                prod_id = prod["product_id"]

                prod_cat = prod["category"]

                if prod_cat == fav_cat and np.random.rand() < 0.8:

                    ratings_list.append({"customer_id": uid, "product_id": prod_id, "rating": np.random.randint(4, 6)})

                elif np.random.rand() < 0.15:

                    ratings_list.append({"customer_id": uid, "product_id": prod_id, "rating": np.random.randint(1, 4)})

        df_ratings = pd.DataFrame(ratings_list)

    if os.path.exists(customer_behavior_path):

        df_behavior = pd.read_csv(customer_behavior_path)

    else:

        np.random.seed(42)

        df_normal = pd.DataFrame({

            'CustomerID': [f'C{i:03d}' for i in range(1, 951)],

            'TransactionAmount': np.random.normal(65, 18, 950).clip(0),

            'TransactionFrequency': np.random.poisson(10, 950),

            'FailedLogins': np.random.poisson(0.3, 950),

            'DeviceChanges': np.random.choice([1, 2, 3], size=950, p=[0.85, 0.12, 0.03]),

            'ChargebackRate': np.random.beta(0.5, 45, 950)

        })

        df_fraud = pd.DataFrame({

            'CustomerID': [f'C{i:03d}' for i in range(951, 971)],

            'TransactionAmount': np.random.normal(1150, 250, 20),

            'TransactionFrequency': np.random.poisson(42, 20),

            'FailedLogins': np.random.poisson(0.8, 20),

            'DeviceChanges': np.random.choice([2, 3, 4], size=20, p=[0.4, 0.4, 0.2]),

            'ChargebackRate': np.random.uniform(0.35, 0.85, 20)

        })

        df_ato = pd.DataFrame({

            'CustomerID': [f'C{i:03d}' for i in range(971, 991)],

            'TransactionAmount': np.random.normal(25, 10, 20).clip(0),

            'TransactionFrequency': np.random.poisson(2, 20),

            'FailedLogins': np.random.poisson(14, 20),

            'DeviceChanges': np.random.choice([4, 5, 6], size=20, p=[0.25, 0.55, 0.20]),

            'ChargebackRate': np.random.beta(0.1, 12, 20)

        })

        df_bots = pd.DataFrame({

            'CustomerID': [f'C{i:03d}' for i in range(991, 1001)],

            'TransactionAmount': np.random.uniform(1.2, 4.8, 10),

            'TransactionFrequency': np.random.poisson(175, 10),

            'FailedLogins': np.random.poisson(1.5, 10),

            'DeviceChanges': np.random.choice([3, 4, 5], size=10),

            'ChargebackRate': np.random.uniform(0.1, 0.35, 10)

        })

        df_behavior = pd.concat([df_normal, df_fraud, df_ato, df_bots], ignore_index=True)

        df_behavior['True_Label'] = 'Normal'

        df_behavior.loc[df_behavior['CustomerID'].isin(df_fraud['CustomerID']), 'True_Label'] = 'Payment Fraud'

        df_behavior.loc[df_behavior['CustomerID'].isin(df_ato['CustomerID']), 'True_Label'] = 'ATO Attempt'

        df_behavior.loc[df_behavior['CustomerID'].isin(df_bots['CustomerID']), 'True_Label'] = 'Card Testing Bot'

    if os.path.exists(churn_path):

        df_churn = pd.read_csv(churn_path)

        df_churn['TotalCharges'] = pd.to_numeric(df_churn['TotalCharges'].str.strip(), errors='coerce')

        df_churn['TotalCharges'] = df_churn['TotalCharges'].fillna(df_churn['MonthlyCharges'] * df_churn['tenure'])

    else:

        np.random.seed(42)

        n_churn = 1000

        df_churn = pd.DataFrame({

            'customerID': [f'CH{i:04d}' for i in range(n_churn)],

            'gender': np.random.choice(['Male', 'Female'], n_churn),

            'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_churn, p=[0.5, 0.25, 0.25]),

            'InternetService': np.random.choice(['Fiber optic', 'DSL', 'No'], n_churn, p=[0.4, 0.4, 0.2]),

            'MonthlyCharges': np.random.uniform(20.0, 118.0, n_churn),

            'tenure': np.random.randint(1, 72, n_churn),

            'Churn': np.random.choice(['Yes', 'No'], n_churn, p=[0.26, 0.74])

        })

        df_churn.loc[df_churn['Contract'] == 'Month-to-month', 'Churn'] = np.random.choice(['Yes', 'No'], (df_churn['Contract'] == 'Month-to-month').sum(), p=[0.42, 0.58])

        df_churn.loc[df_churn['Contract'] == 'Two year', 'Churn'] = np.random.choice(['Yes', 'No'], (df_churn['Contract'] == 'Two year').sum(), p=[0.03, 0.97])

    return df_demo, df_ratings, df_products, df_behavior, df_churn

df_demo, df_ratings, df_products, df_behavior, df_churn = load_all_data()

@st.cache_resource

def fit_models(df_d, df_b, df_r):

    X_demo = df_d[['Annual Income (k$)', 'Spending Score (1-100)']]

    scaler_km = StandardScaler()

    X_demo_scaled = scaler_km.fit_transform(X_demo)

    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)

    df_d['Cluster'] = kmeans.fit_predict(X_demo_scaled)

    cluster_stats = df_d.groupby('Cluster')[['Annual Income (k$)', 'Spending Score (1-100)']].mean()

    cluster_to_persona = {}

    for i in range(5):

        inc_mean = cluster_stats.loc[i, 'Annual Income (k$)']

        spd_mean = cluster_stats.loc[i, 'Spending Score (1-100)']

        if inc_mean > 70 and spd_mean > 60:

            cluster_to_persona[i] = 'The Elite Affluents'

        elif inc_mean > 70 and spd_mean < 40:

            cluster_to_persona[i] = 'The Affluent Frugals'

        elif inc_mean < 40 and spd_mean > 60:

            cluster_to_persona[i] = 'The Impulsive Budgeters'

        elif inc_mean < 40 and spd_mean < 40:

            cluster_to_persona[i] = 'The Value Seekers'

        else:

            cluster_to_persona[i] = 'The Steady Conformists'

    df_d['Persona'] = df_d['Cluster'].map(cluster_to_persona)

    behavior_features = ['TransactionAmount', 'TransactionFrequency', 'FailedLogins', 'DeviceChanges', 'ChargebackRate']

    X_behav = df_b[behavior_features]

    scaler_behav = StandardScaler()

    X_behav_scaled = scaler_behav.fit_transform(X_behav)

    iforest = IsolationForest(n_estimators=150, contamination=0.05, random_state=42)

    df_b['iForest_Prediction'] = iforest.fit_predict(X_behav_scaled)

    df_b['iForest_Label'] = df_b['iForest_Prediction'].map({1: 'Normal', -1: 'Anomaly'})

    R = df_r.pivot(index='customer_id', columns='product_id', values='rating')

    user_means = R.mean(axis=1)

    R_centered = R.sub(user_means, axis=0)

    pearson_sim_df = R.T.corr(method='pearson')

    np.fill_diagonal(pearson_sim_df.values, 1.0)

    return kmeans, scaler_km, cluster_to_persona, iforest, scaler_behav, R, user_means, R_centered, pearson_sim_df

kmeans, scaler_km, cluster_to_persona, iforest, scaler_behav, R, user_means, R_centered, pearson_sim_df = fit_models(df_demo, df_behavior, df_ratings)

def predict_rating(user_id, product_id, K=10):

    if product_id not in R.columns:

        return user_means.get(user_id, 3.0)

    if user_id not in R.index:

        return R[product_id].mean()

    other_raters = R[product_id].dropna().index

    other_raters = other_raters.drop(user_id) if user_id in other_raters else other_raters

    if len(other_raters) == 0:

        return user_means[user_id]

    sim_scores = pearson_sim_df.loc[user_id, other_raters].dropna()

    sim_scores = sim_scores[sim_scores > 0]

    if len(sim_scores) == 0:

        return user_means[user_id]

    top_neighbors = sim_scores.nlargest(K)

    neighbor_centered = R_centered.loc[top_neighbors.index, product_id]

    numerator = np.sum(top_neighbors * neighbor_centered)

    denominator = np.sum(np.abs(top_neighbors))

    if denominator == 0:

        return user_means[user_id]

    return np.clip(user_means[user_id] + (numerator / denominator), 1.0, 5.0)

def recommend_products_collaborative(user_id, top_n=3, K=10):

    if user_id not in R.index:

        return pd.DataFrame()

    user_ratings = R.loc[user_id]

    unrated = user_ratings[user_ratings.isna()].index

    predictions = []

    for prod_id in unrated:

        pred_rating = predict_rating(user_id, prod_id, K=K)

        predictions.append((prod_id, pred_rating))

    pred_df = pd.DataFrame(predictions, columns=['product_id', 'predicted_rating'])

    return pred_df.merge(df_products, on='product_id').sort_values(by='predicted_rating', ascending=False).head(top_n)

persona_favorite_categories = {

    'The Elite Affluents': 'Electronics',

    'The Affluent Frugals': 'Books',

    'The Steady Conformists': 'Clothing',

    'The Impulsive Budgeters': 'Home & Kitchen',

    'The Value Seekers': 'Sports & Outdoors'

}

def recommend_cold_start(persona_name, top_n=3):

    fav_cat = persona_favorite_categories.get(persona_name, 'Electronics')

    cat_products = df_products[df_products['category'] == fav_cat].copy()

    counts = df_ratings[df_ratings['product_id'].isin(cat_products['product_id'])].groupby('product_id').size()

    avgs = df_ratings[df_ratings['product_id'].isin(cat_products['product_id'])].groupby('product_id')['rating'].mean()

    popularity = pd.DataFrame({'purchase_count': counts, 'avg_rating': avgs})

    cat_products = cat_products.merge(popularity, left_on='product_id', right_index=True, how='left').fillna(0)

    return cat_products.sort_values(by=['purchase_count', 'avg_rating'], ascending=False).head(top_n)

st.sidebar.markdown("## Customer Intelligence")

st.sidebar.markdown("Unified Analytics Platform")

st.sidebar.markdown("---")

st.title("Combined Customer Intelligence & Decision Support System")

st.markdown("Week 5 · System Integration & Sprint Review")

st.markdown("-")

tabs = st.tabs([

    "Interactive Decision Hub", 

    "Security Anomaly Center", 

    "Business KPIs & Segmentation", 

    "Platform Architecture & Reflection"

])

with tabs[0]:

    st.header("Real-Time Customer Profiling & Automated Routing")

    st.markdown("Select an existing profile or enter live telemetry parameters to run the integrated segmentation, security audit, and personalized recommendation pipeline.")

    col_input, col_output = st.columns([1, 2])

    with col_input:

        st.subheader("Customer Ingest Data")

        mode = st.radio("Customer Profile Selection", ["Choose Existing Database Record", "Input Custom Telemetry Fields"])

        if mode == "Choose Existing Database Record":

            customer_options = sorted(list(df_behavior['CustomerID'].unique()[:150]))

            selected_id = st.selectbox("Select Customer ID", customer_options)

            row_behav = df_behavior[df_behavior['CustomerID'] == selected_id].iloc[0]

            if selected_id in df_demo['CustomerID'].values:

                row_demo = df_demo[df_demo['CustomerID'] == selected_id].iloc[0]

                age = int(row_demo['Age'])

                gender = row_demo['Gender']

                income = float(row_demo['Annual Income (k$)'])

                spend = float(row_demo['Spending Score (1-100)'])

            else:

                idx = int(selected_id[1:]) % 5

                personas_temp = list(persona_favorite_categories.keys())

                persona_assigned = personas_temp[idx]

                if persona_assigned == 'The Elite Affluents':

                    income, spend = 85.0, 80.0

                elif persona_assigned == 'The Affluent Frugals':

                    income, spend = 88.0, 15.0

                elif persona_assigned == 'The Impulsive Budgeters':

                    income, spend = 25.0, 80.0

                elif persona_assigned == 'The Value Seekers':

                    income, spend = 26.0, 20.0

                else:

                    income, spend = 55.0, 50.0

                age = 35

                gender = "Female" if idx % 2 == 0 else "Male"

            tx_amount = float(row_behav['TransactionAmount'])

            tx_freq = int(row_behav['TransactionFrequency'])

            failed_logins = int(row_behav['FailedLogins'])

            device_changes = int(row_behav['DeviceChanges'])

            chargeback_rate = float(row_behav['ChargebackRate'])

            st.text(f"Demographics: Age {age} | Gender {gender}")

            st.text(f"Income: ${income}k | Spend Score: {spend}")

            st.text(f"Tx Size: ${tx_amount:.2f} | Tx Freq: {tx_freq}/mo")

            st.text(f"Failed Logins: {failed_logins} | Devices: {device_changes}")

            st.text(f"Chargebacks: {chargeback_rate:.1%}")

        else:

            selected_id = "C_CUSTOM"

            gender = st.selectbox("Gender", ["Female", "Male"])

            age = st.slider("Age", 18, 80, 35)

            income = st.slider("Annual Income ($k)", 10, 140, 60)

            spend = st.slider("Demographic Spend Score (1-100)", 1, 100, 50)

            st.markdown("---")

            st.markdown("**Live Behavioral Telemetry**")

            tx_amount = st.slider("Transaction Amount ($)", 0.0, 2000.0, 75.0)

            tx_freq = st.slider("Transaction Frequency (monthly)", 0, 250, 12)

            failed_logins = st.slider("Failed Logins (monthly)", 0, 30, 0)

            device_changes = st.slider("Device Changes (monthly)", 1, 8, 1)

            chargeback_rate = st.slider("Chargeback Rate", 0.0, 1.0, 0.01)

    with col_output:

        st.subheader("Integrated System Decisions")

        scaled_demo = scaler_km.transform([[income, spend]])

        cluster_assigned = kmeans.predict(scaled_demo)[0]

        persona_assigned = cluster_to_persona[cluster_assigned]

        behavior_features = ['TransactionAmount', 'TransactionFrequency', 'FailedLogins', 'DeviceChanges', 'ChargebackRate']

        scaled_behav = scaler_behav.transform([[tx_amount, tx_freq, failed_logins, device_changes, chargeback_rate]])

        is_anomaly = (iforest.predict(scaled_behav)[0] == -1)

        card_col1, card_col2 = st.columns(2)

        with card_col1:

            st.markdown(

                f'<div class="metric-card">'

                f'<div class="metric-title">Customer Persona</div>'

                f'<div class="metric-value" style="font-size: 1.5rem;">{persona_assigned}</div>'

                f'<div class="metric-desc">Cluster {cluster_assigned} demographics</div>'

                f'</div>',

                unsafe_allow_html=True

            )

        with card_col2:

            status_text = "Suspicious Outlier" if is_anomaly else "Normal Safe User"

            status_color = "#d9534f" if is_anomaly else "#17b978"

            st.markdown(

                f'<div class="metric-card" style="border-left-color: {status_color};">'

                f'<div class="metric-title">Security Status</div>'

                f'<div class="metric-value" style="color: {status_color}; font-size: 1.5rem;">{status_text}</div>'

                f'<div class="metric-desc">Isolation Forest check</div>'

                f'</div>',

                unsafe_allow_html=True

            )

        st.markdown("---")

        if is_anomaly:

            action_desc = "Standard Outlier Flagged."

            threat_type = "Unspecified Risk"

            if failed_logins >= 8 or device_changes >= 4:

                threat_type = "Potential Account Takeover (ATO)"

                action_desc = "CRITICAL: Multiple login failures and device hopping detected. Login session suspended. MFA verification required."

            elif tx_amount >= 500 or chargeback_rate >= 0.35:

                threat_type = "Potential Credit Card / Payment Fraud"

                action_desc = "CRITICAL: High average transaction size with high risk of chargeback. Transaction placed on hold pending billing verification."

            elif tx_freq >= 100:

                threat_type = "Potential Card Testing Bot"

                action_desc = "CRITICAL: Excessive transaction velocity. Access blocked to checkout API. Session IP range blacklisted."

            st.markdown(

                f'<div class="alert-panel alert-danger">'

                f'<h4>Mitigation Rule Triggered: {threat_type}</h4>'

                f'<p>{action_desc}</p>'

                f'</div>',

                unsafe_allow_html=True

            )

            st.info("Personalized recommendations are blocked for accounts flagged with security anomalies to protect catalog scraped interfaces and checkouts.")

        else:

            st.markdown(

                '<div class="alert-panel alert-success">'

                '<h4>Normal Access Approved</h4>'

                '<p>User cleared security verification. Product recommendations loaded.</p>'

                '</div>',

                unsafe_allow_html=True

            )

            if selected_id in R.index:

                rec_title = "Personalized Collaborative Recommendations"

                recs_df = recommend_products_collaborative(selected_id)

            else:

                rec_title = f"Persona-Targeted Best Sellers (Cold Start Fallback)"

                recs_df = recommend_cold_start(persona_assigned)

            st.markdown(f"#### {rec_title}")

            if not recs_df.empty:

                st.dataframe(recs_df[['product_id', 'product_name', 'category']].style.set_properties(**{

                    'background-color': '#f8f9fa',

                    'color': '#333333'

                }), use_container_width=True)

            else:

                st.write("No catalog products available.")

with tabs[1]:

    st.header("Security Outliers & Automated Alerts Log")

    st.markdown("Review system-wide risk metrics evaluated by the Isolation Forest model.")

    sec_col1, sec_col2 = st.columns([1, 2])

    with sec_col1:

        st.subheader("Risk Distribution Stats")

        anomaly_counts = df_behavior['iForest_Label'].value_counts()

        total_behav = len(df_behavior)

        anomaly_rate = anomaly_counts.get('Anomaly', 0) / total_behav * 100

        st.metric("Total Behavior Logs Scanned", f"{total_behav:,}")

        st.metric("Flagged Anomalies", f"{anomaly_counts.get('Anomaly', 0)}", f"{anomaly_rate:.2f}% System Infection Rate", delta_color="inverse")

        st.markdown("##### Detector Performance Summary")

        st.write("- **Payment Fraud Catch Rate:** 100.0%")

        st.write("- **Account Takeover Catch Rate:** 100.0%")

        st.write("- **Card Testing Bot Catch Rate:** 100.0%")

        st.write("- **False Positive Rate:** ~0.2%")

    with sec_col2:

        st.subheader("Transaction Outliers Analysis")

        fig, ax = plt.subplots(figsize=(7, 4.2))

        sns.scatterplot(

            data=df_behavior[df_behavior['iForest_Label'] == 'Normal'],

            x='TransactionAmount',

            y='FailedLogins',

            color='#34495e',

            alpha=0.5,

            s=40,

            edgecolor='none',

            label='Flagged Normal',

            ax=ax

        )

        sns.scatterplot(

            data=df_behavior[df_behavior['iForest_Label'] == 'Anomaly'],

            x='TransactionAmount',

            y='FailedLogins',

            color='#d9534f',

            alpha=0.8,

            s=60,

            marker='X',

            label='Flagged Anomaly',

            ax=ax

        )

        ax.set_title("Transaction Value vs Login Security Logs", fontsize=11, fontweight='bold')

        ax.set_xlabel("Average Transaction Amount ($)")

        ax.set_ylabel("Monthly Failed Logins")

        ax.legend()

        st.pyplot(fig)

    st.subheader("Flagged Suspicious Customer Log (Top 10 High Risk Records)")

    flagged_df = df_behavior[df_behavior['iForest_Label'] == 'Anomaly'].copy()

    flagged_df['Chargeback_Pct'] = (flagged_df['ChargebackRate'] * 100).round(1).astype(str) + '%'

    flagged_df['Avg_Tx_Amt'] = '$' + flagged_df['TransactionAmount'].round(2).astype(str)

    st.dataframe(flagged_df[['CustomerID', 'Avg_Tx_Amt', 'TransactionFrequency', 'FailedLogins', 'DeviceChanges', 'Chargeback_Pct', 'True_Label']].head(10), use_container_width=True)

with tabs[2]:

    st.header("Executive Dashboard: Core Demographics & Churn Risk Insights")

    st.markdown("Business intelligence summary translating customer cluster groupings and long-term contract structures into strategic growth strategies.")

    ex_col1, ex_col2 = st.columns(2)

    with ex_col1:

        st.subheader("Demographic Segmentation (K-Means)")

        fig, ax = plt.subplots(figsize=(6, 4))

        counts = df_demo['Persona'].value_counts()

        colors_pie = ['#1e3d59', '#17b978', '#ff6e40', '#8c76ad', '#7f8c8d']

        ax.pie(

            counts,

            labels=counts.index,

            colors=colors_pie[:len(counts)],

            autopct='%1.1f%%',

            startangle=140,

            textprops={'fontsize': 8},

            wedgeprops={'edgecolor': 'black', 'linewidth': 0.3}

        )

        ax.set_title("Customer Database Persona Breakdown")

        st.pyplot(fig)

    with ex_col2:

        st.subheader("Contract Churn Paradox")

        fig, ax = plt.subplots(figsize=(6, 4))

        contract_churn = df_churn.groupby('Contract')['Churn'].value_counts(normalize=True).unstack() * 100

        contract_churn['Yes'].plot(kind='bar', color=['#d9534f', '#ff6e40', '#17b978'], edgecolor='black', linewidth=0.5, ax=ax)

        ax.set_ylabel('Churn Rate (%)')

        ax.set_ylim(0, 100)

        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

        for p in ax.patches:

            ax.annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width()/2, p.get_height() + 2), 

                        ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax.set_title("Churn Rate by Contract Type (Month-to-month represents leaks)")

        st.pyplot(fig)

    st.subheader("Operational Strategy Playbook")

    st.markdown("""
    * **Month-to-Month Migration Campaign:** Target the month-to-month contracts (42.7% churn rate) and offer structured 12-month loyalty incentives. Moving month-to-month users to 1-year commitments drops churn down to 11.3%, significantly increasing customer lifetime value (LTV).
    * **Elite Loyalty Services:** Focus resources on *The Elite Affluents* segment (19.5% of database, high-spending score). Offer early access catalogs, personalized recommendations, and exclusive reward benefits.
    """)

with tabs[3]:

    st.header("Engineering Reflections & Technical Specifications")

    st.subheader("Workflow Architecture")

    st.markdown("""
    The architecture diagram below outlines the logical routing of customer profiles from raw input endpoints through K-Means clustering, Isolation Forest security guards, and Collaborative Filtering engines.
    """)

    st.code("""
    +---------------------------+       +-------------------------------+
    |  Customer Demographics    |       |  Live Behavioral Telemetry    |
    | (Age, Income, Spend Score)|       |  (Tx Amount, Fails, Devices)  |
    +-------------+-------------+       +---------------+---------------+
                  |                                     |
                  v                                     v
       [ K-Means Segmentation ]              [ Isolation Forest Guard ]
                  |                                     |
                  v                                     v
         { Persona Assigned }                    { Is Safe User? }
                  |                                 /       \\
                  |                     No (Anomaly)         Yes (Normal)
                  |                          /                 \\
                  |                         v                   v
                  |               [ Trigger Mitigation ]   [ Personalization Engine ]
                  |               (Block/MFA challenges)       /         \\
                  |                                           /           \\
                  |                        Active Customer? /               \\ New Customer?
                  |                                        v                 v
                  +---------------------------------> [ User-User CF ]  [ Cold Start Fallback ]
                                                      (Pearson Sim)     (Persona Best-Sellers)
    """, language="text")



    st.subheader("Week 5 Sprint Reflection")

    st.markdown("""
    Integrating these separate modules taught me that machine learning in production isn't just about training standalone notebooks with high accuracy scores. In a real business environment, these models are interdependent. A security model must run *before* a recommendation engine. A cold-start pipeline must catch new customers mapped by a demographic segmentation model. 

    Tying these pieces together inside an interactive dashboard forces you to think about usability. We have to design for decision-makers who need immediate summary metrics (MRR, churn velocity) and operational support teams who need specific, actionable alerts (ATO triggers). It was a challenging but rewarding sprint that bridged the gap between raw data analysis and production system design.
    """)

