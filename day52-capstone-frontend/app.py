import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(page_title="Customer Intelligence Platform", page_icon="📊", layout="wide")

# Apply Custom CSS for better UX
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 1rem;
    }
    .kpi-card {
        background-color: #F3F4F6;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    .kpi-title {
        font-size: 1.1rem;
        color: #4B5563;
        font-weight: 600;
    }
    .kpi-value {
        font-size: 2rem;
        font-weight: 700;
        color: #2563EB;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #F3F4F6;
        border-radius: 5px 5px 0px 0px;
        padding: 10px 20px;
        color: #1F2937;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2563EB !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    data_path = "../day49-capstone-baseline/customer_data.csv"
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
    else:
        # Dummy data if actual not found
        np.random.seed(42)
        df = pd.DataFrame({
            'customer_id': range(1, 1001),
            'age': np.random.randint(18, 70, 1000),
            'tenure_months': np.random.randint(1, 60, 1000),
            'monthly_charges': np.random.uniform(20, 150, 1000),
            'total_charges': np.random.uniform(100, 5000, 1000),
            'churn': np.random.choice([0, 1], p=[0.75, 0.25], size=1000),
            'segment': np.random.choice(['High Value', 'Medium Value', 'Low Value', 'At Risk'], 1000)
        })
    return df

df = load_data()

# Sidebar Navigation & Filters
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3206/3206016.png", width=100)
st.sidebar.title("Navigation & Filters")
st.sidebar.markdown("---")

nav_selection = st.sidebar.radio("Go to:", ["Overview", "Churn Analysis", "Customer Segments", "Explainability (XAI)"])

st.sidebar.markdown("---")
st.sidebar.subheader("Global Filters")
age_range = st.sidebar.slider("Age Range", int(df['age'].min()), int(df['age'].max()), (int(df['age'].min()), int(df['age'].max())))
churn_filter = st.sidebar.multiselect("Churn Status", ["Retained", "Churned"], default=["Retained", "Churned"])

if 'segment' in df.columns:
    segment_filter = st.sidebar.multiselect("Customer Segments", df['segment'].unique(), default=df['segment'].unique())

# Apply Filters
churn_map = {"Retained": 0, "Churned": 1}
selected_churn = [churn_map[c] for c in churn_filter]
filtered_df = df[(df['age'] >= age_range[0]) & (df['age'] <= age_range[1])]
filtered_df = filtered_df[filtered_df['churn'].isin(selected_churn)]
if 'segment' in filtered_df.columns:
    filtered_df = filtered_df[filtered_df['segment'].isin(segment_filter)]

st.markdown('<div class="main-header">Customer Intelligence Platform</div>', unsafe_allow_html=True)

if nav_selection == "Overview":
    st.subheader("Executive Overview")
    
    # KPI Row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'''
        <div class="kpi-card">
            <div class="kpi-title">Total Customers</div>
            <div class="kpi-value">{len(filtered_df):,}</div>
        </div>
        ''', unsafe_allow_html=True)
    with col2:
        churn_rate = filtered_df['churn'].mean() * 100 if len(filtered_df) > 0 else 0
        st.markdown(f'''
        <div class="kpi-card">
            <div class="kpi-title">Churn Rate</div>
            <div class="kpi-value" style="color: #DC2626;">{churn_rate:.1f}%</div>
        </div>
        ''', unsafe_allow_html=True)
    with col3:
        avg_ltv = filtered_df['total_charges'].mean() if len(filtered_df) > 0 else 0
        st.markdown(f'''
        <div class="kpi-card">
            <div class="kpi-title">Avg Lifetime Value</div>
            <div class="kpi-value">${avg_ltv:,.0f}</div>
        </div>
        ''', unsafe_allow_html=True)
    with col4:
        avg_tenure = filtered_df['tenure_months'].mean() if len(filtered_df) > 0 else 0
        st.markdown(f'''
        <div class="kpi-card">
            <div class="kpi-title">Avg Tenure (Months)</div>
            <div class="kpi-value">{avg_tenure:.1f}</div>
        </div>
        ''', unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Charts Row
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Monthly Revenue by Churn Status")
        if not filtered_df.empty:
            fig = px.histogram(filtered_df, x="monthly_charges", color="churn", nbins=30,
                               color_discrete_map={0: "#10B981", 1: "#EF4444"},
                               labels={"monthly_charges": "Monthly Charges ($)", "churn": "Churn Status"},
                               title="Distribution of Monthly Charges")
            fig.update_layout(legend_title_text='Status',
                              legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
                              margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)
            
    with c2:
        st.markdown("#### Customer Segments")
        if 'segment' in filtered_df.columns and not filtered_df.empty:
            seg_counts = filtered_df['segment'].value_counts().reset_index()
            seg_counts.columns = ['Segment', 'Count']
            fig2 = px.pie(seg_counts, names='Segment', values='Count', hole=0.4,
                          color_discrete_sequence=px.colors.qualitative.Pastel,
                          title="Customer Distribution by Segment")
            fig2.update_layout(margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig2, use_container_width=True)

elif nav_selection == "Churn Analysis":
    st.subheader("Deep Dive: Churn Analysis")
    
    tab1, tab2 = st.tabs(["Demographics", "Financials"])
    
    with tab1:
        if not filtered_df.empty:
            fig_age = px.box(filtered_df, x="churn", y="age", color="churn",
                             color_discrete_map={0: "#10B981", 1: "#EF4444"},
                             labels={"churn": "Churn Status (0=Retained, 1=Churned)", "age": "Age"},
                             title="Age Distribution vs Churn")
            st.plotly_chart(fig_age, use_container_width=True)
            
    with tab2:
        if not filtered_df.empty:
            fig_tenure = px.scatter(filtered_df, x="tenure_months", y="total_charges", color="churn",
                                    color_discrete_map={0: "#10B981", 1: "#EF4444"},
                                    opacity=0.6,
                                    labels={"tenure_months": "Tenure (Months)", "total_charges": "Total Charges ($)"},
                                    title="Tenure vs Total Charges by Churn Status")
            st.plotly_chart(fig_tenure, use_container_width=True)

elif nav_selection == "Customer Segments":
    st.subheader("Customer Segment Profiles")
    st.info("Select a segment from the global filters to focus this view.")
    if not filtered_df.empty:
        st.dataframe(filtered_df.head(100), use_container_width=True)
    
elif nav_selection == "Explainability (XAI)":
    st.subheader("Model Explainability")
    st.write("Understand what drives customer churn based on our Random Forest model.")
    st.markdown("""
    * **High Monthly Charges** are the strongest predictor of churn.
    * **Short Tenure** increases churn risk significantly.
    * **Contract Type** (Month-to-month) is highly associated with churn.
    """)
    st.info("Check Day 51 notebook for detailed SHAP plots.")

st.markdown("---")
st.caption("© 2026 Analytics Platform UI Redesign | Day 52 Capstone Project")
