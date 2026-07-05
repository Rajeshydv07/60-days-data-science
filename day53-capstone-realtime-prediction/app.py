import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
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
    .predict-box-low {
        background-color: #D1FAE5;
        border-left: 5px solid #10B981;
        padding: 20px;
        border-radius: 5px;
        margin-top: 20px;
    }
    .predict-box-high {
        background-color: #FEE2E2;
        border-left: 5px solid #EF4444;
        padding: 20px;
        border-radius: 5px;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    data_path = "../day49-capstone-baseline/customer_data.csv"
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
    else:
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

st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3206/3206016.png", width=100)
st.sidebar.title("Navigation")
nav_selection = st.sidebar.radio("Go to:", ["Overview", "Real-Time Prediction", "Churn Analysis"])

st.markdown('<div class="main-header">Customer Intelligence Platform</div>', unsafe_allow_html=True)

if nav_selection == "Overview":
    st.subheader("Executive Overview")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="kpi-card"><div class="kpi-title">Total Customers</div><div class="kpi-value">{len(df):,}</div></div>', unsafe_allow_html=True)
    with col2:
        churn_rate = df['churn'].mean() * 100 if len(df) > 0 else 0
        st.markdown(f'<div class="kpi-card"><div class="kpi-title">Churn Rate</div><div class="kpi-value" style="color: #DC2626;">{churn_rate:.1f}%</div></div>', unsafe_allow_html=True)
    with col3:
        avg_ltv = df['total_charges'].mean() if len(df) > 0 else 0
        st.markdown(f'<div class="kpi-card"><div class="kpi-title">Avg Lifetime Value</div><div class="kpi-value">${avg_ltv:,.0f}</div></div>', unsafe_allow_html=True)
    with col4:
        avg_tenure = df['tenure_months'].mean() if len(df) > 0 else 0
        st.markdown(f'<div class="kpi-card"><div class="kpi-title">Avg Tenure (Months)</div><div class="kpi-value">{avg_tenure:.1f}</div></div>', unsafe_allow_html=True)
        
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Monthly Revenue by Churn Status")
        fig = px.histogram(df, x="monthly_charges", color="churn", nbins=30, color_discrete_map={0: "#10B981", 1: "#EF4444"})
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown("#### Customer Segments")
        if 'segment' in df.columns:
            seg_counts = df['segment'].value_counts().reset_index()
            seg_counts.columns = ['Segment', 'Count']
            fig2 = px.pie(seg_counts, names='Segment', values='Count', hole=0.4)
            st.plotly_chart(fig2, use_container_width=True)

elif nav_selection == "Real-Time Prediction":
    st.subheader("Live Customer Churn Prediction")
    st.markdown("Enter customer details to instantly predict their churn risk and evaluate actionable insights.")
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, value=35)
            tenure = st.number_input("Tenure (Months)", min_value=0, max_value=120, value=12)
            contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
            
        with col2:
            monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=500.0, value=65.0)
            total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=780.0)
            support_calls = st.number_input("Recent Support Calls", min_value=0, max_value=20, value=1)
            
        submit_button = st.form_submit_button("Predict Churn Risk")
        
    if submit_button:
        # Dummy scoring based on XAI insights (Day 51)
        risk_score = 0.2
        if contract == "Month-to-month":
            risk_score += 0.35
        if tenure < 12:
            risk_score += 0.20
        if monthly_charges > 80:
            risk_score += 0.15
        if support_calls > 2:
            risk_score += 0.10
            
        risk_score = min(risk_score, 0.99)
        
        if risk_score > 0.5:
            st.markdown(f'''
            <div class="predict-box-high">
                <h3 style="color: #991B1B; margin: 0;">High Churn Risk Detected ({(risk_score*100):.1f}%)</h3>
                <p style="color: #7F1D1D; margin-top: 10px;">This customer has a high likelihood of churning. Immediate retention action is recommended.</p>
                <b>Recommended Actions:</b>
                <ul style="color: #7F1D1D;">
                    <li>Offer a targeted 15% discount for upgrading to an annual contract.</li>
                    <li>Schedule a proactive outreach call from the Customer Success team.</li>
                </ul>
            </div>
            ''', unsafe_allow_html=True)
        else:
            st.markdown(f'''
            <div class="predict-box-low">
                <h3 style="color: #065F46; margin: 0;">Low Churn Risk ({(risk_score*100):.1f}%)</h3>
                <p style="color: #064E3B; margin-top: 10px;">This customer is currently stable and highly likely to be retained.</p>
                <b>Recommended Actions:</b>
                <ul style="color: #064E3B;">
                    <li>Explore cross-selling opportunities for premium addons.</li>
                    <li>Enroll in standard automated engagement campaigns.</li>
                </ul>
            </div>
            ''', unsafe_allow_html=True)

elif nav_selection == "Churn Analysis":
    st.subheader("Deep Dive: Churn Analysis")
    st.write("Visualizations from Day 52 to analyze demographic and financial churn patterns.")
