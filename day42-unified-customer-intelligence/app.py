import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Set page config
st.set_page_config(page_title="Unified Customer Intelligence", layout="wide", page_icon="📊")

# Custom CSS for styling
st.markdown("""
<style>
    .kpi-card {
        background-color: #1e1e1e;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.5);
    }
    .kpi-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: #4CAF50;
    }
    .kpi-title {
        font-size: 1.2rem;
        color: #aaaaaa;
    }
</style>
""", unsafe_allow_html=True)

# Generate Mock Data
@st.cache_data
def generate_data():
    # KPI Data
    kpis = {
        "Total MRR": "$1.2M",
        "Active Customers": "8,450",
        "Avg LTV": "$12,400",
        "Overall Churn": "3.2%"
    }
    
    # Forecasting Data
    dates = pd.date_range(start="2023-01-01", end="2024-12-31", freq='M')
    historical_revenue = np.linspace(500000, 1200000, len(dates[:18])) + np.random.normal(0, 20000, 18)
    forecast_revenue = np.linspace(1200000, 1500000, len(dates[18:])) + np.random.normal(0, 25000, len(dates[18:]))
    
    revenue_df = pd.DataFrame({
        'Date': dates,
        'Revenue': np.concatenate([historical_revenue, forecast_revenue]),
        'Type': ['Historical'] * 18 + ['Forecast'] * 6
    })
    
    # Retention Data
    months = np.arange(0, 25)
    retention_rate = np.exp(-0.05 * months) * 100
    retention_df = pd.DataFrame({'Month': months, 'Retention Rate (%)': retention_rate})
    
    # Risk Scoring Data
    customers = [f"Cust_{i}" for i in range(100)]
    risk_scores = np.random.beta(a=2, b=5, size=100) * 100
    ltv = np.random.normal(10000, 3000, 100).clip(1000)
    risk_df = pd.DataFrame({
        'Customer ID': customers,
        'Risk Score': risk_scores,
        'LTV ($)': ltv,
        'Last Active (Days)': np.random.randint(1, 45, 100)
    })
    risk_df['Risk Category'] = pd.cut(risk_df['Risk Score'], bins=[0, 30, 70, 100], labels=['Low', 'Medium', 'High'])
    risk_df = risk_df.sort_values('Risk Score', ascending=False).reset_index(drop=True)
    
    return kpis, revenue_df, retention_df, risk_df

kpis, revenue_df, retention_df, risk_df = generate_data()

st.title("📊 Unified Customer Intelligence Decision System")
st.markdown("An integrated dashboard for Executive Analytics, Forecasting, and Churn Risk Scoring.")

st.divider()

# 1. KPI Section
st.header("Executive KPIs")
cols = st.columns(4)
for i, (title, value) in enumerate(kpis.items()):
    with cols[i]:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">{title}</div>
            <div class="kpi-value">{value}</div>
        </div>
        """, unsafe_allow_html=True)

st.divider()

# 2. Forecasting & Retention Section
col1, col2 = st.columns(2)

with col1:
    st.header("📈 Revenue Forecasting")
    st.markdown("Historical and predicted Monthly Recurring Revenue (MRR).")
    fig_forecast = px.line(revenue_df, x='Date', y='Revenue', color='Type', 
                           color_discrete_map={'Historical': '#2196F3', 'Forecast': '#FF9800'},
                           markers=True)
    fig_forecast.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_forecast, use_container_width=True)

with col2:
    st.header("⏳ Survival & Retention")
    st.markdown("Overall customer retention curve (Kaplan-Meier estimate).")
    fig_retention = px.area(retention_df, x='Month', y='Retention Rate (%)', 
                            color_discrete_sequence=['#9C27B0'])
    fig_retention.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_retention, use_container_width=True)

st.divider()

# 3. Predictive Risk Scoring
st.header("🚨 At-Risk Customers (Predictive Scoring)")
st.markdown("Customers flagged by the machine learning churn classifier, prioritized by risk and LTV.")

risk_cols = st.columns([1, 2])

with risk_cols[0]:
    st.subheader("Risk Distribution")
    risk_dist = risk_df['Risk Category'].value_counts().reset_index()
    risk_dist.columns = ['Category', 'Count']
    fig_pie = px.pie(risk_dist, values='Count', names='Category', hole=0.4,
                     color='Category', color_discrete_map={'Low': '#4CAF50', 'Medium': '#FFEB3B', 'High': '#F44336'})
    fig_pie.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_pie, use_container_width=True)

with risk_cols[1]:
    st.subheader("High-Risk Customer Queue")
    high_risk_df = risk_df[risk_df['Risk Category'] == 'High'].head(10).copy()
    high_risk_df['Risk Score'] = high_risk_df['Risk Score'].apply(lambda x: f"{x:.1f}%")
    high_risk_df['LTV ($)'] = high_risk_df['LTV ($)'].apply(lambda x: f"${x:,.2f}")
    st.dataframe(high_risk_df, use_container_width=True, hide_index=True)

st.markdown("---")
st.caption("Generated by the Unified Intelligence Engine | Model Version: v2.4 (XGBoost/Prophet)")
