import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Churn Risk Dashboard", layout="wide")

st.title("Customer Churn Risk Dashboard")
st.markdown("Monitor and rank customers based on their predicted churn risk.")

# Load data
@st.cache_data
def load_data():
    try:
        return pd.read_csv('../data/churn_scored.csv')
    except FileNotFoundError:
        return None

df = load_data()

if df is None:
    st.warning("Data not found. Please ensure the model has been trained and data is saved at `data/churn_scored.csv`.")
    st.stop()

# KPIs
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Customers", len(df))
col2.metric("Critical Risk Customers", len(df[df['risk_group'] == 'Critical']))
col3.metric("High Risk Customers", len(df[df['risk_group'] == 'High']))
col4.metric("Avg Churn Risk", f"{df['churn_risk_score'].mean():.2%}")

st.markdown("---")

col_viz1, col_viz2 = st.columns(2)

with col_viz1:
    st.subheader("Risk Group Distribution")
    risk_counts = df['risk_group'].value_counts().reset_index()
    risk_counts.columns = ['Risk Group', 'Count']
    fig1 = px.pie(risk_counts, names='Risk Group', values='Count', hole=0.4, 
                  color='Risk Group',
                  color_discrete_map={'Critical':'darkred', 'High':'red', 'Medium':'orange', 'Low':'green'})
    st.plotly_chart(fig1, use_container_width=True)

with col_viz2:
    st.subheader("Churn Risk by Contract Type")
    fig2 = px.box(df, x='contract_type', y='churn_risk_score', color='contract_type')
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

st.subheader("High-Risk Customer List")
st.markdown("Review the top customers who are most likely to churn.")

# Filtering
risk_filter = st.multiselect("Filter by Risk Group", options=df['risk_group'].unique(), default=['Critical', 'High'])
filtered_df = df[df['risk_group'].isin(risk_filter)].sort_values('churn_risk_score', ascending=False)

st.dataframe(filtered_df[['customer_id', 'churn_risk_score', 'risk_group', 'contract_type', 'tenure_months', 'monthly_charges', 'satisfaction_score', 'support_tickets']].head(100), use_container_width=True)
