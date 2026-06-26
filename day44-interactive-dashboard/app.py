import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Ensure page config is the very first Streamlit command
st.set_page_config(
    page_title="Customer Intelligence Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling the dashboard
st.markdown("""
<style>
    .kpi-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
        border-left: 5px solid #4e73df;
    }
    .kpi-title {
        color: #5a5c69;
        font-size: 1.1rem;
        font-weight: 700;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }
    .kpi-value {
        color: #2e59d9;
        font-size: 2.5rem;
        font-weight: 800;
    }
    .st-emotion-cache-16txtl3 {
        padding-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Generate Mock Data for the Dashboard
@st.cache_data
def load_data():
    np.random.seed(42)
    n_samples = 1500
    
    # Generate features
    customer_ids = [f"CUST-{str(i).zfill(5)}" for i in range(n_samples)]
    age = np.random.normal(loc=45, scale=15, size=n_samples).astype(int)
    age = np.clip(age, 18, 85)
    
    tenure = np.random.uniform(low=1, high=72, size=n_samples).astype(int)
    monthly_charges = np.random.uniform(low=20.0, high=120.0, size=n_samples)
    total_charges = monthly_charges * tenure * np.random.uniform(0.9, 1.1, size=n_samples)
    
    segments = np.random.choice(['Basic', 'Standard', 'Premium'], size=n_samples, p=[0.5, 0.35, 0.15])
    contract = np.random.choice(['Month-to-month', 'One year', 'Two year'], size=n_samples, p=[0.55, 0.25, 0.20])
    
    # Simulate Churn logic based on features
    churn_prob = np.zeros(n_samples)
    churn_prob += np.where(contract == 'Month-to-month', 0.3, 0.05)
    churn_prob += np.where(monthly_charges > 80, 0.1, 0)
    churn_prob -= np.where(tenure > 40, 0.15, 0)
    churn_prob -= np.where(segments == 'Premium', 0.05, 0)
    
    churn_prob = np.clip(churn_prob, 0.05, 0.8)
    churn = np.random.binomial(1, churn_prob)
    churn_status = ['Yes' if c == 1 else 'No' for c in churn]
    
    data = pd.DataFrame({
        'CustomerID': customer_ids,
        'Age': age,
        'Tenure (Months)': tenure,
        'Monthly Charges ($)': np.round(monthly_charges, 2),
        'Total Charges ($)': np.round(total_charges, 2),
        'Segment': segments,
        'Contract': contract,
        'Churn': churn_status
    })
    return data

# Main Title
st.title("📊 Customer Intelligence & Analytics Dashboard")
st.markdown("Monitor KPIs, analyze customer segments, and evaluate churn risks in real-time.")
st.markdown("---")

# Sidebar Configuration
st.sidebar.header("Filter Data")

# Allow user to upload data or use simulated
uploaded_file = st.sidebar.file_uploader("Upload custom CSV (Optional)", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success("Data uploaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Error loading file: {e}")
        df = load_data()
else:
    df = load_data()

st.sidebar.markdown("### Dashboard Controls")

# Filters
# Age Filter
min_age, max_age = int(df['Age'].min()), int(df['Age'].max())
age_range = st.sidebar.slider("Select Age Range", min_value=min_age, max_value=max_age, value=(min_age, max_age))

# Segment Filter
all_segments = list(df['Segment'].unique())
selected_segments = st.sidebar.multiselect("Select Customer Segments", options=all_segments, default=all_segments)

# Churn Filter
all_churn = list(df['Churn'].unique())
selected_churn = st.sidebar.multiselect("Select Churn Status", options=all_churn, default=all_churn)

# Apply Filters
filtered_df = df[
    (df['Age'] >= age_range[0]) & 
    (df['Age'] <= age_range[1]) & 
    (df['Segment'].isin(selected_segments)) &
    (df['Churn'].isin(selected_churn))
]

if filtered_df.empty:
    st.warning("No data matches the selected filters. Please adjust your criteria.")
    st.stop()

# ==========================================
# KPI METRICS SECTION
# ==========================================
col1, col2, col3, col4 = st.columns(4)

total_customers = len(filtered_df)
churn_rate = (len(filtered_df[filtered_df['Churn'] == 'Yes']) / total_customers) * 100 if total_customers > 0 else 0
avg_monthly = filtered_df['Monthly Charges ($)'].mean()
high_value = len(filtered_df[filtered_df['Segment'] == 'Premium'])

with col1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Total Customers</div>
        <div class="kpi-value">{total_customers:,}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="kpi-card" style="border-left-color: {'#e74a3b' if churn_rate > 20 else '#1cc88a'};">
        <div class="kpi-title">Churn Rate</div>
        <div class="kpi-value">{churn_rate:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="kpi-card" style="border-left-color: #f6c23e;">
        <div class="kpi-title">Avg Monthly Rev</div>
        <div class="kpi-value">${avg_monthly:.2f}</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="kpi-card" style="border-left-color: #36b9cc;">
        <div class="kpi-title">Premium Customers</div>
        <div class="kpi-value">{high_value:,}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ==========================================
# VISUALIZATIONS SECTION
# ==========================================
row1_col1, row1_col2 = st.columns(2)

with row1_col1:
    st.subheader("Customer Segmentation")
    # Pie Chart for Segments
    segment_counts = filtered_df['Segment'].value_counts().reset_index()
    segment_counts.columns = ['Segment', 'Count']
    fig_seg = px.pie(segment_counts, values='Count', names='Segment', 
                     color='Segment',
                     color_discrete_map={'Basic':'#4e73df', 'Standard':'#36b9cc', 'Premium':'#1cc88a'},
                     hole=0.4)
    fig_seg.update_layout(margin=dict(t=20, b=20, l=20, r=20))
    st.plotly_chart(fig_seg, use_container_width=True)

with row1_col2:
    st.subheader("Churn Risk by Contract Type")
    # Bar Chart for Churn by Contract
    churn_contract = filtered_df.groupby(['Contract', 'Churn']).size().reset_index(name='Count')
    fig_churn = px.bar(churn_contract, x='Contract', y='Count', color='Churn', 
                       barmode='group',
                       color_discrete_map={'No':'#1cc88a', 'Yes':'#e74a3b'})
    fig_churn.update_layout(margin=dict(t=20, b=20, l=20, r=20))
    st.plotly_chart(fig_churn, use_container_width=True)

st.markdown("---")

row2_col1, row2_col2 = st.columns(2)

with row2_col1:
    st.subheader("Revenue vs Tenure Correlation")
    # Scatter plot for Tenure vs Total Charges
    fig_scatter = px.scatter(filtered_df, x="Tenure (Months)", y="Total Charges ($)", 
                             color="Segment", opacity=0.7,
                             color_discrete_map={'Basic':'#4e73df', 'Standard':'#36b9cc', 'Premium':'#1cc88a'},
                             trendline="ols")
    fig_scatter.update_layout(margin=dict(t=20, b=20, l=20, r=20))
    st.plotly_chart(fig_scatter, use_container_width=True)

with row2_col2:
    st.subheader("Customer Demographics (Age Distribution)")
    # Histogram for Age
    fig_hist = px.histogram(filtered_df, x="Age", nbins=20, color="Churn",
                            marginal="box",
                            color_discrete_map={'No':'#1cc88a', 'Yes':'#e74a3b'})
    fig_hist.update_layout(margin=dict(t=20, b=20, l=20, r=20))
    st.plotly_chart(fig_hist, use_container_width=True)

# ==========================================
# RAW DATA SECTION
# ==========================================
st.markdown("---")
st.subheader("Detailed Customer Records")
with st.expander("View Raw Data Table"):
    st.dataframe(filtered_df.style.highlight_max(axis=0))
    
    # Download Button for Filtered Data
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Filtered Data as CSV",
        data=csv,
        file_name='filtered_customer_data.csv',
        mime='text/csv',
    )
