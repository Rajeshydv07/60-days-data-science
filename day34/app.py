import os
import sys

try:
    import streamlit as st
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
except ModuleNotFoundError:
    print("Error: Streamlit or standard analytical libraries are missing. Please run:")
    print("pip install streamlit pandas numpy matplotlib seaborn")
    sys.exit(1)

# Page Configuration
st.set_page_config(
    page_title="Executive Customer Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Style styling adjustments via Markdown injection
st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .kpi-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 5px solid #1e3d59;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .kpi-title {
        font-size: 0.9rem;
        color: #6c757d;
        font-weight: bold;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }
    .kpi-value {
        font-size: 2.2rem;
        color: #1e3d59;
        font-weight: bold;
        line-height: 1.2;
    }
    .kpi-desc {
        font-size: 0.8rem;
        color: #adb5bd;
        margin-top: 0.5rem;
        font-style: italic;
    }
</style>
""", unsafe_style_allowed=True)

# Datasets Loading
@st.cache_data
def load_data():
    churn_path = "day15/telco_customer_churn.csv"
    segment_path = "day31/Mall_Customers_Labeled_Personas.csv"
    
    # Try parent directory paths if not found directly
    if not os.path.exists(churn_path):
        churn_path = "../day15/telco_customer_churn.csv"
    if not os.path.exists(segment_path):
        segment_path = "../day31/Mall_Customers_Labeled_Personas.csv"
        
    df_c = pd.read_csv(churn_path)
    df_s = pd.read_csv(segment_path)
    
    # Preprocess Churn TotalCharges
    df_c['TotalCharges'] = pd.to_numeric(df_c['TotalCharges'].str.strip(), errors='coerce')
    df_c['TotalCharges'] = df_c['TotalCharges'].fillna(df_c['MonthlyCharges'] * df_c['tenure'])
    
    return df_c, df_s

try:
    df_churn, df_segments = load_data()
except Exception as e:
    st.error(f"Failed to load datasets. Ensure you run this script from the workspace root directory. Details: {e}")
    st.stop()

# --- SIDEBAR FILTERS ---
st.sidebar.image("https://img.icons8.com/clouds/100/dashboard.png", width=70)
st.sidebar.title("Executive Filters")
st.sidebar.markdown("Filter the datasets below to update the KPIs and Churn analysis in real-time.")

# Contract filter
contract_list = ['All'] + sorted(list(df_churn['Contract'].unique()))
selected_contract = st.sidebar.selectbox("Select Contract Type", contract_list)

# Internet service filter
internet_list = ['All'] + sorted(list(df_churn['InternetService'].unique()))
selected_internet = st.sidebar.selectbox("Select Internet Tech", internet_list)

# Gender filter
gender_list = ['All'] + sorted(list(df_churn['gender'].unique()))
selected_gender = st.sidebar.selectbox("Select Customer Gender", gender_list)

# Apply filters to churn data
df_churn_filtered = df_churn.copy()
if selected_contract != 'All':
    df_churn_filtered = df_churn_filtered[df_churn_filtered['Contract'] == selected_contract]
if selected_internet != 'All':
    df_churn_filtered = df_churn_filtered[df_churn_filtered['InternetService'] == selected_internet]
if selected_gender != 'All':
    df_churn_filtered = df_churn_filtered[df_churn_filtered['gender'] == selected_gender]

# --- MAIN DASHBOARD HEADER ---
st.title("📈 Executive Customer Analytics Dashboard")
st.markdown("### Day 34 · Business Intelligence & Churn Risk Insights")
st.markdown("---")

# --- KPI METRICS CARD ROW ---
col1, col2, col3, col4 = st.columns(4)

total_cust_num = len(df_churn_filtered)
if len(df_churn_filtered) > 0:
    churn_rate_num = (df_churn_filtered['Churn'] == 'Yes').sum() / total_cust_num * 100
    avg_bill_num = df_churn_filtered['MonthlyCharges'].mean()
    mrr_num = df_churn_filtered[df_churn_filtered['Churn'] == 'No']['MonthlyCharges'].sum()
else:
    churn_rate_num = 0.0
    avg_bill_num = 0.0
    mrr_num = 0.0

with col1:
    st.markdown(
        f'<div class="kpi-card">'
        f'<div class="kpi-title">👤 Total Customer Base</div>'
        f'<div class="kpi-value">{total_cust_num:,}</div>'
        f'<div class="kpi-desc">Filtered active & churned accounts</div>'
        f'</div>',
        unsafe_allow_html=True
    )

with col2:
    color_churn = "#d9534f" if churn_rate_num > 15 else "#1e3d59"
    st.markdown(
        f'<div class="kpi-card" style="border-left-color: {color_churn};">'
        f'<div class="kpi-title">🚨 Churn Rate</div>'
        f'<div class="kpi-value" style="color: {color_churn};">{churn_rate_num:.2f}%</div>'
        f'<div class="kpi-desc">Global Base Average: 26.54%</div>'
        f'</div>',
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        f'<div class="kpi-card">'
        f'<div class="kpi-title">💵 Avg Monthly Charge</div>'
        f'<div class="kpi-value">${avg_bill_num:.2f}</div>'
        f'<div class="kpi-desc">Average billing tier</div>'
        f'</div>',
        unsafe_allow_html=True
    )

with col4:
    st.markdown(
        f'<div class="kpi-card" style="border-left-color: #17b978;">'
        f'<div class="kpi-title">📈 Estimated MRR</div>'
        f'<div class="kpi-value" style="color: #17b978;">${mrr_num/1000:.1f}k</div>'
        f'<div class="kpi-desc">Active account recurring value</div>'
        f'</div>',
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)

# --- TABS FOR DIFFERENT VIEWPORTS ---
tab1, tab2, tab3 = st.tabs(["📉 Customer Churn Risk Analysis", "👥 Demographic Segments & Personas", "🎯 Strategic Storytelling & Playbook"])

with tab1:
    st.subheader("Where is Churn Leaking? (Contract & Connection Risks)")
    
    if total_cust_num == 0:
        st.warning("No data matches the selected filters. Please adjust the sidebar filters.")
    else:
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            st.markdown("#### Churn Rate by Contract Type")
            contract_churn = df_churn_filtered.groupby('Contract')['Churn'].value_counts(normalize=True).unstack().fillna(0) * 100
            
            fig, ax = plt.subplots(figsize=(6, 4))
            colors_con = ['#1e3d59', '#ff6e40', '#d9534f']
            if 'Yes' in contract_churn.columns:
                contract_churn['Yes'].plot(kind='bar', color=colors_con[:len(contract_churn)], ax=ax, edgecolor='black', linewidth=0.5)
                ax.set_ylabel('Churn Rate (%)')
                ax.set_ylim(0, 100)
                for p in ax.patches:
                    ax.annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width()/2, p.get_height() + 3), 
                                ha='center', va='bottom', fontsize=9, fontweight='bold')
            else:
                ax.text(0.5, 0.5, "No Churned Customers in Selection", ha='center', va='center')
            sns.despine()
            st.pyplot(fig)
            
        with chart_col2:
            st.markdown("#### Churn Rate by Internet Technology")
            internet_churn = df_churn_filtered.groupby('InternetService')['Churn'].value_counts(normalize=True).unstack().fillna(0) * 100
            
            fig, ax = plt.subplots(figsize=(6, 4))
            colors_net = ['#ff6e40', '#d9534f', '#17b978']
            if 'Yes' in internet_churn.columns:
                internet_churn['Yes'].plot(kind='bar', color=colors_net[:len(internet_churn)], ax=ax, edgecolor='black', linewidth=0.5)
                ax.set_ylabel('Churn Rate (%)')
                ax.set_ylim(0, 100)
                for p in ax.patches:
                    ax.annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width()/2, p.get_height() + 3), 
                                ha='center', va='bottom', fontsize=9, fontweight='bold')
            else:
                ax.text(0.5, 0.5, "No Churned Customers in Selection", ha='center', va='center')
            sns.despine()
            st.pyplot(fig)
            
        st.markdown("#### Contract & Internet Service Distribution Detail")
        df_details = df_churn_filtered.groupby(['Contract', 'InternetService']).agg(
            Total_Customers=('customerID', 'count'),
            Churned_Customers=('Churn', lambda x: (x == 'Yes').sum()),
            Average_Tenure_Months=('tenure', 'mean')
        ).reset_index()
        
        df_details['Churn_Rate'] = (df_details['Churned_Customers'] / df_details['Total_Customers'] * 100).round(2)
        df_details['Average_Tenure_Months'] = df_details['Average_Tenure_Months'].round(1)
        
        st.dataframe(df_details.style.format({
            "Total_Customers": "{:,}",
            "Churned_Customers": "{:,}",
            "Average_Tenure_Months": "{:.1f} months",
            "Churn_Rate": "{:.2f}%"
        }), use_container_width=True)

with tab2:
    st.subheader("Demographic Clustering (K-Means Personas)")
    
    # Checkbox filter inside Tab 2
    show_table = st.checkbox("Show Raw Persona Profiles Table", value=False)
    
    chart_col1, chart_col2 = st.columns([2, 1])
    
    with chart_col1:
        st.markdown("#### Income vs. Spending Profile Matrix")
        
        persona_colors = {
            'The Value Seekers': '#7f8c8d',
            'The Impulsive Budgeters': '#e74c3c',
            'The Steady Conformists': '#f1c40f',
            'The Elite Affluents': '#2c3e50',
            'The Affluent Frugals': '#1abc9c'
        }
        
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.scatterplot(
            data=df_segments,
            x='Annual Income (k$)',
            y='Spending Score (1-100)',
            hue='Persona',
            palette=persona_colors,
            s=70,
            alpha=0.8,
            ax=ax,
            edgecolor='black',
            linewidth=0.5
        )
        ax.set_title("Target Markets: Elite Affluents represent high priority targets", fontsize=10, style='italic', color='#555555')
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.legend(title='Persona Profiles', bbox_to_anchor=(1.02, 1), loc='upper left')
        st.pyplot(fig)
        
    with chart_col2:
        st.markdown("#### Persona Breakdown")
        counts = df_segments['Persona'].value_counts()
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.pie(
            counts,
            labels=counts.index,
            colors=[persona_colors.get(p, '#cccccc') for p in counts.index],
            autopct='%1.1f%%',
            startangle=140,
            textprops={'fontsize': 8},
            wedgeprops={'edgecolor': 'black', 'linewidth': 0.3}
        )
        st.pyplot(fig)
        
    if show_table:
        st.dataframe(df_segments)

with tab3:
    st.subheader("Executive Storytelling & Operational Playbook")
    
    st.markdown("""
    ### 📊 Key Dashboard Storytelling Insights
    
    1. **Contract Structure is the Single Strongest Churn Indicator**:
       - **Month-to-month contracts** show an overwhelming churn rate of **42.71%**.
       - **One-year contracts** fall sharply to **11.27%**.
       - **Two-year contracts** fall to a negligible **2.83%**.
       - *Story:* Customer attrition is not an organic product-failure issue; it is a transactional friction issue. Single-month billing represents the point of maximum cancellation exposure.
       
    2. **The Fiber Optic Churn Paradox**:
       - Fiber Optic internet customers have a **41.89%** churn rate, compared to DSL customers at only **18.96%**.
       - *Story:* While Fiber optic represents a premium billing tier (meaning high revenue), customers are leaving rapidly. This suggests price sensitivity, poor post-sales onboarding, or network stability issues that do not justify the high monthly costs.
       
    3. **Targeting the Right Personas**:
       - **The Elite Affluents** (High Income, High Spending Score) represent **23.2%** of Mall Customers.
       - These users are less price sensitive and require high-quality services. Up-selling them to long-term commitments (1 or 2-year contracts) with premium service guarantees will secure stable, high-margin revenue streams.
    
    ---
    
    ### 🎯 Recommended Operational Action Plan
    
    *   **Campaign 1: 'Month-to-Month Migration' (High Impact)**
        - **Target:** Existing Month-to-Month accounts (especially those reaching 3+ months tenure).
        - **Action:** Offer a small monthly discount (e.g., $5 off) or a free streaming service option in exchange for signing a 12-month contract. Reducing contract churn from 42% to 11% will save millions of dollars in customer acquisition costs (CAC).
        
    *   **Campaign 2: 'Fiber Optic Health Audit' (Urgent)**
        - **Target:** All Fiber Optic accounts that have completed less than 90 days of tenure.
        - **Action:** Customer Success teams must conduct proactive check-ins to ensure line stability and customer satisfaction. Introduce a $10 temporary credit for the first 3 months to ease initial billing sticker-shock.
        
    *   **Campaign 3: 'Elite Loyalty Program' (Medium Term)**
        - **Target:** High-spending customer profiles matching *The Elite Affluents*.
        - **Action:** Design customized reward schemes and high-priority customer service support to prevent competitor poaching.
    """)
