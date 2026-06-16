import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Style configuration for clean, corporate executive reporting
sns.set_theme(style="white")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.titlesize'] = 16
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['grid.color'] = '#f0f0f0'

PRIMARY_COLOR = '#1e3d59'
ACCENT_COLOR = '#ff6e40'
MUTED_DARK = '#17b978'
DANGER_COLOR = '#d9534f'
BG_CARD = '#f5f7fa'

# 1. Ingest Datasets
churn_data_path = 'day15/telco_customer_churn.csv'
segment_data_path = 'day31/Mall_Customers_Labeled_Personas.csv'

if not os.path.exists(churn_data_path):
    raise FileNotFoundError(f"Missing churn dataset at: {churn_data_path}")
if not os.path.exists(segment_data_path):
    raise FileNotFoundError(f"Missing personas dataset at: {segment_data_path}")

df_churn = pd.read_csv(churn_data_path)
df_segments = pd.read_csv(segment_data_path)

# 2. Preprocess and Calculate KPIs
df_churn['TotalCharges'] = pd.to_numeric(df_churn['TotalCharges'].str.strip(), errors='coerce')
df_churn['TotalCharges'] = df_churn['TotalCharges'].fillna(df_churn['MonthlyCharges'] * df_churn['tenure'])

total_customers = len(df_churn)
churn_rate = (df_churn['Churn'] == 'Yes').sum() / total_customers * 100
avg_monthly_bill = df_churn['MonthlyCharges'].mean()
active_mrr = df_churn[df_churn['Churn'] == 'No']['MonthlyCharges'].sum()

print("--- Executive Dashboard Statistics ---")
print(f"Total Customer Accounts: {total_customers:,}")
print(f"Customer Churn Rate: {churn_rate:.2f}%")
print(f"Average Monthly Bill: ${avg_monthly_bill:.2f}")
print(f"Active MRR: ${active_mrr:,.2f}\n")

# 3. Plot 1: KPI Cards
fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
fig.patch.set_facecolor('#ffffff')

kpis = [
    {"title": "TOTAL CUSTOMERS", "val": f"{total_customers:,}", "desc": "Active + Churned Accounts", "color": PRIMARY_COLOR},
    {"title": "CHURN RATE", "val": f"{churn_rate:.1f}%", "desc": "Industry Benchmark ~ 5%", "color": DANGER_COLOR},
    {"title": "AVG MONTHLY BILL", "val": f"${avg_monthly_bill:.2f}", "desc": "Per active user average", "color": PRIMARY_COLOR},
    {"title": "ACTIVE MRR", "val": f"${active_mrr/1000:.1f}k", "desc": "Non-churned monthly charges", "color": MUTED_DARK}
]

for i, kpi in enumerate(kpis):
    ax = axes[i]
    ax.set_facecolor(BG_CARD)
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    for spine in ax.spines.values():
        spine.set_color('#e0e0e0')
        spine.set_linewidth(1.5)
        
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.axvline(x=0.01, color=kpi['color'], linewidth=8, ymin=0, ymax=1)
    
    ax.text(0.08, 0.78, kpi['title'], fontsize=10, color='#666666', fontweight='bold', ha='left')
    ax.text(0.08, 0.42, kpi['val'], fontsize=28, color=kpi['color'], fontweight='bold', ha='left')
    ax.text(0.08, 0.15, kpi['desc'], fontsize=8.5, color='#999999', style='italic', ha='left')

plt.suptitle('EXECUTIVE BRIEF: CORE CUSTOMER INSIGHTS', fontsize=14, fontweight='bold', y=1.05, color=PRIMARY_COLOR)
plt.tight_layout()
plt.savefig('day34/executive_kpis.png', dpi=150, bbox_inches='tight')
plt.close()

# 4. Plot 2: Churn Risks by Category
contract_churn = df_churn.groupby('Contract')['Churn'].value_counts(normalize=True).unstack() * 100
internet_churn = df_churn.groupby('InternetService')['Churn'].value_counts(normalize=True).unstack() * 100

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

contract_churn['Yes'].plot(kind='bar', ax=axes[0], color=[DANGER_COLOR, '#f9a825', MUTED_DARK], edgecolor='black', linewidth=0.5)
axes[0].set_title('Churn Rate (%) by Contract Type', fontweight='bold', pad=12, color=PRIMARY_COLOR)
axes[0].set_xlabel('Contract Structure')
axes[0].set_ylabel('Churn Rate (%)')
axes[0].set_ylim(0, 60)
axes[0].grid(axis='y', linestyle='--', alpha=0.5)
for p in axes[0].patches:
    axes[0].annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width()/2, p.get_height() + 1.5), 
                    ha='center', va='bottom', fontsize=10, fontweight='bold', color='#333333')

internet_churn['Yes'].plot(kind='bar', ax=axes[1], color=['#f9a825', DANGER_COLOR, MUTED_DARK], edgecolor='black', linewidth=0.5)
axes[1].set_title('Churn Rate (%) by Internet Connection Type', fontweight='bold', pad=12, color=PRIMARY_COLOR)
axes[1].set_xlabel('Internet Technology')
axes[1].set_ylabel('Churn Rate (%)')
axes[1].set_ylim(0, 60)
axes[1].grid(axis='y', linestyle='--', alpha=0.5)
for p in axes[1].patches:
    axes[1].annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width()/2, p.get_height() + 1.5), 
                    ha='center', va='bottom', fontsize=10, fontweight='bold', color='#333333')

plt.suptitle('WHERE IS CHURN LEAKING? TARGET SEGMENTS', fontsize=15, fontweight='bold', y=1.02, color=PRIMARY_COLOR)
plt.tight_layout()
plt.savefig('day34/churn_risk_analysis.png', dpi=150, bbox_inches='tight')
plt.close()

# 5. Plot 3: K-Means Demographic Segmentation
plt.figure(figsize=(10, 6.5))
persona_colors = {
    'The Value Seekers': '#7f8c8d',
    'The Impulsive Budgeters': '#e74c3c',
    'The Steady Conformists': '#f1c40f',
    'The Elite Affluents': '#2c3e50',
    'The Affluent Frugals': '#1abc9c'
}

sns.scatterplot(
    data=df_segments,
    x='Annual Income (k$)',
    y='Spending Score (1-100)',
    hue='Persona',
    palette=persona_colors,
    s=80,
    alpha=0.85,
    edgecolor='black',
    linewidth=0.5
)

plt.title('Customer Persona Distribution Matrix (Income vs. Spending)', fontweight='bold', pad=15, fontsize=13, color=PRIMARY_COLOR)
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.grid(True, linestyle='--', alpha=0.3)
plt.legend(title='Customer Persona Profiles', loc='upper right', frameon=True, facecolor='white', edgecolor='#e0e0e0')

plt.annotate(
    "Key Premium Value Target\n(The Elite Affluents)",
    xy=(86, 82),
    xytext=(55, 95),
    arrowprops=dict(facecolor='black', shrink=0.08, width=1.2, headwidth=7),
    fontweight='bold', fontsize=9.5, bbox=dict(boxstyle='round,pad=0.3', fc='#fdfefe', ec='#2c3e50', alpha=0.9)
)

plt.tight_layout()
plt.savefig('day34/customer_segmentation.png', dpi=150, bbox_inches='tight')
plt.close()

print("All dashboard visualization images successfully saved in day34/.")
