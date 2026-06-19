import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
import subprocess
import sys
import os

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

def md(src):
    return new_markdown_cell(src)

def code(src):
    return new_code_cell(src)

cells = []

# Cell 1: Header
cells.append(md("""\
# Day 37 - Predicting Future Revenue with Time Series Models
### 60 Days Data Science | Phase: Business Forecasting

**Date:** 19 June 2026  
**Name:** Rajesh Yadav

---
"""))


# Cell 2: Step 1 Imports Header
cells.append(md("## Step 1 - Imports"))

# Cell 3: Step 1 Imports Code
cells.append(code("""\
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Make plots look professional and clean
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 10

print("numpy version:", np.__version__)
print("pandas version:", pd.__version__)
"""))

# Cell 4: Step 2 Simulate Data Header
cells.append(md("## Step 2 - Simulating Revenue Data"))


# Cell 5: Step 2 Simulate Data Code
cells.append(code("""\
np.random.seed(42)
dates = pd.date_range(start='2023-01-01', end='2025-12-31', freq='D')
n_days = len(dates)

# 1. Base revenue + Trend
base_rev = 1500
trend = 2.2 * np.arange(n_days)

# 2. Weekly Seasonality (0=Monday, 6=Sunday)
# B2B SaaS: high activity mid-week, drops significantly on weekends
weekly_pattern = {0: 120, 1: 250, 2: 300, 3: 280, 4: 150, 5: -450, 6: -550}
weekly_seasonality = np.array([weekly_pattern[d.dayofweek] for d in dates])

# 3. Yearly Seasonality (Q4 holiday spikes + seasonal wave)
yearly_wave = 300 * np.sin(2 * np.pi * dates.dayofyear / 365.25)
holiday_boost = np.zeros(n_days)
for i, date in enumerate(dates):
    if date.month == 11 and date.day >= 20: # Black Friday / Cyber Monday promo boost
        holiday_boost[i] = 800 + np.random.normal(0, 100)
    elif date.month == 12 and date.day <= 25: # December holiday rush
        holiday_boost[i] = 1200 + np.random.normal(0, 150)

# 4. Outliers/Shocks (Outages & Promo Spikes)
shocks = np.zeros(n_days)
# Server outages (sharp revenue drop)
outage_days = np.random.choice(n_days, size=5, replace=False)
shocks[outage_days] = -1200
# Successful promo campaigns (sharp revenue spike)
promo_days = np.random.choice(n_days, size=5, replace=False)
shocks[promo_days] = 2000

# 5. Gaussian Noise
noise = np.random.normal(0, 180, n_days)

# Combine components into daily revenue
revenue = base_rev + trend + weekly_seasonality + yearly_wave + holiday_boost + shocks + noise
# Clip negative revenue values to make them realistic (min $200 daily revenue)
revenue = np.clip(revenue, 200, None)

df = pd.DataFrame({'Date': dates, 'Revenue': revenue})
df.set_index('Date', inplace=True)

print(f"Dataset generated. Shape: {df.shape}")
print(df.head())
"""))

# Cell 6: Step 3 EDA Header
cells.append(md("## Step 3 - Exploratory Data Analysis (EDA)"))


# Cell 7: Step 3 EDA Code
cells.append(code("""\
df['7d_MA'] = df['Revenue'].rolling(window=7, min_periods=1).mean()
df['30d_MA'] = df['Revenue'].rolling(window=30, min_periods=1).mean()

plt.figure(figsize=(14, 6))
plt.plot(df.index, df['Revenue'], label='Daily Revenue', color='lightskyblue', alpha=0.5, linewidth=1)
plt.plot(df.index, df['7d_MA'], label='7-Day Moving Average', color='royalblue', linewidth=1.5)
plt.plot(df.index, df['30d_MA'], label='30-Day Moving Average', color='darkblue', linewidth=2)
plt.title('Daily Business Revenue & Rolling Trends (2023-2025)', fontsize=14, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Revenue ($)', fontsize=12)
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
"""))

# Cell 8: Step 3 EDA Weekly Boxplot Code
cells.append(code("""\
# Check day of week pattern
df['DayOfWeek'] = df.index.day_name()
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

plt.figure(figsize=(10, 5))
sns.boxplot(x='DayOfWeek', y='Revenue', data=df, order=day_order, palette='Set2')
plt.title('Daily Revenue Distribution by Day of the Week', fontsize=14, fontweight='bold')
plt.xlabel('Day of the Week', fontsize=12)
plt.ylabel('Revenue ($)', fontsize=12)
plt.tight_layout()
plt.show()
"""))

# Cell 9: Step 4 Stationarity Header
cells.append(md("## Step 4 - Stationarity Check (ADF Test)"))


# Cell 10: Step 4 Stationarity Code
cells.append(code("""\
def run_adf_test(series, name):
    result = adfuller(series.dropna())
    print(f"ADF Test Results for: {name}")
    print(f"ADF Statistic: {result[0]:.4f}")
    print(f"p-value: {result[1]:.4e}")
    print(f"Lags Used: {result[2]}")
    print(f"Observations: {result[3]}")
    print("Critical Values:")
    for key, value in result[4].items():
        print(f"   {key}: {value:.4f}")
    if result[1] <= 0.05:
        print("Conclusion: Reject H0. The series is STATIONARY.\\n")
    else:
        print("Conclusion: Fail to reject H0. The series is NON-STATIONARY.\\n")

run_adf_test(df['Revenue'], 'Raw Revenue')
"""))

# Cell 11: Step 4 Stationarity Diff Header
cells.append(md("## Step 5 - Stationarity Check on Differenced Series"))


# Cell 12: Step 4 Stationarity Diff Code
cells.append(code("""\
df['Revenue_Diff'] = df['Revenue'].diff()
run_adf_test(df['Revenue_Diff'], '1st Differenced Revenue')
"""))

# Cell 13: Step 5 Train Test Split Header
cells.append(md("## Step 6 - Train-Test Split & Baseline Model"))


# Cell 14: Step 5 Train Test Split Code
cells.append(code("""\
# Split into Train/Test
train = df.iloc[:-31].copy()
test = df.iloc[-31:].copy()

print(f"Train size: {len(train)} days (up to {train.index.max().date()})")
print(f"Test size:  {len(test)} days (from {test.index.min().date()} to {test.index.max().date()})")

# Seasonal Naive baseline
test['Baseline_Forecast'] = df['Revenue'].shift(7).loc[test.index]
print(test[['Revenue', 'Baseline_Forecast']].head())
"""))

# Cell 15: Step 6 SARIMAX Header
cells.append(md("## Step 7 - SARIMAX Model"))


# Cell 16: Step 6 SARIMAX Code
cells.append(code("""\
# Fit SARIMAX on train data
model = SARIMAX(train['Revenue'],
                order=(1, 1, 1),
                seasonal_order=(1, 1, 1, 7),
                enforce_stationarity=False,
                enforce_invertibility=False)
model_fit = model.fit(disp=False)

print(model_fit.summary())
"""))

# Cell 17: Step 7 Test Set Comparison Header
cells.append(md("## Step 8 - Predictions vs Test Set"))


# Cell 18: Step 7 Test Set Comparison Code
cells.append(code("""\
# Get forecast and confidence intervals
forecast_res = model_fit.get_forecast(steps=31)
test['SARIMAX_Forecast'] = forecast_res.predicted_mean
ci = forecast_res.conf_int()

plt.figure(figsize=(14, 6))
plt.plot(train.index[-60:], train['Revenue'].iloc[-60:], label='Historical Train Revenue (Last 60d)', color='gray', alpha=0.5)
plt.plot(test.index, test['Revenue'], label='Actual Test Revenue', color='black', marker='o', markersize=4)
plt.plot(test.index, test['Baseline_Forecast'], label='Seasonal Naive (7d shift)', color='orange', linestyle='--')
plt.plot(test.index, test['SARIMAX_Forecast'], label='SARIMAX Forecast', color='royalblue', linewidth=2)
plt.fill_between(test.index, ci.iloc[:, 0], ci.iloc[:, 1], color='royalblue', alpha=0.15, label='95% Confidence Interval')

plt.title('Validation Comparison: December 2025 Revenue', fontsize=14, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Revenue ($)', fontsize=12)
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
"""))

# Cell 19: Step 8 Metrics Header
cells.append(md("## Step 9 - Model Evaluation Metrics"))


# Cell 20: Step 8 Metrics Code
cells.append(code("""\
def calculate_metrics(actual, predicted, model_name):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    return pd.Series({'MAE': mae, 'RMSE': rmse, 'MAPE (%)': mape}, name=model_name)

metrics_df = pd.DataFrame([
    calculate_metrics(test['Revenue'], test['Baseline_Forecast'], 'Seasonal Naive Baseline'),
    calculate_metrics(test['Revenue'], test['SARIMAX_Forecast'], 'SARIMAX Model')
])

print("Validation Set Model Comparison:")
print(metrics_df.round(2))
"""))

# Cell 21: Step 9 Future Forecast Header
cells.append(md("## Step 10 - Future Forecasting (January 2026)"))


# Cell 22: Step 9 Future Forecast Code
cells.append(code("""\
# Retrain on full history
final_model = SARIMAX(df['Revenue'],
                      order=(1, 1, 1),
                      seasonal_order=(1, 1, 1, 7),
                      enforce_stationarity=False,
                      enforce_invertibility=False)
final_model_fit = final_model.fit(disp=False)

# Forecast 30 days ahead
forecast_steps = 30
future_forecast = final_model_fit.get_forecast(steps=forecast_steps)
future_dates = pd.date_range(start='2026-01-01', periods=forecast_steps, freq='D')

forecast_df = pd.DataFrame({
    'Forecast': future_forecast.predicted_mean.values,
}, index=future_dates)

future_ci = future_forecast.conf_int()
forecast_df['Lower_CI'] = future_ci.iloc[:, 0].values
forecast_df['Upper_CI'] = future_ci.iloc[:, 1].values

# Plot the historical series (last 90 days) and the 30-day forecast
plt.figure(figsize=(14, 6))
plt.plot(df.index[-90:], df['Revenue'].iloc[-90:], label='Historical Revenue (Last 90d)', color='black')
plt.plot(forecast_df.index, forecast_df['Forecast'], label='Future Revenue Forecast (Jan 2026)', color='forestgreen', linewidth=2.5, marker='o', markersize=4)
plt.fill_between(forecast_df.index, forecast_df['Lower_CI'], forecast_df['Upper_CI'], color='forestgreen', alpha=0.15, label='95% Confidence Interval')

plt.title('30-Day Future Revenue Forecast (January 2026)', fontsize=14, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Revenue ($)', fontsize=12)
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
"""))

# Cell 23: Step 9 Forecast Insights Code
cells.append(code("""\
# Calculate total forecasted revenue and print insights
print("First 7 Days of January 2026 Forecast:")
print(forecast_df.head(7).round(2))

total_forecast = forecast_df['Forecast'].sum()
lower_total = forecast_df['Lower_CI'].sum()
upper_total = forecast_df['Upper_CI'].sum()

print("-" * 50)
print(f"Total Forecasted Revenue for January 2026: ${total_forecast:,.2f}")
print(f"95% Confidence Range: ${lower_total:,.2f} to ${upper_total:,.2f}")
print("-" * 50)
"""))

# Cell 24: Step 10 Reflection Header
cells.append(md("""\
## Step 11 - Key Takeaways
- SARIMAX(1, 1, 1)x(1, 1, 1)7 captures trend and weekly cycles.
- Under-forecasting results in staffing/capacity limits, while over-forecasting drains capital runway.
"""))


nb = new_notebook(cells=cells)
nb.metadata['kernelspec'] = {
    'display_name': 'Python 3',
    'language': 'python',
    'name': 'python3'
}
nb.metadata['language_info'] = {'name': 'python', 'version': '3.10.0'}

NB_PATH = 'day37/day37_revenue_forecasting.ipynb'
os.makedirs(os.path.dirname(NB_PATH), exist_ok=True)

with open(NB_PATH, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print(f"Saved notebook: {NB_PATH}")
print("Running execution via nbconvert...")

result = subprocess.run(
    [sys.executable, '-m', 'nbconvert', '--to', 'notebook',
     '--execute', '--inplace',
     '--ExecutePreprocessor.timeout=300',
     '--ExecutePreprocessor.kernel_name=python3',
     NB_PATH],
    capture_output=True, text=True
)

if result.returncode == 0:
    print("Execution complete! Notebook outputs have been saved inline.")
else:
    print("Error executing notebook:")
    print(result.stderr[-3000:])
    sys.exit(1)
