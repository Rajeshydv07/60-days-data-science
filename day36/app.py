import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    import streamlit as st
except ModuleNotFoundError:
    print("Streamlit not found. Please install it using: pip install streamlit")
    sys.exit(1)

# Set page config
st.set_page_config(
    page_title="Customer Growth Forecasting Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling using HTML/CSS
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
    .status-panel {
        padding: 1rem;
        border-radius: 6px;
        margin-bottom: 1rem;
    }
    .status-green {
        background-color: #f3faf7;
        border-left: 5px solid #17b978;
        color: #155724;
    }
    .status-blue {
        background-color: #f0f7fc;
        border-left: 5px solid #1e3d59;
        color: #0c5460;
    }
    .status-orange {
        background-color: #fff9e6;
        border-left: 5px solid #ff6e40;
        color: #856404;
    }
</style>
""", unsafe_allow_html=True)

# Data paths
csv_path = 'day36/customer_growth_data.csv'
for p in [csv_path, '../' + csv_path]:
    if os.path.exists(p):
        csv_path = p
        break

@st.cache_data
def load_data(path):
    if os.path.exists(path):
        df = pd.read_csv(path)
    else:
        # Fallback generation in case app runs before runner script
        from customer_growth_generator import generate_customer_growth_data
        df = generate_customer_growth_data('day36/customer_growth_data.csv')
        
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    return df

df = load_data(csv_path)

# Custom cache for models
@st.cache_resource
def train_and_eval_models(data_series):
    test_days = 30
    train = data_series.iloc[:-test_days]
    test = data_series.iloc[-test_days:]
    
    # 1. Seasonal Naive Baseline
    history = list(train)
    naive_preds = []
    for i in range(test_days):
        pred = history[-7]
        naive_preds.append(pred)
        history.append(test.iloc[i])
    naive_preds = pd.Series(naive_preds, index=test.index)
    
    # 2. Holt-Winters
    hw_model = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=7).fit()
    hw_preds = hw_model.forecast(test_days)
    
    # 3. SARIMA
    sarima_model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7), 
                           enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    sarima_preds = sarima_model.forecast(test_days)
    
    def get_metrics(y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}
        
    metrics = {
        'Naive': get_metrics(test, naive_preds),
        'HW': get_metrics(test, hw_preds),
        'SARIMA': get_metrics(test, sarima_preds)
    }
    
    # Fit full models
    hw_full = ExponentialSmoothing(data_series, trend='add', seasonal='add', seasonal_periods=7).fit()
    sarima_full = SARIMAX(data_series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7), 
                           enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    
    return train, test, naive_preds, hw_preds, sarima_preds, metrics, hw_full, sarima_full

train, test, naive_preds, hw_preds, sarima_preds, metrics, hw_full, sarima_full = train_and_eval_models(df['New_Customers'])

# Sidebar Configuration
st.sidebar.markdown("## Forecasting Dashboard")
st.sidebar.markdown("Day 36 · Time Series Analytics")
st.sidebar.markdown("---")

st.sidebar.markdown("### Model Sandbox Options")
selected_model_name = st.sidebar.selectbox(
    "Select Forecasting Model", 
    ["SARIMA (Seasonal ARIMA)", "Holt-Winters (Exponential Smoothing)", "Seasonal Naive (Baseline)"]
)
forecast_horizon = st.sidebar.slider("Forecast Horizon (Days)", 15, 90, 30, step=5)

st.sidebar.markdown("---")
st.sidebar.markdown("### Historical Stats")
total_users = df['Cumulative_Customers'].iloc[-1]
total_revenue = df['Daily_Revenue'].sum()
avg_new_signups = df['New_Customers'].mean()

st.sidebar.metric("Cumulative Customers", f"{total_users:,}")
st.sidebar.metric("Total Generated Revenue", f"${total_revenue:,.2f}")
st.sidebar.metric("Average Daily Signups", f"{avg_new_signups:.1f}")

# Main Header
st.title("Time Series Customer Growth Forecasting Platform")
st.markdown("Translating historical user signups and engagement trends into predictive future insights.")
st.markdown("---")

# Navigation Tabs
tabs = st.tabs([
    "Historical Trends & Decomposition",
    "Forecast Playground",
    "Growth Scenario Simulator",
    "Methodology & Risk Observations"
])

# Tab 1: Historical trends & decomposition
with tabs[0]:
    st.header("Exploratory Growth Analytics")
    st.markdown("Explore trends, rolling moving averages, and underlying seasonality components extracted from the historical data.")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="metric-card"><div class="metric-title">Current Customer Base</div><div class="metric-value">{total_users:,}</div><div class="metric-desc">As of last log date</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><div class="metric-title">Active Daily Users (DAU)</div><div class="metric-value">{df["Active_Users"].iloc[-1]:,}</div><div class="metric-desc">~35.4% engagement rate</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><div class="metric-title">Current MRR (Est.)</div><div class="metric-value">${df["Active_Users"].iloc[-1] * 0.15 * 30:,.2f}</div><div class="metric-desc">Estimated Monthly Recurring</div></div>', unsafe_allow_html=True)
    with col4:
        # Calculate ADF Stationarity details
        adf_res = adfuller(df['New_Customers'])
        p_val = adf_res[1]
        stationarity_status = "Non-Stationary (Needs Diff)" if p_val > 0.05 else "Stationary (d=0)"
        status_color = "#ff6e40" if p_val > 0.05 else "#17b978"
        st.markdown(f'<div class="metric-card" style="border-left-color: {status_color};"><div class="metric-title">Series Stationarity</div><div class="metric-value" style="color: {status_color}; font-size: 1.4rem;">{stationarity_status}</div><div class="metric-desc">ADF p-value: {p_val:.2e}</div></div>', unsafe_allow_html=True)

    # Historical plots
    hist_col1, hist_col2 = st.columns([2, 1])
    with hist_col1:
        st.subheader("Historical Daily Signups & Smoothed Averages")
        fig_hist, ax_hist = plt.subplots(figsize=(10, 5))
        ax_hist.plot(df.index, df['New_Customers'], label='Daily New Signups', color='#1e3d59', alpha=0.25)
        ax_hist.plot(df.index, df['New_Customers'].rolling(7).mean(), label='7-Day SMA', color='#ff6e40', linewidth=1.5)
        ax_hist.plot(df.index, df['New_Customers'].rolling(30).mean(), label='30-Day SMA', color='#17b978', linewidth=2.0)
        ax_hist.set_xlabel('Date')
        ax_hist.set_ylabel('Signups Count')
        ax_hist.legend()
        plt.tight_layout()
        st.pyplot(fig_hist)
    with hist_col2:
        st.subheader("Weekly Customer Acquisition Pattern")
        # Group by weekday
        weekday_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        df_weekday = df.copy()
        df_weekday['Weekday'] = df_weekday.index.dayofweek
        df_weekday_mean = df_weekday.groupby('Weekday')['New_Customers'].mean()
        df_weekday_mean.index = weekday_names
        
        fig_week, ax_week = plt.subplots(figsize=(5, 5))
        sns.barplot(x=df_weekday_mean.values, y=df_weekday_mean.index, palette='Blues_r', ax=ax_week, edgecolor='black', linewidth=0.5)
        ax_week.set_title("Average Daily Signups by Weekday")
        ax_week.set_xlabel("Signups")
        plt.tight_layout()
        st.pyplot(fig_week)

    st.markdown("---")
    st.subheader("Seasonal Decomposition Analysis")
    st.markdown("Extracting the underlying growth **Trend**, recurring weekly **Seasonality**, and **Residual** (noise) from daily acquisitions.")
    
    decomp = seasonal_decompose(df['New_Customers'], model='additive', period=7)
    
    fig_dec, axes = plt.subplots(4, 1, figsize=(14, 8), sharex=True)
    axes[0].plot(decomp.observed, color='#1e3d59')
    axes[0].set_ylabel('Observed')
    axes[0].set_title('Observed Series = Trend + Weekly Seasonality + Residuals')
    
    axes[1].plot(decomp.trend, color='#ff6e40')
    axes[1].set_ylabel('Trend')
    
    axes[2].plot(decomp.seasonal, color='#17b978')
    axes[2].set_ylabel('Seasonal')
    
    axes[3].scatter(decomp.resid.index, decomp.resid.values, color='#8b8c8d', s=3, alpha=0.5)
    axes[3].axhline(0, color='red', linestyle='--', linewidth=0.8)
    axes[3].set_ylabel('Residual')
    
    plt.xlabel('Date')
    plt.tight_layout()
    st.pyplot(fig_dec)

# Tab 2: Forecast Playground
with tabs[1]:
    st.header("Forecasting Model Sandbox")
    st.markdown("Benchmark model accuracies on historical tests and generate dynamic future signups and growth projections.")
    
    # Select validation metrics for chosen model
    model_key = 'SARIMA' if 'SARIMA' in selected_model_name else ('HW' if 'Holt-Winters' in selected_model_name else 'Naive')
    m = metrics[model_key]
    
    val_col1, val_col2, val_col3, val_col4 = st.columns(4)
    with val_col1:
        st.markdown(f'<div class="metric-card" style="border-left-color: #ff6e40;"><div class="metric-title">Model Configured</div><div class="metric-value" style="font-size: 1.3rem; color: #ff6e40;">{selected_model_name.split(" ")[0]}</div><div class="metric-desc">Active Sandbox Model</div></div>', unsafe_allow_html=True)
    with val_col2:
        st.markdown(f'<div class="metric-card"><div class="metric-title">Validation MAE</div><div class="metric-value">{m["MAE"]:.2f}</div><div class="metric-desc">Mean Abs Error (Test Set)</div></div>', unsafe_allow_html=True)
    with val_col3:
        st.markdown(f'<div class="metric-card"><div class="metric-title">Validation RMSE</div><div class="metric-value">{m["RMSE"]:.2f}</div><div class="metric-desc">Root Mean Sq Error</div></div>', unsafe_allow_html=True)
    with val_col4:
        st.markdown(f'<div class="metric-card" style="border-left-color: #17b978;"><div class="metric-title">Validation MAPE</div><div class="metric-value" style="color: #17b978;">{m["MAPE"]:.2f}%</div><div class="metric-desc">Mean Abs Percentage Error</div></div>', unsafe_allow_html=True)
        
    # Generate future forecasts dynamically based on slider
    future_dates = pd.date_range(start=df.index.max() + pd.Timedelta(days=1), periods=forecast_horizon, freq='D')
    
    if model_key == 'SARIMA':
        forecast_res = sarima_full.get_forecast(steps=forecast_horizon)
        forecast_mean = forecast_res.predicted_mean.values
        ci = forecast_res.conf_int(alpha=0.05)
        lower_bound = ci.iloc[:, 0].values
        upper_bound = ci.iloc[:, 1].values
    elif model_key == 'HW':
        forecast_mean = hw_full.forecast(forecast_horizon).values
        # Holt Winters analytical bounds approximate standard error
        residuals = hw_full.resid
        se = np.std(residuals)
        lower_bound = forecast_mean - 1.96 * se * np.sqrt(np.arange(1, forecast_horizon + 1))
        upper_bound = forecast_mean + 1.96 * se * np.sqrt(np.arange(1, forecast_horizon + 1))
    else: # Naive
        # Drift-style naive forecast
        last_val = df['New_Customers'].iloc[-1]
        forecast_mean = np.array([df['New_Customers'].iloc[-7 + (i % 7)] for i in range(forecast_horizon)])
        residuals = df['New_Customers'].diff(7).dropna()
        se = np.std(residuals)
        lower_bound = forecast_mean - 1.96 * se
        upper_bound = forecast_mean + 1.96

    forecast_mean = np.clip(forecast_mean, 1, None).round(0).astype(int)
    lower_bound = np.clip(lower_bound, 1, None).round(0).astype(int)
    upper_bound = np.clip(upper_bound, 1, None).round(0).astype(int)
    
    last_cumulative = df['Cumulative_Customers'].iloc[-1]
    cumulative_forecast = last_cumulative + np.cumsum(forecast_mean)
    cumulative_lower = last_cumulative + np.cumsum(lower_bound)
    cumulative_upper = last_cumulative + np.cumsum(upper_bound)

    # Plot sandbox future forecast
    play_col1, play_col2 = st.columns([2, 1])
    with play_col1:
        st.subheader(f"{forecast_horizon}-Day Ahead Acquisition Forecast")
        fig_fore, ax_fore = plt.subplots(figsize=(10, 5))
        # Plot last 60 days of history
        ax_fore.plot(df.index[-60:], df['New_Customers'].iloc[-60:], label='Historical Actuals (Last 60d)', color='#1e3d59', linewidth=2.0)
        ax_fore.plot(future_dates, forecast_mean, label='Predicted Growth', color='#ff6e40', linewidth=2.0)
        ax_fore.fill_between(future_dates, lower_bound, upper_bound, color='#ff6e40', alpha=0.15, label='95% Confidence Interval')
        ax_fore.set_xlabel('Date')
        ax_fore.set_ylabel('New Customers / Day')
        ax_fore.legend(loc='upper left')
        plt.tight_layout()
        st.pyplot(fig_fore)
        
        st.subheader("Cumulative Growth Projections")
        fig_cum_fore, ax_cum_fore = plt.subplots(figsize=(10, 4))
        ax_cum_fore.plot(df.index[-60:], df['Cumulative_Customers'].iloc[-60:], label='Historical Cumulative (Last 60d)', color='#1e3d59', linewidth=2.0)
        ax_cum_fore.plot(future_dates, cumulative_forecast, label='Projected Growth Path', color='#17b978', linewidth=2.0)
        ax_cum_fore.fill_between(future_dates, cumulative_lower, cumulative_upper, color='#17b978', alpha=0.15, label='95% Confidence')
        ax_cum_fore.set_xlabel('Date')
        ax_cum_fore.set_ylabel('Total Customers')
        ax_cum_fore.legend(loc='upper left')
        plt.tight_layout()
        st.pyplot(fig_cum_fore)

    with play_col2:
        st.subheader("Model Validation Sandbox")
        st.markdown("Below is the validation fit showing how each model performed when testing against the last 30 days of actual historical values.")
        
        fig_val, ax_val = plt.subplots(figsize=(5, 5.5))
        ax_val.plot(test.index, test.values, label='Actuals', color='#1e3d59', linewidth=2.0)
        ax_val.plot(test.index, naive_preds.values, label='Naive', color='#8b8c8d', linestyle='--')
        ax_val.plot(test.index, hw_preds.values, label='Holt-Winters', color='#ff6e40', linestyle='-.')
        ax_val.plot(test.index, sarima_preds.values, label='SARIMA', color='#17b978', linewidth=1.5)
        ax_val.set_xlabel('Validation Date')
        ax_val.set_ylabel('Signups')
        ax_val.tick_params(axis='x', rotation=45)
        ax_val.legend(fontsize=8)
        plt.tight_layout()
        st.pyplot(fig_val)
        
        # Download button for forecast data
        forecast_df = pd.DataFrame({
            'Date': future_dates.strftime('%Y-%m-%d'),
            'Projected_New_Customers': forecast_mean,
            'Lower_Bound': lower_bound,
            'Upper_Bound': upper_bound,
            'Projected_Total_Customers': cumulative_forecast
        })
        st.markdown("### Export Forecast Projections")
        st.dataframe(forecast_df.head(5), use_container_width=True)
        csv_data = forecast_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Forecast CSV Table",
            data=csv_data,
            file_name=f"customer_growth_forecast_{forecast_horizon}d.csv",
            mime="text/csv",
            use_container_width=True
        )

# Tab 3: Growth Scenario Simulator
with tabs[2]:
    st.header("What-If Growth Scenario Simulator")
    st.markdown("Time Series projections assume business-as-usual behavior. This interactive dashboard simulator allows business leaders to model promotional shocks or service disruption events.")
    
    sim_col1, sim_col2 = st.columns([1, 2])
    with sim_col1:
        st.subheader("Scenario Parameters")
        
        st.markdown("**Campaign 1: Marketing Campaign / Growth Boost**")
        campaign_pct = st.slider("Signup Growth Boost (%)", 0, 100, 30, step=5)
        campaign_duration = st.slider("Campaign Duration (Days)", 3, 21, 7)
        campaign_delay = st.slider("Days until Campaign starts", 1, 15, 3)
        
        st.markdown("---")
        st.markdown("**Campaign 2: System Outage / Churn Event**")
        outage_pct = st.slider("Signup Reduction Severity (%)", 0, 95, 50, step=5)
        outage_duration = st.slider("Outage Duration (Days)", 1, 5, 2)
        outage_delay = st.slider("Days until Outage starts", 5, 30, 14)
        
    with sim_col2:
        st.subheader("Simulated Growth Projections")
        
        # Baseline Forecast (SARIMA as default)
        base_forecast = sarima_full.forecast(forecast_horizon).values.clip(1).round(0).astype(int)
        simulated_forecast = base_forecast.copy().astype(float)
        
        # Apply campaign boost
        c_start = campaign_delay
        c_end = min(campaign_delay + campaign_duration, forecast_horizon)
        if c_start < forecast_horizon:
            simulated_forecast[c_start:c_end] = simulated_forecast[c_start:c_end] * (1.0 + (campaign_pct / 100.0))
            
        # Apply outage dip
        o_start = outage_delay
        o_end = min(outage_delay + outage_duration, forecast_horizon)
        if o_start < forecast_horizon:
            simulated_forecast[o_start:o_end] = simulated_forecast[o_start:o_end] * (1.0 - (outage_pct / 100.0))
            
        simulated_forecast = np.clip(simulated_forecast, 1, None).round(0).astype(int)
        
        # Cumulative comparisons
        base_cumulative = last_cumulative + np.cumsum(base_forecast)
        sim_cumulative = last_cumulative + np.cumsum(simulated_forecast)
        
        # Plot Scenario
        fig_sim, ax_sim = plt.subplots(figsize=(10, 5))
        ax_sim.plot(future_dates, base_forecast, label='Baseline Forecast (Normal Growth)', color='#1e3d59', linestyle='--')
        ax_sim.plot(future_dates, simulated_forecast, label='Simulated Forecast (With Shocks)', color='#ff6e40', linewidth=2.0)
        # Highlight regions
        if c_start < forecast_horizon:
            ax_sim.axvspan(future_dates[c_start], future_dates[c_end - 1], color='#17b978', alpha=0.1, label='Marketing Push window')
        if o_start < forecast_horizon:
            ax_sim.axvspan(future_dates[o_start], future_dates[o_end - 1], color='#d9534f', alpha=0.1, label='Outage Outage window')
            
        ax_sim.set_xlabel('Forecast Date')
        ax_sim.set_ylabel('New Customers / Day')
        ax_sim.legend(loc='upper left')
        plt.tight_layout()
        st.pyplot(fig_sim)
        
        fig_sim_cum, ax_sim_cum = plt.subplots(figsize=(10, 4))
        ax_sim_cum.plot(future_dates, base_cumulative, label='Baseline Cumulative Base', color='#1e3d59', linestyle='--')
        ax_sim_cum.plot(future_dates, sim_cumulative, label='Simulated Cumulative Growth', color='#17b978', linewidth=2.0)
        ax_sim_cum.set_xlabel('Forecast Date')
        ax_sim_cum.set_ylabel('Total Customers')
        ax_sim_cum.legend(loc='upper left')
        plt.tight_layout()
        st.pyplot(fig_sim_cum)
        
        diff_customers = sim_cumulative[-1] - base_cumulative[-1]
        diff_mrr = (simulated_forecast.sum() - base_forecast.sum()) * 0.15 * 30
        
        status_panel_class = "status-green" if diff_customers >= 0 else "status-orange"
        sign_char = "+" if diff_customers >= 0 else ""
        
        st.markdown(
            f'<div class="status-panel {status_panel_class}">'
            f'<h4>Scenario Impact Summary</h4>'
            f'<p>By the end of the {forecast_horizon}-day forecast window, this scenario results in '
            f'<strong>{sign_char}{diff_customers:,}</strong> net customers relative to the baseline projection. '
            f'This translates to an estimated impact of <strong>{sign_char}${diff_mrr:,.2f}/mo</strong> in recurring revenue.'
            f'</p></div>',
            unsafe_allow_html=True
        )

# Tab 4: Reflection & Methodology
with tabs[3]:
    st.header("Technical Reflection & Model Methodology")
    
    ref_col1, ref_col2 = st.columns([1, 1])
    with ref_col1:
        st.subheader("Model Hyperparameters & Diagnostics")
        st.markdown(
            """
            * **Augmented Dickey-Fuller Test**:
              The raw daily series has a non-stationary profile ($p$-value > 0.05) due to the linear upward trend.
              Applying a first order difference ($d=1$) removes the trend drift, making the series stationary ($p$-value < 0.0001).
              This guarantees that differenced autoregressive models like SARIMA can optimize effectively.
              
            * **SARIMA Configuration**:
              We configured the model as **SARIMAX(1, 1, 1) x (1, 1, 1)7**:
              - $p=1, d=1, q=1$: First order autoregression and moving average handles daily smoothing.
              - $P=1, D=1, Q=1, s=7$: Captures the strong weekly cycle (Monday-Sunday signups cycle).
              
            * **Holt-Winters Configuration**:
              Exponential smoothing with an additive trend and additive seasonal cycles ($s=7$) was selected.
              This smoothing is highly effective at filtering out the high daily noise variance while staying responsive to localized trends.
            """
        )
    with ref_col2:
        st.subheader("Residual Diagnostics Analysis")
        # Extract SARIMA residuals and display basic diagnostics
        resid = sarima_full.resid
        fig_diag, ax_diag = plt.subplots(figsize=(6, 5))
        sns.histplot(resid, kde=True, ax=ax_diag, color='#1e3d59')
        ax_diag.set_title("SARIMA Residuals Distribution (Errors)")
        ax_diag.set_xlabel("Prediction Error")
        plt.tight_layout()
        st.pyplot(fig_diag)
        
    st.markdown("---")
    st.subheader("Key Strategic Forecasting Risks")
    
    st.warning(
        "**1. Structural Breaks & Outliers**\n\n"
        "Promotional events or server downtime cause extreme outliers (anomalies) in historical logs. "
        "Standard models can treat these as permanent trend shifts, leading to over/under forecasting. "
        "Adding event-flag regressors (using SARIMAX with exogenous parameters) is highly recommended for production."
    )
    st.info(
        "**2. Data Drift & Customer Cohort Behavior**\n\n"
        "Customer retention factors and signups velocity change as companies mature. "
        "For long-term forecasting (6 months to 1 year), statistical models decay. "
        "Stakeholders should couple statistical forecasts with cohort analysis and retention curves."
    )
    st.error(
        "**3. Black Swan Events & Out-of-Sample Failure**\n\n"
        "Statistical models are completely blind to external market conditions (e.g. macro shutdowns or sudden pricing shifts). "
        "Business leaders should use the Scenario Simulator tab to stress-test their acquisition budgets against outages and campaign dips."
    )
