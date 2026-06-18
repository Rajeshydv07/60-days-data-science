import os
import pandas as pd
import numpy as np

def generate_customer_growth_data(output_path='day36/customer_growth_data.csv'):
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 3 years of daily data (from 2023-06-18 to 2026-06-18)
    dates = pd.date_range(start='2023-06-18', end='2026-06-18', freq='D')
    n_days = len(dates)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # 1. Base customer signups and growth trend
    base_signups = 35.0
    # Linear trend: starts at 0 and adds up to 120 signups/day by the end of 3 years
    trend = np.linspace(0, 120, n_days)
    
    # 2. Weekly Seasonality (B2B SaaS style: busy weekdays, quiet weekends)
    # 0 = Monday, 6 = Sunday
    weekly_pattern = {
        0: 12.0,   # Monday
        1: 22.0,   # Tuesday
        2: 25.0,   # Wednesday
        3: 20.0,   # Thursday
        4: 8.0,    # Friday
        5: -22.0,  # Saturday
        6: -18.0   # Sunday
    }
    weekly_seasonality = np.array([weekly_pattern[d.weekday()] for d in dates])
    
    # 3. Yearly Seasonality (Q4 holiday push, Q1 slow start)
    # Cosine wave peaking around late November (day 325 of year)
    day_of_year = dates.dayofyear.to_numpy()
    yearly_seasonality = 28.0 * np.cos(2 * np.pi * (day_of_year - 325) / 365.25)
    
    # 4. Noise
    noise = np.random.normal(0, 8.0, n_days)
    
    # Combine signals
    new_customers = base_signups + trend + weekly_seasonality + yearly_seasonality + noise
    
    # 5. Inject Special Events (Outliers) to show model resilience
    # Event A: Major Product Launch / Marketing Campaign (Sept 2024)
    # Dates: 2024-09-10 to 2024-09-14
    launch_mask = (dates >= '2024-09-10') & (dates <= '2024-09-14')
    new_customers[launch_mask] += np.random.uniform(70, 110, size=launch_mask.sum())
    
    # Event B: Major Server Outage / DNS Failure (March 2025)
    # Dates: 2025-03-24 to 2025-03-25
    outage_mask = (dates >= '2025-03-24') & (dates <= '2025-03-25')
    new_customers[outage_mask] = np.random.uniform(2, 6, size=outage_mask.sum())
    
    # Event C: New Year Promotion (Dec 28, 2025 to Jan 3, 2026)
    promo_mask = (dates >= '2025-12-28') & (dates <= '2026-01-03')
    new_customers[promo_mask] += np.random.uniform(50, 90, size=promo_mask.sum())

    # Ensure integer values and clip to positive numbers
    new_customers = np.clip(new_customers, 1, None).astype(int)
    
    # Calculate cumulative metrics
    cumulative_customers = np.cumsum(new_customers)
    
    # Generate secondary metrics: Active Users & Daily Revenue
    # Daily Active Users (DAU) has a weekly cycle + grows with cumulative customers
    # Retention factor decays slightly as base grows, simulating churn/inactivity
    retention_factor = 0.45 - np.linspace(0, 0.1, n_days) # starts at 45% of cumulative, decays to 35%
    base_dau = cumulative_customers * retention_factor
    # Add weekly active user spikes (more active mid-week)
    dau_weekly_pattern = {0: 1.0, 1: 1.05, 2: 1.08, 3: 1.04, 4: 0.95, 5: 0.75, 6: 0.82}
    dau_multiplier = np.array([dau_weekly_pattern[d.weekday()] for d in dates])
    active_users = (base_dau * dau_multiplier + np.random.normal(0, 50, n_days)).clip(10).astype(int)
    
    # Daily Revenue (USD): Base fee per active user + new customer acquisition spend
    # Simulates $0.15 recurring revenue per active user per day, plus $5.0 signup fee
    daily_revenue = (active_users * 0.15 + new_customers * 5.0 + np.random.normal(0, 15.0, n_days)).clip(5.0).round(2)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Date': dates.strftime('%Y-%m-%d'),
        'New_Customers': new_customers,
        'Cumulative_Customers': cumulative_customers,
        'Active_Users': active_users,
        'Daily_Revenue': daily_revenue
    })
    
    # Save CSV
    df.to_csv(output_path, index=False)
    print(f"Dataset generated at '{output_path}' with {len(df)} rows.")
    print(f"Date range: {df['Date'].iloc[0]} to {df['Date'].iloc[-1]}")
    print(f"Final cumulative customers: {df['Cumulative_Customers'].iloc[-1]}")
    return df

if __name__ == '__main__':
    generate_customer_growth_data()
