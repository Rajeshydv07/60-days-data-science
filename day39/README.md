# Day 39: KPI Monitoring System for Executives

## Project Overview
This project is part of my **60 Days of Data Science** portfolio. It implements a student-level, beginner-friendly **KPI (Key Performance Indicator) Monitoring System** designed for business executives. Executives need to monitor the pulse of a company's operations without getting lost in complicated statistical modeling or raw logs. 

Using transaction-level data, this system aggregates individual purchase records to calculate high-level performance metrics, creates clean visual explanations of business trends, and provides actionable recommendations.

---

## Dataset Description
The analysis is based on `business_kpi_data.csv`, a synthetic dataset of **1,500 records** spanning the calendar year 2025. It simulates a retail environment with the following fields:
* **Customer_ID**: Unique identifier for individual customers.
* **Date**: Purchase timestamp.
* **Revenue**: Order value in USD.
* **Orders**: Product quantity bought in that transaction.
* **Region**: The geographic zone of the sale (North, South, East, West).
* **Category**: The product category (Electronics, Apparel, Home & Kitchen, Books, Beauty).
* **Retained**: Customer loyalty status (Yes/No).

---

## Executive KPIs Calculated
The system computes the following primary metrics:

1. **Total Revenue**: The total monetary volume generated from all sales.
   * *Formula*: $\sum \text{Revenue}$
   * *Value*: **$252,018.05**
2. **Active Customers**: The count of unique buyers.
   * *Formula*: $\text{Count of Unique Customer\_IDs}$
   * *Value*: **339**
3. **Total Orders (Items Purchased)**: The total volume of items sold.
   * *Formula*: $\sum \text{Orders}$
   * *Value*: **2,613**
4. **Customer Retention Rate**: The percentage of unique customers who are marked as retained.
   * *Formula*: $\left( \frac{\text{Unique Retained Customers}}{\text{Total Unique Customers}} \right) \times 100$
   * *Value*: **71.09%**
5. **Customer Churn Rate**: The percentage of unique customers who have stopped purchasing.
   * *Formula*: $100\% - \text{Retention Rate}$
   * *Value*: **28.91%**
6. **Average Order Value (AOV)**: The average financial value of a transaction.
   * *Formula*: $\frac{\text{Total Revenue}}{\text{Total Transactions (Rows)}}$
   * *Value*: **$168.01** per transaction
7. **Average Revenue Per Customer (ARPU)**: The average amount of money spent per customer.
   * *Formula*: $\frac{\text{Total Revenue}}{\text{Active Customers}}$
   * *Value*: **$743.42**

---

## Visualizations
The system outputs four high-resolution visualizations to support the executive summary (saved in the project folder):

1. **Revenue Trend Over Time (`revenue_trend_over_time.png`)**:
   * A line plot demonstrating monthly revenue over 2025. It reveals strong sales spikes in Q4 (specifically October, November, and December).
2. **Revenue by Region (`revenue_by_region.png`)**:
   * A vertical bar chart indicating regional performance. The North region is the dominant driver with **$91,891.39** in sales, while the West is the lowest with **$46,994.17**.
3. **Retention vs Churn Pie Chart (`retention_vs_churn.png`)**:
   * A clean pie chart showing that **71.1%** of unique customers are active (retained) vs **28.9%** inactive (churned).
4. **Top Categories by Revenue (`revenue_by_category.png`)**:
   * A horizontal bar chart demonstrating revenue by category. Electronics is the clear leader with **$154,580.12**, followed by Home & Kitchen and Apparel.

---

## Executive Business Insights
Based on the metrics and visual analysis, we offer the following key insights for the leadership team:

1. **Pronounced Q4 Seasonality**: Monthly revenue peaks in Q4, with October generating **$42,627.85** (representing a ~140% increase compared to the early Q1 baseline of ~$11.6k). This seasonal pattern indicates a heavy dependency on holiday promotional events and shopping habits.
2. **Northern Sales Concentration**: The **North** region generates **$91,891.39** of the total $252k revenue, representing over 36% of sales. The East and West regions together make up less than 40%, indicating potential under-penetration in these areas.
3. **Electronics is the Core Anchor**: **Electronics** represents **$154,580.12** (more than 61%) of total company revenue. Other categories like Books and Beauty contribute very little ($8.7k and $10.5k respectively).
4. **Healthy Customer Base Stability**: A **71.09% retention rate** is strong for retail and shows solid loyalty. However, the **28.91% churn** represents a leakage of nearly 3 out of every 10 customers that should be optimized.
5. **High Average Order Value**: The Average Order Value of **$168.01** shows strong average transaction values, indicating that shoppers are either purchasing multiple items or focusing on high-priced products (e.g. Electronics).
6. **High Customer Lifetime Value**: The Average Revenue Per Customer is **$743.42**, which is more than four times the AOV. This indicates that customers are return buyers who buy multiple times throughout the year.

---

## Technologies Used
* **Pandas**: For reading, sorting, and aggregating transaction-level records.
* **NumPy**: For array logic and statistical allocations.
* **Matplotlib**: For structure and layout controls of the visualization charts.
* **Seaborn**: For advanced color palettes, formatting, and high-level plot interfaces.
