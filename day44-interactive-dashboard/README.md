# Day 44: Interactive Customer Intelligence Dashboard

This project contains a comprehensive, interactive business intelligence dashboard built with Streamlit and Plotly. It is designed to provide actionable insights into customer behavior, retention, churn risk, and overall business performance in real time.

## Dashboard Architecture

The dashboard is built on a simple, scalable architecture:

1. **Frontend / Presentation Layer**: Built with **Streamlit**. It provides the UI components including the sidebar for filters, KPI metric cards, and layout containers for charts. Streamlit's reactivity model ensures that any change in the filters immediately updates all affected visualizations without a full page reload.
2. **Data Layer**: Powered by **Pandas** and **NumPy**. Data is either generated dynamically using a built-in simulation function or loaded via a user-uploaded CSV. The data is cached using `@st.cache_data` to ensure smooth performance across reruns.
3. **Visualization Layer**: Uses **Plotly Graph Objects** and **Plotly Express** to render interactive charts. Plotly provides highly customizable, interactive (hover, zoom, pan) charts that are embedded seamlessly into Streamlit.
4. **Analytics Logic**: The application uses filtering masks and aggregation functions to compute real-time metrics (like churn rate or average customer lifetime value) based on the user's current sidebar selections.

## Features

- **Real-Time KPIs**: High-level metrics for Total Customers, Churn Rate, Average Monthly Revenue, and High-Value Customer Count.
- **Customer Segmentation Analysis**: Visualizes the distribution of customers across different segments (e.g., Premium, Standard, Basic).
- **Churn Analytics**: Breaks down churn risk by different customer demographics and contract types.
- **Revenue Modeling**: Shows the relationship between customer tenure and total revenue, identifying the most profitable customer cohorts.
- **Dynamic Filtering**: A sidebar that allows stakeholders to filter data by Age Range, Customer Segment, and Churn Status, instantly updating the entire dashboard.
- **Built-in Mock Data**: The app automatically generates realistic customer intelligence data if a custom dataset is not uploaded, allowing for immediate demonstration and testing.

## Setup and Usage

### Prerequisites

Ensure you have Python installed, then install the required dependencies:

```bash
pip install -r requirements.txt
```

### Running the Dashboard

Navigate to this directory and start the Streamlit server:

```bash
cd day44-interactive-dashboard
streamlit run app.py
```

The application will automatically open in your default web browser at `http://localhost:8501`.
