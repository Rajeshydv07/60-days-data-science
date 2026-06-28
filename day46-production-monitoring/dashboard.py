import streamlit as st
import pandas as pd
import json
import os
import plotly.express as px

st.set_page_config(page_title="API Monitoring Dashboard", layout="wide")
st.title("🛡️ API Production Monitoring & Reliability Dashboard")

log_file = "logs/prediction_tracking.jsonl"

if os.path.exists(log_file):
    data = []
    with open(log_file, "r") as f:
        for line in f:
            try:
                record = json.loads(line)
                # Flatten the nested dictionaries
                flat_record = {
                    "timestamp": record["timestamp"],
                    "request_id": record["request_id"],
                    "churn_probability": record["output"]["churn_probability"],
                    "risk_category": record["output"]["risk_category"],
                    "prediction": record["output"]["prediction"],
                }
                # Add inputs
                for k, v in record["input"].items():
                    flat_record[f"input_{k}"] = v
                
                data.append(flat_record)
            except:
                pass
                
    if data:
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total API Requests", len(df))
        col2.metric("High Risk Predictions", len(df[df['risk_category'].isin(['High', 'Critical'])]))
        col3.metric("Avg Churn Probability", f"{df['churn_probability'].mean():.2f}")
        col4.metric("Unique Customers", df['input_age'].count()) # Just as a proxy
        
        st.markdown("---")
        
        # Charts
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Prediction Distribution")
            fig1 = px.pie(df, names='risk_category', title='Risk Categories', 
                          color_discrete_map={"Low":"#28a745", "Medium":"#ffc107", "High":"#fd7e14", "Critical":"#dc3545"})
            st.plotly_chart(fig1, use_container_width=True)
            
        with c2:
            st.subheader("Prediction Timeline")
            # Group by minute or hour if many, here we just scatter
            fig2 = px.scatter(df, x='timestamp', y='churn_probability', color='risk_category', title='Churn Probability over Time')
            st.plotly_chart(fig2, use_container_width=True)
            
        st.subheader("Recent API Requests (Logs)")
        st.dataframe(df.sort_values(by='timestamp', ascending=False).head(50))
    else:
        st.info("Log file exists but is empty. Make some API requests to see data.")
else:
    st.warning("No tracking logs found. Please run the API and make some requests.")
