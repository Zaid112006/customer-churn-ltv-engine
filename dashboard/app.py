import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

st.set_page_config(page_title="Customer Churn & LTV Dashboard", layout="wide")

st.title("📊 Customer Churn & LTV Dashboard")

# Load processed data (already cleaned + feature engineered)
df = pd.read_csv("data/processed/telco_processed.csv")

# --- Section 1: Overview metrics ---
col1, col2, col3 = st.columns(3)
col1.metric("Total Customers", len(df))
col2.metric("Churn Rate", f"{df['Churn'].mean()*100:.1f}%")
col3.metric("Avg Monthly Charges", f"${df['MonthlyCharges'].mean():.2f}")

st.divider()

# --- Section 2: Churn by contract type ---
st.subheader("Churn by Contract Type")
contract_map = {0: 'Month-to-month', 1: 'One year', 2: 'Two year'}
df['Contract_Label'] = df['Contract'].map(contract_map)
contract_churn = df.groupby('Contract_Label')['Churn'].mean().reset_index()
fig1 = px.bar(contract_churn, x='Contract_Label', y='Churn', title="Churn Rate by Contract Type")
st.plotly_chart(fig1, use_container_width=True)

# --- Section 3: Tenure distribution ---
st.subheader("Tenure Distribution by Churn Status")
fig2 = px.histogram(df, x='tenure', color='Churn', barmode='overlay', title="Tenure vs Churn")
st.plotly_chart(fig2, use_container_width=True)

# --- Section 4: Raw data table ---
st.subheader("Customer Data")
st.dataframe(df.head(50))
