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

st.divider()

# --- Section 5: LTV Segmentation ---
st.subheader("Customer Segmentation by Predicted LTV")

# Load the trained LTV model to generate predictions for all customers
ltv_model = joblib.load("models/ltv_model.pkl")
X_ltv = df.drop(columns=['customerID', 'Churn', 'TotalCharges', 'Contract_Label'])
df['Predicted_LTV'] = ltv_model.predict(X_ltv)

# Segment into Low/Medium/High value tiers
df['LTV_Segment'] = pd.qcut(df['Predicted_LTV'], q=3, labels=['Low Value', 'Medium Value', 'High Value'])

col1, col2 = st.columns(2)
with col1:
    fig3 = px.pie(df, names='LTV_Segment', title="Customer Distribution by LTV Segment")
    st.plotly_chart(fig3, use_container_width=True)

with col2:
    segment_churn = df.groupby('LTV_Segment')['Churn'].mean().reset_index()
    fig4 = px.bar(segment_churn, x='LTV_Segment', y='Churn', title="Churn Rate by LTV Segment")
    st.plotly_chart(fig4, use_container_width=True)

# --- Section 6: High-risk, high-value customers (the key business insight) ---
st.subheader("🚨 Priority Retention List: High-Value Customers at Risk")

churn_model = joblib.load("models/churn_model.pkl")
scaler = joblib.load("models/scaler.pkl")
X_churn = df.drop(columns=['customerID', 'Churn', 'Predicted_LTV', 'LTV_Segment', 'Contract_Label'])
X_churn_scaled = scaler.transform(X_churn)
df['Churn_Probability'] = churn_model.predict_proba(X_churn_scaled)[:, 1]

priority_list = df[(df['LTV_Segment'] == 'High Value') & (df['Churn_Probability'] > 0.4)]
priority_list = priority_list.sort_values('Churn_Probability', ascending=False)

st.write(f"**{len(priority_list)} high-value customers** flagged as at-risk — these should be prioritized for retention campaigns.")
st.dataframe(priority_list[['customerID', 'Predicted_LTV', 'Churn_Probability', 'tenure', 'MonthlyCharges']].head(20))
