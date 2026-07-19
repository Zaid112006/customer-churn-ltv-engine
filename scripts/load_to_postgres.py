import pandas as pd
from sqlalchemy import create_engine

# Update the password below to match what you set during PostgreSQL install
engine = create_engine("postgresql://postgres:Zaid_2326@localhost:5432/churn_db")

df = pd.read_csv("data/raw/telco_data.csv")
df.to_sql("customers", engine, if_exists="replace", index=False)

print(f"Loaded {len(df)} rows into churn_db.customers")
