from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import joblib
import pandas as pd

app = FastAPI(title="Customer Churn & LTV API")

churn_model = joblib.load("models/churn_model.pkl")
scaler = joblib.load("models/scaler.pkl")
ltv_model = joblib.load("models/ltv_model.pkl")

@app.get("/")
def read_root():
    return {"message": "Churn & LTV API is running"}

class Customer(BaseModel):
    gender: int
    SeniorCitizen: int
    Partner: int
    Dependents: int
    tenure: float
    PhoneService: int
    MultipleLines: int
    InternetService: int
    OnlineSecurity: int
    OnlineBackup: int
    DeviceProtection: int
    TechSupport: int
    StreamingTV: int
    StreamingMovies: int
    Contract: int
    PaperlessBilling: int
    PaymentMethod: int
    MonthlyCharges: float
    TotalCharges: float
    charge_ratio: float
    tenure_group: int
    total_services: int
    charges_difference: float

def make_prediction(data: pd.DataFrame):
    data_scaled = scaler.transform(data)
    churn_pred = churn_model.predict(data_scaled)
    churn_prob = churn_model.predict_proba(data_scaled)[:, 1]

    data_for_ltv = data.drop(columns=['TotalCharges'])
    ltv_pred = ltv_model.predict(data_for_ltv)

    results = []
    for i in range(len(data)):
        results.append({
            "churn_prediction": int(churn_pred[i]),
            "churn_probability": round(float(churn_prob[i]), 3),
            "predicted_ltv": round(float(ltv_pred[i]), 2)
        })
    return results

@app.post("/predict/single")
def predict_single(customer: Customer):
    data = pd.DataFrame([customer.model_dump()])
    return make_prediction(data)[0]

@app.post("/predict/batch")
def predict_batch(customers: List[Customer]):
    data = pd.DataFrame([c.model_dump() for c in customers])
    results = make_prediction(data)
    return {"predictions": results, "count": len(results)}
