from fastapi import FastAPI
from pydantic import BaseModel
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

@app.post("/predict/single")
def predict_single(customer: Customer):
    data = pd.DataFrame([customer.model_dump()])

    # Churn model expects all 23 columns including TotalCharges
    data_scaled = scaler.transform(data)
    churn_pred = churn_model.predict(data_scaled)[0]
    churn_prob = churn_model.predict_proba(data_scaled)[0][1]

    # LTV model expects TotalCharges EXCLUDED (that's what it predicts)
    data_for_ltv = data.drop(columns=['TotalCharges'])
    ltv_pred = ltv_model.predict(data_for_ltv)[0]

    return {
        "churn_prediction": int(churn_pred),
        "churn_probability": round(float(churn_prob), 3),
        "predicted_ltv": round(float(ltv_pred), 2)
    }
