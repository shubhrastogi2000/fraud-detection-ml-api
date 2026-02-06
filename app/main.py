from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from pathlib import Path
import time

# ------------------ Load Artifacts Safely ------------------

BASE_DIR = Path(__file__).resolve().parent.parent

model_path = BASE_DIR / "models" / "fraud_model.pkl"
scaler_path = BASE_DIR / "models" / "scaler.pkl"

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

app = FastAPI(title="Fraud Detection API")

# ------------------ Input Schema ------------------

class Transaction(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float

# ------------------ Simplified UI Input Schema ------------------

class TransactionUI(BaseModel):
    amount: float
    hour: int
    card_present: int

# ----------Feature Builder for UI Input----------
def  build_feature_vector(ui_data: TransactionUI):
    """
    Convert simplified UI input into full feature vector.
    PCA features are filled with zeros for demo purposes.
    """
    features = np.zeros(30)
    features[0] = ui_data.hour
    features[29] = ui_data.amount
    return features.reshape(1, -1)

# ------------------ Prediction Endpoint ------------------

@app.post("/predict")
def predict(transaction: Transaction):
    data = np.array([[
        transaction.Time,
        transaction.V1,
        transaction.V2,
        transaction.V3,
        transaction.V4,
        transaction.V5,
        transaction.V6,
        transaction.V7,
        transaction.V8,
        transaction.V9,
        transaction.V10,
        transaction.V11,
        transaction.V12,
        transaction.V13,
        transaction.V14,
        transaction.V15,
        transaction.V16,
        transaction.V17,
        transaction.V18,
        transaction.V19,
        transaction.V20,
        transaction.V21,
        transaction.V22,
        transaction.V23,
        transaction.V24,
        transaction.V25,
        transaction.V26,
        transaction.V27,
        transaction.V28,
        transaction.Amount
    ]])

    # Scale input
    data_scaled = scaler.transform(data)

    # Predict
    start = time.time()
    prediction = model.predict(data_scaled)[0]
    probability = model.predict_proba(data_scaled)[0][1]
    latency = round((time.time() - start)*1000, 2)

    return {
        "fraud_prediction": int(prediction),
        "fraud_probability": float(probability),
        "inference_ms": latency
    }

#---------------UI Friendly Prediction Endpoint----------------

@app.post("predict_ui")
def predict_ui(transaction: TransactionUI):
    data = build_feature_vector(transaction)
    data_scaled = scaler.transform(data)
    start = time.time()
    prediction = model.predict(data_scaled)[0]
    probability = model.predict_proba(data_scaled)[0][1]
    latency = round((time.time() - start)*1000, 2)

    return {
        "fraud": int(prediction),
        "probability": float(probability),
        "inference_ms": latency
    }

#--------health check endpoint--------
@app.get("/health")
def health():
    return {"status": "ok"}
