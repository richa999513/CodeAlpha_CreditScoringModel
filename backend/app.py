#  python -m uvicorn app:app --reload
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np
import os

app = FastAPI(title="Credit Risk Prediction API")

# ==========================================================
# 1️⃣ LOAD THE ENTIRE PIPELINE
# ==========================================================
# We use joblib because that's how you saved it in model.py
model_pipeline = None
try:
    # build a path relative to this file so it works regardless of cwd
    base_dir = os.path.dirname(__file__)
    model_path = os.path.normpath(os.path.join(base_dir, "..", "loan", "credit_risk_model.pkl"))
    model_pipeline = joblib.load(model_path)
except Exception as e:
    print(f"Error loading model ({model_path}): {e}")
    model_pipeline = None

# ==========================================================
# 2️⃣ DEFINE REQUEST BODY
# ==========================================================
# Using a dict allows flexibility, but you can also define 
# specific fields if you want stricter validation.
class LoanRequest(BaseModel):
    data: dict 

# ==========================================================
# 3️⃣ ROUTES
# ==========================================================

@app.get("/")
def health_check():
    return {"status": "online", "model_loaded": model_pipeline is not None}

@app.post("/predict")
def predict(request: LoanRequest):
    if model_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Convert input dict to DataFrame
        # The pipeline expects a DataFrame because it uses ColumnTransformer
        input_df = pd.DataFrame([request.data])

        # We don't need to manually scale or encode! 
        # The pipeline does: Impute -> OneHot -> Variance -> LogReg
        probability = model_pipeline.predict_proba(input_df)[0, 1]
        prediction = int(probability > 0.5)

        # build response with both old and new keys for compatibility
        resp = {
            "prediction": "Bad Loan" if prediction == 1 else "Good Loan",
            "probability": round(float(probability), 4),
            # keep original field for any existing clients
            "probability_of_default": round(float(probability), 4),
            "status_code": 1 if prediction == 1 else 0
        }
        return resp

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")