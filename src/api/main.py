
from fastapi import FastAPI
import mlflow
import pandas as pd
from src.api.pydantic_models import CreditRiskInput, CreditRiskOutput
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI()

# Load the model at startup
logged_model = '/app/mlruns/788380042057566915/4bdb10531f4f498eb09695ae774fd8b7/artifacts/GradientBoosting'
model = mlflow.pyfunc.load_model(logged_model)

@app.post("/predict", response_model=CreditRiskOutput)
def predict(input_data: CreditRiskInput):
    """Predict credit risk based on input data."""
    df = pd.DataFrame([input_data.dict()])
    
    # The model pipeline includes the preprocessor, so we can directly predict
    prediction = model.predict_proba(df)
    
    # The output of predict_proba is a 2D array, we want the probability of the positive class
    risk_probability = prediction[0][1]
    
    return {"risk_probability": risk_probability}

@app.get("/")
def read_root():
    return {"message": "Credit Risk API is running"}
