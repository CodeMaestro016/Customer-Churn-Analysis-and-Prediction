# File: main.py (FastAPI application)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
from typing import Optional, List
import uvicorn

# Initialize FastAPI app
app = FastAPI(title="Telco Customer Churn Prediction API")

# Load the original dataset to get expected columns
try:
    df = pd.read_csv('Telco_Customer_Churn_Dataset.csv')
    expected_columns = df.drop(['Churn', 'customerID'], axis=1).columns.tolist()
except FileNotFoundError:
    raise Exception("Telco_Customer_Churn_Dataset.csv not found in the API directory.")

# Define input data model using Pydantic
class CustomerData(BaseModel):
    customerID: Optional[str] = None
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

# Define predict_churn function
def predict_churn(new_data, pipeline_path='logistic_regression_pipeline_fixed.pkl'):
    """
    Predict churn for new data using the saved pipeline.
    
    Parameters:
    new_data (pd.DataFrame or dict): New data with the same features as the original dataset
    pipeline_path (str): Path to the saved pipeline
    
    Returns:
    dict: Predictions with customer IDs, churn predictions, and probabilities
    """
    try:
        # Load the pipeline
        pipeline = joblib.load(pipeline_path)
    except Exception as e:
        raise Exception(f"Failed to load pipeline: {str(e)}")
    
    # Ensure new_data is a DataFrame
    if isinstance(new_data, dict):
        new_data = pd.DataFrame([new_data])
    
    # Store customerID if present
    customer_ids = new_data['customerID'] if 'customerID' in new_data.columns else None
    
    # Drop 'customerID' if present
    if 'customerID' in new_data.columns:
        new_data = new_data.drop('customerID', axis=1)
    
    # Ensure new_data has the same columns as the original dataset (excluding 'Churn')
    new_data = new_data.reindex(columns=expected_columns, fill_value=0)
    
    # Make predictions
    predictions = pipeline.predict(new_data)
    probabilities = pipeline.predict_proba(new_data)[:, 1]
    
    # Format results
    results = {
        'customerID': customer_ids.tolist() if customer_ids is not None else [f'Customer_{i+1}' for i in range(len(predictions))],
        'Predicted_Churn': ['Yes' if pred == 1 else 'No' for pred in predictions],
        'Churn_Probability': probabilities.tolist()
    }
    
    return results

# Root endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to the Telco Customer Churn Prediction API"}

# Prediction endpoint for single customer
@app.post("/predict/")
async def predict_single(customer: CustomerData):
    try:
        # Convert Pydantic model to dictionary
        customer_dict = customer.dict()
        # Make prediction
        result = predict_churn(customer_dict)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error predicting churn: {str(e)}")

# Prediction endpoint for multiple customers
@app.post("/predict/batch/")
async def predict_batch(customers: List[CustomerData]):
    try:
        # Convert list of Pydantic models to DataFrame
        customer_dicts = [customer.dict() for customer in customers]
        customer_df = pd.DataFrame(customer_dicts)
        # Make predictions
        result = predict_churn(customer_df)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error predicting churn: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)