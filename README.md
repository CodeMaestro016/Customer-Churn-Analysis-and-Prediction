#  Telco Customer Churn Prediction
This project provides a machine learning-based solution to predict customer churn for a telecommunications company using the Telco Customer Churn dataset. It includes a FastAPI application to serve predictions via a REST API and a Streamlit dashboard for an interactive user interface to visualize churn predictions.
Project Structure

## Features

Machine Learning Model: Logistic Regression model to predict customer churn.
FastAPI: REST API with endpoints:

GET /: Welcome message.

POST /predict/: Predict churn for a single customer.

POST /predict/batch/: Predict churn for multiple customers.

-------------------------------------------------

## Prerequisites

Python: 3.8 or higher.

VS Code: Recommended for development.

Dataset: Telco_Customer_Churn_Dataset.csv.
------------------------------------------------

### Setup Instructions

Clone the Repository:

git clone https://github.com/CodeMaestro016/Customer-Churn-Analysis-and-Prediction.git

cd customer-churn-analysis-and-prediction/API


Create and Activate Virtual Environment (Windows):

python -m venv venv

venv\Scripts\activate


### Install Dependencies:Ensure requirements.txt contains:

fastapi==0.115.0

uvicorn==0.30.6

pandas==2.2.2

joblib==1.4.2

scikit-learn==1.5.1

pydantic==2.8.2

numpy==1.26.4

streamlit==1.39.0

plotly==5.24.1

requests==2.32.3


Install:
pip install -r requirements.txt

--------------------------------------------------------
This generates logistic_regression_pipeline_fixed.pkl using the local scikit-learn version.


Running Locally

FastAPI Application

---------------------------------------------------------

Start the API:python main.py

Streamlit Dashboard

Ensure the FastAPI app is running.

Start the dashboard:streamlit run dashboard.py


Access at http://localhost:8501.

Test at http://localhost:8000/docs.

## Troubleshooting

API Errors:

Ensure logistic_regression_pipeline_fixed.pkl is generated with recreate_pipeline.py using scikit-learn==1.5.1.

----------------------------------------------------------------------

Performance

Model Metrics:

F1-score: 0.6160

Recall: 0.7914 

ROC-AUC: 0.8371

Precision: 0.5043


Dataset: Telco Customer Churn dataset (7043 records, 21 columns).

For issues or suggestions, open an issue on GitHub or contact avishkapiyumal16@gmail.com.
