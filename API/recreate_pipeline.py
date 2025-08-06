# File: recreate_pipeline.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import joblib

# Load the dataset
try:
    df = pd.read_csv('Telco_Customer_Churn_Dataset.csv')
except FileNotFoundError:
    print("Error: 'Telco_Customer_Churn_Dataset.csv' not found in the API directory.")
    raise

# Debugging: Print column names
print("Dataset columns:", df.columns.tolist())

# Preprocessing
if 'customerID' in df.columns:
    df = df.drop('customerID', axis=1)
else:
    print("Warning: 'customerID' not found in DataFrame.")

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

# Identify columns
categorical_columns = df.select_dtypes(include=['object']).columns
binary_columns = [col for col in ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling'] if col in df.columns]
multi_category_columns = [col for col in categorical_columns if col not in binary_columns and col != 'Churn']
numerical_columns = [col for col in df.select_dtypes(include=['int64', 'float64']).columns if col in df.columns]

# Debugging: Print identified columns
print("Binary columns:", binary_columns)
print("Multi-category columns:", multi_category_columns)
print("Numerical columns:", numerical_columns)

# Define preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('binary', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), binary_columns),
        ('multi_cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), multi_category_columns),
        ('num', 'passthrough', numerical_columns)
    ],
    remainder='drop'  # Explicitly drop any unspecified columns to avoid custom attributes
)

# Create pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'))
])

# Fit pipeline
try:
    X_train_full = df.drop('Churn', axis=1)
    y_train_full = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
    pipeline.fit(X_train_full, y_train_full)
    print("Pipeline fitted successfully.")
except Exception as e:
    print(f"Error fitting pipeline: {str(e)}")
    raise

# Save pipeline
joblib.dump(pipeline, 'logistic_regression_pipeline_fixed.pkl')
print("Pipeline saved as 'logistic_regression_pipeline_fixed.pkl'")