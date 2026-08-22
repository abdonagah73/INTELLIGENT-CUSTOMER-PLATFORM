import os
import shutil
import json
import pandas as pd
import numpy as np
import pytest
from src.data.validate_schema import (
    clean_and_validate_telco,
    clean_and_validate_bank,
    check_join_overlap
)
from src.data.etl_pipeline import run_etl_pipeline

def test_telco_null_charges_resolution():
    # Create test dataframe with a blank TotalCharges for a tenure=0 customer
    test_df = pd.DataFrame({
        "customerID": ["1", "2"],
        "gender": ["Male", "Female"],
        "SeniorCitizen": [0, 1],
        "Partner": ["Yes", "No"],
        "Dependents": ["No", "Yes"],
        "tenure": [0, 12],
        "PhoneService": ["Yes", "Yes"],
        "MultipleLines": ["No", "Yes"],
        "InternetService": ["DSL", "Fiber optic"],
        "OnlineSecurity": ["No", "Yes"],
        "OnlineBackup": ["Yes", "No"],
        "DeviceProtection": ["No", "Yes"],
        "TechSupport": ["No", "No"],
        "StreamingTV": ["Yes", "Yes"],
        "StreamingMovies": ["No", "Yes"],
        "Contract": ["Month-to-month", "One year"],
        "PaperlessBilling": ["Yes", "No"],
        "PaymentMethod": ["Electronic check", "Mailed check"],
        "MonthlyCharges": [50.0, 80.0],
        "TotalCharges": [" ", "960.0"], # Note the space in index 0
        "Churn": ["No", "Yes"]
    })
    
    schema_path = "config/schema_contracts/telco_schema.json"
    cleaned = clean_and_validate_telco(test_df, schema_path)
    
    # Assertions
    assert cleaned.shape[0] == 2
    # The empty string for tenure=0 should resolve to 0.0 * 50.0 = 0.0
    assert cleaned.loc[cleaned["customerID"] == "1", "TotalCharges"].values[0] == 0.0
    # The normal TotalCharges should be converted to float
    assert cleaned.loc[cleaned["customerID"] == "2", "TotalCharges"].values[0] == 960.0

def test_bank_uniqueness_check():
    # Create dataset with duplicate CustomerId
    test_df = pd.DataFrame({
        "RowNumber": [1, 2],
        "CustomerId": ["1000", "1000"], # Duplicate ID
        "Surname": ["Doe", "Doe"],
        "CreditScore": [600, 600],
        "Geography": ["France", "France"],
        "Gender": ["Male", "Male"],
        "Age": [30, 30],
        "Tenure": [5, 5],
        "Balance": [1000.0, 1000.0],
        "NumOfProducts": [1, 1],
        "HasCrCard": [1, 1],
        "IsActiveMember": [1, 1],
        "EstimatedSalary": [50000.0, 50000.0],
        "Exited": [0, 0]
    })
    
    schema_path = "config/schema_contracts/bank_schema.json"
    cleaned = clean_and_validate_bank(test_df, schema_path)
    
    # Assertions
    # It should drop the duplicate CustomerId
    assert len(cleaned) == 1

def test_overlap_calculation():
    telco = pd.DataFrame({"customerID": ["1", "2", "3"]})
    bank = pd.DataFrame({"CustomerId": ["2", "3", "4"]})
    
    rate = check_join_overlap(telco, bank)
    # Overlap is {"2", "3"} out of {"1", "2", "3"} = 2/3 = 66.67%
    assert abs(rate - 0.666666) < 1e-4
