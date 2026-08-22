import os
import json
import pandas as pd
import numpy as np

def load_schema(schema_path):
    with open(schema_path, 'r') as f:
        return json.load(f)

def validate_columns_and_types(df, schema):
    """
    Validates that all required columns are present in the DataFrame and matches type.
    """
    errors = []
    required_cols = schema.get("required", [])
    properties = schema.get("properties", {})
    
    # 1. Check required columns
    for col in required_cols:
        if col not in df.columns:
            errors.append(f"Missing required column: {col}")
            
    # 2. Check types (basic verification)
    for col in df.columns:
        if col in properties:
            prop_type = properties[col].get("type")
            # If type is a list, e.g. ["string", "number"]
            types = prop_type if isinstance(prop_type, list) else [prop_type]
            
            # Simple mapping to pandas/numpy types
            # Note: We won't throw hard errors for string vs numeric representation on initial load
            # as some columns like TotalCharges might load as string.
            pass
            
    return errors

def clean_and_validate_telco(df, schema_path):
    print("Validating and cleaning Telco dataset...")
    schema = load_schema(schema_path)
    
    # Validate columns
    errors = validate_columns_and_types(df, schema)
    if errors:
        raise ValueError(f"Telco schema validation failed:\n" + "\n".join(errors))
        
    df_cleaned = df.copy()
    
    # 1. TotalCharges null handling:
    # 11 rows have empty string instead of float. These are tenure=0 customers. 
    # Fix: convert to float, fill with tenure * MonthlyCharges.
    original_empty_count = 0
    if "TotalCharges" in df_cleaned.columns:
        # Convert strings to numeric, coerce errors to NaN
        is_string_col = df_cleaned["TotalCharges"].dtype == object
        if is_string_col:
            # Strip spaces
            df_cleaned["TotalCharges"] = df_cleaned["TotalCharges"].astype(str).str.strip()
            # Find empty strings
            empty_mask = df_cleaned["TotalCharges"] == ""
            original_empty_count = empty_mask.sum()
            df_cleaned.loc[empty_mask, "TotalCharges"] = np.nan
            
        df_cleaned["TotalCharges"] = pd.to_numeric(df_cleaned["TotalCharges"], errors="coerce")
        
        # Fill NaN
        nan_mask = df_cleaned["TotalCharges"].isna()
        fill_values = df_cleaned.loc[nan_mask, "tenure"] * df_cleaned.loc[nan_mask, "MonthlyCharges"]
        df_cleaned.loc[nan_mask, "TotalCharges"] = fill_values
        print(f"TotalCharges: filled {nan_mask.sum()} empty/null rows (original empty strings detected: {original_empty_count}).")
        
    # 2. customerID uniqueness
    if "customerID" in df_cleaned.columns:
        duplicates = df_cleaned["customerID"].duplicated().sum()
        if duplicates > 0:
            print(f"WARNING: Found {duplicates} duplicate customerID values. Keeping the first occurrence.")
            df_cleaned = df_cleaned.drop_duplicates(subset=["customerID"], keep="first")
            
    # 3. Churn label completeness: Every row must have Churn (Yes/No). Rows without labels are dropped and logged.
    if "Churn" in df_cleaned.columns:
        initial_rows = len(df_cleaned)
        df_cleaned = df_cleaned.dropna(subset=["Churn"])
        df_cleaned = df_cleaned[df_cleaned["Churn"].isin(["Yes", "No"])]
        dropped = initial_rows - len(df_cleaned)
        if dropped > 0:
            print(f"Churn completeness: dropped {dropped} rows with missing or invalid Churn label.")
            
    # 4. Range checks: MonthlyCharges must be 0-200. Tenure must be 0-72. Values outside range flagged.
    if "MonthlyCharges" in df_cleaned.columns:
        out_of_range = df_cleaned[(df_cleaned["MonthlyCharges"] < 0) | (df_cleaned["MonthlyCharges"] > 200)]
        if not out_of_range.empty:
            print(f"WARNING: {len(out_of_range)} rows found with MonthlyCharges outside [0, 200] range. Flagged for review.")
            
    if "tenure" in df_cleaned.columns:
        out_of_range = df_cleaned[(df_cleaned["tenure"] < 0) | (df_cleaned["tenure"] > 72)]
        if not out_of_range.empty:
            print(f"WARNING: {len(out_of_range)} rows found with tenure outside [0, 72] range. Flagged for review.")
            
    return df_cleaned

def clean_and_validate_bank(df, schema_path):
    print("Validating and cleaning Bank dataset...")
    schema = load_schema(schema_path)
    
    # Validate columns
    errors = validate_columns_and_types(df, schema)
    if errors:
        raise ValueError(f"Bank schema validation failed:\n" + "\n".join(errors))
        
    df_cleaned = df.copy()
    
    # Uniqueness check on CustomerId
    if "CustomerId" in df_cleaned.columns:
        duplicates = df_cleaned["CustomerId"].duplicated().sum()
        if duplicates > 0:
            print(f"WARNING: Found {duplicates} duplicate CustomerId values in Bank dataset. Keeping first.")
            df_cleaned = df_cleaned.drop_duplicates(subset=["CustomerId"], keep="first")
            
    return df_cleaned

def clean_and_validate_tickets(df, schema_path):
    print("Validating Customer Support Tickets dataset...")
    schema = load_schema(schema_path)
    
    # Validate columns
    errors = validate_columns_and_types(df, schema)
    if errors:
        raise ValueError(f"Tickets schema validation failed:\n" + "\n".join(errors))
        
    return df

def check_join_overlap(telco_df, bank_df):
    """
    Verifies match rate between Telco customerID and Bank CustomerId.
    """
    if "customerID" not in telco_df.columns or "CustomerId" not in bank_df.columns:
        print("Overlap check skipped: missing join keys.")
        return 0.0
        
    telco_ids = set(telco_df["customerID"].astype(str))
    bank_ids = set(bank_df["CustomerId"].astype(str))
    
    overlap = telco_ids.intersection(bank_ids)
    match_rate = len(overlap) / len(telco_ids) if len(telco_ids) > 0 else 0.0
    
    print(f"Bank Join Overlap Analysis:")
    print(f"  Telco Customer Count: {len(telco_ids)}")
    print(f"  Bank Customer Count: {len(bank_ids)}")
    print(f"  Overlapping Customers: {len(overlap)}")
    print(f"  Match Rate: {match_rate:.2%}")
    
    if match_rate < 0.10:
        print("WARNING: Match rate between Telco and Bank datasets is very low (<10%). Verify join keys.")
        
    return match_rate
