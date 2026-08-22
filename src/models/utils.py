import os
import pickle
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_merged_snapshot(snapshot_date="2024-03-01", base_dir="data/feature_store"):
    """
    Loads all five feature store parquet files and merges them on customerID.
    """
    groups = ["demographic", "behavioral", "billing", "llm_features", "market_signals"]
    dfs = []
    
    for group in groups:
        path = os.path.join(base_dir, group, f"snapshot_{snapshot_date}.parquet")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Feature store snapshot not found: {path}")
        df = pd.read_parquet(path)
        dfs.append(df)
        
    # Merge all dataframes on customerID
    merged = dfs[0]
    for df in dfs[1:]:
        # Avoid duplicate columns except join key
        cols_to_use = [col for col in df.columns if col == "customerID" or col not in merged.columns]
        merged = pd.merge(merged, df[cols_to_use], on="customerID", how="inner")
        
    return merged

def prepare_train_test_split(df, test_size=0.2, random_state=42):
    """
    Preprocesses the merged dataframe and returns X_train, X_test, y_train, y_test, 
    along with the feature names list and preprocessors.
    """
    # 1. Load scaler and encoder fitted in Milestone 1
    feature_store_dir = "data/feature_store"
    with open(os.path.join(feature_store_dir, "scaler_v1.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(feature_store_dir, "encoder_v1.pkl"), "rb") as f:
        encoder = pickle.load(f)
        
    df_proc = df.copy()
    
    # 2. Map ordinal and binary features
    # Map urgency_level
    urgency_map = {"low": 0, "medium": 1, "high": 2, "critical": 3}
    if "urgency_level" in df_proc.columns:
        df_proc["urgency_level_encoded"] = df_proc["urgency_level"].map(urgency_map).fillna(0)
    if "ticket_urgency_level" in df_proc.columns:
        df_proc["ticket_urgency_level_encoded"] = df_proc["ticket_urgency_level"].map(urgency_map).fillna(0)
        
    # Map TechSupport and OnlineSecurity Yes/No to 1/0
    yes_no_map = {"Yes": 1, "No": 0}
    for col in ["TechSupport", "OnlineSecurity"]:
        if col in df_proc.columns:
            df_proc[col] = df_proc[col].map(yes_no_map).fillna(0).astype(int)
        
    # 3. Apply One-Hot Encoder on categorical columns
    categorical_cols = ["PaymentMethod", "InternetService"]
    encoded_cats = encoder.transform(df_proc[categorical_cols])
    encoded_cat_df = pd.DataFrame(
        encoded_cats, 
        columns=encoder.get_feature_names_out(categorical_cols),
        index=df_proc.index
    )
    
    # 4. Apply StandardScaler on numerical columns
    numerical_cols = [
        "tenure", "MonthlyCharges", "TotalCharges", 
        "CreditScore", "Balance", "EstimatedSalary"
    ]
    scaled_nums = scaler.transform(df_proc[numerical_cols])
    scaled_num_df = pd.DataFrame(
        scaled_nums, 
        columns=numerical_cols,
        index=df_proc.index
    )
    
    # 5. Extract target and ID
    y = df_proc["Churn"].values
    c_ids = df_proc["customerID"].values
    
    # 6. Gather remaining numeric/binary columns that do not need scaling/one-hot encoding
    # These are either binary flags (1/0) or already scaled ratios/averages
    other_cols = [
        "SeniorCitizen", "Partner", "Dependents", "PaperlessBilling",
        "contract_type_encoded", "service_bundle_size", "TechSupport", "OnlineSecurity",
        "logins_7d_rolling_avg", "logins_30d_rolling_avg", "login_trend",
        "support_contacts_30d", "days_since_last_login", "usage_delta_mom",
        "sentiment_score", "escalation_flag", "public_complaint_flag",
        "charges_per_tenure_month", "billing_delta_mom", "is_high_risk_contract",
        "ticket_sentiment_score", "ticket_complaint_topics_billing", 
        "ticket_complaint_topics_competitor", "ticket_complaint_topics_performance",
        "ticket_escalation_flag", "review_sentiment_monthly", "urgency_level_encoded",
        "competitor_news_volume_7d", "public_review_sentiment_monthly",
        "one_star_review_rate_monthly", "price_competitiveness_ratio", "competitor_promotion_flag"
    ]
    
    # Fill remaining NaNs if any
    df_other = df_proc[other_cols].fillna(0)
    
    # Combine all processed features
    X = pd.concat([scaled_num_df, encoded_cat_df, df_other], axis=1)
    feature_names = list(X.columns)
    
    # Train-test split
    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        X, y, c_ids, test_size=test_size, random_state=random_state, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, ids_train, ids_test, feature_names
