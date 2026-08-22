import os
import sys
import pickle
import json
import uuid
import sqlite3
import pandas as pd
import numpy as np
import shap
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional, Union, Any

# Add models and RAG paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "RAG")))

from utils import load_merged_snapshot, prepare_train_test_split
from rag_pipeline import RAGPipeline

app = FastAPI(title="Intelligent Customer Churn API", version="1.0")

@app.get("/")
def read_root():
    index_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "index.html"))
    if not os.path.exists(index_path):
        index_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "web", "index.html"))
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Intelligent Customer Platform Serving API is active. Go to /docs for API documentation."}

@app.get("/Infographic_Official.html")
@app.get("/infographic")
def read_infographic():
    info_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Infographic_Official.html"))
    if os.path.exists(info_path):
        return FileResponse(info_path)
    raise HTTPException(status_code=404, detail="Infographic file not found")

@app.get("/presentation.html")
@app.get("/presentation")
def read_presentation():
    pres_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "presentation.html"))
    if os.path.exists(pres_path):
        return FileResponse(pres_path)
    raise HTTPException(status_code=404, detail="Presentation file not found")


# Global variables loaded at startup
model = None
scaler = None
encoder = None
feature_names = None
optimal_threshold = 0.32
snapshot_date = "2024-03-01"
explainer = None
rag_pipeline = None
merged_features_df = None
db_path = "data/rlhf_outcomes.db"

class PredictRequest(BaseModel):
    customerID: str
    gender: Optional[str] = None
    SeniorCitizen: Optional[int] = None
    Partner: Optional[str] = None
    Dependents: Optional[str] = None
    tenure: Optional[int] = None
    PhoneService: Optional[str] = None
    MultipleLines: Optional[str] = None
    InternetService: Optional[str] = None
    OnlineSecurity: Optional[str] = None
    OnlineBackup: Optional[str] = None
    DeviceProtection: Optional[str] = None
    TechSupport: Optional[str] = None
    StreamingTV: Optional[str] = None
    StreamingMovies: Optional[str] = None
    Contract: Optional[str] = None
    PaperlessBilling: Optional[str] = None
    PaymentMethod: Optional[str] = None
    MonthlyCharges: Optional[float] = None
    TotalCharges: Optional[Union[str, float, int]] = None

class OutcomeRequest(BaseModel):
    customer_id: str
    intervention_type: str
    outcome: int # 1 = retained, 0 = churned

@app.on_event("startup")
def startup_event():
    global model, scaler, encoder, feature_names, optimal_threshold, explainer, rag_pipeline, merged_features_df
    
    feature_store_dir = "data/feature_store"
    config_dir = "config"
    
    # 1. Load preprocessors and feature list
    with open(os.path.join(feature_store_dir, "scaler_v1.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(feature_store_dir, "encoder_v1.pkl"), "rb") as f:
        encoder = pickle.load(f)
    with open(os.path.join(feature_store_dir, "feature_names.json"), "r") as f:
        feature_names = json.load(f)
        
    # 2. Load model (prefer staging model if it exists for test runs)
    model_name = "xgboost_model_staging.pkl" if os.path.exists(os.path.join(feature_store_dir, "xgboost_model_staging.pkl")) else "xgboost_model.pkl"
    with open(os.path.join(feature_store_dir, model_name), "rb") as f:
        model = pickle.load(f)
        
    # 3. Load threshold config
    threshold_path = os.path.join(config_dir, "optimal_threshold.json")
    if os.path.exists(threshold_path):
        with open(threshold_path, "r") as f:
            t_data = json.load(f)
            optimal_threshold = t_data.get("threshold", 0.32)
            
    # 4. Initialize SHAP Explainer
    explainer = shap.TreeExplainer(model)
    
    # 5. Initialize RAG
    rag_pipeline = RAGPipeline()
    
    # 6. Load central feature store snapshot to retrieve historical features for customer IDs
    try:
        merged_features_df = load_merged_snapshot(snapshot_date)
        print(f"Loaded feature store snapshot with {len(merged_features_df)} customer rows.")
    except Exception as e:
        print(f"Error loading snapshot: {e}. Will fall back to request attributes.")
        
    # 7. Initialize RLHF DB
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS rlhf_outcomes (
                id TEXT PRIMARY KEY,
                customer_id TEXT,
                intervention_type TEXT,
                outcome INTEGER,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
    print("FastAPI serving layer loaded successfully.")

def preprocess_single_customer(customer_data: dict):
    """
    Transforms raw customer input attributes using the standard MinMaxScaler and Encoder.
    """
    # Create single-row DataFrame
    cust_df = pd.DataFrame([customer_data])
    
    # Map ordinals and Yes/No values matching etl_pipeline/utils clean logic
    yes_no_cols = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]
    for col in yes_no_cols:
        if col in cust_df.columns:
            cust_df[col] = cust_df[col].replace({"Yes": 1, "No": 0})
            
    replace_cols = ["MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
    for col in replace_cols:
        if col in cust_df.columns:
            cust_df[col] = cust_df[col].replace({"No internet service": "No", "No phone service": "No"})
            
    # Convert Yes/No strings to integers for TechSupport and OnlineSecurity
    yes_no_map = {"Yes": 1, "No": 0}
    for col in ["TechSupport", "OnlineSecurity"]:
        if col in cust_df.columns:
            if cust_df[col].dtype == object:
                cust_df[col] = cust_df[col].map(yes_no_map).fillna(0).astype(int)
            else:
                cust_df[col] = cust_df[col].fillna(0).astype(int)
                
    contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
    if "Contract" in cust_df.columns:
        cust_df["contract_type_encoded"] = cust_df["Contract"].map(contract_map).fillna(0).astype(int)
        
    # Fill bank columns if not provided
    bank_defaults = {"CreditScore": 650, "Balance": 50000.0, "NumOfProducts": 1, "EstimatedSalary": 80000.0}
    for col, default in bank_defaults.items():
        if col not in cust_df.columns or pd.isna(cust_df.loc[0, col]):
            cust_df[col] = default
            
    # Transform categoricals
    categorical_cols = ["PaymentMethod", "InternetService"]
    # Provide defaults if missing
    for col in categorical_cols:
        if col not in cust_df.columns or pd.isna(cust_df.loc[0, col]):
            cust_df[col] = "No" if col == "InternetService" else "Mailed check"
            
    encoded_cats = encoder.transform(cust_df[categorical_cols])
    encoded_cat_df = pd.DataFrame(
        encoded_cats, 
        columns=encoder.get_feature_names_out(categorical_cols),
        index=cust_df.index
    )
    
    # Scale numericals
    numerical_cols = ["tenure", "MonthlyCharges", "TotalCharges", "CreditScore", "Balance", "EstimatedSalary"]
    # Cast TotalCharges
    cust_df["TotalCharges"] = pd.to_numeric(cust_df["TotalCharges"].astype(str).str.strip(), errors="coerce").fillna(0.0)
    scaled_nums = scaler.transform(cust_df[numerical_cols])
    scaled_num_df = pd.DataFrame(
        scaled_nums, 
        columns=numerical_cols,
        index=cust_df.index
    )
    
    # Fill behavioral, LLM, and market default values if not in record
    defaults = {
        "SeniorCitizen": 0, "Partner": 0, "Dependents": 0, "PhoneService": 0, "PaperlessBilling": 0,
        "contract_type_encoded": 0, "service_bundle_size": 1, "TechSupport": 0, "OnlineSecurity": 0,
        "logins_7d_rolling_avg": 0.5, "logins_30d_rolling_avg": 0.5, "login_trend": 0.0,
        "support_contacts_30d": 0, "days_since_last_login": 5, "usage_delta_mom": 0.0,
        "sentiment_score": 0.0, "escalation_flag": 0, "public_complaint_flag": 0,
        "charges_per_tenure_month": 10.0, "billing_delta_mom": 0.0, "is_high_risk_contract": 0,
        "ticket_sentiment_score": 0.0, "ticket_complaint_topics_billing": 0, 
        "ticket_complaint_topics_competitor": 0, "ticket_complaint_topics_performance": 0,
        "ticket_escalation_flag": 0, "review_sentiment_monthly": 0.05, "urgency_level_encoded": 0,
        "competitor_news_volume_7d": 15, "public_review_sentiment_monthly": 0.12,
        "one_star_review_rate_monthly": 0.08, "price_competitiveness_ratio": 1.0, "competitor_promotion_flag": 0
    }
    
    other_df_data = {}
    for col, default in defaults.items():
        if col in cust_df.columns:
            other_df_data[col] = cust_df[col].values[0]
        else:
            other_df_data[col] = default
            
    df_other = pd.DataFrame([other_df_data], index=cust_df.index)
    
    # Combine processed features matching model feature names contract
    X_cust = pd.concat([scaled_num_df, encoded_cat_df, df_other], axis=1)
    
    # Reindex to match the training feature names contract exactly
    X_cust = X_cust.reindex(columns=feature_names, fill_value=0)
    X_cust = X_cust.apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)
    
    return X_cust, cust_df.to_dict(orient="records")[0]

@app.post("/predict")
def predict_churn(request: PredictRequest):
    global model, explainer, rag_pipeline, merged_features_df
    
    customer_id = request.customerID
    
    # 1. Fetch features from snapshot first
    customer_row = None
    if merged_features_df is not None and "customerID" in merged_features_df.columns:
        match = merged_features_df[merged_features_df["customerID"] == customer_id]
        if not match.empty:
            customer_row = match.to_dict(orient="records")[0]
            
    if customer_row is None:
        # Fall back to using request raw values
        customer_row = request.model_dump() if hasattr(request, "model_dump") else request.dict()
        print(f"Customer {customer_id} not found in Feature Store. Using request attributes.")
        
    try:
        # Preprocess feature vector
        X_cust, customer_clean_dict = preprocess_single_customer(customer_row)
        
        # 2. Churn score prediction
        prob = float(model.predict_proba(X_cust)[0, 1])
        
        # Calculate confidence: distance from decision threshold
        confidence = float(np.abs(prob - optimal_threshold) * 2.0)
        confidence = min(1.0, max(0.0, confidence)) # bound between 0 and 1
        
        # 3. Calculate top 3 SHAP drivers
        shap_raw = explainer.shap_values(X_cust)
        # Handle different SHAP version return types for binary classifiers
        if isinstance(shap_raw, list):
            shap_vals = shap_raw[1][0] if len(shap_raw) > 1 else shap_raw[0][0]
        else:
            shap_vals = shap_raw[0] if len(shap_raw.shape) > 1 else shap_raw
        top_indices = np.argsort(np.abs(shap_vals))[::-1][:3]
        top_drivers = [feature_names[idx] for idx in top_indices]
        
        # 4. Generate RAG retention briefing if score exceeds optimal threshold
        recommendation_text = "No action required. Customer exhibits stable behavior."
        if prob >= optimal_threshold:
            recommendation_text = rag_pipeline.generate_briefing(customer_id, customer_clean_dict, top_drivers)
            
        return {
            "customer_id": customer_id,
            "churn_score": prob,
            "confidence": confidence,
            "top_3_shap_drivers": top_drivers,
            "recommended_action": recommendation_text
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/outcome")
def log_outcome(request: OutcomeRequest):
    outcome_id = str(uuid.uuid4())
    
    try:
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "INSERT INTO rlhf_outcomes (id, customer_id, intervention_type, outcome) VALUES (?, ?, ?, ?)",
                (outcome_id, request.customer_id, request.intervention_type, request.outcome)
            )
            conn.commit()
        return {"logged": True, "outcome_id": outcome_id}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Database write failure: {e}")

@app.get("/model_info")
def get_model_info():
    global optimal_threshold, snapshot_date
    return {
        "model_name": "churn_predictor",
        "model_version": "v1.0",
        "feature_snapshot_date": snapshot_date,
        "optimal_threshold": optimal_threshold,
        "evaluation_metrics": {
            "test_roc_auc": 0.4766, # Mock snapshot metrics
            "test_pr_auc": 0.8252,
            "recall_at_optimal_threshold": 1.0
        }
    }
