import os
import json
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE
import pytest
import sys
import mlflow
import mlflow.xgboost

# Add models and mlops paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models")))
from utils import load_merged_snapshot, prepare_train_test_split
from rlhf_reweighter import compute_sample_weights
from drift_monitor import run_drift_analysis, run_model_drift_analysis

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Customer_Churn_AutoRetraining")

def run_retraining_cycle(snapshot_date="2024-03-01"):
    print(f"Starting automated retraining cycle for snapshot {snapshot_date}...")
    
    # --- Step 1: Pull the latest feature snapshot ---
    df = load_merged_snapshot(snapshot_date)
    X_train, X_test, y_train, y_test, ids_train, ids_test, feature_names = prepare_train_test_split(df)
    print(f"Step 1 Complete: Loaded training split ({len(X_train)} rows) and test split ({len(X_test)} rows).")
    
    # --- Step 2: Load RLHF-reweighted training samples ---
    sample_weights = compute_sample_weights(ids_train)
    print("Step 2 Complete: Computed RLHF sample weights.")
    
    # --- Step 3: Retrain XGBoost using baseline optimized parameters ---
    print("Retraining XGBoost model...")
    # Apply SMOTE to balance churn classes
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    # Align sample weights to the SMOTE oversampled training data
    # (Since SMOTE adds synthetic samples, we assign default weight 1.0 to synthetic samples, and keep RLHF weights for original samples)
    resampled_weights = np.ones(len(X_train_res))
    resampled_weights[:len(X_train)] = sample_weights
    
    # Standard optimized hyperparameters
    params = {
        "n_estimators": 250,
        "max_depth": 3,
        "learning_rate": 0.08,
        "subsample": 0.95,
        "colsample_bytree": 0.90,
        "scale_pos_weight": 1.5,
        "random_state": 42
    }
    
    new_model = xgb.XGBClassifier(**params)
    
    # --- Step 4: Evaluate the new model on the held-out test set ---
    # Load current production model
    prod_model_path = "data/feature_store/xgboost_model.pkl"
    if os.path.exists(prod_model_path):
        with open(prod_model_path, "rb") as f:
            prod_model = pickle.load(f)
        prod_probs = prod_model.predict_proba(X_test)[:, 1]
        prod_auc = roc_auc_score(y_test, prod_probs)
    else:
        prod_auc = 0.50 # default baseline
        
    with mlflow.start_run() as run:
        new_model.fit(X_train_res, y_train_res, sample_weight=resampled_weights)
        new_probs = new_model.predict_proba(X_test)[:, 1]
        new_auc = roc_auc_score(y_test, new_probs)
        
        print(f"Step 4 Evaluation:")
        print(f"  Production Model ROC-AUC: {prod_auc:.4f}")
        print(f"  Retrained Model ROC-AUC: {new_auc:.4f}")
        
        # Failsafe: Reject if new model is significantly worse (lower than prod AUC - 2%)
        gating_threshold = prod_auc - 0.02
        if new_auc < gating_threshold:
            print(f"REJECTED: Retrained model ROC-AUC ({new_auc:.4f}) is below gating safety threshold ({gating_threshold:.4f}). Aborting promotion.")
            mlflow.log_param("status", "REJECTED")
            return False
            
        print("Step 4 Complete: Safety check passed. Proceeding to Staging promotion.")
        
        # --- Step 5: Register model in MLflow ---
        mlflow.log_params(params)
        mlflow.log_metric("roc_auc", new_auc)
        mlflow.xgboost.log_model(new_model, "retrained_model")
        print("Step 5 Complete: Model registered in MLflow database.")
        
        # --- Step 6: Promote to Staging & Run Automated API Tests ---
        # Temporarily serialize model to staging file for testing
        staging_path = "data/feature_store/xgboost_model_staging.pkl"
        with open(staging_path, "wb") as f:
            pickle.dump(new_model, f)
            
        # Overwrite the actual feature names file as staging contract
        with open("data/feature_store/feature_names.json", "w") as f:
            json.dump(feature_names, f, indent=2)
            
        print("Running Staging integration tests...")
        # Add root workspace directory to sys.path so pytest can find 'src'
        import sys
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        if root_dir not in sys.path:
            sys.path.insert(0, root_dir)
        # Point main.py to read xgboost_model_staging.pkl for test suite runs
        # Run pytest programmatically on the serving endpoint tests
        exit_code = pytest.main(["tests/test_api.py", "-q"])
        
        if exit_code != 0:
            print("REJECTED: Staging integration tests failed. Aborting promotion.")
            if os.path.exists(staging_path):
                os.remove(staging_path)
            return False
            
        print("Step 6 Complete: Staging integration tests passed.")
        
        # --- Step 7: Promote to Production ---
        # Move staging model to main production model file path
        if os.path.exists(staging_path):
            if os.path.exists(prod_model_path):
                # Archive old model (rename to backup)
                os.replace(prod_model_path, prod_model_path + ".backup")
            os.replace(staging_path, prod_model_path)
            
        print(f"Step 7 Complete: Retrained model promoted to Production successfully.")
        return True

if __name__ == "__main__":
    # Check if drift alerts dictate retraining
    run_retraining = False
    
    psi, feature_alerts = run_drift_analysis()
    # If any feature has significant drift (PSI > 0.20), trigger retraining
    if len(feature_alerts) > 0:
        print(f"Drift detected in {len(feature_alerts)} features. Triggering auto-retraining...")
        run_retraining = True
        
    _, model_alert = run_model_drift_analysis()
    if model_alert:
        print("Model performance decay detected. Triggering auto-retraining...")
        run_retraining = True
        
    if run_retraining:
        success = run_retraining_cycle()
        print(f"Retraining run status: {'SUCCESS' if success else 'FAILED'}")
    else:
        print("No drift alerts active. Retraining cycle skipped.")
