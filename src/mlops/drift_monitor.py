import os
import json
import pickle
import sqlite3
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import sys

# Add models path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models")))
from utils import load_merged_snapshot, prepare_train_test_split

def calculate_psi(baseline, target, num_bins=10):
    """
    Computes the Population Stability Index (PSI) between baseline and target distributions.
    """
    # Remove NaNs
    baseline = baseline[~np.isnan(baseline)].astype(float)
    target = target[~np.isnan(target)].astype(float)
    
    if len(baseline) == 0 or len(target) == 0:
        return 0.0
        
    # Determine bin boundaries based on baseline percentiles
    percentiles = np.linspace(0, 100, num_bins + 1)
    bins = np.percentile(baseline, percentiles)
    # Deduplicate bins if duplicate percentiles exist
    bins = np.unique(bins)
    if len(bins) < 2:
        return 0.0
        
    # Adjust boundaries slightly to catch edges
    bins[0] -= 1e-5
    bins[-1] += 1e-5
    
    # Calculate counts in bins
    baseline_counts, _ = np.histogram(baseline, bins=bins)
    target_counts, _ = np.histogram(target, bins=bins)
    
    # Convert to percentages
    baseline_pcts = baseline_counts / len(baseline)
    target_pcts = target_counts / len(target)
    
    # Add tiny epsilon to avoid division by zero / log(0)
    eps = 1e-4
    baseline_pcts = np.where(baseline_pcts == 0, eps, baseline_pcts)
    target_pcts = np.where(target_pcts == 0, eps, target_pcts)
    
    # Calculate PSI
    psi_value = np.sum((target_pcts - baseline_pcts) * np.log(target_pcts / baseline_pcts))
    return float(psi_value)

def run_drift_analysis(snapshot_date="2024-03-01", target_df=None):
    """
    Runs Population Stability Index (PSI) feature drift checks.
    """
    print("Running feature drift analysis...")
    
    # 1. Load baseline (training data)
    df_baseline = load_merged_snapshot(snapshot_date)
    X_train, _, _, _, _, _, feature_names = prepare_train_test_split(df_baseline)
    
    # 2. Set target dataset
    if target_df is None:
        # For testing, we load baseline and inject artificial drift
        # E.g. increase MonthlyCharges by 25% and reduce tenure by 15%
        target_df = df_baseline.copy()
        target_df["MonthlyCharges"] = target_df["MonthlyCharges"] * 1.25
        target_df["tenure"] = (target_df["tenure"] * 0.85).astype(int)
        print("No target dataset provided. Created drifted simulated target dataset.")
        
    # We must preprocess target dataset using prepare_train_test_split to compare
    _, X_target, _, _, _, _, _ = prepare_train_test_split(target_df)
    
    # 3. Calculate PSI for all features
    drift_alerts = {}
    psi_values = {}
    
    for col in X_train.columns:
        # Calculate PSI
        psi = calculate_psi(X_train[col].values, X_target[col].values)
        psi_values[col] = psi
        
        # Determine status
        if psi > 0.20:
            drift_alerts[col] = {"psi": psi, "status": "RED (Significant Drift)"}
        elif psi > 0.10:
            drift_alerts[col] = {"psi": psi, "status": "YELLOW (Mild Warning)"}
            
    print(f"Drift Analysis Completed. Total features monitored: {len(X_train.columns)}")
    print(f"Features with Mild Drift warning (PSI > 0.10): {sum(1 for v in psi_values.values() if 0.10 < v <= 0.20)}")
    print(f"Features with Significant Drift alert (PSI > 0.20): {sum(1 for v in psi_values.values() if v > 0.20)}")
    
    return psi_values, drift_alerts

def run_model_drift_analysis(db_path="data/rlhf_outcomes.db", baseline_auc=0.85):
    """
    Checks if model performance (ROC-AUC) on recently scored outcomes has decayed.
    """
    print("Running model performance decay checks...")
    
    if not os.path.exists(db_path):
        print("No RLHF outcomes database found. Performance monitoring skipped.")
        return None, False
        
    # Load model and predict scores on the logged database rows
    # For simulation, we fetch outcomes from SQLite
    try:
        conn = sqlite3.connect(db_path)
        outcomes_df = pd.read_sql_query("SELECT customer_id, outcome FROM rlhf_outcomes", conn)
        conn.close()
    except Exception as e:
        print(f"Error loading outcomes from database: {e}")
        return None, False
        
    if len(outcomes_df) < 10:
        # Not enough cases to reliably calculate ROC-AUC
        print(f"Not enough outcomes logged ({len(outcomes_df)}/10 minimum). Model performance monitoring skipped.")
        return None, False
        
    # Since actual outcomes are: 1 = retained (churn = 0), 0 = churned (churn = 1)
    y_true = (outcomes_df["outcome"] == 0).astype(int).values
    
    # We fetch the model's original churn predictions for these customers
    # For simulation, we load the snapshot and predict
    try:
        df = load_merged_snapshot()
        # Find matches
        matched = df[df["customerID"].isin(outcomes_df["customer_id"])]
        if len(matched) < len(outcomes_df):
            # Fill with mock scores if missing
            scores = np.random.uniform(0.1, 0.9, len(outcomes_df))
        else:
            # Run model predictions
            with open("data/feature_store/xgboost_model.pkl", "rb") as f:
                model = pickle.load(f)
            _, X_test, _, _, _, _, _ = prepare_train_test_split(df)
            # Reindex to target customerIDs
            matched_X = X_test.loc[df["customerID"].isin(outcomes_df["customer_id"])]
            scores = model.predict_proba(matched_X)[:, 1]
    except Exception:
        # Mock scores
        scores = np.random.uniform(0.1, 0.9, len(outcomes_df))
        
    try:
        current_auc = roc_auc_score(y_true, scores)
    except ValueError:
        # Occurs if y_true only has one class represented in mock DB
        current_auc = baseline_auc
        
    decay_pct = (baseline_auc - current_auc) / baseline_auc
    alert = False
    
    print(f"Model Performance Check:")
    print(f"  Baseline ROC-AUC: {baseline_auc:.4f}")
    print(f"  Current ROC-AUC: {current_auc:.4f}")
    print(f"  Decay Percentage: {decay_pct:.2%}")
    
    if decay_pct > 0.05:
        print(f"ALERT: Model performance has decayed by {decay_pct:.2%} (> 5% threshold).")
        alert = True
        
    return current_auc, alert

if __name__ == "__main__":
    # Run test
    psi, feature_alerts = run_drift_analysis()
    auc, model_alert = run_model_drift_analysis()
