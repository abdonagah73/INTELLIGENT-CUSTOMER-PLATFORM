import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, recall_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import mlflow
from utils import load_merged_snapshot, prepare_train_test_split

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Customer_Churn_GradBoost")

def main():
    print("Loading feature snapshot for Gradient Boosting comparison...")
    df = load_merged_snapshot()
    X_train, X_test, y_train, y_test, ids_train, ids_test, feature_names = prepare_train_test_split(df)
    
    # Load optimal threshold to evaluate under same threshold
    threshold_path = "config/optimal_threshold.json"
    optimal_t = 0.32 # Default fallback
    if os.path.exists(threshold_path):
        with open(threshold_path, "r") as f:
            t_data = json.load(f)
            optimal_t = t_data.get("threshold", 0.32)
            
    print("Fitting Gradient Boosting Classifier with SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    # Set standard hyperparameters comparable to XGBoost
    model = GradientBoostingClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42
    )
    
    with mlflow.start_run():
        model.fit(X_train_res, y_train_res)
        
        # Predict probabilities
        probs = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, probs)
        precision, recall, _ = precision_recall_curve(y_test, probs)
        pr_auc = auc(recall, precision)
        
        # Evaluate at optimal threshold
        preds_at_t = (probs >= optimal_t).astype(int)
        rec_at_t = recall_score(y_test, preds_at_t)
        tn, fp, fn, tp = confusion_matrix(y_test, preds_at_t).ravel()
        total_cost = (fn * 1680) + (fp * 50)
        
        # Log Metrics
        mlflow.log_metric("test_roc_auc", roc_auc)
        mlflow.log_metric("test_pr_auc", pr_auc)
        mlflow.log_metric("recall_at_optimal_threshold", rec_at_t)
        mlflow.log_metric("business_cost", total_cost)
        
        print("\nGradient Boosting Evaluation:")
        print(f"  Test ROC-AUC: {roc_auc:.4f}")
        print(f"  Test PR-AUC: {pr_auc:.4f}")
        print(f"  Recall @ Threshold ({optimal_t}): {rec_at_t:.4f}")
        print(f"  Business Cost at Threshold: ${total_cost}")
        
        # Serialize model
        feature_store_dir = "data/feature_store"
        with open(os.path.join(feature_store_dir, "gradboost_model.pkl"), "wb") as f:
            pickle.dump(model, f)
            
    print("Gradient Boosting comparison model saved and logged.")

if __name__ == "__main__":
    main()
