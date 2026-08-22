import os
import pickle
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, recall_score, confusion_matrix
import mlflow
from utils import load_merged_snapshot, prepare_train_test_split

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Customer_Churn_RandomForest")

def main():
    print("Loading feature snapshot for Random Forest baseline...")
    df = load_merged_snapshot()
    X_train, X_test, y_train, y_test, ids_train, ids_test, feature_names = prepare_train_test_split(df)
    
    # Load optimal threshold to evaluate under same threshold
    threshold_path = "config/optimal_threshold.json"
    optimal_t = 0.32
    if os.path.exists(threshold_path):
        with open(threshold_path, "r") as f:
            t_data = json.load(f)
            optimal_t = t_data.get("threshold", 0.32)
            
    print("Fitting Random Forest Classifier with balanced class weights...")
    model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    
    with mlflow.start_run():
        model.fit(X_train, y_train)
        
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
        
        print("\nRandom Forest baseline Evaluation:")
        print(f"  Test ROC-AUC: {roc_auc:.4f}")
        print(f"  Test PR-AUC: {pr_auc:.4f}")
        print(f"  Recall @ Threshold ({optimal_t}): {rec_at_t:.4f}")
        print(f"  Business Cost at Threshold: ${total_cost}")
        
        # Compute Permutation Feature Importance
        print("Calculating Permutation Feature Importance...")
        perm_importance = permutation_importance(
            model, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1
        )
        
        # Sort and save top 15 permutation importances
        sorted_idx = perm_importance.importances_mean.argsort()[::-1]
        top_importances = []
        for idx in sorted_idx[:15]:
            top_importances.append({
                "feature": feature_names[idx],
                "importance_mean": float(perm_importance.importances_mean[idx]),
                "importance_std": float(perm_importance.importances_std[idx])
            })
            
        # Save permutation importance JSON
        os.makedirs("config", exist_ok=True)
        with open("config/permutation_importance.json", "w") as f:
            json.dump(top_importances, f, indent=2)
            
        print("Permutation Feature Importance complete. Saved top 15 features.")
        
        # Serialize model
        feature_store_dir = "data/feature_store"
        with open(os.path.join(feature_store_dir, "rf_model.pkl"), "wb") as f:
            pickle.dump(model, f)
            
    print("Random Forest baseline model saved and logged.")

if __name__ == "__main__":
    main()
