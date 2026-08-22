import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Non-interactive backend for file generation
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, recall_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import optuna
import shap
import mlflow
import mlflow.xgboost
from utils import load_merged_snapshot, prepare_train_test_split

# Setup MLflow local registry
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Customer_Churn_XGBoost")

def run_optuna_tuning(X_train, y_train):
    print("Running Bayesian Hyperparameter Tuning with Optuna...")
    
    # Stratified K-Fold for validation
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 300), # Reduced slightly for faster local runs
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 5.0),
            "random_state": 42,
            "verbosity": 0
        }
        
        pr_aucs = []
        for train_idx, val_idx in cv.split(X_train, y_train):
            X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
            X_val, y_val = X_train.iloc[val_idx], y_train[val_idx]
            
            # Apply SMOTE to fold training data only
            smote = SMOTE(random_state=42)
            X_tr_res, y_tr_res = smote.fit_resample(X_tr, y_tr)
            
            model = xgb.XGBClassifier(**params)
            model.fit(X_tr_res, y_tr_res)
            
            # Predict probabilities
            probs = model.predict_proba(X_val)[:, 1]
            precision, recall, _ = precision_recall_curve(y_val, probs)
            pr_aucs.append(auc(recall, precision))
            
        return np.mean(pr_aucs)
        
    study = optuna.create_study(direction="maximize")
    # Limit to 30 trials for faster local execution during pairing session
    study.optimize(objective, n_trials=30)
    print(f"Optuna complete. Best PR-AUC: {study.best_value:.4f}")
    return study.best_params

def tune_business_threshold(y_true, y_probs, ltv=1680, cost_fp=50):
    """
    Finds the optimal threshold that minimizes the business cost:
    Total Cost = (FN * LTV) + (FP * Cost_FP)
    """
    thresholds = np.arange(0.10, 0.90, 0.01)
    best_threshold = 0.50
    min_cost = float('inf')
    best_costs = {}
    
    costs = []
    for t in thresholds:
        preds = (y_probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
        
        # FN is missed churner (we lose LTV)
        # FP is unnecessary intervention (discount / call cost)
        total_cost = (fn * ltv) + (fp * cost_fp)
        costs.append(total_cost)
        
        if total_cost < min_cost:
            min_cost = total_cost
            best_threshold = t
            best_costs = {
                "threshold": float(np.round(t, 2)),
                "cost_fn": ltv,
                "cost_fp": cost_fp,
                "min_cost": int(total_cost),
                "false_negatives": int(fn),
                "false_positives": int(fp),
                "true_positives": int(tp),
                "true_negatives": int(tn)
            }
            
    # Plot cost curve
    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, costs, color="#1e3a8a", linewidth=2)
    plt.axvline(best_threshold, color="#dc2626", linestyle="--", label=f"Optimal Threshold: {best_threshold:.2f}")
    plt.title("Business Cost Curve vs. Classification Threshold", fontsize=12)
    plt.xlabel("Probability Threshold")
    plt.ylabel("Business Cost ($)")
    plt.grid(True, linestyle=":")
    plt.legend()
    os.makedirs("config", exist_ok=True)
    plt.savefig("config/business_cost_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    return best_threshold, best_costs

def generate_shap_visualizations(model, X_test, feature_names):
    print("Generating SHAP explainability plots...")
    os.makedirs("config", exist_ok=True)
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)
    
    # 1. Global Bar Chart
    plt.figure()
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.title("XGBoost Churn Predictor - Global SHAP Importance", fontsize=12)
    plt.savefig("config/shap_summary_bar.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    # 2. Beeswarm Plot
    plt.figure()
    shap.plots.beeswarm(shap_values, max_display=15, show=False)
    plt.title("SHAP Beeswarm Plot (Impact of Feature Values)", fontsize=12)
    plt.savefig("config/shap_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    # 3. Waterfall Plot (For customer index 0)
    plt.figure()
    shap.plots.waterfall(shap_values[0], show=False)
    plt.savefig("config/shap_waterfall_sample.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    # 4. Dependence Plot (MonthlyCharges vs contract_type_encoded)
    # Map index of MonthlyCharges
    try:
        col_idx = feature_names.index("MonthlyCharges")
        plt.figure()
        # Find feature index to color by
        interact_idx = feature_names.index("contract_type_encoded")
        shap.dependence_plot("MonthlyCharges", shap_values.values, X_test, interaction_index="contract_type_encoded", show=False)
        plt.savefig("config/shap_dependence.png", dpi=150, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"Skipping SHAP dependence plot: {e}")

def main():
    print("Loading feature snapshot...")
    df = load_merged_snapshot()
    X_train, X_test, y_train, y_test, ids_train, ids_test, feature_names = prepare_train_test_split(df)
    
    # Run Tuning
    best_params = run_optuna_tuning(X_train, y_train)
    
    # Fit Final Model
    print("Fitting final model with optimal parameters...")
    # Apply SMOTE to the entire training split
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    final_model = xgb.XGBClassifier(**best_params)
    
    # Start MLflow Log
    with mlflow.start_run():
        final_model.fit(X_train_res, y_train_res)
        
        # Log Params
        mlflow.log_params(best_params)
        
        # Evaluate on Test Set
        probs = final_model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, probs)
        precision, recall, _ = precision_recall_curve(y_test, probs)
        pr_auc = auc(recall, precision)
        
        # Threshold Optimization
        best_t, best_costs = tune_business_threshold(y_test, probs)
        
        # Binary prediction at optimal threshold
        preds_at_t = (probs >= best_t).astype(int)
        rec_at_t = recall_score(y_test, preds_at_t)
        
        # Log Metrics
        mlflow.log_metric("test_roc_auc", roc_auc)
        mlflow.log_metric("test_pr_auc", pr_auc)
        mlflow.log_metric("optimal_threshold", best_t)
        mlflow.log_metric("recall_at_optimal_threshold", rec_at_t)
        mlflow.log_metric("min_business_cost", best_costs["min_cost"])
        
        # Log Model to MLflow Registry
        mlflow.xgboost.log_model(final_model, "model")
        
        print("\nFinal Model Evaluation (XGBoost):")
        print(f"  Test ROC-AUC: {roc_auc:.4f}")
        print(f"  Test PR-AUC: {pr_auc:.4f}")
        print(f"  Optimal Threshold: {best_t:.2f}")
        print(f"  Recall @ Threshold: {rec_at_t:.4f}")
        print(f"  Min Business Cost: ${best_costs['min_cost']}")
        
        # Save threshold JSON
        with open("config/optimal_threshold.json", "w") as f:
            json.dump(best_costs, f, indent=2)
            
        # Serialize model artifact
        feature_store_dir = "data/feature_store"
        with open(os.path.join(feature_store_dir, "xgboost_model.pkl"), "wb") as f:
            pickle.dump(final_model, f)
            
        # Overwrite feature names contract with full model training features
        with open(os.path.join(feature_store_dir, "feature_names.json"), "w") as f:
            json.dump(feature_names, f, indent=2)
            
        # Generate SHAP
        generate_shap_visualizations(final_model, X_test, feature_names)
        
        print("Final model saved and registered in MLflow registry.")

if __name__ == "__main__":
    main()
