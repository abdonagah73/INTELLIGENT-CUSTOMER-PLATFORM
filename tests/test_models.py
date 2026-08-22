import os
import pickle
import json
import numpy as np
import pandas as pd
import pytest

def test_preprocessor_loading():
    feature_store_dir = "data/feature_store"
    
    # Assert files exist
    assert os.path.exists(os.path.join(feature_store_dir, "scaler_v1.pkl"))
    assert os.path.exists(os.path.join(feature_store_dir, "encoder_v1.pkl"))
    assert os.path.exists(os.path.join(feature_store_dir, "feature_names.json"))
    
    # Load feature names contract
    with open(os.path.join(feature_store_dir, "feature_names.json"), "r") as f:
        feature_names = json.load(f)
    assert len(feature_names) > 0
    assert "tenure" in feature_names

def test_models_loading_and_inference():
    feature_store_dir = "data/feature_store"
    
    models = ["xgboost_model.pkl", "gradboost_model.pkl", "rf_model.pkl"]
    for m in models:
        model_path = os.path.join(feature_store_dir, m)
        assert os.path.exists(model_path)
        
        with open(model_path, "rb") as f:
            model = pickle.load(f)
            
        # Create a mock processed input record matching model features length
        # Using utils prepare_train_test_split to verify shape
        from src.models.utils import load_merged_snapshot, prepare_train_test_split
        df = load_merged_snapshot()
        X_train, X_test, y_train, y_test, _, _, _ = prepare_train_test_split(df)
        
        # Test prediction shape and output
        probs = model.predict_proba(X_test)[:, 1]
        assert len(probs) == len(X_test)
        assert np.all(probs >= 0.0) and np.all(probs <= 1.0)

def test_optimal_threshold_config():
    threshold_path = "config/optimal_threshold.json"
    assert os.path.exists(threshold_path)
    
    with open(threshold_path, "r") as f:
        data = json.load(f)
        
    assert "threshold" in data
    assert "min_cost" in data
    assert 0.0 <= data["threshold"] <= 1.0
