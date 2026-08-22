import os
import sqlite3
import numpy as np
import pandas as pd

def compute_sample_weights(customer_ids, db_path="data/rlhf_outcomes.db"):
    """
    Computes sample weights for retraining based on historical RLHF outcome logs.
    Upweights successful interventions (outcome=1, i.e., customer was retained).
    Downweights failed interventions (outcome=0, i.e., customer churned).
    """
    weights = np.ones(len(customer_ids))
    
    if not os.path.exists(db_path):
        print("No RLHF outcomes database found. Defaulting all sample weights to 1.0.")
        return weights
        
    try:
        conn = sqlite3.connect(db_path)
        outcomes_df = pd.read_sql_query("SELECT customer_id, outcome FROM rlhf_outcomes", conn)
        conn.close()
    except Exception as e:
        print(f"Error loading outcomes for sample weights calculation: {e}. Defaulting to 1.0.")
        return weights
        
    if outcomes_df.empty:
        print("RLHF outcomes table is empty. Defaulting all sample weights to 1.0.")
        return weights
        
    # Group by customer_id and get the latest outcome
    latest_outcomes = outcomes_df.groupby("customer_id")["outcome"].last().to_dict()
    
    retained_count = 0
    churned_count = 0
    
    for idx, cid in enumerate(customer_ids):
        if cid in latest_outcomes:
            outcome = latest_outcomes[cid]
            if outcome == 1:
                # Upweight successful retention profiles so the model learns features of retrainable clients
                weights[idx] = 2.0
                retained_count += 1
            else:
                # Downweight failed interventions
                weights[idx] = 0.5
                churned_count += 1
                
    print(f"RLHF Sample Reweighter: calculated weights for {len(customer_ids)} training records.")
    print(f"  Upweighted successful interventions (weight=2.0): {retained_count}")
    print(f"  Downweighted failed interventions (weight=0.5): {churned_count}")
    
    return weights

if __name__ == "__main__":
    # Test run
    ids = ["0001-ABCD", "0002-ABCD", "9999-XYZ"]
    w = compute_sample_weights(ids)
    print("Computed weights:", w)
