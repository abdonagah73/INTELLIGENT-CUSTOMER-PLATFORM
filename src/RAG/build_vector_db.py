import os
import json
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import sys

# Add models path to load snapshot
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models")))
from utils import load_merged_snapshot

def load_playbook(playbook_path="data/retention_playbook.txt"):
    """
    Loads retention playbook and splits it into individual sections.
    """
    if not os.path.exists(playbook_path):
        return []
        
    with open(playbook_path, "r") as f:
        content = f.read()
        
    # Split on playbook section tag
    parts = content.split("[PLAYBOOK SECTION:")
    sections = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        sections.append(f"[PLAYBOOK SECTION: {part}")
    return sections

def generate_customer_profiles(df, limit=100):
    """
    Generates natural language summary profiles for historically churned customers.
    """
    # Select churned customers
    churn_df = df[df["Churn"] == 1].head(limit)
    profiles = []
    
    for _, row in churn_df.iterrows():
        cid = row["customerID"]
        ten = row["tenure"]
        charges = row["MonthlyCharges"]
        contract = "month-to-month" if row["contract_type_encoded"] == 0 else "yearly"
        sec = "Yes" if row["OnlineSecurity"] == 1 else "No"
        support = "Yes" if row["TechSupport"] == 1 else "No"
        sent = row["sentiment_score"]
        esc = "Yes" if row["escalation_flag"] == 1 else "No"
        
        # Interventions and outcomes are simulated for historical context
        np.random.seed(hash(cid) % 10000)
        intervention = np.random.choice([
            "Offered 10% monthly discount.",
            "Offered free tech support upgrade.",
            "Offered contract lock price discount.",
            "Offered account waiver credit."
        ])
        outcome = np.random.choice([
            "Customer churned anyway.",
            "Customer was retained for 3 months then churned."
        ])
        
        profile = (
            f"[HISTORICAL CASE] Customer ID: {cid}. "
            f"Profile: tenure {ten} months, contract {contract}, MonthlyCharges ${charges:.2f}, "
            f"OnlineSecurity {sec}, TechSupport {support}. "
            f"Support History: sentiment {sent:.2f}, escalation {esc}. "
            f"Intervention Taken: {intervention} Outcome: {outcome}"
        )
        profiles.append(profile)
        
    return profiles

def main():
    print("Building FAISS Vector Database index...")
    
    # 1. Load playbook
    playbook_docs = load_playbook()
    print(f"Loaded {len(playbook_docs)} playbook sections.")
    
    # 2. Load and generate customer profiles
    try:
        df = load_merged_snapshot()
        customer_docs = generate_customer_profiles(df)
        print(f"Generated {len(customer_docs)} historical churn profile documents.")
    except Exception as e:
        print(f"Could not load snapshot for profiles ({e}). Using playbook docs only.")
        customer_docs = []
        
    all_docs = playbook_docs + customer_docs
    if not all_docs:
        print("No documents found to index.")
        return
        
    # 3. Embed documents using SentenceTransformer
    print("Embedding documents using all-MiniLM-L6-v2...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(all_docs, show_progress_bar=True)
    
    # Convert embeddings to float32
    embeddings = np.array(embeddings).astype("float32")
    dimension = embeddings.shape[1]
    
    # 4. Create FAISS index
    print(f"Creating FAISS index with dimension {dimension}...")
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    # 5. Save FAISS index and metadata mappings
    os.makedirs("data", exist_ok=True)
    faiss.write_index(index, "data/faiss_index.bin")
    
    # Save the original texts corresponding to vector index IDs
    metadata = {
        "dimension": dimension,
        "total_documents": len(all_docs),
        "documents": all_docs
    }
    with open("data/faiss_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
        
    print("FAISS Index and metadata saved successfully to data/faiss_index.bin and data/faiss_metadata.json.")

if __name__ == "__main__":
    main()
