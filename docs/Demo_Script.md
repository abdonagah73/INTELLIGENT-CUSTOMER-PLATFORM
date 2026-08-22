# Live Demo Script & Run Sequence

This document describes the scripted sequence for the 15-minute end-to-end platform demo.

---

## Demo Sequence Overview

| Time | Presenter | Section | Action / Screen Shared |
| :--- | :--- | :--- | :--- |
| **00:00 - 02:00** | Person 1 | **Data Overview** | Show raw folder layout. Open Parquet feature store files, show schema contract compliance, and explain Bank merge join metrics. |
| **02:00 - 05:00** | Person 3 | **Model Demonstration** | Open MLflow local DB console. Show Optuna tuning history, parameter importance, and SHAP beeswarm/cost curves in `/config/`. |
| **05:00 - 08:00** | Person 4 | **Live API Call** | Start Uvicorn service. Run curl `POST /predict` with a mock customer ID. Show returned score, SHAP drivers, and RAG briefing. |
| **08:00 - 11:00** | Person 6 | **Dashboard Walkthrough** | Open Streamlit UI. Show risk scores table, filter by contract, expand customer cards, show RAG briefs, and query NL console. |
| **11:00 - 13:00** | Person 5 | **RLHF Loop Demo** | Log a simulated campaign outcome from the UI. Trigger `drift_monitor.py` drift alert, and run `retrain_pipeline.py`. |
| **13:00 - 15:00** | All | **Roadmap & Q&A** | Display Future Roadmap slides and open the floor to client questions. |

---

## Presenter Scripts

### 1. Data Overview (Person 1)
*"We start by showing the immutable raw data layer where CSV files land from Kaggle, APIs, and BeautifulSoup scraping. When the ETL script runs, it merges these sources on customerID and fits our scaler and encoder. The result is five versioned Parquet feature groups saved under `data/feature_store/` along with `manifest.json` which logs metadata. The matching rate between Telco and Bank is 100%, and TotalCharges null values are fully resolved."*

### 2. Model Demonstration (Person 3)
*"Now looking at the training phase, we fetch snapshot v1.0 data. We run `train_xgboost.py` which executes SMOTE to balance the churn class and runs 30 Optuna trials maximizing PR-AUC. In this run, the best PR-AUC achieved is 0.8329. We apply business cost tuning using LTV ($1,680) and FP cost ($50). The cost curve minimum lands the classification threshold at 0.10. We also calculate per-prediction SHAP values, generating global importance, beeswarm, and waterfall plots."*

### 3. Live API Call (Person 4)
*"With the model registered in MLflow, we run the FastAPI serving container. By querying `POST /predict` with a customer JSON, the API fetches feature store metrics, preprocesses them, calls the booster to get the churn score, identifies the top 3 SHAP drivers, and integrates the RAG pipeline to generate a plain-English briefing detailing why the customer is at risk and the exact strategy to deploy."*

### 4. Dashboard Walkthrough (Person 6)
*"The Streamlit dashboard aggregates these components. Support managers see a sortable risk scoring table, metric KPIs, feature PSI drift warnings, and campaign success stats. Clicking on a customer displays their RAG brief. The natural language search box allows users to type queries like 'Which price-sensitive customers are at risk?' which performs FAISS similarity matching to return results immediately."*

### 5. RLHF Loop Demo (Person 5)
*"Finally, we demonstrate the retraining loop. By logging an intervention outcome in the UI, data is saved to `rlhf_outcomes.db`. When feature drift is detected, `retrain_pipeline.py` starts: it pulls the latest snapshot, computes RLHF sample weights (2.0 for retained, 0.5 for churned), retrains XGBoost, runs staging integration tests, and replaces the production model backup file, closing the pipeline loop."*
