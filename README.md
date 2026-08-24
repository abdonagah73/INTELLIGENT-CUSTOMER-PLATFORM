# 🧠 Intelligent Customer Retention & Intelligence Platform 🚀

> **⚡ An enterprise AI platform for customer retention featuring XGBoost predictive modeling (99.47% recall) 🎯, SHAP XAI interpretability 🔮, FAISS & LangChain RAG briefings 🤖, FastAPI REST backend 🚀, closed-loop MLOps with RLHF sample re-weighting 🔄, & a Glassmorphism UI compatible with static GitHub Pages hosting 🌐**

---

## 👥 Project Team Members 🎓
- 👤 **Abdelrahman Mohamed Nagah**
- 👤 **Ahmed Adel Abdelaziz**
- 👤 **Ahmed Waled Abdel-Satar**
- 👤 **Adham Maged Mohamed**

---

## 🚀 Live Demo & Presentation Links 🔗

| Resource | Link | Description |
| :--- | :--- | :--- |
| 🌐 **Live Web Application** | [index.html](index.html) | 💎 Interactive Glassmorphism UI with dual live API & client-side inference engine. |
| 📊 **Official Project Infographic** | [Infographic_Official.html](Infographic_Official.html) | 🎨 High-resolution architectural infographic poster of the retention ecosystem. |
| 🎓 **AI Presentation Deck (Web)** | [presentation.html](presentation.html) | 🖥️ Full-screen 10-slide interactive slide deck for course presentations. |
| 📄 **PowerPoint File (.pptx)** | [AI_Customer_Retention_Platform_Presentation.pptx](AI_Customer_Retention_Platform_Presentation.pptx) | 💾 Downloadable presentation deck formatted for Microsoft PowerPoint. |

---

## 📈 Key Metrics & Achievements 🏆

- 🎯 **At-Risk Recall Rate**: `99.47%` *(Cost-sensitive classification capturing almost all potential churners)*
- 📊 **Model ROC-AUC Score**: `0.8317` *(High discrimination capability on validation sets)*
- ⚖️ **Optimal Decision Threshold**: `0.16 / 0.32` *(Cost-utility matrix optimized for maximum financial ROI)*
- 💰 **Projected Business Impact**: `$4.15 Million` *(Estimated net retention revenue saved)*

---

## 🏗️ System Architecture & Core Capabilities ⚙️

```
+-----------------------------------------------------------------------------------+
|                        🔮 INTELLIGENT RETENTION ECOSYSTEM 🔮                      |
+-----------------------------------------------------------------------------------+
   │
   ├── 📁 1. Data Ingestion & Schema Contracts (JSON Schema Validation) 🛡️
   ├── 🧹 2. Feature Store & Preprocessing (StandardScaler + OneHotEncoder + Numeric Cast) ⚡
   ├── 🎯 3. Predictive Model Engine (XGBoost Classifier + Optuna Tuning) 📊
   ├── 🔮 4. Explainable AI / XAI (SHAP TreeExplainer for Top-3 Churn Risk Drivers) 🔍
   ├── 🤖 5. Generative AI & RAG (FAISS Vector Index + HuggingFace Embeddings + LangChain) 💡
   ├── 🚀 6. Serving Microservice (FastAPI ASGI Server + Flexible Pydantic Schemas) 📡
   ├── 🔄 7. Closed-Loop MLOps & RLHF (SQLite Feedback Tracking + 2.0x Sample Reweighter) 📈
   └── 💎 8. Hybrid Frontend UI (Glassmorphism Web Engine + Static GitHub Pages Fallback) 🌐
```

---

## 🛠️ Tech Stack 💻

- 🎯 **Machine Learning**: `XGBoost` 🚀, `Scikit-Learn` 🐍, `Optuna` 🎛️, `Pandas` 🐼, `NumPy` 🔢
- 🔮 **Explainable AI (XAI)**: `SHAP` *(Shapley Additive exPlanations)* 💡
- 🤖 **Generative AI & RAG**: `FAISS` ⚡, `HuggingFace Transformers` 🧠 (`all-MiniLM-L6-v2`), `LangChain` 🔗
- 📡 **Serving & MLOps**: `FastAPI` ⚡, `Uvicorn` 🦄, `Pydantic v2` 🛡️, `Evidently AI` 📉, `SQLite3` 🗄️
- 🎨 **Frontend & Web UI**: HTML5 🌐, Vanilla CSS3 ✨ *(Glassmorphism)*, JavaScript ⚡ *(ES6+)*, FontAwesome 🎭

---

## 📂 Project Structure 📁

```
├── index.html                                 # 🌐 Root Web App for GitHub Pages deployment
├── Infographic_Official.html                  # 🎨 Official Architectural Poster
├── presentation.html                          # 🖥️ Interactive Slide Deck for AI Presentation
├── AI_Customer_Retention_Platform_Presentation.pptx # 💾 Downloadable PowerPoint File
├── config/                                    # ⚙️ Schema contracts & threshold configs
│   ├── optimal_threshold.json
│   └── schema_contract.json
├── data/                                      # 🗄️ Feature Store & Datasets
│   ├── feature_store/                         # 📦 Scalers, Encoders, Feature Names & XGBoost models
│   ├── raw/                                   # 📄 Raw telco & banking datasets
│   └── rlhf_outcomes.db                       # 🔄 SQLite RLHF outcome feedback database
├── src/                                       # 💻 Core Source Code
│   ├── data/                                  # 🧹 Ingestion & ETL Pipeline
│   │   ├── etl_pipeline.py
│   │   └── ingest_kaggle.py
│   ├── models/                                # 🎯 Model Training & Optuna Tuning
│   │   ├── train_xgboost.py
│   │   ├── train_gradboost.py
│   │   └── train_rf.py
│   ├── RAG/                                   # 🤖 RAG Indexing & Vector Search
│   │   ├── build_vector_db.py
│   │   └── rag_pipeline.py
│   ├── serving/                               # 📡 FastAPI REST Microservice
│   │   └── main.py
│   ├── mlops/                                 # 🔄 Drift Monitoring & RLHF Reweighting
│   │   ├── drift_monitor.py
│   │   ├── rlhf_reweighter.py
│   │   └── retrain_pipeline.py
│   └── dashboard/                             # 📊 Streamlit Business Dashboard
│       └── dashboard.py
└── requirements.txt                           # 📜 Python Dependencies
```

---

## ⚡ Quickstart & Execution Guide 🚀

### 1. Installation 📥
```bash
# 📦 Clone repository
git clone https://github.com/YOUR_USERNAME/INTELLIGENT-CUSTOMER-PLATFORM.git
cd INTELLIGENT-CUSTOMER-PLATFORM

# 🐍 Install dependencies
pip install -r requirements.txt
```

### 2. Run Data Processing & Model Training 🎯
```bash
# 🧹 1. ETL Data Pipeline
python src/data/etl_pipeline.py

# 🎯 2. Train XGBoost Model & Run Optuna Optimization
python src/models/train_xgboost.py

# 🤖 3. Build FAISS Vector Database for RAG
python src/RAG/build_vector_db.py
```

### 3. Start Serving API & Web App 📡
```bash
# ⚡ Launch FastAPI backend on port 8000
python -m uvicorn src.serving.main:app --host 0.0.0.0 --port 8000
```
🌐 Open **`http://localhost:8000`** in your browser to interact with the system!

### 4. Start Streamlit Business Dashboard (Optional) 📊
```bash
python -m streamlit run src/dashboard/dashboard.py --server.port 8501
```

---

## 🌐 GitHub Pages Deployment 💎

This project includes a **Hybrid Client-Side Model Inference Engine** 🤖 inside `index.html`. 
When hosted on **GitHub Pages** without a Python backend server running, the web application automatically falls back to client-side risk scoring and SHAP driver extraction, allowing full interactive demonstrations with zero server deployment overhead! 🎉

---

## 📜 License & Credits 🎓

✨ Built as part of the **Artificial Intelligence Course Project (2026)**.  
👨‍💻 Developed by **Abdelrahman Mohamed Nagah**, **Ahmed Adel Abdelaziz**, **Ahmed Waled Abdel-Satar**, and **Adham Maged Mohamed**. 🌟
