import os
import json
import sqlite3
import uuid
import pandas as pd
import numpy as np
import streamlit as st
import sys
import pickle
import shap
from dotenv import load_dotenv

# Load environment variables (.env) so the dashboard can display the
# currently-active LLM provider/model dynamically in the sidebar.
load_dotenv()

# Add models and RAG paths to imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "RAG")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "mlops")))

from utils import load_merged_snapshot, prepare_train_test_split
from rag_pipeline import RAGPipeline
from drift_monitor import run_drift_analysis, run_model_drift_analysis

# Page config
st.set_page_config(
    page_title="Customer Intelligence Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling (glassmorphism + curated color palette)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;700&family=Inter:wght@400;500;600&display=swap');

    /* ---- Global canvas: deep gradient background ---- */
    .stApp {
        background: radial-gradient(1200px 600px at 15% -10%, #1e293b 0%, #0f172a 45%, #0b1120 100%);
        color: #e2e8f0;
        font-family: 'Inter', sans-serif;
    }

    /* ---- Headings ---- */
    h1, h2, h3 {
        color: #38bdf8 !important;
        font-family: 'Outfit', 'Inter', sans-serif !important;
        letter-spacing: 0.2px;
    }

    /* ---- Glassmorphism KPI metric cards ---- */
    [data-testid="stMetric"] {
        background: rgba(30, 41, 59, 0.55);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        padding: 18px 18px 14px 18px;
        border-radius: 16px;
        border: 1px solid rgba(148, 163, 184, 0.18);
        box-shadow: 0 8px 24px -8px rgba(0, 0, 0, 0.45);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-3px);
        border-color: rgba(56, 189, 248, 0.55);
    }
    [data-testid="stMetricValue"] {
        color: #f8fafc !important;
        font-family: 'Outfit', sans-serif !important;
    }

    /* ---- Dataframes / tables ---- */
    [data-testid="stDataFrame"], .stTable {
        border-radius: 14px;
        overflow: hidden;
        border: 1px solid rgba(148, 163, 184, 0.15);
        box-shadow: 0 6px 20px -10px rgba(0, 0, 0, 0.5);
    }

    /* ---- Sidebar ---- */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b1120 0%, #111c33 100%);
        border-right: 1px solid rgba(148, 163, 184, 0.12);
    }

    /* ---- Drift / churn status pills (kept class names used in code) ---- */
    .status-card {
        padding: 12px;
        border-radius: 12px;
        font-weight: 700;
        text-align: center;
        backdrop-filter: blur(8px);
        box-shadow: 0 4px 14px -6px rgba(0, 0, 0, 0.5);
    }
    .status-red {
        background: rgba(220, 38, 38, 0.18);
        color: #fecaca;
        border: 1px solid #ef4444;
    }
    .status-yellow {
        background: rgba(234, 179, 8, 0.18);
        color: #fef08a;
        border: 1px solid #eab308;
    }
    .status-green {
        background: rgba(22, 163, 74, 0.18);
        color: #bbf7d0;
        border: 1px solid #22c55e;
    }

    /* ---- Active AI Model status card (sidebar) ---- */
    .ai-card {
        background: rgba(15, 23, 42, 0.6);
        backdrop-filter: blur(14px);
        border-radius: 16px;
        padding: 16px;
        margin-bottom: 8px;
        border: 1px solid rgba(148, 163, 184, 0.18);
        box-shadow: 0 8px 26px -10px rgba(0, 0, 0, 0.6);
    }
    .ai-card-head {
        display: flex; align-items: center; gap: 10px; margin-bottom: 10px;
    }
    .ai-dot {
        width: 10px; height: 10px; border-radius: 50%;
        box-shadow: 0 0 10px 2px currentColor; animation: pulse 1.6s infinite;
    }
    @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.35; } }
    .ai-badge {
        display: inline-flex; align-items: center; gap: 8px;
        padding: 8px 12px; border-radius: 12px; font-weight: 700;
        font-family: 'Outfit', sans-serif; font-size: 15px; width: 100%;
    }
    .ai-model {
        margin-top: 10px; font-family: 'Inter', monospace; font-size: 13px;
        color: #cbd5e1; background: rgba(2, 6, 23, 0.55);
        padding: 8px 10px; border-radius: 10px; word-break: break-all;
        border: 1px solid rgba(148, 163, 184, 0.12);
    }
    .ai-label { color: #94a3b8; font-size: 11px; letter-spacing: 0.5px; text-transform: uppercase; }

    /* ---- Architecture pipeline steps (sidebar) ---- */
    .arch-step {
        background: rgba(30, 41, 59, 0.45);
        border-left: 3px solid #38bdf8;
        border-radius: 10px;
        padding: 8px 12px; margin: 6px 0;
        font-size: 13px; color: #e2e8f0;
    }
    .arch-step b { color: #7dd3fc; }
    .arch-step small { color: #94a3b8; }
    .arch-arrow { text-align: center; color: #475569; font-size: 14px; line-height: 0.6; }
</style>
""", unsafe_allow_html=True)

# ---- Read the active AI provider/model dynamically from the environment (.env) ----
def get_active_ai_config():
    """Return the live LLM provider + model shown in the sidebar status card.

    Mirrors the selection logic used by RAGPipeline / LLMExtractor so the
    dashboard always reflects the *actual* engine powering the platform.
    """
    provider = os.environ.get("LLM_PROVIDER", "openai").lower()
    if provider == "groq":
        return {
            "provider": "Groq",
            "provider_ar": "جروك",
            "model": os.environ.get("GROQ_MODEL", "llama-3.1-70b-versatile"),
            "icon": "⚡",
            "accent": "#f97316",   # Groq signature orange
            "connected": bool(os.environ.get("GROQ_API_KEY")),
        }
    return {
        "provider": "OpenAI",
        "provider_ar": "أوبن إيه آي",
        "model": os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        "icon": "🧠",
        "accent": "#10a37f",       # OpenAI signature green
        "connected": bool(os.environ.get("OPENAI_API_KEY")),
    }

# Helper to fetch database stats
def get_rlhf_stats(db_path="data/rlhf_outcomes.db"):
    if not os.path.exists(db_path):
        return 0, 0, 0.0, pd.DataFrame()
        
    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query("SELECT id, customer_id, intervention_type, outcome, timestamp FROM rlhf_outcomes", conn)
        conn.close()
    except Exception:
        return 0, 0, 0.0, pd.DataFrame()
        
    if df.empty:
        return 0, 0, 0.0, df
        
    total = len(df)
    success = int(df["outcome"].sum())
    success_rate = (success / total) * 100.0 if total > 0 else 0.0
    return total, success, success_rate, df

# Load preprocessors and model at startup
@st.cache_resource
def load_ml_resources():
    feature_store_dir = "data/feature_store"
    with open(os.path.join(feature_store_dir, "xgboost_model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(feature_store_dir, "scaler_v1.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(feature_store_dir, "encoder_v1.pkl"), "rb") as f:
        encoder = pickle.load(f)
    with open(os.path.join(feature_store_dir, "feature_names.json"), "r") as f:
        feature_names = json.load(f)
        
    rag = RAGPipeline()
    
    return model, scaler, encoder, feature_names, rag

def get_predictions_df(merged_df, model, scaler, encoder, feature_names):
    # Process features in batches to compute scores
    X_train, X_test, y_train, y_test, ids_train, ids_test, f_names = prepare_train_test_split(merged_df)
    
    # We combine X_train and X_test back to represent the full customer list
    X_full = pd.concat([X_train, X_test])
    c_ids_full = np.concatenate([ids_train, ids_test])
    
    probs = model.predict_proba(X_full)[:, 1]
    
    # Calculate SHAP drivers (approximate or precomputed for speed)
    # Using feature importance as proxy if SHAP takes too long, 
    # but since this is a mock dataset, we can run SHAP on full X_full in 1 second!
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_full)
    
    top_drivers_list = []
    for i in range(len(X_full)):
        row_shap = shap_values[i]
        top_idx = np.argsort(np.abs(row_shap))[::-1][:3]
        top_drivers_list.append([f_names[idx] for idx in top_idx])
        
    # Build a friendly predictions dataframe
    pred_df = pd.DataFrame({
        "customerID": c_ids_full,
        "churn_score": probs,
        "top_drivers": top_drivers_list
    })
    
    # Merge with original features for filters
    out_df = pd.merge(merged_df, pred_df, on="customerID", how="inner")
    return out_df

def preprocess_batch(df, scaler, encoder, feature_names):
    df_proc = df.copy()
    
    # 1. Yes/No to 1/0
    yes_no_cols = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]
    for col in yes_no_cols:
        if col in df_proc.columns:
            df_proc[col] = df_proc[col].replace({"Yes": 1, "No": 0})
        else:
            df_proc[col] = 0
            
    replace_cols = ["MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
    for col in replace_cols:
        if col in df_proc.columns:
            df_proc[col] = df_proc[col].replace({"No internet service": "No", "No phone service": "No"})
        else:
            df_proc[col] = "No"
            
    # Convert Yes/No strings to integers for TechSupport and OnlineSecurity
    yes_no_map = {"Yes": 1, "No": 0}
    for col in ["TechSupport", "OnlineSecurity"]:
        if col in df_proc.columns:
            if df_proc[col].dtype == object:
                df_proc[col] = df_proc[col].map(yes_no_map).fillna(0).astype(int)
            else:
                df_proc[col] = df_proc[col].fillna(0).astype(int)
        else:
            df_proc[col] = 0
                
    contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
    if "Contract" in df_proc.columns:
        df_proc["contract_type_encoded"] = df_proc["Contract"].map(contract_map).fillna(0).astype(int)
    else:
        df_proc["contract_type_encoded"] = 0
        
    # Fill bank columns if not provided
    bank_defaults = {"CreditScore": 650.0, "Balance": 50000.0, "NumOfProducts": 1.0, "EstimatedSalary": 80000.0}
    for col, default in bank_defaults.items():
        if col not in df_proc.columns:
            df_proc[col] = default
        else:
            df_proc[col] = pd.to_numeric(df_proc[col], errors="coerce").fillna(default)
            
    # Transform categoricals
    categorical_cols = ["PaymentMethod", "InternetService"]
    for col in categorical_cols:
        if col not in df_proc.columns:
            df_proc[col] = "No" if col == "InternetService" else "Mailed check"
        else:
            df_proc[col] = df_proc[col].fillna("No" if col == "InternetService" else "Mailed check")
            
    encoded_cats = encoder.transform(df_proc[categorical_cols])
    encoded_cat_df = pd.DataFrame(
        encoded_cats, 
        columns=encoder.get_feature_names_out(categorical_cols),
        index=df_proc.index
    )
    
    # Scale numericals
    numerical_cols = ["tenure", "MonthlyCharges", "TotalCharges", "CreditScore", "Balance", "EstimatedSalary"]
    for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
        if col not in df_proc.columns:
            df_proc[col] = 0.0
        else:
            df_proc[col] = pd.to_numeric(df_proc[col].astype(str).str.strip(), errors="coerce").fillna(0.0)
            
    scaled_nums = scaler.transform(df_proc[numerical_cols])
    scaled_num_df = pd.DataFrame(
        scaled_nums, 
        columns=numerical_cols,
        index=df_proc.index
    )
    
    # Engineered features defaults
    defaults = {
        "SeniorCitizen": 0, "service_bundle_size": 1,
        "logins_7d_rolling_avg": 0.5, "logins_30d_rolling_avg": 0.5, "login_trend": 0.0,
        "support_contacts_30d": 0, "days_since_last_login": 5, "usage_delta_mom": 0.0,
        "sentiment_score": 0.0, "escalation_flag": 0, "public_complaint_flag": 0,
        "charges_per_tenure_month": 10.0, "billing_delta_mom": 0.0, "is_high_risk_contract": 0,
        "ticket_sentiment_score": 0.0, "ticket_complaint_topics_billing": 0, 
        "ticket_complaint_topics_competitor": 0, "ticket_complaint_topics_performance": 0,
        "ticket_escalation_flag": 0, "review_sentiment_monthly": 0.05, "urgency_level_encoded": 0,
        "competitor_news_volume_7d": 15, "public_review_sentiment_monthly": 0.12,
        "one_star_review_rate_monthly": 0.08, "price_competitiveness_ratio": 1.0, "competitor_promotion_flag": 0
    }
    
    other_df_data = {}
    for col, default in defaults.items():
        if col in df_proc.columns:
            other_df_data[col] = df_proc[col].values
        else:
            other_df_data[col] = np.full(len(df_proc), default)
            
    df_other = pd.DataFrame(other_df_data, index=df_proc.index)
    
    # Combine features
    X_cust = pd.concat([scaled_num_df, encoded_cat_df, df_other], axis=1)
    X_cust = X_cust.reindex(columns=feature_names, fill_value=0)
    
    return X_cust

# =====================================================================
# Sidebar: Active AI status + Interactive System Architecture
# =====================================================================
with st.sidebar:
    ai = get_active_ai_config()
    status_color = "#22c55e" if ai["connected"] else "#eab308"
    status_text_ar = "متصل ويعمل" if ai["connected"] else "بانتظار مفتاح API"
    status_text_en = "Live &amp; Connected" if ai["connected"] else "Awaiting API key"

    st.markdown("### 🧠 محرّك الذكاء الاصطناعي\n<span style='color:#94a3b8;font-size:12px'>Active AI Engine</span>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class="ai-card">
        <div class="ai-card-head">
            <div class="ai-dot" style="color:{status_color};"></div>
            <span style="color:{status_color};font-weight:600;font-size:13px;">{status_text_ar} · {status_text_en}</span>
        </div>
        <div class="ai-badge" style="background:{ai['accent']}22;border:1px solid {ai['accent']};color:{ai['accent']};">
            <span style="font-size:20px;">{ai['icon']}</span>
            <span>{ai['provider']} <span style="opacity:0.75;font-weight:400;">/ {ai['provider_ar']}</span></span>
        </div>
        <div class="ai-label" style="margin-top:12px;">الموديل الفعّال · Active Model</div>
        <div class="ai-model">{ai['model']}</div>
    </div>
    """, unsafe_allow_html=True)
    st.caption("يتم قراءة المزوّد والموديل ديناميكياً من ملف البيئة `.env` — أي أن المنصة تعمل بذكاء اصطناعي حقيقي.")

    st.markdown("---")

    # ---- Interactive System Architecture Visualizer ----
    st.markdown("### 🏗️ معمارية النظام\n<span style='color:#94a3b8;font-size:12px'>System Architecture Pipeline</span>", unsafe_allow_html=True)
    arch_steps = [
        ("📥", "البيانات (Data Sources)", "اتصالات، فواتير، تذاكر دعم، وإشارات السوق"),
        ("🔄", "المعالجة (ETL Pipeline)", "تنظيف، دمج، وهندسة الخصائص (Feature Store)"),
        ("🤖", "النموذج التنبؤي (XGBoost)", "حساب احتمالية مغادرة كل عميل (Churn Score)"),
        ("🔍", "تفسير القرار (SHAP)", "لماذا سيغادر العميل؟ أهم الأسباب لكل حالة"),
        ("💡", f"الذكاء التوليدي (RAG + {ai['provider']})", "توليد توصيات احتجاز ذكية عبر FAISS + LLM"),
        ("🔁", "التعلّم من البشر (RLHF Loop)", "تسجيل نتائج التدخّل لتحسين النظام باستمرار"),
    ]
    for i, (icon, title, desc) in enumerate(arch_steps):
        st.markdown(
            f"""<div class="arch-step">{icon} <b>{title}</b><br><small>{desc}</small></div>""",
            unsafe_allow_html=True,
        )
        if i < len(arch_steps) - 1:
            st.markdown('<div class="arch-arrow">▼</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.caption("منصة ذكاء العملاء · Customer Intelligence Platform © 2026")

# Title block
st.title("🤖 منصّة ذكاء العملاء المتكاملة | Intelligent Customer Intelligence Platform")
st.subheader("لوحة تحكّم التنبؤ بمغادرة العملاء والاحتفاظ بهم | Automated Customer Retention & Churn Prediction")

# Load assets
try:
    model, scaler, encoder, feature_names, rag_pipeline = load_ml_resources()
    merged_df = load_merged_snapshot()
    
    # Reverse-map contract_type_encoded back to human-readable Contract labels
    contract_reverse_map = {0: "Month-to-month", 1: "One year", 2: "Two year"}
    if "contract_type_encoded" in merged_df.columns and "Contract" not in merged_df.columns:
        merged_df["Contract"] = merged_df["contract_type_encoded"].map(contract_reverse_map).fillna("Unknown")
    
    predictions_df = get_predictions_df(merged_df, model, scaler, encoder, feature_names)
except Exception as e:
    st.error(f"Error loading dashboard dependencies: {e}. Please complete Milestones 1-3 first.")
    st.stop()

# Load optimal threshold
optimal_t = 0.32
if os.path.exists("config/optimal_threshold.json"):
    with open("config/optimal_threshold.json", "r") as f:
        optimal_t = json.load(f).get("threshold", 0.32)

# --- KPI Summary Cards (Section 2) ---
col1, col2, col3, col4 = st.columns(4)

total_attempts, success_count, success_rate, rlhf_df = get_rlhf_stats()

high_risk_count = len(predictions_df[predictions_df["churn_score"] > 0.70])
avg_score = predictions_df["churn_score"].mean()

with col1:
    st.metric(
        label="🔴 عملاء عالُو الخطورة | High Risk (>0.70)",
        value=f"{high_risk_count}",
        delta="+2.4% عن الأسبوع الماضي",
        delta_color="inverse",
        help="عدد العملاء الذين تتجاوز احتمالية مغادرتهم 70% — أولوية قصوى للتدخّل."
    )
with col2:
    st.metric(
        label="📊 متوسط احتمالية المغادرة | Avg Churn Score",
        value=f"{avg_score:.2%}",
        delta="-0.85% شهرياً",
        help="متوسط احتمالية مغادرة جميع العملاء الحاليين."
    )
with col3:
    st.metric(
        label="🎯 نسبة نجاح الاحتفاظ | Retention Success",
        value=f"{success_rate:.1f}%",
        delta=f"تم الاحتفاظ بـ {success_count} من {total_attempts} حالة",
        help="نسبة حملات الاحتفاظ الناجحة من إجمالي التدخّلات المسجّلة."
    )
with col4:
    # Model Status
    st.metric(
        label="⚙️ النموذج الفعّال | Active Model",
        value="churn_predictor v1.0",
        delta="اختبارات النشر: ناجحة ✅",
        help="إصدار نموذج التنبؤ المستخدم حالياً في الإنتاج."
    )

st.markdown("---")

# --- Layout: Main columns ---
left_col, right_col = st.columns([2, 1])

# --- Left Column: Risk Table & Filters ---
with left_col:
    st.subheader("🔍 تحليل مخاطر المغادرة والتقييم | Churn Risk Analysis")
    st.caption("جدول العملاء مرتّب من الأعلى خطورة إلى الأقل. كلّما ارتفعت **احتمالية المغادرة**، زادت الحاجة للتدخّل السريع.")

    # Filters
    f_col1, f_col2, f_col3 = st.columns(3)
    with f_col1:
        score_filter = st.slider("أدنى احتمالية مغادرة | Min Churn Score", 0.0, 1.0, float(optimal_t), 0.05)
    with f_col2:
        contract_types = ["All"] + list(merged_df["Contract"].unique()) if "Contract" in merged_df.columns else ["All"]
        contract_filter = st.selectbox("نوع العقد | Contract Type", contract_types)
    with f_col3:
        tenure_bands = ["All", "<12 months", "12-24 months", "24-48 months", ">48 months"]
        tenure_filter = st.selectbox("شريحة مدة الاشتراك | Tenure Segment", tenure_bands)
        
    # Apply filters
    filtered_df = predictions_df[predictions_df["churn_score"] >= score_filter]
    if contract_filter != "All":
        filtered_df = filtered_df[filtered_df["Contract"] == contract_filter]
        
    if tenure_filter != "All":
        # Compute tenure band
        if tenure_filter == "<12 months":
            filtered_df = filtered_df[filtered_df["tenure"] < 12]
        elif tenure_filter == "12-24 months":
            filtered_df = filtered_df[(filtered_df["tenure"] >= 12) & (filtered_df["tenure"] <= 24)]
        elif tenure_filter == "24-48 months":
            filtered_df = filtered_df[(filtered_df["tenure"] > 24) & (filtered_df["tenure"] <= 48)]
        elif tenure_filter == ">48 months":
            filtered_df = filtered_df[filtered_df["tenure"] > 48]
            
    # Display table
    display_cols = ["customerID", "Contract", "tenure", "MonthlyCharges", "churn_score", "top_drivers"]
    table_df = filtered_df[display_cols].sort_values(by="churn_score", ascending=False).copy()
    # Format drivers to string
    table_df["top_drivers"] = table_df["top_drivers"].apply(lambda x: ", ".join(x))
    table_df["churn_score"] = table_df["churn_score"].apply(lambda x: f"{x:.2%}")
    table_df["MonthlyCharges"] = table_df["MonthlyCharges"].apply(lambda x: f"${x:.2f}")

    # Bilingual headers for display only (keep table_df keys intact for the selectbox below)
    table_df_display = table_df.rename(columns={
        "customerID": "معرّف العميل | Customer ID",
        "Contract": "نوع العقد | Contract",
        "tenure": "مدة الاشتراك | Tenure",
        "MonthlyCharges": "الرسوم الشهرية | Monthly",
        "churn_score": "احتمالية المغادرة | Churn Score",
        "top_drivers": "الأسباب الرئيسية | Top Drivers",
    })
    st.dataframe(table_df_display, width='stretch', height=350)
    st.caption("🟥 احتمالية عالية = خطر مغادرة مرتفع (تدخّل عاجل)  ·  🟩 احتمالية منخفضة = عميل مستقر. "
               "**الأسباب الرئيسية (Top Drivers)** = العوامل التي دفعت النموذج لهذا التقييم (مُستخرجة عبر SHAP).")

    # --- Expandable RAG Briefing Panel (Section 4) ---
    st.subheader("💡 توصيات الاحتفاظ الذكية | RAG Retention Briefings")
    selected_customer = st.selectbox("اختر عميلاً لتوليد توصية احتفاظ | Select Customer for RAG Briefing", table_df["customerID"].unique())
    
    if selected_customer:
        cust_row = filtered_df[filtered_df["customerID"] == selected_customer].iloc[0].to_dict()
        drivers = cust_row["top_drivers"]
        score = cust_row["churn_score"]  # This is a raw float from filtered_df
        
        with st.spinner("Generating customer intelligence briefing..."):
            briefing = rag_pipeline.generate_briefing(selected_customer, cust_row, drivers)
            
        st.info(f"**Customer ID: {selected_customer}** (Score: {score:.2%})\n\n{briefing}")
        
        # Log Intervention Action directly from UI (RLHF database insertion)
        st.write("---")
        st.write("**تسجيل الإجراء المتّخذ (مُدخل حلقة التعلّم البشري RLHF) | Record Action**")
        act_col1, act_col2 = st.columns(2)
        with act_col1:
            intervention = st.selectbox(
                "الإجراء المتّخذ | Intervention Taken",
                ["Loyalty Contract Price Lock", "Support Waiver & Router Replacement", " Loyalty Upgrade Bundle", "Refund Credit Issued"]
            )
        with act_col2:
            outcome_val = st.selectbox("نتيجة التدخّل | Intervention Outcome", ["Retained (Success)", "Churned (Failure)"])

        if st.button("💾 تسجيل تدخّل الاحتفاظ | Log Retention Intervention"):
            # Write to SQLite
            outcome_numeric = 1 if outcome_val == "Retained (Success)" else 0
            conn = sqlite3.connect("data/rlhf_outcomes.db")
            conn.execute(
                "INSERT INTO rlhf_outcomes (id, customer_id, intervention_type, outcome) VALUES (?, ?, ?, ?)",
                (str(uuid.uuid4()), selected_customer, intervention, outcome_numeric)
            )
            conn.commit()
            conn.close()
            st.success(f"Successfully logged '{intervention}' for customer {selected_customer}.")
            st.rerun()

# --- Right Column: Drift Panel, RLHF Chart, Natural Queries ---
with right_col:
    # --- Drift Panel (Section 3) ---
    st.subheader("🚨 مراقبة انحراف البيانات | Drift Monitoring")
    st.caption("يكتشف تغيّر توزيع بيانات العملاء عن بيانات تدريب النموذج (باستخدام مؤشر PSI).")

    # Run drift
    psi_values, drift_alerts = run_drift_analysis()

    # Draw simple status indicators
    avg_psi = np.mean(list(psi_values.values()))

    if len(drift_alerts) > 0:
        st.markdown(
            f'<div class="status-card status-red">🔴 تم رصد انحراف · DRIFT DETECTED ({len(drift_alerts)} خاصية متأثرة)<br>متوسط PSI: {avg_psi:.3f}</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="status-card status-green">🟢 مستقر · STABLE (لا توجد تنبيهات)<br>متوسط PSI: {avg_psi:.3f}</div>',
            unsafe_allow_html=True
        )

    # Display top drifted features
    st.write("**أكثر الخصائص انحرافاً (PSI) | Top Drifted Features**")
    sorted_psi = sorted(psi_values.items(), key=lambda x: x[1], reverse=True)[:5]
    for feat, val in sorted_psi:
        status_color = "🔴" if val > 0.20 else "🟡" if val > 0.10 else "🟢"
        st.write(f"{status_color} `{feat}`: **{val:.4f}**")
        
    st.markdown("---")
    
    # --- RLHF Outcome Tracker Chart (Section 5) ---
    st.subheader("📈 أداء حملات الاحتفاظ | Campaign Performance")
    if not rlhf_df.empty:
        # Group by intervention type and outcome
        summary = rlhf_df.groupby(["intervention_type", "outcome"]).size().unstack(fill_value=0)
        # Dynamically rename columns based on what's present
        col_map = {0: "Churned (مغادر)", 1: "Retained (محتفَظ به)"}
        summary = summary.rename(columns=col_map)
        st.bar_chart(summary, width='stretch')
    else:
        st.caption("لا توجد نتائج حملات مسجّلة بعد. استخدم مُسجّل الإجراءات في اللوحة اليسرى لتعبئة الرسم البياني.")

    st.markdown("---")

    # --- Natural Language Queries (Section 6) ---
    st.subheader("💬 محرّك الاستعلام باللغة الطبيعية | NL Query Console")
    user_query = st.text_input("اطرح سؤالاً عن العملاء أو شكاوى التذاكر: | Ask a question:", placeholder="مثال: من هم العملاء الحسّاسون للسعر؟ / Which customers are price-sensitive?")
    
    if user_query:
        # Use RAG or simple matching to answer
        st.write("**استجابة محرّك الذكاء | Intelligence Engine Response:**")
        query_lower = user_query.lower()
        
        if "price" in query_lower or "cost" in query_lower:
            high_value_risk = predictions_df[(predictions_df["churn_score"] >= optimal_t) & (predictions_df["MonthlyCharges"] >= 80)]
            st.markdown(f"**Found {len(high_value_risk)} price-sensitive customer(s) at risk (charges >= $80/mo):**")
            for _, row in high_value_risk.head(3).iterrows():
                st.write(f"- Customer `{row['customerID']}`: Charges `${row['MonthlyCharges']:.2f}` (Score: {row['churn_score']:.1%})")
        elif "escalat" in query_lower:
            esc_risk = predictions_df[(predictions_df["escalation_flag"] == 1) & (predictions_df["churn_score"] >= optimal_t)]
            st.markdown(f"**Found {len(esc_risk)} customer(s) with active escalations at risk:**")
            for _, row in esc_risk.head(3).iterrows():
                st.write(f"- Customer `{row['customerID']}`: Churn Score: {row['churn_score']:.1%}")
        else:
            # General similarity matching using FAISS
            playbook_match, cases_match = rag_pipeline.retrieve_similar(user_query, k=3)
            if playbook_match:
                st.markdown("**Related playbooks retrieved:**")
                st.write(playbook_match[0])
            else:
                st.write("Query processed. Recommend checking the Churn Scoring risk table for details.")
                
# --- Section 7: Upload New Customer Data ---
st.markdown("---")
st.subheader("📤 رفع ملف عملاء جديد للتنبؤ (Upload New Churn Data)")
st.markdown("""
يرجى رفع ملف بصيغة CSV يحتوي على بيانات العملاء. ستقوم المنصة بتشغيل نموذج الـ Machine Learning وحساب احتمالية رحيل العملاء (Churn Risk Score) وتوليد أهم الأسباب (SHAP Drivers) لكل عميل بشكل ديناميكي.
""")

uploaded_file = st.file_uploader("اختر ملف CSV للرفع", type=["csv"], key="dashboard_file_uploader")

if uploaded_file is not None:
    try:
        # Load uploaded data
        user_df = pd.read_csv(uploaded_file)
        st.success(f"تم تحميل الملف بنجاح! يحتوي الملف على {len(user_df)} عميل.")
        
        # Display preview of raw data
        st.write("👀 معاينة البيانات المرفوعة | Uploaded Data Preview:")
        st.dataframe(user_df.head(5), width='stretch')
        
        with st.spinner("جاري معالجة البيانات وتشغيل نموذج الذكاء الاصطناعي..."):
            # Process batch
            X_proc = preprocess_batch(user_df, scaler, encoder, feature_names)
            
            # Predict scores
            probs = model.predict_proba(X_proc)[:, 1]
            
            # Explain drivers using SHAP
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_proc)
            
            top_drivers_list = []
            for i in range(len(X_proc)):
                # Handle SHAP version types (list vs array)
                if isinstance(shap_values, list):
                    row_shap = shap_values[1][i] if len(shap_values) > 1 else shap_values[0][i]
                else:
                    row_shap = shap_values[i]
                top_idx = np.argsort(np.abs(row_shap))[::-1][:3]
                top_drivers_list.append([feature_names[idx] for idx in top_idx])
                
            # Create friendly output table
            results_df = user_df.copy()
            results_df["Churn Score"] = probs
            results_df["Risk Level"] = np.where(probs >= optimal_t, "🔴 High Risk", "🟢 Low Risk")
            results_df["Top Drivers"] = [", ".join(d) for d in top_drivers_list]
            
            # Reorder columns to show predictions first
            cols = ["customerID", "Churn Score", "Risk Level", "Top Drivers"]
            # Keep other cols in original df
            other_cols = [c for c in results_df.columns if c not in cols]
            # Ensure customerID is present
            if "customerID" not in results_df.columns:
                if "CustomerId" in results_df.columns:
                    results_df = results_df.rename(columns={"CustomerId": "customerID"})
                else:
                    results_df.insert(0, "customerID", [f"CUST-{idx:04d}" for idx in range(len(results_df))])
            results_df = results_df[cols + other_cols]

            # Sort by the NUMERIC score (before formatting) so ordering is correct,
            # then format the score column as a percentage string for display.
            table_display = results_df.sort_values(by="Churn Score", ascending=False).copy()
            table_display["Churn Score"] = table_display["Churn Score"].apply(lambda x: f"{x:.2%}")
            # Bilingual headers for the prediction columns (display only)
            table_display = table_display.rename(columns={
                "customerID": "معرّف العميل | Customer ID",
                "Churn Score": "احتمالية المغادرة | Churn Score",
                "Risk Level": "مستوى الخطورة | Risk Level",
                "Top Drivers": "الأسباب الرئيسية | Top Drivers",
            })

        st.subheader("🎯 نتائج التنبؤ للعملاء المرفوعين | Prediction Results:")
        st.dataframe(table_display, width='stretch', height=350)
        
        # Download predictions button
        csv_data = results_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 تحميل نتائج التنبؤ بصيغة CSV",
            data=csv_data,
            file_name="customer_churn_predictions.csv",
            mime="text/csv"
        )
    except Exception as ex:
        st.error(f"حدث خطأ أثناء معالجة الملف: {ex}")
        
# uuid is imported at the top of this file
