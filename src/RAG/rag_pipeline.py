import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

class RAGPipeline:
    def __init__(self, index_path="data/faiss_index.bin", metadata_path="data/faiss_metadata.json"):
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.model = None
        self.index = None
        self.documents = []
        
        provider = os.environ.get("LLM_PROVIDER", "openai").lower()
        if provider == "groq":
            self.api_key = os.environ.get("GROQ_API_KEY")
            self.base_url = "https://api.groq.com/openai/v1"
            self.model_name = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
        else:
            self.api_key = os.environ.get("OPENAI_API_KEY")
            self.base_url = None
            self.model_name = "gpt-4o-mini"
            
        self.client = None
        
        self._load_index()
        self._init_openai()

    def _load_index(self):
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            print("Loading FAISS index and metadata...")
            self.index = faiss.read_index(self.index_path)
            with open(self.metadata_path, "r") as f:
                metadata = json.load(f)
                self.documents = metadata.get("documents", [])
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
        else:
            print("FAISS index or metadata not found. RAG pipeline will use heuristic fallback.")

    def _init_openai(self):
        if self.api_key:
            try:
                from openai import OpenAI
                if self.base_url:
                    self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
                else:
                    self.client = OpenAI(api_key=self.api_key)
            except ImportError:
                print("OpenAI package not installed. Will use rule-based template fallback.")

    def build_query_string(self, customer_row, top_3_drivers):
        """
        Builds a natural language query from the customer record and top 3 SHAP drivers.
        """
        query_parts = ["customer"]
        
        for driver in top_3_drivers:
            # Map features to friendly search terms
            if "contract" in driver.lower():
                val = customer_row.get("contract_type_encoded", 0)
                term = "month-to-month contract" if val == 0 else "one-year contract" if val == 1 else "two-year contract"
                query_parts.append(term)
            elif "tenure" in driver.lower():
                val = customer_row.get("tenure", 0)
                query_parts.append(f"tenure {int(val)} months")
            elif "charges" in driver.lower() or "charges_per_tenure_month" in driver.lower():
                val = customer_row.get("MonthlyCharges", 0)
                query_parts.append(f"monthly charges ${val:.2f}")
            elif "techsupport" in driver.lower():
                val = customer_row.get("TechSupport", 0)
                term = "has tech support" if val == 1 else "no tech support"
                query_parts.append(term)
            elif "onlinesecurity" in driver.lower():
                val = customer_row.get("OnlineSecurity", 0)
                term = "has online security" if val == 1 else "no online security"
                query_parts.append(term)
            elif "escalation" in driver.lower():
                val = customer_row.get("escalation_flag", 0)
                term = "escalated support ticket" if val == 1 else "no escalation"
                query_parts.append(term)
            elif "sentiment" in driver.lower():
                val = customer_row.get("sentiment_score", 0.0)
                term = f"negative ticket sentiment {val:.2f}" if val < 0 else f"neutral sentiment {val:.2f}"
                query_parts.append(term)
                
        # Fallback to general terms if query is too short
        if len(query_parts) <= 1:
            query_parts.append("high churn risk")
            
        return " ".join(query_parts)

    def retrieve_similar(self, query, k=10):
        """
        Queries FAISS and separates playbook sections from historical customer cases.
        """
        if not self.index or not self.model:
            return [], []
            
        # Embed query
        query_vector = self.model.encode([query]).astype("float32")
        
        # Search FAISS index
        distances, indices = self.index.search(query_vector, k)
        
        playbooks = []
        historical_cases = []
        
        for idx in indices[0]:
            if idx < 0 or idx >= len(self.documents):
                continue
            doc = self.documents[idx]
            if "[PLAYBOOK SECTION:" in doc:
                playbooks.append(doc)
            elif "[HISTORICAL CASE]" in doc:
                historical_cases.append(doc)
                
        # Return at most 1 playbook section and 5 historical cases
        return playbooks[:1], historical_cases[:5]

    def generate_heuristic_briefing(self, customer_id, query_str, playbook, cases):
        """
        Rule-based heuristic fallback generator if OpenAI is not available.
        """
        playbook_clean = playbook[0].split("Strategy:")[1].split("Key drivers:")[0].strip() if playbook else "Offer standard retention outreach."
        case_ids = []
        for case in cases:
            # Extract customer ID
            try:
                cid = case.split("Customer ID:")[1].split(".")[0].strip()
                case_ids.append(cid)
            except Exception:
                pass
                
        precedent = f"similar past cases (IDs: {', '.join(case_ids[:3])})" if case_ids else "historical precedents"
        
        briefing = (
            f"This customer is at high risk of churn due to characteristics matching: '{query_str}'. "
            f"Historical cases show that customers with similar features were successfully retained or churned depending on immediate outreach. "
            f"Recommended Action: {playbook_clean} (estimated success probability is 80%)."
        )
        return briefing

    def generate_briefing(self, customer_id, customer_row, top_3_drivers):
        """
        Generates retention briefing using RAG.
        """
        query_str = self.build_query_string(customer_row, top_3_drivers)
        playbook, cases = self.retrieve_similar(query_str)
        
        # Build prompt
        playbook_text = playbook[0] if playbook else "No specific playbook section found."
        cases_text = "\n".join(cases) if cases else "No historical precedent cases found."
        
        prompt = f"""
You are a retention coordinator briefing a customer support agent.
Generate a plain-English retention briefing for Customer {customer_id}.

Current Customer Context:
- Churn Risk Score: {customer_row.get('churn_score', 0.85):.2f}
- Top 3 SHAP drivers: {', '.join(top_3_drivers)}
- Support Sentiment: {customer_row.get('sentiment_score', 0.0):.2f}

Retrieved Playbook Strategy:
{playbook_text}

Retrieved Historical Similar Cases:
{cases_text}

Write a 3-4 sentence plain-English briefing using this structure:
1. One sentence explaining WHY the customer is at risk based on their top drivers.
2. One sentence describing the historical precedent from similar cases (mentioning what worked or failed).
3. One sentence detailing the recommended retention action and its estimated success probability based on the playbook.

Write ONLY the briefing text. No headings, no intro, no labels.
"""
        if self.client:
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a professional customer retention assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2,
                    max_tokens=150
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                print(f"OpenAI call failed in RAG pipeline ({e}). Falling back to heuristic.")
                return self.generate_heuristic_briefing(customer_id, query_str, playbook, cases)
        else:
            return self.generate_heuristic_briefing(customer_id, query_str, playbook, cases)

if __name__ == "__main__":
    # Test RAG pipeline
    pipeline = RAGPipeline()
    mock_row = {
        "contract_type_encoded": 0,
        "tenure": 3,
        "MonthlyCharges": 85.00,
        "TechSupport": 0,
        "OnlineSecurity": 0,
        "escalation_flag": 1,
        "sentiment_score": -0.6
    }
    drivers = ["contract_type_encoded", "tenure", "TechSupport"]
    briefing = pipeline.generate_briefing("1234-ABCD", mock_row, drivers)
    print("Generated Retention Briefing Preview:")
    print(briefing)
