import os
import hashlib
import json
import sqlite3
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class LLMExtractor:
    def __init__(self, db_path="data/llm_cache.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()
        provider = os.environ.get("LLM_PROVIDER", "openai").lower()
        if provider == "groq":
            self.api_key = os.environ.get("GROQ_API_KEY")
            base_url = "https://api.groq.com/openai/v1"
            self.model_name = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
        else:
            self.api_key = os.environ.get("OPENAI_API_KEY")
            base_url = None
            self.model_name = "gpt-4o-mini"
            
        self.client = None
        if self.api_key:
            try:
                from openai import OpenAI
                if base_url:
                    self.client = OpenAI(api_key=self.api_key, base_url=base_url)
                else:
                    self.client = OpenAI(api_key=self.api_key)
            except ImportError:
                print("OpenAI package not installed or import error. Will use rule-based fallback.")

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ticket_cache (
                    ticket_hash TEXT PRIMARY KEY,
                    sentiment_score REAL,
                    urgency_level TEXT,
                    complaint_topics TEXT,
                    escalation_flag INTEGER
                )
            """)
            conn.commit()

    def get_hash(self, text):
        return hashlib.sha256(text.strip().encode('utf-8')).hexdigest()

    def get_cached(self, ticket_hash):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT sentiment_score, urgency_level, complaint_topics, escalation_flag FROM ticket_cache WHERE ticket_hash = ?",
                (ticket_hash,)
            )
            row = cursor.fetchone()
            if row:
                return {
                    "sentiment_score": row[0],
                    "urgency_level": row[1],
                    "complaint_topics": json.loads(row[2]),
                    "escalation_flag": bool(row[3])
                }
        return None

    def save_cache(self, ticket_hash, result):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO ticket_cache (ticket_hash, sentiment_score, urgency_level, complaint_topics, escalation_flag) VALUES (?, ?, ?, ?, ?)",
                (
                    ticket_hash,
                    result["sentiment_score"],
                    result["urgency_level"],
                    json.dumps(result["complaint_topics"]),
                    1 if result["escalation_flag"] else 0
                )
            )
            conn.commit()

    def heuristic_extract(self, text):
        """
        Rule-based local fallback parser when OpenAI API key is unavailable.
        """
        text_lower = text.lower()
        
        # 1. Topics
        topics = []
        if any(w in text_lower for w in ["bill", "invoice", "charge", "refund", "price", "cost"]):
            topics.append("billing")
        if any(w in text_lower for w in ["slow", "speed", "outage", "down", "network", "disruption", "power"]):
            topics.append("performance")
        if any(w in text_lower for w in ["competitor", "verizon", "t-mobile", "att", "cheaper", "match"]):
            topics.append("competitor")
        if any(w in text_lower for w in ["cancel", "terminate", "expire", "leaving", "stop"]):
            topics.append("cancellation")
        if not topics:
            topics.append("other")
            
        # 2. Urgency
        urgency = "low"
        if any(w in text_lower for w in ["immediate", "quick", "asap", "hurry"]):
            urgency = "high"
        if any(w in text_lower for w in ["cancel", "legal", "lawyer", "regulator", "court", "threat", "unacceptable"]):
            urgency = "critical"
        elif any(w in text_lower for w in ["error", "down", "outage", "double charge"]):
            urgency = "high"
        elif any(w in text_lower for w in ["question", "help", "how to"]):
            urgency = "medium"
            
        # 3. Sentiment
        sentiment = 0.0
        neg_words = ["terrible", "bad", "unacceptable", "angry", "disruption", "broken", "hate", "worst"]
        pos_words = ["happy", "good", "great", "thank", "love", "satisfied", "helpful"]
        neg_count = sum(1 for w in neg_words if w in text_lower)
        pos_count = sum(1 for w in pos_words if w in text_lower)
        
        if neg_count > 0:
            sentiment = -0.5 - (0.1 * min(neg_count, 5))
        elif pos_count > 0:
            sentiment = 0.3 + (0.1 * min(pos_count, 5))
            
        # Bounds check
        sentiment = max(-1.0, min(1.0, sentiment))
        
        # 4. Escalation
        escalation = False
        if any(w in text_lower for w in ["cancel", "leave", "switch", "court", "sue", "legal", "regulator", "publicly"]):
            escalation = True
            
        return {
            "sentiment_score": float(sentiment),
            "urgency_level": urgency,
            "complaint_topics": topics,
            "escalation_flag": escalation
        }

    def call_llm(self, text):
        prompt = f"""
You are a customer intelligence AI. Parse the following support ticket text and return a JSON object with four fields:
1. "sentiment_score": (float, between -1.0 and 1.0. -1.0 is extremely negative, 1.0 is extremely positive, 0.0 is neutral)
2. "urgency_level": (string, exactly one of: "low", "medium", "high", "critical")
3. "complaint_topics": (array of strings, containing zero or more of: "billing", "performance", "competitor", "cancellation", "other")
4. "escalation_flag": (boolean, true if user explicitly threatens to cancel, switch to competitor, contact regulator, or take legal action)

Return ONLY a valid JSON object. No preamble, no explanation, no markdown blocks.

Ticket Text:
{text}
"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a precise data extractor that returns only structured JSON."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            raw_content = response.choices[0].message.content
            result = json.loads(raw_content)
            # Ensure correct types and fields
            return {
                "sentiment_score": float(result.get("sentiment_score", 0.0)),
                "urgency_level": str(result.get("urgency_level", "low")).lower(),
                "complaint_topics": list(result.get("complaint_topics", ["other"])),
                "escalation_flag": bool(result.get("escalation_flag", False))
            }
        except Exception as e:
            print(f"OpenAI API call failed ({e}). Falling back to heuristic extractor.")
            return self.heuristic_extract(text)

    def extract_ticket_features(self, text):
        ticket_hash = self.get_hash(text)
        cached_result = self.get_cached(ticket_hash)
        if cached_result:
            return cached_result
            
        if self.client:
            result = self.call_llm(text)
        else:
            result = self.heuristic_extract(text)
            
        self.save_cache(ticket_hash, result)
        return result

    def process_dataframe(self, df, text_column="Ticket Description"):
        """
        Processes a DataFrame containing support tickets and appends the extracted columns.
        """
        sentiments = []
        urgencies = []
        topics_list = []
        escalations = []
        
        print(f"Extracting LLM features for {len(df)} tickets...")
        for i, text in enumerate(df[text_column]):
            if pd.isna(text):
                res = {
                    "sentiment_score": 0.0,
                    "urgency_level": "low",
                    "complaint_topics": [],
                    "escalation_flag": False
                }
            else:
                res = self.extract_ticket_features(str(text))
            sentiments.append(res["sentiment_score"])
            urgencies.append(res["urgency_level"])
            topics_list.append(res["complaint_topics"])
            escalations.append(res["escalation_flag"])
            
        df_out = df.copy()
        df_out["sentiment_score"] = sentiments
        df_out["urgency_level"] = urgencies
        df_out["complaint_topics"] = topics_list
        df_out["escalation_flag"] = escalations
        
        return df_out

if __name__ == "__main__":
    # Quick self-test
    extractor = LLMExtractor()
    sample_text = "I am very angry! My bill was $30 extra this month and I want to cancel my subscription immediately!"
    features = extractor.extract_ticket_features(sample_text)
    print("Self-Test Output:")
    print(json.dumps(features, indent=2))
