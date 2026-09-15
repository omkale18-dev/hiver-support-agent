"""
intents.py
---------------------------------------------------------------------------
Intent taxonomy for AmazonHelp, arrived at by reading a sample of ~150
customer-inbound messages and clustering on recurring "what does the
customer want" patterns (see report.md, "Problem framing" section, for
the actual process and why we stopped at 8 buckets instead of going
finer-grained like Banking77's 77).

    order_status_delay     - "where is my stuff" / tracking not moving
    refund_return          - wants money back or wants to send something back
    damaged_wrong_item     - what arrived is broken / not what they ordered
    account_payment        - login, password, duplicate/failed charges
    cancel_order            - stop an order before/after it ships
    product_question        - pre-purchase info, sizing, restock, compatibility
    complaint_escalation    - repeat contact, explicit anger, wants a manager
    positive_feedback       - thanks / praise, no action needed
    other_uncategorized     - doesn't fit the above (fallback bucket)

Two classifiers are provided:
  * TfidfIntentClassifier - a trainable bag-of-words + logistic regression
    model. This is the "simple baseline" tier: fast, cheap, no API calls,
    but brittle to phrasing it hasn't seen.
  * LlmIntentClassifier - zero/few-shot classification via llm_client.
    This is the system under test.
"""
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

from src.llm_client import call_llm

TAXONOMY = [
    "order_status_delay",
    "refund_return",
    "damaged_wrong_item",
    "account_payment",
    "cancel_order",
    "product_question",
    "complaint_escalation",
    "positive_feedback",
    "other_uncategorized",
]

TAXONOMY_DESCRIPTIONS = {
    "order_status_delay": "Customer is asking where their order is or why tracking hasn't updated.",
    "refund_return": "Customer wants a refund, wants to return an item, or is asking about refund status.",
    "damaged_wrong_item": "The item that arrived is broken, defective, incomplete, or not what was ordered.",
    "account_payment": "Login/password/account access issues, or duplicate/incorrect/failed charges.",
    "cancel_order": "Customer wants to cancel an order that hasn't been delivered yet.",
    "product_question": "Pre-purchase question: sizing, compatibility, availability, restock timing.",
    "complaint_escalation": "Customer is expressing significant frustration, repeat contact, or explicitly asking for a manager/escalation.",
    "positive_feedback": "Customer is thanking the brand or praising a resolution; no action needed.",
    "other_uncategorized": "Doesn't clearly fit any of the above (chit-chat, unrelated, too ambiguous).",
}


@dataclass
class IntentPrediction:
    label: str
    confidence: float
    method: str


class TfidfIntentClassifier:
    def __init__(self):
        self.vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=3000)
        self.clf = LogisticRegression(max_iter=1000, C=2.0)
        self.fitted = False

    def fit(self, texts, labels):
        X = self.vec.fit_transform(texts)
        self.clf.fit(X, labels)
        self.fitted = True
        return self

    def predict(self, text):
        if not self.fitted:
            raise RuntimeError("Call .fit() first")
        X = self.vec.transform([text])
        pred = self.clf.predict(X)[0]
        proba = self.clf.predict_proba(X)[0]
        conf = float(max(proba))
        return IntentPrediction(label=pred, confidence=conf, method="tfidf_logreg")

    def save(self, path):
        joblib.dump({"vec": self.vec, "clf": self.clf}, path)

    def load(self, path):
        obj = joblib.load(path)
        self.vec, self.clf = obj["vec"], obj["clf"]
        self.fitted = True
        return self


class LlmIntentClassifier:
    """Zero/few-shot classification. Falls back to the offline mock scorer
    in llm_client.py when no API key is configured."""

    SYSTEM_PROMPT = (
        "You are an intent classifier for a customer support team at an e-commerce brand. "
        "Classify the customer's message into exactly one of these intents:\n"
        + "\n".join(f"- {k}: {v}" for k, v in TAXONOMY_DESCRIPTIONS.items())
        + "\n\nRespond ONLY with JSON: {\"label\": \"<intent>\", \"confidence\": <0-1 float>, \"reasoning\": \"<one short sentence>\"}"
    )

    def predict(self, text, thread_context=""):
        user_prompt = f"Thread context: {thread_context}\n\nCustomer message: {text}"
        label, confidence = call_llm(self.SYSTEM_PROMPT, user_prompt, mode="classify", labels=TAXONOMY)
        if label not in TAXONOMY:
            label = "other_uncategorized"
        return IntentPrediction(label=label, confidence=confidence, method="llm")
