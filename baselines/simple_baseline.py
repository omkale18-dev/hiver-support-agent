"""
simple_baseline.py
---------------------------------------------------------------------------
What a competent engineer would ship in an afternoon without any ML/LLM:
if/elif keyword rules for intent, one canned reply template per intent
(no grounding, no retrieval), and escalation only on an explicit keyword
hit. This is the bar the real system needs to clear to justify the extra
complexity of retrieval + an LLM classifier/generator.
"""
import re

RULES = [
    ("cancel_order", re.compile(r"\bcancel\b", re.I)),
    ("damaged_wrong_item", re.compile(r"\b(broken|damaged|smashed|defective|wrong item|missing pieces)\b", re.I)),
    ("refund_return", re.compile(r"\b(refund|return|money back|reimburse|exchange)\b", re.I)),
    ("account_payment", re.compile(r"\b(log ?in|password|locked|charged twice|payment failed)\b", re.I)),
    ("order_status_delay", re.compile(r"\b(tracking|late|delay|where is|still waiting|hasn'?t (arrived|moved))\b", re.I)),
    ("product_question", re.compile(r"\b(compatible|size|back in stock|when will)\b", re.I)),
    ("positive_feedback", re.compile(r"\b(thank you|thanks|appreciate|smooth|shoutout)\b", re.I)),
]

ESCALATION_KEYWORDS = re.compile(r"\b(manager|unacceptable|third time|lawsuit|lawyer|sue)\b", re.I)

CANNED_REPLIES = {
    "cancel_order": "We've received your cancellation request and will confirm shortly.",
    "damaged_wrong_item": "Sorry to hear that! We'll get a replacement or refund started for you.",
    "refund_return": "We can help with that -- a refund/return will be processed shortly.",
    "account_payment": "Sorry for the trouble -- we're looking into your account issue now.",
    "order_status_delay": "Thanks for flagging -- we're checking on your order's status now.",
    "product_question": "Thanks for the question -- let us look into that for you.",
    "positive_feedback": "Thank you so much for letting us know!",
    "other_uncategorized": "Thanks for reaching out, we'll get back to you shortly.",
}


class SimpleBaseline:
    def predict_intent(self, text):
        for label, pattern in RULES:
            if pattern.search(text):
                return label, 1.0  # rule-based: always "certain" (this overconfidence is
                                    # itself a finding, discussed in report.md)
        return "other_uncategorized", 1.0

    def generate_reply(self, text):
        label, _ = self.predict_intent(text)
        return CANNED_REPLIES[label]

    def decide_escalation(self, text):
        if ESCALATION_KEYWORDS.search(text):
            return True, "Matched an explicit escalation keyword."
        return False, "No escalation keyword matched."
