"""
escalation.py
---------------------------------------------------------------------------
Decides auto-handle vs. escalate-to-human, and always returns a reason
string -- "trust me" is not an acceptable output for a system that touches
refunds and angry customers.

Escalation triggers (any one fires -> escalate), in priority order:
  1. Intent is complaint_escalation (customer already explicitly upset /
     asking for a manager / repeat contact).
  2. Classifier confidence below threshold (default 0.55) -- we'd rather
     hand an ambiguous case to a human than auto-send a confidently wrong
     reply.
  3. Message contains a monetary amount above a configurable ceiling
     (default $50) combined with refund/return intent -- large refunds
     get a human sign-off regardless of how confident the model is.
  4. Message contains legal/safety-adjacent language (e.g. "lawyer",
     "lawsuit", "allergic", "injury", "unsafe") -- always human, no
     exceptions, regardless of confidence.
  5. No sufficiently similar historical resolution was retrieved (top
     retrieval score below threshold) -- nothing to ground the reply in.

Anything that doesn't trip a trigger is auto-handled.
"""
import re
from dataclasses import dataclass

LEGAL_SAFETY_PATTERN = re.compile(
    r"\b(lawyer|lawsuit|sue|attorney|allergic reaction|injur(?:ed|y)|unsafe|fraud|police|legal action)\b",
    re.IGNORECASE,
)
MONEY_PATTERN = re.compile(r"\$\s?(\d{2,5}(?:\.\d{2})?)")

CONFIDENCE_THRESHOLD = 0.55
MONEY_CEILING = 50.0
RETRIEVAL_SCORE_FLOOR = 0.05


@dataclass
class EscalationDecision:
    escalate: bool
    reason: str


def decide(intent_label, intent_confidence, customer_text, top_retrieval_score):
    if intent_label == "complaint_escalation":
        return EscalationDecision(True, "Customer language indicates repeat contact or explicit frustration/escalation request.")

    if LEGAL_SAFETY_PATTERN.search(customer_text):
        return EscalationDecision(True, "Message contains legal or safety-adjacent language; always routed to a human regardless of confidence.")

    money_match = MONEY_PATTERN.search(customer_text)
    if money_match and intent_label in ("refund_return", "damaged_wrong_item") and float(money_match.group(1)) > MONEY_CEILING:
        return EscalationDecision(True, f"Refund-adjacent request references an amount (${money_match.group(1)}) above the ${MONEY_CEILING:.0f} auto-approval ceiling.")

    if intent_confidence < CONFIDENCE_THRESHOLD:
        return EscalationDecision(True, f"Intent classifier confidence ({intent_confidence:.2f}) is below the {CONFIDENCE_THRESHOLD:.2f} auto-handle threshold.")

    if top_retrieval_score < RETRIEVAL_SCORE_FLOOR:
        return EscalationDecision(True, f"No sufficiently similar historical resolution found (top retrieval score {top_retrieval_score:.2f}); nothing to ground a reply in.")

    return EscalationDecision(False, "Confident intent match, no red-flag language, and a well-grounded historical resolution was found.")
