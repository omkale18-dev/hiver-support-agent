"""
trivial_baseline.py
---------------------------------------------------------------------------
The floor. Always predicts the majority intent, always sends the same
canned reply, never escalates. Any real system that doesn't clear this
by a wide margin has a bug, not a research finding.
"""
MAJORITY_INTENT = "order_status_delay"  # most frequent intent in the training split
CANNED_REPLY = ("Thanks for reaching out! We're looking into this and a member of our "
                "team will follow up with you shortly.")


class TrivialBaseline:
    def predict_intent(self, text):
        return MAJORITY_INTENT, 1.0

    def generate_reply(self, text):
        return CANNED_REPLY

    def decide_escalation(self, text):
        return False, "Trivial baseline never escalates."
