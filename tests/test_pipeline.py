"""
Minimal smoke tests -- run with:  PYTHONPATH=. python3 -m pytest tests/ -q
Not a substitute for the golden-set eval; just checks nothing is broken.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.escalation import decide
from src.retrieval import ReplyRetriever
from baselines.trivial_baseline import TrivialBaseline
from baselines.simple_baseline import SimpleBaseline


def test_escalation_legal_language_always_escalates():
    d = decide("refund_return", 0.99, "I'm going to get a lawyer involved", 0.9)
    assert d.escalate is True
    assert "legal" in d.reason.lower()


def test_escalation_low_confidence_escalates():
    d = decide("order_status_delay", 0.2, "where is my order", 0.9)
    assert d.escalate is True


def test_escalation_confident_clean_case_auto_handles():
    d = decide("cancel_order", 0.9, "please cancel my order", 0.9)
    assert d.escalate is False


def test_retrieval_returns_k_results():
    corpus = [
        {"customer_text": "where is my order", "agent_reply_text": "checking now", "intent": "order_status_delay"},
        {"customer_text": "refund please", "agent_reply_text": "refund started", "intent": "refund_return"},
        {"customer_text": "cancel my order", "agent_reply_text": "cancelled", "intent": "cancel_order"},
    ]
    r = ReplyRetriever().fit(corpus)
    out = r.retrieve("where is my package", intent="order_status_delay", k=2)
    assert len(out) == 2
    assert out[0].intent == "order_status_delay"


def test_trivial_baseline_never_escalates():
    b = TrivialBaseline()
    esc, _ = b.decide_escalation("anything at all")
    assert esc is False


def test_simple_baseline_keyword_routing():
    b = SimpleBaseline()
    label, _ = b.predict_intent("please cancel my order")
    assert label == "cancel_order"
