"""
metrics.py
---------------------------------------------------------------------------
Automated (non-LLM-judge) metrics. Two families:
  1. Intent classification: accuracy, macro-F1, per-class confusion matrix.
  2. Escalation decision: precision/recall/F1 treating "should escalate"
     as the positive class, PLUS a separate "unsafe auto-handle rate" --
     the fraction of escalation-worthy cases the system did NOT catch.
     This second number matters more than F1 here: a false escalation
     just costs a human's time, a missed escalation can mean the model
     confidently auto-sent something it shouldn't have.
"""
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support
import pandas as pd


def intent_metrics(y_true, y_pred, labels):
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    per_class = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    per_class_df = pd.DataFrame({
        "intent": labels,
        "precision": per_class[0],
        "recall": per_class[1],
        "f1": per_class[2],
        "support": per_class[3],
    })
    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": per_class_df,
        "confusion_matrix": pd.DataFrame(cm, index=labels, columns=labels),
    }


def escalation_metrics(gold_escalate, pred_escalate):
    p, r, f1, _ = precision_recall_fscore_support(gold_escalate, pred_escalate, average="binary", zero_division=0)
    gold_escalate = list(gold_escalate)
    pred_escalate = list(pred_escalate)
    missed = sum(1 for g, p_ in zip(gold_escalate, pred_escalate) if g and not p_)
    n_gold_positive = sum(gold_escalate) or 1
    unsafe_auto_handle_rate = missed / n_gold_positive
    return {
        "precision": round(p, 4),
        "recall": round(r, 4),
        "f1": round(f1, 4),
        "missed_escalations": missed,
        "gold_escalation_worthy_count": sum(gold_escalate),
        "unsafe_auto_handle_rate": round(unsafe_auto_handle_rate, 4),
    }
