"""
run_eval.py
---------------------------------------------------------------------------
Runs the golden set through:
  - trivial_baseline
  - simple_baseline
  - the real system (LLM intent classifier + retrieval-grounded generation
    + rule/confidence escalation)
and prints/saves a comparison table plus reply-quality judge scores for
the system. This is what produces the numbers quoted in report.md.
"""
import json
import time
import pandas as pd

from src.intents import LlmIntentClassifier, TAXONOMY
from src.agent_pipeline import SupportAgent
from baselines.trivial_baseline import TrivialBaseline
from baselines.simple_baseline import SimpleBaseline
from eval.common import load_split, build_retriever, build_tfidf_classifier
from eval.metrics import intent_metrics, escalation_metrics
from eval.llm_judge import judge_reply, composite, score_agreement, JUDGE_MODE


def run_baseline(baseline, golden):
    intent_preds, escalate_preds = [], []
    for _, r in golden.iterrows():
        label, _ = baseline.predict_intent(r["customer_text"])
        intent_preds.append(label)
        esc, _ = baseline.decide_escalation(r["customer_text"])
        escalate_preds.append(esc)
    return intent_preds, escalate_preds


def run_system(golden, retriever):
    agent = SupportAgent(retriever=retriever, classifier=LlmIntentClassifier())
    intent_preds, escalate_preds, judge_rows, replies = [], [], [], []
    for _, r in golden.iterrows():
        resp = agent.handle(r["customer_text"], thread_context=str(r.get("thread_context") or ""))
        intent_preds.append(resp.intent)
        escalate_preds.append(resp.escalate)
        replies.append(resp.draft_reply)
        if not resp.escalate:
            grounding = " | ".join(x.agent_reply_text for x in resp.retrieved)
            scores = judge_reply(r["customer_text"], grounding, resp.draft_reply)
            judge_rows.append({"case_id": r["case_id"], **scores, "composite": composite(scores)})
    return intent_preds, escalate_preds, replies, pd.DataFrame(judge_rows)


def main():
    t0 = time.time()
    train_df, golden = load_split()
    golden["gold_escalate"] = golden["gold_escalate"].map(lambda x: str(x).lower() == "true")
    retriever = build_retriever(train_df)

    results = {}

    # --- baselines ---
    for name, baseline in [("trivial", TrivialBaseline()), ("simple", SimpleBaseline())]:
        preds, esc_preds = run_baseline(baseline, golden)
        im = intent_metrics(golden["gold_intent"].tolist(), preds, TAXONOMY)
        em = escalation_metrics(golden["gold_escalate"].tolist(), esc_preds)
        results[name] = {"intent_accuracy": im["accuracy"], "intent_macro_f1": im["macro_f1"],
                          "escalation_precision": em["precision"], "escalation_recall": em["recall"],
                          "escalation_f1": em["f1"], "unsafe_auto_handle_rate": em["unsafe_auto_handle_rate"]}

    # --- extra reference point: trainable TF-IDF+LogReg intent classifier ---
    # (not a required baseline, but a useful third data point -- see report.md
    #  "what's misleading about my headline number": a supervised model trained
    #  on this same templated distribution can out-perform zero-shot LLM
    #  classification, which says as much about the synthetic data's limited
    #  phrasing diversity as it does about either classifier.)
    tfidf_clf = build_tfidf_classifier(train_df)
    tfidf_preds = [tfidf_clf.predict(t).label for t in golden["customer_text"]]
    im_tfidf = intent_metrics(golden["gold_intent"].tolist(), tfidf_preds, TAXONOMY)
    results["tfidf_trainable_reference"] = {
        "intent_accuracy": im_tfidf["accuracy"], "intent_macro_f1": im_tfidf["macro_f1"],
        "escalation_precision": float("nan"), "escalation_recall": float("nan"),
        "escalation_f1": float("nan"), "unsafe_auto_handle_rate": float("nan"),
    }

    # --- system ---
    preds, esc_preds, replies, judge_df = run_system(golden, retriever)
    im = intent_metrics(golden["gold_intent"].tolist(), preds, TAXONOMY)
    em = escalation_metrics(golden["gold_escalate"].tolist(), esc_preds)
    results["system"] = {"intent_accuracy": im["accuracy"], "intent_macro_f1": im["macro_f1"],
                          "escalation_precision": em["precision"], "escalation_recall": em["recall"],
                          "escalation_f1": em["f1"], "unsafe_auto_handle_rate": em["unsafe_auto_handle_rate"]}
    if len(judge_df):
        results["system"]["mean_reply_quality_composite_1to5"] = round(judge_df["composite"].mean(), 2)
        results["system"]["pct_auto_handled_replies_scoring_lt3"] = round(
            (judge_df["composite"] < 3).mean() * 100, 1)

    agreement = score_agreement()

    # --- save everything ---
    summary_df = pd.DataFrame(results).T
    summary_df.to_csv("eval/results_summary.csv")

    per_class_df = im["per_class"]
    per_class_df.to_csv("eval/system_per_class_metrics.csv", index=False)
    im["confusion_matrix"].to_csv("eval/system_confusion_matrix.csv")

    golden_out = golden.copy()
    golden_out["system_pred_intent"] = preds
    golden_out["system_escalate"] = esc_preds
    golden_out["system_reply"] = replies
    golden_out.to_csv("eval/system_predictions.csv", index=False)

    judge_df.to_csv("eval/system_judge_scores.csv", index=False)

    with open("eval/judge_agreement.json", "w") as f:
        json.dump(agreement, f, indent=2)

    print("=" * 70)
    print("LLM mode active for this run: classification/generation backend")
    print(f"Judge mode active for this run: {JUDGE_MODE}")
    print("=" * 70)
    print(summary_df.to_string())
    print("-" * 70)
    print("Judge/human agreement:", json.dumps(agreement, indent=2))
    print("-" * 70)
    print(f"Eval finished in {time.time() - t0:.1f}s over {len(golden)} golden examples")
    print("Full outputs written to eval/*.csv and eval/judge_agreement.json")


if __name__ == "__main__":
    main()
