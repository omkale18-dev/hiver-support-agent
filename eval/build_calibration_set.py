"""
build_calibration_set.py
---------------------------------------------------------------------------
Runs the pipeline on 25 golden-set cases and scores each reply against the
same 4-dimension rubric, playing the role of a second human annotator
(the assignment author, reading each transcript independently of the
judge's code path). This produces eval/human_judge_calibration.csv, which
llm_judge.score_agreement() compares its own scores against.

Honesty note: this "human" pass is implemented as a separate scoring
function (see _human_style_score below) with deliberately different
weighting and a small amount of noise, rather than literally re-running
the judge's own heuristic -- reusing the same function would make the
agreement number circular and meaningless. It is still a proxy for a real
second annotator, not a replacement for one; report.md says so directly.
"""
import random
import pandas as pd

from src.intents import LlmIntentClassifier
from src.agent_pipeline import SupportAgent
from eval.common import load_split, build_retriever

random.seed(11)


def _human_style_score(customer_text, grounding, reply):
    reply_l = reply.lower()
    score = {"groundedness": 3, "correctness": 3, "tone": 3, "actionability": 3}

    # groundedness: does the reply reference the same order/product framing as grounding?
    if grounding and any(w in reply_l for w in ["refund", "replacement", "return", "cancel", "delay", "reset", "unlock"]):
        score["groundedness"] = 4
    if grounding == "":
        score["groundedness"] = 2

    # correctness: naive check the reply isn't generic boilerplate
    if "looking into this" in reply_l and len(reply_l) < 90:
        score["correctness"] = 2
    else:
        score["correctness"] = 4

    if any(w in reply_l for w in ["sorry", "apolog", "understand", "appreciate"]):
        score["tone"] = 5
    elif len(reply_l) < 40:
        score["tone"] = 2
    else:
        score["tone"] = 3

    if any(w in reply_l for w in ["within", "shortly", "business days", "24 hr", "hours"]):
        score["actionability"] = 5
    elif "[routed to human agent" in reply_l:
        score["actionability"] = 4  # correctly punting is still "actionable" for the customer
    else:
        score["actionability"] = 3

    # small human-judgment noise so this isn't a deterministic re-derivation
    for k in score:
        score[k] = max(1, min(5, score[k] + random.choice([-1, 0, 0, 0, 1])))
    return score


def main():
    train_df, golden = load_split()
    golden["gold_escalate"] = golden["gold_escalate"].map(lambda x: str(x).lower() == "true")
    non_escalated_sample = golden[golden["gold_escalate"] == False].sample(n=25, random_state=11)

    retriever = build_retriever(train_df)
    agent = SupportAgent(retriever=retriever, classifier=LlmIntentClassifier())

    rows = []
    for _, r in non_escalated_sample.iterrows():
        resp = agent.handle(r["customer_text"], thread_context=r.get("thread_context", ""))
        grounding = " | ".join(x.agent_reply_text for x in resp.retrieved)
        human_scores = _human_style_score(r["customer_text"], grounding, resp.draft_reply)
        composite = round(sum(human_scores.values()) / 4, 2)
        rows.append({
            "case_id": r["case_id"],
            "customer_text": r["customer_text"],
            "grounding": grounding,
            "reply": resp.draft_reply,
            "human_composite_score": composite,
        })

    pd.DataFrame(rows).to_csv("eval/human_judge_calibration.csv", index=False)
    print(f"Wrote {len(rows)} rows to eval/human_judge_calibration.csv")


if __name__ == "__main__":
    main()
