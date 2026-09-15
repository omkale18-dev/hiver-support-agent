"""
llm_judge.py
---------------------------------------------------------------------------
Judges a drafted reply on a 4-dimension rubric, 1-5 each:
  - groundedness   : consistent with the retrieved historical resolution
                      (doesn't invent a policy/promise not seen before)
  - correctness    : actually addresses what the customer asked
  - tone           : empathetic, professional, not robotic/curt
  - actionability  : gives the customer a concrete next step or timeframe

Uses a real Claude call when ANTHROPIC_API_KEY is set (see llm_client.py);
otherwise falls back to a heuristic scorer (JUDGE_MODE below reports
which one ran). The heuristic fallback is intentionally simple lexical
scoring -- it exists so `make eval` is reproducible without credentials,
NOT as a claim that it's as good as a real LLM judge. report.md's
"what's misleading" section calls this out explicitly.

Human-agreement calibration: score_agreement() compares judge scores
against a small hand-scored subsample (eval/human_judge_calibration.csv,
built once by the assignment author reading 25 transcripts) and reports
Cohen's kappa (scores binned to agree/disagree at a >=4 "good reply"
threshold) plus mean absolute error on the raw 1-5 scale.
"""
import re
import json
import pandas as pd
from sklearn.metrics import cohen_kappa_score, mean_absolute_error

from src.llm_client import MODE as LLM_MODE, _client

JUDGE_MODE = "anthropic" if LLM_MODE == "anthropic" else "heuristic_fallback"

RUBRIC_PROMPT = """You are grading a customer support reply on 4 dimensions, each 1-5:
- groundedness: does it stay consistent with the provided historical resolution examples, without inventing new policy?
- correctness: does it actually address what the customer asked?
- tone: empathetic, professional, human -- not robotic or curt?
- actionability: does the customer know what happens next and roughly when?

Customer message: {customer_text}
Historical resolution examples used for grounding: {grounding}
Drafted reply: {reply}

Respond ONLY with JSON: {{"groundedness": <1-5>, "correctness": <1-5>, "tone": <1-5>, "actionability": <1-5>}}"""

EMPATHY_MARKERS = ["sorry", "apolog", "understand", "thanks for", "appreciate", "no problem", "happy to help"]
ACTION_MARKERS = ["will", "within", "shortly", "business days", "refund", "replacement", "email", "shipped", "escalat"]
ROBOTIC_MARKERS = ["your request has been logged", "ticket number", "per policy"]


def _heuristic_score(customer_text, grounding, reply):
    reply_l = reply.lower()
    # groundedness: token overlap between reply and grounding text
    ground_tokens = set(re.findall(r"[a-z]+", grounding.lower()))
    reply_tokens = set(re.findall(r"[a-z]+", reply_l))
    overlap = len(ground_tokens & reply_tokens) / (len(reply_tokens) or 1)
    groundedness = 1 + round(min(4, overlap * 10))

    # correctness: overlap with the customer's own content words (order ids, key nouns)
    cust_tokens = set(re.findall(r"[a-z0-9#]+", customer_text.lower()))
    corr_overlap = len(cust_tokens & reply_tokens) / (len(cust_tokens) or 1)
    correctness = 1 + round(min(4, corr_overlap * 8))

    tone = 2 + sum(1 for m in EMPATHY_MARKERS if m in reply_l)
    tone = min(5, tone) - (1 if any(m in reply_l for m in ROBOTIC_MARKERS) else 0)
    tone = max(1, tone)

    actionability = 1 + sum(1 for m in ACTION_MARKERS if m in reply_l)
    actionability = min(5, actionability)

    return {"groundedness": groundedness, "correctness": correctness, "tone": tone, "actionability": actionability}


def judge_reply(customer_text, grounding, reply):
    if JUDGE_MODE == "anthropic":
        prompt = RUBRIC_PROMPT.format(customer_text=customer_text, grounding=grounding, reply=reply)
        msg = _client.messages.create(
            model="claude-sonnet-4-6", max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        txt = msg.content[0].text
        try:
            return json.loads(re.search(r"\{.*\}", txt, re.S).group(0))
        except Exception:
            return _heuristic_score(customer_text, grounding, reply)
    return _heuristic_score(customer_text, grounding, reply)


def composite(scores):
    return round(sum(scores.values()) / len(scores), 2)


def score_agreement(calibration_csv="eval/human_judge_calibration.csv"):
    """Compares judge scores to a small hand-scored subsample."""
    cal = pd.read_csv(calibration_csv)
    judge_scores, human_scores = [], []
    for _, row in cal.iterrows():
        s = judge_reply(row["customer_text"], row["grounding"], row["reply"])
        judge_scores.append(composite(s))
        human_scores.append(row["human_composite_score"])

    judge_bin = [1 if s >= 4 else 0 for s in judge_scores]
    human_bin = [1 if s >= 4 else 0 for s in human_scores]
    kappa = cohen_kappa_score(judge_bin, human_bin) if len(set(human_bin)) > 1 and len(set(judge_bin)) > 1 else float("nan")
    mae = mean_absolute_error(human_scores, judge_scores)
    return {
        "n": len(cal),
        "judge_mode": JUDGE_MODE,
        "cohens_kappa_good_reply_threshold": round(kappa, 3) if kappa == kappa else "undefined (insufficient class variance)",
        "mean_absolute_error_1to5_scale": round(mae, 3),
    }
