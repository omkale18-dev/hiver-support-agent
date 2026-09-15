"""
common.py
---------------------------------------------------------------------------
Everything downstream (calibration set, full eval run) needs the SAME
train/holdout split: golden-set case_ids must never appear in the
retrieval corpus or the TF-IDF classifier's training data, or the eval
numbers are measuring memorization, not generalization. This module is
the single place that split happens.
"""
import pandas as pd

from src.retrieval import ReplyRetriever
from src.intents import TfidfIntentClassifier


def load_split():
    cases = pd.read_csv("data/cases.csv", dtype=str)
    gt = pd.read_csv("data/ground_truth_intents.csv", dtype=str)
    golden = pd.read_csv("eval/golden_set.csv", dtype=str)

    labeled = cases.merge(gt, left_on="case_id", right_on="tweet_id", how="inner")
    golden_ids = set(golden["case_id"].astype(str))

    train = labeled[~labeled["case_id"].astype(str).isin(golden_ids)].reset_index(drop=True)
    return train, golden


def build_retriever(train_df):
    corpus = train_df.rename(columns={"true_intent": "intent"})[
        ["customer_text", "agent_reply_text", "intent"]
    ].to_dict("records")
    return ReplyRetriever().fit(corpus)


def build_tfidf_classifier(train_df):
    return TfidfIntentClassifier().fit(train_df["customer_text"].tolist(), train_df["true_intent"].tolist())
