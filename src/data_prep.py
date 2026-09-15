"""
data_prep.py
---------------------------------------------------------------------------
Loads a twcs-schema CSV (real Kaggle file OR our synthetic stand-in --
identical columns), filters to one brand, and reconstructs threads by
following in_response_to_tweet_id / response_tweet_id links.

Output unit: a "case" = one customer inbound message + the brand's first
reply + any following back-and-forth in that thread. This is the unit
the rest of the pipeline (classification, retrieval, escalation) operates
on, matching how a real support agent triages one incoming message at a
time.

To point this at the REAL Kaggle dataset instead of the synthetic one:
    python src/data_prep.py --csv path/to/twcs.csv --brand AmazonHelp
No other file needs to change -- schema is identical.
"""
import argparse
import pandas as pd


def load_raw(csv_path):
    df = pd.read_csv(csv_path, dtype={"tweet_id": str, "response_tweet_id": str,
                                       "in_response_to_tweet_id": str})
    df["inbound"] = df["inbound"].astype(str).str.lower().isin(["true", "1"])
    return df


def build_cases(df, brand):
    df = df.copy()
    df_indexed = df.set_index("tweet_id")

    brand_replies = df[(df["inbound"] == False) & (df["author_id"] == brand)]
    cases = []
    for _, reply_row in brand_replies.iterrows():
        parent_id = reply_row["in_response_to_tweet_id"]
        if pd.isna(parent_id) or parent_id == "" or parent_id not in df_indexed.index:
            continue
        parent = df_indexed.loc[parent_id]
        if not parent["inbound"]:
            continue  # brand replying to itself / another agent, skip

        # walk forward for a follow-up customer turn + second agent turn, if any
        followup_customer, followup_agent = None, None
        child_rows = df[df["in_response_to_tweet_id"] == reply_row["tweet_id"]]
        cust_children = child_rows[child_rows["inbound"] == True]
        if len(cust_children):
            followup_customer = cust_children.iloc[0]["text"]
            gc = df[df["in_response_to_tweet_id"] == cust_children.iloc[0]["tweet_id"]]
            agent_gc = gc[(gc["inbound"] == False) & (gc["author_id"] == brand)]
            if len(agent_gc):
                followup_agent = agent_gc.iloc[0]["text"]

        cases.append({
            "case_id": f"{parent.name}",
            "customer_text": parent["text"],
            "agent_reply_text": reply_row["text"],
            "followup_customer_text": followup_customer,
            "followup_agent_text": followup_agent,
            "created_at": parent["created_at"],
        })
    return pd.DataFrame(cases).drop_duplicates(subset=["case_id"]).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/twcs_synthetic.csv")
    ap.add_argument("--brand", default="AmazonHelp")
    ap.add_argument("--out", default="data/cases.csv")
    args = ap.parse_args()

    df = load_raw(args.csv)
    cases = build_cases(df, args.brand)
    cases.to_csv(args.out, index=False)
    print(f"Built {len(cases)} cases for brand={args.brand} -> {args.out}")


if __name__ == "__main__":
    main()
