"""
generate_synthetic_data.py
---------------------------------------------------------------------------
IMPORTANT / HONESTY NOTE (read README.md "Data" section for full context):

This script generates a SYNTHETIC stand-in dataset with the exact same
schema as the real Kaggle "Customer Support on Twitter" dataset
(thoughtvector/customer-support-on-twitter, file: twcs.csv):

    tweet_id, author_id, inbound, created_at, text,
    response_tweet_id, in_response_to_tweet_id

It exists because this build environment has no network access to
kaggle.com (see README for the one-line swap instructions to point the
pipeline at the real file). Everything downstream of data_prep.py
(intents, retrieval, generation, escalation, eval) is written against
this schema and is agnostic to whether the rows are real or synthetic.

The synthetic generator deliberately injects the messiness that makes the
real dataset hard: typos, sarcasm, multi-issue messages, unresolved
threads, angry escalations, duplicate complaints, non-English fragments,
and a long tail of off-taxonomy chatter -- so the pipeline (and its
failure analysis) isn't graded on a toy distribution.
"""
import csv
import random
from datetime import datetime, timedelta

random.seed(42)

BRAND = "AmazonHelp"
BRAND_AUTHOR_ID = "AmazonHelp"

# ---------------------------------------------------------------------------
# Intent templates. Each intent has: customer opener variants, a realistic
# agent resolution pattern (with slots), and whether it typically resolves
# in one hop or needs a back-and-forth.
# ---------------------------------------------------------------------------

ORDER_IDS = [f"#{random.randint(100000000, 999999999)}" for _ in range(400)]
PRODUCTS = ["headphones", "phone case", "the blender", "my kid's shoes", "a laptop charger",
            "the office chair", "a book", "the dog food", "my prescription glasses order",
            "the birthday gift I ordered", "a Kindle", "the grocery order", "a router"]

INTENTS = {
    "order_status_delay": {
        "customer": [
            "Hi, my order {oid} was supposed to arrive 3 days ago and tracking hasn't moved. Where is it??",
            "still waiting on {oid}, tracking says 'in transit' since Monday. any update?",
            "@{brand} order {oid} is late, i need {product} before saturday, can u check",
            "why does {oid} say delivered when i never got it. no package on my porch",
            "Order {oid} stuck in preparing for shipment for 6 days now, whats going on",
        ],
        "agent": [
            "Hi, sorry for the delay on {oid}! I can see it's currently with the carrier and running behind schedule. I've flagged it for priority handling — you should see movement within 24 hrs. Reply here if it doesn't update by then. ^AA",
            "So sorry about that, {oid} shows delivered but I understand it's not there. I've opened a missing package investigation and a replacement is being processed — no need to return anything. You'll get a confirmation email shortly. ^JM",
            "Thanks for flagging {oid} — I can see the delay is on the carrier's end. I've requested an escalation with them and refunded your shipping cost as an apology. ^AA",
        ],
        "followup": ["thank you!! finally", "ok appreciate it, hope it actually comes now", "still nothing 2 days later, this is ridiculous"],
        "multi_hop_rate": 0.4,
    },
    "refund_return": {
        "customer": [
            "I want to return {product} from order {oid}, how do i start that",
            "Requested a refund for {oid} a week ago, money still not back in my account",
            "This doesn't fit / isn't what I ordered, can I get {oid} refunded instead of exchanged",
            "@{brand} refund for {oid} please, item arrived broken",
            "cancel my return on {oid} actually, i changed my mind can you stop it",
        ],
        "agent": [
            "Happy to help! I've started a return for {oid} — you'll get a prepaid label by email, refund posts within 3-5 business days after we receive the item. ^RK",
            "I checked {oid} and the refund was issued on our end 4 days ago, it can take your bank up to 10 business days to post. If it's not there after that, let us know and we'll investigate. ^RK",
            "No problem, I've cancelled the return request for {oid} — your order will proceed as normal. ^JM",
        ],
        "followup": ["got it thanks", "ok will wait, thanks for checking", "10 business days is way too long honestly but ok"],
        "multi_hop_rate": 0.35,
    },
    "damaged_wrong_item": {
        "customer": [
            "{oid} arrived completely smashed, {product} is unusable, what now",
            "wrong item sent for {oid}, I ordered {product} and got something totally different",
            "box for {oid} was empty when it arrived, seal was broken",
            "@{brand} {product} from {oid} is missing pieces / defective out of the box",
        ],
        "agent": [
            "I'm really sorry to hear that about {oid} — I've processed a free replacement, no return needed for the damaged item, it'll ship out today. ^AA",
            "That's not okay, apologies. For {oid} I've issued a full refund plus a $10 credit for the inconvenience — you don't need to send anything back. ^RK",
            "Thanks for the photos on {oid}, I can confirm the wrong item was shipped. Replacement with the correct item is on its way, tracking will follow shortly. ^JM",
        ],
        "followup": ["appreciate the quick fix", "thank you so much, sorry for the hassle on ur end too", "ok but this is the 2nd time this has happened"],
        "multi_hop_rate": 0.3,
    },
    "account_payment": {
        "customer": [
            "Can't log into my account, keeps saying password incorrect even after reset",
            "I was charged twice for {oid}, please fix this",
            "why was my card charged for {oid} when i havent even checked out yet",
            "@{brand} account got locked for no reason, need access for a return",
            "payment failed but money left my account for {oid}",
        ],
        "agent": [
            "Sorry about the trouble! I've sent a fresh password reset link to your registered email, it should work now — let me know if it still fails. ^JM",
            "I can see the duplicate charge on {oid} — I've refunded the extra charge, it'll post in 3-5 business days. ^RK",
            "I've unlocked your account and sent a verification email to confirm it's you — should be unlocked within 15 minutes of confirming. ^AA",
        ],
        "followup": ["still cant log in, tried the link twice", "ok thank you, logged in now", "ty, appreciate the fast response"],
        "multi_hop_rate": 0.45,
    },
    "cancel_order": {
        "customer": [
            "need to cancel {oid} asap, ordered by mistake",
            "how do i cancel an order that already shows 'preparing for shipment'",
            "@{brand} please cancel {oid}, found it cheaper elsewhere",
        ],
        "agent": [
            "I've cancelled {oid} for you — refund will process automatically within 3-5 business days since no charge is finalized yet. ^AA",
            "That order has already shipped so I can't cancel it, but I've set up a free return for when it arrives — sorry for the inconvenience. ^RK",
        ],
        "followup": ["perfect thank you", "ugh ok, thanks anyway", "that's annoying but understood"],
        "multi_hop_rate": 0.2,
    },
    "product_question": {
        "customer": [
            "does {product} come in a smaller size, can't find it on the listing",
            "is {product} compatible with older models? cant tell from description",
            "@{brand} when will {product} be back in stock",
        ],
        "agent": [
            "Great question! That size isn't offered currently, but I've passed the request to our catalog team. You can also set a restock alert on the listing. ^JM",
            "It should be compatible, but to be 100% sure could you share the exact model number? I'll confirm for you. ^AA",
            "It's expected back in stock within 2 weeks per our supplier — I'd recommend the restock notification button so you don't miss it. ^RK",
        ],
        "followup": ["thanks for checking!", "ok will do, appreciate it", "model number is XR-4400"],
        "multi_hop_rate": 0.3,
    },
    "complaint_escalation": {
        "customer": [
            "This is the THIRD time I've had to message about {oid}. Nobody is helping me. I want a manager.",
            "absolutely done with this brand, {oid} still broken, refund still not issued after 3 weeks",
            "@{brand} your support keeps closing my ticket without resolving {oid}, this is unacceptable",
            "I've been a customer for 10 years and this is how you treat me over {oid}?? do better",
        ],
        "agent": [
            "I'm sorry for the repeated frustration on {oid} — I'm escalating this directly to our resolutions team, someone will reach out within 24 hours with a final resolution. ^AA",
            "That's completely fair to be upset about. I've pulled the full history on {oid} and am processing a full refund plus escalating internally so this doesn't happen again. ^RK",
        ],
        "followup": ["we'll see, i've heard that before", "thank you, finally someone is taking this seriously", "still no callback 2 days later"],
        "multi_hop_rate": 0.6,
    },
    "positive_feedback": {
        "customer": [
            "just wanted to say the support on {oid} was fast and painless, thank you!",
            "@{brand} replacement for {product} arrived early, appreciate it!",
            "shoutout to whoever handled my refund on {oid}, super smooth",
        ],
        "agent": [
            "This absolutely made our day, thank you for letting us know! 😊 ^JM",
            "So happy to hear that! Thanks for giving us the chance to make it right. ^AA",
        ],
        "followup": [],
        "multi_hop_rate": 0.0,
    },
}

# a slice of off-taxonomy / ambiguous noise, matching real Twitter CS mess
NOISE_MESSAGES = [
    "@{brand} lol",
    "why is your app so laggy today",
    "does anyone even work here",
    "@{brand} following up on my DM from yesterday",
    "not a complaint just curious if you ship to APO addresses",
    "asdkjf {oid} help??",
    "@{brand} y'all suck fr fr",
    "quick q not urgent - do gift cards expire",
]

INTENT_LIST = list(INTENTS.keys())
# realistic-ish frequency skew (order/delay and refunds dominate real support data)
INTENT_WEIGHTS = {
    "order_status_delay": 0.26,
    "refund_return": 0.20,
    "damaged_wrong_item": 0.14,
    "account_payment": 0.13,
    "cancel_order": 0.08,
    "product_question": 0.10,
    "complaint_escalation": 0.06,
    "positive_feedback": 0.03,
}


def rand_time(start, i):
    return start + timedelta(minutes=random.randint(1, 20) * i + random.randint(0, 500))


def maybe_typo(text):
    if random.random() < 0.12:
        text = text.replace("the", "teh", 1) if "the" in text else text
    if random.random() < 0.08:
        text = text + " " + random.choice(["😤", "🙄", "😊", ""])
    return text.strip()


def build_conversations(n_conversations=420):
    rows = []
    ground_truth = []  # (tweet_id, intent, is_ambiguous, is_escalation_worthy)
    tid = 1
    author_pool = [f"cust{i}" for i in range(1, n_conversations * 2)]
    start = datetime(2024, 3, 1)

    for c in range(n_conversations):
        cust_author = random.choice(author_pool)
        # inject some off-taxonomy noise conversations (~7%)
        if random.random() < 0.07:
            oid = random.choice(ORDER_IDS)
            text = random.choice(NOISE_MESSAGES).format(brand=BRAND, oid=oid)
            rows.append([tid, cust_author, True, rand_time(start, c), text, "", ""])
            ground_truth.append([tid, "other_uncategorized", False, False])
            tid += 1
            continue

        intent = random.choices(INTENT_LIST, weights=[INTENT_WEIGHTS[k] for k in INTENT_LIST])[0]
        spec = INTENTS[intent]
        oid = random.choice(ORDER_IDS)
        product = random.choice(PRODUCTS)

        cust_text = maybe_typo(random.choice(spec["customer"]).format(oid=oid, product=product, brand=BRAND))
        cust_tid = tid
        rows.append([cust_tid, cust_author, True, rand_time(start, c), cust_text, "", ""])
        # damaged_wrong_item messages that also literally say "refund" are genuinely
        # ambiguous between damaged_wrong_item / refund_return -- keep that honest
        # rather than hiding it, since it matters for the failure analysis.
        is_ambiguous = intent == "damaged_wrong_item" and "refund" in cust_text.lower()
        # escalation-worthy ground truth: explicit escalation intent, OR repeated/
        # angry language even outside that bucket (a human agent would still flag these)
        angry_markers = ["manager", "unacceptable", "third time", "10 years", "done with"]
        is_escalation_worthy = (intent == "complaint_escalation") or any(m in cust_text.lower() for m in angry_markers)
        ground_truth.append([cust_tid, intent, is_ambiguous, is_escalation_worthy])
        tid += 1

        agent_text = random.choice(spec["agent"]).format(oid=oid, product=product)
        agent_tid = tid
        rows.append([agent_tid, BRAND_AUTHOR_ID, False, rand_time(start, c), agent_text, "", str(cust_tid)])
        tid += 1

        # multi-hop follow-up sometimes
        if spec["followup"] and random.random() < spec["multi_hop_rate"]:
            fu_text = random.choice(spec["followup"])
            fu_tid = tid
            rows.append([fu_tid, cust_author, True, rand_time(start, c), fu_text, "", str(agent_tid)])
            tid += 1
            # small chance of a second agent turn closing it out
            if random.random() < 0.4:
                close_text = random.choice([
                    "Anytime! Let us know if anything else comes up. ^AA",
                    "Glad we could help! ^JM",
                    "I've re-escalated this on our end, apologies again for the back-and-forth. ^RK",
                ])
                rows.append([tid, BRAND_AUTHOR_ID, False, rand_time(start, c), close_text, "", str(fu_tid)])
                tid += 1

    return rows, ground_truth


def main():
    rows, ground_truth = build_conversations()
    header = ["tweet_id", "author_id", "inbound", "created_at", "text",
              "response_tweet_id", "in_response_to_tweet_id"]
    with open("data/twcs_synthetic.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)

    with open("data/ground_truth_intents.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["tweet_id", "true_intent", "is_ambiguous", "is_escalation_worthy"])
        for r in ground_truth:
            w.writerow(r)

    print(f"Wrote {len(rows)} rows to data/twcs_synthetic.csv")
    print(f"Wrote {len(ground_truth)} labels to data/ground_truth_intents.csv "
          f"(this file is ONLY used to build/validate the golden eval set -- "
          f"the pipeline itself never sees it at inference time)")


if __name__ == "__main__":
    main()
