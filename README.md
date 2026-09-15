# 🤖 AmazonHelp Support Agent

An AI agent that reads a customer support tweet, figures out what they need,
drafts a reply based on how the brand actually resolved similar issues before,
and decides whether to send it or hand it to a human.

Built for the **Hiver SDE Intern take-home assignment**.

📄 **[Read the full report](report.md)** · 📝 **[Decision log](decision_log.md)**

---

## What it does

```
Customer tweet  →  Classify intent  →  Find similar past resolutions  →  Draft reply  →  Auto-send or escalate?
```

| Step | How |
|---|---|
| **Classify** | Sorts each message into 1 of 8 intents (refund, damaged item, order delay, etc.) |
| **Draft a reply** | Grounds the reply in real past resolutions — doesn't invent policy |
| **Escalate or not** | Rule-based checks (confidence, legal language, refund size) — always with a stated reason |

---

## Results

Tested on a 199-example hand-labeled set, against two baselines:

| System | Intent accuracy | Never misses an escalation? |
|---|---|---|
| Trivial (canned reply, guesses majority intent) | 24% | ❌ |
| Simple (keyword rules) | 61% | ❌ (misses half) |
| **This system** | **89%** | ✅ (0 missed) |

Full numbers, failure examples, and an honest **"what's misleading about this
number"** section are in [`report.md`](report.md).

---

## ⚠️ One thing to know about the data

The real dataset (Kaggle's *Customer Support on Twitter*) wasn't reachable
from the build environment, so this uses a **synthetic stand-in** with the
identical file format — same columns, same messiness (typos, sarcasm, noise).
Swapping in the real file is one flag:

```bash
python src/data_prep.py --csv path/to/real_twcs.csv --brand AmazonHelp
```

Full explanation in the report.

---

## Quickstart (~10 seconds, no API key needed)

```bash
git clone <this repo>
cd hiver-support-agent
pip install -r requirements.txt
make all
```

That's it — this generates the data, builds the eval set, and prints a
results table. Add `ANTHROPIC_API_KEY` as an env var to swap in a real Claude
model instead of the offline demo mode.

```bash
PYTHONPATH=. python3 -m pytest tests/ -q   # run tests
```

---

## Project layout

```
src/          the agent — classifier, retrieval, reply drafting, escalation
baselines/    the two comparison systems
eval/         golden set, metrics, LLM-judge, agreement check
tests/        smoke tests
report.md     results, failure analysis, honest caveats
decision_log  15 non-obvious calls made, and why
```

---

Built with `pandas`, `scikit-learn`, and (optionally) the `anthropic` SDK.
No external code was copied — see [`report.md`](report.md) for details.
