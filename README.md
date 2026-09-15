# AmazonHelp Support Agent — Hiver SDE Intern Take-Home

An AI support agent for **AmazonHelp** (Twitter customer support) that
classifies incoming messages into a support taxonomy, drafts a reply
grounded in how the brand has historically resolved similar issues, and
decides auto-handle vs. escalate-to-human with a stated reason.

**Start with `report.md`** for problem framing, results vs. baselines,
failure analysis, the mandatory "what's misleading about my headline
number" section, and next steps. **`decision_log.md`** lists the 15
non-obvious calls made and why.

## Data — read this before looking at any numbers

This build environment has no network access to `kaggle.com`, so the real
*Customer Support on Twitter* dataset couldn't be downloaded here. Instead,
`src/generate_synthetic_data.py` generates a stand-in dataset with the
**exact schema** of the real `twcs.csv` for brand `AmazonHelp`, with
deliberately injected realism (typos, sarcasm/emoji mismatches, multi-intent
messages, unresolved threads, ~7% off-taxonomy noise). Every module from
`data_prep.py` onward only cares about the schema, so pointing at the real
file is a one-line swap:

```bash
python src/data_prep.py --csv /path/to/real_twcs.csv --brand AmazonHelp --out data/cases.csv
```

All numbers in `report.md` are real outputs of this code run against the
synthetic stand-in — see report.md §4 for exactly how and why they'd move
on the real dataset.

## Reproduce the headline results (~10 seconds, no API key required)

```bash
git clone <this repo>
cd hiver-support-agent
pip install -r requirements.txt
make all
```

`make all` = `make data` (generate data → build cases → build golden set →
build judge-calibration set) + `make eval` (run system + both baselines +
reply-quality judging, write `eval/results_summary.csv`).

Expect output ending in a comparison table like:

```
         intent_accuracy  intent_macro_f1  escalation_precision  escalation_recall  ...
trivial           0.2412           0.0432                0.0000                0.0
simple            0.6131           0.5803                1.0000                0.5
system            0.8945           0.8098                0.0699                1.0
```

Run the smoke tests:

```bash
PYTHONPATH=. python3 -m pytest tests/ -q
```

## Running with a real model instead of the offline mock

By default the pipeline runs in a deterministic **offline mock mode**
(`src/llm_client.py`) so `make eval` finishes in seconds with zero
credentials — required for the "reproduce in under 15 minutes" bar. The
mock mode is a template/keyword-heuristic engine, **not a real LLM**, and
`report.md` calls this out explicitly wherever it affects a number
(especially reply-quality judging — see report.md §4, point 3).

To use a real model instead:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
pip install anthropic
make eval   # re-run; llm_client.py auto-detects the key and switches modes
```

`src/llm_client.py` also supports `OPENAI_API_KEY` as a fallback path.

## Repo layout

```
src/
  generate_synthetic_data.py   synthetic twcs-schema data (see Data note above)
  data_prep.py                 load twcs csv -> filter brand -> build "cases"
  intents.py                   taxonomy + TF-IDF classifier + LLM classifier
  retrieval.py                 TF-IDF retrieval over historical resolutions
  escalation.py                rule/confidence escalation decision + reason
  agent_pipeline.py            orchestrates the above into SupportAgent
  llm_client.py                unified LLM wrapper (real model or offline mock)
baselines/
  trivial_baseline.py          majority class, canned reply, never escalates
  simple_baseline.py           keyword rules, per-intent canned reply
eval/
  build_golden_set.py          stratified sampling + labeling methodology
  build_calibration_set.py     hand-scored subsample for judge-agreement check
  common.py                    train/holdout split (golden set never leaks in)
  metrics.py                   intent accuracy/F1/confusion, escalation P/R/F1
  llm_judge.py                 4-dimension reply-quality rubric + agreement calc
  run_eval.py                  runs everything, writes eval/results_summary.csv
tests/
  test_pipeline.py             smoke tests (pytest)
report.md                      the required report
decision_log.md                the required decision log
```

## What's borrowed vs. original

Nothing in this repo is copied from an external source. Libraries used:
`pandas`, `numpy`, `scikit-learn` (TF-IDF vectorizer, LogisticRegression,
cosine similarity, standard classification metrics), `joblib`. Optional:
`anthropic` / `openai` SDKs for the real-model path. The Kaggle dataset
schema (column names) is reproduced from the public dataset card for
`thoughtvector/customer-support-on-twitter` — no dataset content is
reproduced since the real file was never accessible in this environment.
