# Decision log

Plain list of the non-obvious calls made while building this, and why.

1. **Used a schema-identical synthetic dataset instead of the real Kaggle file.**
   `kaggle.com` isn't reachable from this build sandbox's network allowlist.
   Rather than either fabricate "results" against data I never touched, or
   silently skip the eval, I generated a stand-in with the same columns and
   pointed every downstream module at it, with a one-flag swap documented in
   the README and report to point at the real file. Flagged prominently, not
   buried.

2. **Deliberately injected mess into the synthetic data (typos, sarcasm,
   multi-intent messages, off-taxonomy noise, unresolved threads) instead of
   generating clean templated examples.** A pipeline that's only ever graded
   against easy data produces a misleadingly high headline number — the
   whole point of the assignment is proving the system is trustworthy on
   *real*, noisy input.

3. **"Case" = one customer inbound + brand's reply (+ optional follow-up),
   not a full multi-turn dialogue.** Matches how a human triager actually
   works: one incoming message at a time. Simplifies escalation logic
   (no need to model conversational state) at the cost of not handling
   long negotiated threads well — acceptable for v1.

4. **Held out the entire golden set from both the retrieval corpus and the
   TF-IDF classifier's training data** (`eval/common.py`), rather than
   letting eval examples double as training/grounding data. Retrieval
   leakage (a golden example's own historical reply showing up as its own
   "grounding evidence") would have inflated both intent accuracy and
   reply-groundedness scores invisibly.

5. **8 intents, not Banking77's 77.** A human triager for a Twitter support
   account works off a small bucket list; 77 fine-grained banking intents
   is the wrong granularity for this brand/channel. Explicitly a scope
   decision, discussed in report.md section 1.

6. **TF-IDF cosine retrieval instead of embeddings.** At this corpus size
   (a few hundred historical resolutions), TF-IDF is fast, needs no API
   calls, and is fully debuggable (you can see exactly which n-grams
   matched). Embeddings would help once paraphrase-heavy queries start
   missing lexical overlap — flagged as a "next week" item, not implemented
   pre-emptively.

7. **Escalation is a rule/threshold system layered on top of the intent
   classifier's confidence, not a separate learned classifier.** For a
   system this consequential (refunds, legal language, angry customers),
   I wanted every escalation decision to have a human-auditable reason
   string, not a black-box score. Explicit rules also let a brand ops team
   tune thresholds (confidence cutoff, dollar ceiling) without retraining
   anything.

8. **Legal/safety language always escalates regardless of classifier
   confidence.** A confident wrong classification is exactly the failure
   mode to guard hardest against for language like "lawyer" / "injured" /
   "allergic reaction" — confidence-gating alone isn't enough here.

9. **Unescalated cases still get a full reply drafted and judged; escalated
   cases get a placeholder "routed to human" string instead of a draft.**
   Drafting a full reply for a case the system has already decided to route
   to a human would be wasted generation cost and risks the draft leaking
   into an auto-send path by mistake later.

10. **Built an offline deterministic "mock" LLM mode as the default, with a
    real Claude/OpenAI path auto-selected when credentials are present.**
    The assignment requires the README to let a grader reproduce headline
    results in under 15 minutes; requiring API credentials to even run the
    eval would violate that for anyone without keys handy. The mock mode is
    explicitly labeled as a lower quality bar throughout the report, not
    presented as equivalent to a real model.

11. **The "human" side of the judge-agreement check is a second, differently-
    weighted heuristic function with injected noise, not a literal second
    run of the judge's own scorer.** Comparing a function to itself would
    produce a meaningless perfect-agreement number. This is still a proxy
    for a real second annotator (documented as a limitation), not a
    replacement for one.

12. **Deliberately perturbed ~8% of golden-set labels away from the
    generator's "true" label** to simulate real annotator disagreement,
    rather than shipping a golden set where every label is definitionally
    correct by construction. (One of these perturbations turned out, on
    inspection, to be a bad perturbation — see report.md section 4, point 5
    — which I kept in rather than quietly fixing, because it's a real
    example of how eval-methodology noise gets caught in failure analysis.)

13. **Golden set uses stratified sampling with a floor of 10 per intent**,
    not pure random sampling, plus force-included every case flagged
    ambiguous by the generator and a fixed sample of off-taxonomy noise.
    Pure random sampling from the raw frequency distribution would have
    produced a golden set that's ~26% `order_status_delay` and nearly
    empty for rare-but-important intents like `complaint_escalation`,
    making per-class recall unmeasurable for exactly the intents where
    recall matters most.

14. **Reported an "unsafe auto-handle rate" (missed escalations / true
    escalation-worthy cases) as its own metric, separate from
    precision/recall/F1.** F1 treats false escalations and missed
    escalations as symmetric costs; they are not (see report.md problem
    framing). A single blended F1 number would hide exactly the risk this
    system most needs to avoid.

15. **Added a third, non-required reference point (trainable TF-IDF +
    LogisticRegression) alongside the two mandated baselines.** It ended up
    being the most important number in the "misleading headline" section —
    outperforming the LLM classifier on this dataset — which only surfaced
    because I benchmarked something beyond the assignment's minimum bar.
