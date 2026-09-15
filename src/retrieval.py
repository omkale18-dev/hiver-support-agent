"""
retrieval.py
---------------------------------------------------------------------------
Builds a searchable index of *historical, already-sent* agent replies
(scoped by intent) so the reply generator can ground new drafts in how
this brand has actually resolved similar issues before, instead of the
LLM inventing a policy on the spot.

Deliberately simple (TF-IDF cosine, not embeddings): the corpus here is
small, replies are short and template-like, and the honest reason to
prefer this over an embedding index is stated in decision_log.md, not
hidden.
"""
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class RetrievedCase:
    customer_text: str
    agent_reply_text: str
    intent: str
    score: float


class ReplyRetriever:
    def __init__(self):
        self.vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=4000)
        self.matrix = None
        self.corpus = None  # list of dicts: customer_text, agent_reply_text, intent

    def fit(self, corpus):
        """corpus: list of {"customer_text", "agent_reply_text", "intent"}"""
        self.corpus = corpus
        texts = [c["customer_text"] for c in corpus]
        self.matrix = self.vec.fit_transform(texts)
        return self

    def retrieve(self, query_text, intent=None, k=3):
        if self.matrix is None:
            raise RuntimeError("Call .fit() first")
        q = self.vec.transform([query_text])
        sims = cosine_similarity(q, self.matrix)[0]

        candidates = []
        for i, sim in enumerate(sims):
            c = self.corpus[i]
            if intent is not None and c["intent"] != intent:
                sim = sim * 0.35  # soft penalty rather than hard filter --
                # real historical resolutions are still useful signal even
                # if our own intent classifier disagrees with the label.
            candidates.append((sim, i))
        candidates.sort(key=lambda x: -x[0])

        out = []
        for sim, i in candidates[:k]:
            c = self.corpus[i]
            out.append(RetrievedCase(customer_text=c["customer_text"],
                                      agent_reply_text=c["agent_reply_text"],
                                      intent=c["intent"], score=round(float(sim), 3)))
        return out
