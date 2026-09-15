"""
agent_pipeline.py
---------------------------------------------------------------------------
Ties intents.py + retrieval.py + escalation.py + llm_client.py into one
call: SupportAgent.handle(customer_text) -> AgentResponse.
"""
from dataclasses import dataclass
from typing import List, Optional

from src.intents import LlmIntentClassifier, TAXONOMY_DESCRIPTIONS
from src.retrieval import ReplyRetriever, RetrievedCase
from src.escalation import decide, EscalationDecision
from src.llm_client import call_llm

REPLY_SYSTEM_PROMPT = (
    "You are drafting a customer support reply for AmazonHelp on Twitter. "
    "Write a short (2-4 sentence), empathetic, specific reply. Ground your answer in the "
    "example historical resolutions provided -- match this brand's actual policy and tone, "
    "don't invent a different policy. If the historical examples don't clearly cover this "
    "case, say you're looking into it rather than guessing at a resolution. "
    "Do not sign with a name/initials unless the examples do."
)


@dataclass
class AgentResponse:
    customer_text: str
    intent: str
    intent_confidence: float
    intent_method: str
    retrieved: List[RetrievedCase]
    draft_reply: str
    escalate: bool
    escalation_reason: str


class SupportAgent:
    def __init__(self, retriever: ReplyRetriever, classifier: Optional[LlmIntentClassifier] = None):
        self.retriever = retriever
        self.classifier = classifier or LlmIntentClassifier()

    def handle(self, customer_text, thread_context="") -> AgentResponse:
        pred = self.classifier.predict(customer_text, thread_context=thread_context)
        retrieved = self.retriever.retrieve(customer_text, intent=pred.label, k=3)
        top_score = retrieved[0].score if retrieved else 0.0

        esc: EscalationDecision = decide(pred.label, pred.confidence, customer_text, top_score)

        if esc.escalate:
            draft = ("[Routed to human agent -- no auto-reply sent] "
                     f"Reason: {esc.reason}")
        else:
            grounding_text = "\n".join(
                f"- Similar past case: \"{r.customer_text}\" -> Resolution sent: \"{r.agent_reply_text}\""
                for r in retrieved
            )
            user_prompt = (
                f"Customer intent: {pred.label} ({TAXONOMY_DESCRIPTIONS.get(pred.label, '')})\n"
                f"Customer message: {customer_text}\n\n"
                f"Historical resolutions for similar cases:\n{grounding_text}\n\n"
                f"Draft the reply now."
            )
            draft = call_llm(REPLY_SYSTEM_PROMPT, user_prompt, mode="generate",
                              grounding_snippets=[r.agent_reply_text for r in retrieved])

        return AgentResponse(
            customer_text=customer_text,
            intent=pred.label,
            intent_confidence=pred.confidence,
            intent_method=pred.method,
            retrieved=retrieved,
            draft_reply=draft,
            escalate=esc.escalate,
            escalation_reason=esc.reason,
        )
