"""
Scope Guard — two-tier scope classification.

Tier 1 (fast):  Keyword heuristics — no LLM cost.
Tier 2 (fallback):  LLM classification — only when Tier 1 is ambiguous.

Out-of-scope messages are rejected politely.  After 3+ out-of-scope
attempts the agent offers to transfer to a human representative.
"""

import logging

logger = logging.getLogger(__name__)


class ScopeGuard:
    """Filters out-of-scope messages before they reach the main LLM call."""

    # Insurance-related positive signals
    IN_SCOPE_KEYWORDS = {
        "claim", "policy", "insurance", "denied", "denial", "appeal",
        "coverage", "deductible", "reimbursement", "document", "submit",
        "verification", "verify", "identity", "name", "dob", "date of birth",
        "ssn", "social security", "phone", "email", "policy number",
        "healthcare", "dental", "auto", "medical", "pathology", "office note",
        "status", "payment", "amount", "fee", "deadline", "representative",
        "claim id", "case", "policyholder", "premium", "beneficiary",
        "copay", "out of pocket", "network", "provider", "in-network",
        "eob", "explanation of benefits", "pre-authorization", "referral",
        # Conversational tokens that are always fine
        "yes", "no", "okay", "sure", "thanks", "thank you", "please",
        "help", "hi", "hello", "hey", "can you", "i need", "i want",
        "i have", "my", "send", "summary", "skip", "that's all",
        "nothing else", "i'm good", "got it",
    }

    # Strong out-of-scope signals
    OUT_OF_SCOPE_SIGNALS = [
        "what is rl", "reinforcement learning", "machine learning",
        "neural network", "python code", "javascript", "write me",
        "tell me a joke", "who is the president", "what's the weather",
        "recipe", "movie recommendation", "song", "game", "sports score",
        "stock price", "crypto", "bitcoin", "translate this",
        "homework", "essay", "math problem", "what is ai",
        "explain quantum", "how does gravity",
    ]

    def __init__(self, llm_client=None):
        self.llm_client = llm_client

    def classify(self, message: str, state) -> dict:
        """Classify as in-scope or out-of-scope.

        Returns dict with:
          in_scope (bool), reason (str), tier (int), count (int, if OOS)
        """
        msg_lower = message.lower().strip()

        # Very short messages are always in-scope (greetings, yes/no, etc.)
        if len(msg_lower.split()) <= 3:
            return {"in_scope": True, "reason": "short message", "tier": 1}

        # Tier 1 — strong out-of-scope signals
        for signal in self.OUT_OF_SCOPE_SIGNALS:
            if signal in msg_lower:
                state.out_of_scope_count += 1
                logger.info("Scope guard: OOS signal '%s' (count=%d)", signal, state.out_of_scope_count)
                return {
                    "in_scope": False,
                    "reason": f"out-of-scope signal: {signal}",
                    "tier": 1,
                    "count": state.out_of_scope_count,
                }

        # Tier 1 — in-scope keywords
        for keyword in self.IN_SCOPE_KEYWORDS:
            if keyword in msg_lower:
                return {"in_scope": True, "reason": f"keyword: {keyword}", "tier": 1}

        # Mid-conversation messages with pronouns/conversational words
        if state.message_history:
            conversational = {"it", "that", "this", "what", "how", "when",
                              "why", "can", "do", "did", "will", "would",
                              "is", "are", "could", "should"}
            words = set(msg_lower.split())
            if words & conversational:
                return {"in_scope": True, "reason": "conversational context", "tier": 1}

        # Tier 2 — LLM classification (only if truly ambiguous)
        if self.llm_client:
            return self._llm_classify(message, state)

        # Default pass-through
        return {"in_scope": True, "reason": "default pass-through", "tier": 1}

    def _llm_classify(self, message: str, state) -> dict:
        """Lightweight LLM call for ambiguous messages."""
        result = self.llm_client.chat(
            system_prompt=(
                "You are a scope classifier.  Reply with exactly "
                "'IN_SCOPE' or 'OUT_OF_SCOPE'.  A message is IN_SCOPE if it "
                "relates to insurance claims, policies, identity verification, "
                "or is a natural conversational response in a customer-service "
                "call.  Everything else is OUT_OF_SCOPE."
            ),
            messages=[
                {"role": "user", "content": f'Classify: "{message}"'}
            ],
            temperature=0.0,
        )

        is_in_scope = "IN_SCOPE" in result.get("content", "").upper()

        if not is_in_scope:
            state.out_of_scope_count += 1

        return {
            "in_scope": is_in_scope,
            "reason": "LLM classification",
            "tier": 2,
            "count": state.out_of_scope_count if not is_in_scope else 0,
        }

    @staticmethod
    def get_rejection_message(state) -> str:
        """Graduated rejection messages based on repeat count."""
        count = state.out_of_scope_count

        if count >= 3:
            return (
                "I appreciate your patience, but I'm only able to assist with "
                "insurance claims and policy-related questions. Since I may not "
                "be the right resource for your other questions, would you like "
                "me to transfer you to a human representative who might be able "
                "to help?"
            )
        if count == 2:
            return (
                "I understand, but I'm specifically designed to help with "
                "insurance claims and policy matters. I'm not able to help "
                "with other topics. Is there anything about your insurance I "
                "can assist with?"
            )
        return (
            "I appreciate the question, but I'm only able to help with "
            "insurance claims and policy-related topics. Is there anything "
            "about your insurance I can help with?"
        )
