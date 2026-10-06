"""
Cross-Phase Memory — structured memory that persists across SOP phases.

Unlike raw chat history (which remembers everything but doesn't structure
it), this module:
  1. Extracts labelled signals (intent hints, case hints, emotions).
  2. Stores them without acting — the SOP phase controls when they're used.
  3. Injects them into prompts at the right moment with the right framing.

Example: During VERIFY_ID the caller says "I'm calling about my denied
healthcare claim from January."  The memory stores this as an intent_hint
and case_hint.  During VERIFY_ID these are NOT injected into the claim-
lookup context.  When RESOLVE_INTENT begins, they ARE injected, so the
LLM can resolve intent without re-asking.
"""


class CrossPhaseMemory:
    """Manages structured cross-phase memory for a session."""

    def __init__(self, state):
        self.state = state

    # --------------------------------------------------------- Storage

    def store_hint(self, hint_data: dict):
        """Store a hint extracted by the LLM's store_memory_hint tool call."""
        if hint_data.get("intent_hint"):
            hint = hint_data["intent_hint"]
            if hint not in self.state.memory_hints["intent_hints"]:
                self.state.memory_hints["intent_hints"].append(hint)

        if hint_data.get("case_hint"):
            hint = hint_data["case_hint"]
            if hint not in self.state.memory_hints["case_hints"]:
                self.state.memory_hints["case_hints"].append(hint)

        if hint_data.get("emotion_signal"):
            self.state.memory_hints["emotion_signals"].append(
                hint_data["emotion_signal"]
            )

    # ------------------------------------------------------- Retrieval

    def get_memory_context(self) -> str:
        """Format stored hints for injection into a phase prompt."""
        parts = []

        intent_hints = self.state.memory_hints.get("intent_hints", [])
        if intent_hints:
            parts.append(f"Intent signals from caller: {'; '.join(intent_hints)}")

        case_hints = self.state.memory_hints.get("case_hints", [])
        if case_hints:
            parts.append(f"Case references from caller: {'; '.join(case_hints)}")

        emotions = self.state.memory_hints.get("emotion_signals", [])
        if emotions:
            recent = emotions[-3:]
            parts.append(f"Recent emotional trajectory: {' → '.join(recent)}")

        return "\n".join(parts) if parts else "No memory hints stored yet."

    def get_latest_emotion(self) -> str:
        emotions = self.state.memory_hints.get("emotion_signals", [])
        return emotions[-1] if emotions else "neutral"

    # ------------------------------------------------ Conversation log

    def add_conversation_summary_point(self, point: str):
        """Add a key point for the post-process email summary."""
        self.state.conversation_summary.append(point)

    def get_conversation_summary(self) -> str:
        if not self.state.conversation_summary:
            return "No summary points recorded."
        return "\n".join(f"- {p}" for p in self.state.conversation_summary)
