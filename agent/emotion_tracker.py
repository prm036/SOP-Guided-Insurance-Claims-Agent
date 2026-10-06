"""
Emotion Tracker — emotional intelligence for the SOP agent.

Tracks emotional trajectory across turns, generates empathy-aware
context for prompt injection, and decides when to offer human escalation.

Works alongside the LLM's emotion detection (via store_memory_hint) to
provide phase-specific de-escalation guidance.
"""


class EmotionTracker:
    """Tracks caller emotions and provides de-escalation guidance."""

    NEGATIVE_EMOTIONS = frozenset({
        "frustrated", "angry", "anxious", "confused", "refusing"
    })

    _GUIDANCE = {
        "frustrated": (
            "Validate frustration, explain the purpose of the current step, "
            "offer alternatives."
        ),
        "angry": (
            "Remain calm and professional.  Acknowledge anger without being "
            "defensive.  Offer concrete next steps or human transfer."
        ),
        "anxious": (
            "Be reassuring.  Explain the process clearly.  Give timeline "
            "expectations."
        ),
        "confused": (
            "Simplify explanations.  Break things into smaller steps.  Ask "
            "if they'd like clarification."
        ),
        "refusing": (
            "Respect their position.  Explain why the step matters.  Offer "
            "alternatives.  If they continue refusing, offer human transfer."
        ),
    }

    def __init__(self, state):
        self.state = state

    def get_emotion_context(self) -> str:
        """Generate emotion-aware context for injection into the prompt."""
        emotions = self.state.memory_hints.get("emotion_signals", [])
        if not emotions:
            return ""

        latest = emotions[-1]
        trajectory = emotions[-5:]

        negative_count = sum(
            1 for e in trajectory if e in self.NEGATIVE_EMOTIONS
        )

        parts: list[str] = []

        if latest in self.NEGATIVE_EMOTIONS:
            parts.append(
                f"⚠️  EMOTIONAL CONTEXT: Caller is currently {latest}."
            )
            parts.append(
                "IMPORTANT: Acknowledge their feelings FIRST before any "
                "business response."
            )
            guidance = self._GUIDANCE.get(latest)
            if guidance:
                parts.append(f"De-escalation: {guidance}")

            if negative_count >= 3:
                parts.append(
                    "NOTE: Caller has been in a negative state for multiple "
                    "turns. Consider offering human transfer."
                )

        elif latest == "neutral" and negative_count > 0:
            parts.append(
                "EMOTIONAL CONTEXT: Caller was previously upset but has "
                "calmed down. Maintain warmth without over-apologising."
            )

        return "\n".join(parts)

    def should_offer_escalation(self) -> bool:
        """True if negative emotions have persisted long enough to
        warrant proactively offering human transfer."""
        emotions = self.state.memory_hints.get("emotion_signals", [])
        recent = emotions[-5:] if len(emotions) >= 5 else emotions
        negative_count = sum(
            1 for e in recent if e in self.NEGATIVE_EMOTIONS
        )
        return negative_count >= 3
