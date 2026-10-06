"""
SOP State Machine — deterministic phase controller.

The LLM NEVER decides phase transitions.  This controller checks hard
gates and advances phases based on server-side verified state only.

Phases:  VERIFY_ID → RESOLVE_INTENT → PROCESS_CASE → POST_PROCESS → COMPLETE
"""

import uuid
from enum import Enum


class Phase(str, Enum):
    VERIFY_ID = "VERIFY_ID"
    RESOLVE_INTENT = "RESOLVE_INTENT"
    PROCESS_CASE = "PROCESS_CASE"
    POST_PROCESS = "POST_PROCESS"
    COMPLETE = "COMPLETE"


PHASE_ORDER = list(Phase)


class SessionState:
    """Holds all mutable state for a single conversation session."""

    def __init__(self):
        self.session_id: str = str(uuid.uuid4())
        self.current_phase: Phase = Phase.VERIFY_ID

        # ---------- Identity verification ----------
        self.verified: bool = False
        self.identity_collected: dict = {
            "name": None,
            "dob": None,
            "phone": None,
            "email": None,
            "ssn_last4": None,
            "policy_number": None,
        }
        self.identity_match_count: int = 0
        self.matched_policyholder: dict | None = None

        # ---------- Cross-phase memory ----------
        self.memory_hints: dict = {
            "intent_hints": [],
            "case_hints": [],
            "emotion_signals": [],
        }

        # ---------- Intent resolution ----------
        self.resolved_intent: str | None = None
        self.resolved_case_id: str | None = None
        self.resolved_case: dict | None = None

        # ---------- Post-process ----------
        self.email_offered: bool = False
        self.email_decision: str | None = None  # "send" | "skip"
        self.email_content: str | None = None

        # ---------- Conversation ----------
        self.conversation_summary: list[str] = []
        self.message_history: list[dict] = []  # for LLM context window

        # ---------- Counters ----------
        self.out_of_scope_count: int = 0
        self.verification_attempts: int = 0
        self.escalation_offered: bool = False

    def to_dict(self) -> dict:
        """Public-safe serialisation for API responses."""
        return {
            "session_id": self.session_id,
            "current_phase": self.current_phase.value,
            "verified": self.verified,
            "identity_match_count": self.identity_match_count,
            "matched_policyholder_name": (
                self.matched_policyholder.get("name")
                if self.matched_policyholder
                else None
            ),
            "resolved_intent": self.resolved_intent,
            "resolved_case_id": self.resolved_case_id,
            "email_decision": self.email_decision,
            "is_complete": self.current_phase == Phase.COMPLETE,
            "out_of_scope_count": self.out_of_scope_count,
        }


class StateMachine:
    """Deterministic SOP phase controller.

    Checks hard gates and advances phases.  The LLM only reports
    extracted data — this class decides what happens.
    """

    def check_and_advance(self, state: SessionState) -> tuple[bool, str]:
        """Check current phase gate; advance if satisfied.

        Returns (advanced: bool, reason: str).
        """
        handler = {
            Phase.VERIFY_ID: self._check_verify_gate,
            Phase.RESOLVE_INTENT: self._check_resolve_gate,
            Phase.PROCESS_CASE: self._check_process_gate,
            Phase.POST_PROCESS: self._check_post_process_gate,
        }.get(state.current_phase)

        if handler is None:
            return False, "already complete"
        return handler(state)

    # ---- individual gates ------------------------------------------------

    @staticmethod
    def _check_verify_gate(state: SessionState) -> tuple[bool, str]:
        """Gate: ≥ 3 PII fields verified against the same policyholder."""
        if state.verified and state.identity_match_count >= 3:
            state.current_phase = Phase.RESOLVE_INTENT
            return True, (
                f"Identity verified with {state.identity_match_count} fields"
            )
        return False, (
            f"Need ≥3 verified fields, have {state.identity_match_count}"
        )

    @staticmethod
    def _check_resolve_gate(state: SessionState) -> tuple[bool, str]:
        """Gate: intent + case_id resolved and validated."""
        if state.resolved_intent and state.resolved_case_id and state.resolved_case:
            state.current_phase = Phase.PROCESS_CASE
            return True, (
                f"Intent resolved: {state.resolved_intent} "
                f"for {state.resolved_case_id}"
            )
        return False, "Intent or case not yet resolved"

    @staticmethod
    def _check_process_gate(_state: SessionState) -> tuple[bool, str]:
        """PROCESS_CASE → POST_PROCESS is triggered by the harness
        when it detects conversational wrap-up signals."""
        return False, "Waiting for case processing to complete"

    @staticmethod
    def _check_post_process_gate(state: SessionState) -> tuple[bool, str]:
        """Gate: email decision made (send or skip)."""
        if state.email_decision is not None:
            state.current_phase = Phase.COMPLETE
            return True, f"Email decision: {state.email_decision}"
        return False, "Waiting for email decision"

    @staticmethod
    def force_advance_to_post_process(state: SessionState) -> bool:
        """Called by the harness when PROCESS_CASE wraps up."""
        if state.current_phase == Phase.PROCESS_CASE:
            state.current_phase = Phase.POST_PROCESS
            return True
        return False
