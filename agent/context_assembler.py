"""
Context Assembler — phase-gated data injection.

Controls WHAT data the LLM can see in each phase.  This is a critical
guardrail layer:

  VERIFY_ID    → NO claim data.  Only emotion context.
  RESOLVE_INTENT → Memory hints + claim list (for disambiguation).
  PROCESS_CASE → Full claim record + document guidelines + follow-ups.
  POST_PROCESS → Conversation summary + caller info + resolved claim.
"""

import json


class ContextAssembler:
    """Builds phase-specific context for LLM prompt injection."""

    def __init__(self, data_store):
        self.data_store = data_store

    # -------------------------------------------------------- Public API

    def assemble(self, state, memory, emotion_tracker) -> dict:
        """Return a dict of context variables for the current phase."""
        assemblers = {
            "VERIFY_ID":      self._assemble_verify,
            "RESOLVE_INTENT": self._assemble_resolve,
            "PROCESS_CASE":   self._assemble_process,
            "POST_PROCESS":   self._assemble_post_process,
        }
        handler = assemblers.get(state.current_phase.value, lambda *_: {})
        ctx = handler(state, memory, emotion_tracker)

        # Emotion context is always available
        emotion_ctx = emotion_tracker.get_emotion_context()
        if emotion_ctx:
            ctx["emotion_context"] = emotion_ctx

        return ctx

    # ------------------------------------------------- Phase assemblers

    def _assemble_verify(self, _state, _memory, _emotion_tracker):
        """VERIFY_ID: NO claim data — only emotion context."""
        return {}

    def _assemble_resolve(self, state, memory, _emotion_tracker):
        """RESOLVE_INTENT: memory hints + caller's claim list."""
        claims = self.data_store.get_claims_for_party(
            state.matched_policyholder["party_id"]
        )
        return {
            "memory_hints": memory.get_memory_context(),
            "claims_list": self._format_claims_list(claims),
        }

    def _assemble_process(self, state, memory, _emotion_tracker):
        """PROCESS_CASE: full claim + guidelines + follow-ups."""
        claim = state.resolved_case
        return {
            "claim_data": json.dumps(claim, indent=2),
            "field_descriptions": json.dumps(
                self.data_store.get_claim_schema().get("field_descriptions", {}),
                indent=2,
            ),
            "document_guidance": self._get_document_guidance(claim),
            "followup_guidance": self._get_followup_guidance(claim, state),
        }

    def _assemble_post_process(self, state, memory, _emotion_tracker):
        """POST_PROCESS: conversation summary + caller + claim."""
        caller = f"Name: {state.matched_policyholder.get('name', 'Unknown')}"
        if state.matched_policyholder.get("email"):
            caller += f"\nEmail: {state.matched_policyholder['email']}"

        resolved = "None"
        if state.resolved_case:
            resolved = json.dumps(
                {
                    k: state.resolved_case.get(k)
                    for k in ("case_id", "case_type", "status", "summary",
                              "denial_reason", "documents_needed",
                              "appeal_deadline")
                    if state.resolved_case.get(k) is not None
                },
                indent=2,
            )

        return {
            "conversation_summary": memory.get_conversation_summary(),
            "caller_info": caller,
            "resolved_claim": resolved,
        }

    # ---------------------------------------------------------- Helpers

    @staticmethod
    def _format_claims_list(claims: list[dict]) -> str:
        if not claims:
            return "No claims found for this policyholder."
        lines = []
        for c in claims:
            line = (
                f"- {c['case_id']}: {c['case_type']} claim, "
                f"created {c['created_at']}, status: {c['status']}"
            )
            if c.get("summary"):
                line += f" — {c['summary']}"
            lines.append(line)
        return "\n".join(lines)

    def _get_document_guidance(self, claim: dict) -> str:
        parts = []

        # Case-type guidance
        case_type = claim.get("case_type", "")
        type_g = self.data_store.get_case_type_guidance(case_type)
        if type_g:
            parts.append(f"General {case_type} guidance: {type_g}")

        # Default guidance
        default_g = self.data_store.get_default_guidance()
        if default_g:
            parts.append(f"Default submission guidance: {default_g}")

        # Per-document guidance
        for doc in claim.get("documents_needed", []):
            for variant in self._doc_name_variants(doc):
                g = self.data_store.get_document_guidance(variant)
                a = self.data_store.get_document_alternative_guidance(variant)
                if g:
                    parts.append(f"\nGuidance for '{variant}':\n{g}")
                if a:
                    parts.append(f"Alternative options for '{variant}':\n{a}")

        return "\n".join(parts) if parts else "No specific document guidance."

    def _get_followup_guidance(self, claim: dict, state) -> str:
        all_guidance = self.data_store.get_followup_guidance()
        settings = self.data_store.get_followup_settings()

        user_text = " ".join(
            m["content"] for m in state.message_history if m["role"] == "user"
        ).lower()

        matched = []
        for topic in all_guidance:
            intent_match = state.resolved_intent in topic.get("intent_hints", [])
            kw_list = topic.get("match_any", [])
            keyword_match = any(kw in user_text for kw in kw_list)

            # Include if intent matches and either no keywords required or keywords match
            if intent_match and (not kw_list or keyword_match):
                text = self._resolve_template(
                    topic.get("en", ""), claim, settings
                )
                matched.append(f"[{topic['topic']}]: {text}")
            elif keyword_match and intent_match:
                text = self._resolve_template(
                    topic.get("en", ""), claim, settings
                )
                matched.append(f"[{topic['topic']}]: {text}")

        if not matched:
            fb = self.data_store.get_followup_fallback()
            return f"General guidance: {fb}"

        return "\n\n".join(matched)

    @staticmethod
    def _resolve_template(text: str, claim: dict, settings: dict) -> str:
        text = text.replace("{case_id}", claim.get("case_id", ""))
        docs = claim.get("documents_needed", [])
        text = text.replace(
            "{documents}", ", ".join(docs) if docs else "required documents"
        )
        avg = (
            settings
            .get("average_processing_time_after_submission", {})
            .get("en", "")
        )
        text = text.replace("{average_processing_time_after_submission}", avg)
        return text

    @staticmethod
    def _doc_name_variants(doc_name: str) -> list[str]:
        """Generate plausible key variants for document name lookup."""
        variants = [doc_name]
        prefixes = ["original ", "treating provider ", "supplemental "]
        for prefix in prefixes:
            variants.append(prefix + doc_name)
        return variants
