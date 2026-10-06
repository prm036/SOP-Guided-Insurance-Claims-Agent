"""
SOP Harness — the master orchestrator.

This is the core of the agentic architecture.  It wires together every
component and implements the main message-processing pipeline:

    User message
      → Scope Guard         (reject off-topic)
      → Phase Gate           (determine available tools / data)
      → Context Assembly     (build phase-specific LLM context)
      → LLM Invocation       (generate response + function calls)
      → Tool Processing      (verify identity, store memory, resolve intent …)
      → Gate Check           (advance phase if conditions met)
      → Response Assembly    (return response + updated state)

Design principle:
  The harness controls WHAT the LLM is allowed to do.
  The LLM controls HOW it talks.
"""

import json
import logging

from agent.state_machine import SessionState, StateMachine, Phase
from agent.identity_verifier import IdentityVerifier
from agent.memory import CrossPhaseMemory
from agent.scope_guard import ScopeGuard
from agent.emotion_tracker import EmotionTracker
from agent.context_assembler import ContextAssembler
from agent.email_generator import EmailGenerator
from llm.client import LLMClient
from llm.prompts import build_system_prompt
from llm.tools import get_tools_for_phase
from data.store import DataStore

logger = logging.getLogger(__name__)


class SOPHarness:
    """The SOP-guided conversational-agent harness."""

    def __init__(self, data_store: DataStore, llm_client: LLMClient):
        self.data_store = data_store
        self.llm_client = llm_client
        self.state_machine = StateMachine()
        self.identity_verifier = IdentityVerifier(data_store)
        self.scope_guard = ScopeGuard(llm_client)
        self.email_generator = EmailGenerator()

        self.sessions: dict[str, SessionState] = {}

    # ============================================================ API

    def create_session(self) -> SessionState:
        state = SessionState()
        self.sessions[state.session_id] = state
        return state

    def get_session(self, session_id: str) -> SessionState | None:
        return self.sessions.get(session_id)

    def process_message(self, session_id: str, user_message: str) -> dict:
        """Process one user message through the full SOP pipeline."""

        state = self.sessions.get(session_id)
        if not state:
            return {"error": "Session not found"}

        # Bind per-session helpers
        memory = CrossPhaseMemory(state)
        emotion_tracker = EmotionTracker(state)
        ctx_assembler = ContextAssembler(self.data_store)

        # ── 1. Scope Guard ──────────────────────────────────────────
        scope = self.scope_guard.classify(user_message, state)
        if not scope["in_scope"]:
            rejection = self.scope_guard.get_rejection_message(state)
            state.message_history.append({"role": "user", "content": user_message})
            state.message_history.append({"role": "assistant", "content": rejection})
            return self._build_response(rejection, state, out_of_scope=True)

        # ── 2. Record user message ──────────────────────────────────
        state.message_history.append({"role": "user", "content": user_message})

        # ── 3. Handle POST_PROCESS email decision ──────────────────
        if state.current_phase == Phase.POST_PROCESS:
            decision = self._detect_email_decision(user_message)
            if decision == "skip":
                state.email_decision = "skip"
                memory.add_conversation_summary_point("Customer declined email summary")
                self.state_machine.check_and_advance(state)
                farewell = (
                    "No problem at all! Thank you for calling, and I hope I was "
                    "able to help. If you have any other questions in the future, "
                    "don't hesitate to reach out. Have a great day!"
                )
                state.message_history.append({"role": "assistant", "content": farewell})
                return self._build_response(farewell, state)

        # ── 4. Assemble context + prompt ────────────────────────────
        context = ctx_assembler.assemble(state, memory, emotion_tracker)
        system_prompt = build_system_prompt(state.current_phase.value, **context)
        tools = get_tools_for_phase(state.current_phase.value)

        # ── 5. LLM invocation ──────────────────────────────────────
        llm_result = self.llm_client.chat(
            system_prompt=system_prompt,
            messages=state.message_history,
            tools=tools,
        )

        response_text = llm_result["content"]
        tool_calls = llm_result.get("tool_calls", [])

        # ── 6. Process tool calls ──────────────────────────────────
        for tc in tool_calls:
            self._dispatch_tool(tc, state, memory)

        # ── 7. If tool-only response, do a follow-up LLM call ─────
        if tool_calls and not response_text:
            response_text = self._follow_up_after_tools(
                tool_calls, state, memory, emotion_tracker, ctx_assembler
            )

        # ── 8. Check gates & advance phases ────────────────────────
        advanced, reason = self.state_machine.check_and_advance(state)
        if advanced:
            memory.add_conversation_summary_point(reason)
            logger.info("Phase advanced: %s — %s", state.current_phase.value, reason)

            # Fast-forward through RESOLVE_INTENT if memory has enough
            if state.current_phase == Phase.RESOLVE_INTENT:
                response_text = self._auto_resolve_intent(
                    state, memory, emotion_tracker, ctx_assembler, response_text
                )

        # ── 9. PROCESS_CASE → POST_PROCESS wrap-up detection ──────
        if state.current_phase == Phase.PROCESS_CASE:
            if self._detect_wrap_up(user_message):
                self.state_machine.force_advance_to_post_process(state)
                memory.add_conversation_summary_point("Case processing completed")
                # Append the email offer
                offer = self._get_post_process_offer(
                    state, memory, emotion_tracker, ctx_assembler
                )
                if offer:
                    response_text = (
                        f"{response_text}\n\n{offer}" if response_text else offer
                    )

        # ── 10. POST_PROCESS email generation ──────────────────────
        if state.current_phase == Phase.POST_PROCESS and state.email_decision == "send":
            self.state_machine.check_and_advance(state)

        # ── 11. Record assistant response ──────────────────────────
        if response_text:
            state.message_history.append({"role": "assistant", "content": response_text})

        return self._build_response(response_text, state)

    # ======================================================= Internals

    def _dispatch_tool(self, tc: dict, state: SessionState, memory: CrossPhaseMemory):
        """Route a single tool call to the right handler."""
        name = tc["name"]
        args = tc["arguments"]

        if name == "extract_identity_info":
            self._handle_identity_extraction(state, args)
        elif name == "store_memory_hint":
            memory.store_hint(args)
        elif name == "resolve_intent":
            self._handle_intent_resolution(state, args)
        elif name == "generate_email_summary":
            self._handle_email_generation(state, args)

    def _handle_identity_extraction(self, state: SessionState, extracted: dict):
        """Accumulate PII and run server-side verification."""
        for field, value in extracted.items():
            if value and isinstance(value, str) and value.strip():
                state.identity_collected[field] = value.strip()

        result = self.identity_verifier.verify(state.identity_collected)
        state.identity_match_count = result["fields_verified"]

        if result["matched"]:
            state.verified = True
            state.matched_policyholder = result["policyholder"]
            logger.info(
                "Identity VERIFIED (%d fields): %s",
                result["fields_verified"],
                result["matched_fields"],
            )

        state.verification_attempts += 1

    def _handle_intent_resolution(self, state: SessionState, resolved: dict):
        """Validate the resolved intent + case against the data store."""
        intent = resolved.get("intent")
        case_id = resolved.get("case_id")
        if not intent or not case_id:
            return

        claim = self.data_store.get_claim_by_id(case_id)
        if claim and state.matched_policyholder and \
                claim["party_id"] == state.matched_policyholder["party_id"]:
            state.resolved_intent = intent
            state.resolved_case_id = case_id
            state.resolved_case = claim
            logger.info("Intent resolved: %s → %s", intent, case_id)

    def _handle_email_generation(self, state: SessionState, email_data: dict):
        state.email_content = self.email_generator.format_email(email_data)
        state.email_decision = "send"

    # ────────────────────────────────────────────────── Follow-up calls

    def _follow_up_after_tools(
        self, tool_calls, state, memory, emotion_tracker, ctx_assembler
    ) -> str:
        """When the LLM returns only tool calls, make a follow-up call
        that includes tool results so it can generate a natural reply."""

        tool_messages = []
        for tc in tool_calls:
            tool_messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": json.dumps(tc["arguments"]),
                    },
                }],
            })
            tool_messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": self._tool_result(tc, state),
            })

        # Rebuild context (phase may have changed after tool processing)
        context = ctx_assembler.assemble(state, memory, emotion_tracker)
        system_prompt = build_system_prompt(state.current_phase.value, **context)
        tools = get_tools_for_phase(state.current_phase.value)

        followup = self.llm_client.chat(
            system_prompt=system_prompt,
            messages=state.message_history + tool_messages,
            tools=tools,
        )

        # Process any secondary tool calls
        for tc in followup.get("tool_calls", []):
            self._dispatch_tool(tc, state, memory)

        return followup.get("content", "")

    def _tool_result(self, tc: dict, state: SessionState) -> str:
        """Synthetic tool-result message for the LLM."""
        name = tc["name"]

        if name == "extract_identity_info":
            if state.verified:
                return json.dumps({
                    "status": "verified",
                    "message": (
                        "Identity verified. You may now inform the caller "
                        "and proceed to help with their inquiry."
                    ),
                })
            return json.dumps({
                "status": "pending",
                "message": (
                    "Some information received. Still need more identifying "
                    "details to complete verification."
                ),
            })

        if name == "store_memory_hint":
            return json.dumps({"status": "stored"})

        if name == "resolve_intent":
            if state.resolved_case:
                return json.dumps({
                    "status": "resolved",
                    "message": (
                        f"Intent resolved: {state.resolved_intent} "
                        f"for claim {state.resolved_case_id}."
                    ),
                })
            return json.dumps({
                "status": "failed",
                "message": "Could not validate. Ask the caller to clarify.",
            })

        if name == "generate_email_summary":
            return json.dumps({
                "status": "generated",
                "message": "Email summary generated.",
            })

        return json.dumps({"status": "ok"})

    # ──────────────────────────────────────────── Auto intent resolution

    def _auto_resolve_intent(
        self, state, memory, emotion_tracker, ctx_assembler, existing_text
    ) -> str:
        """When entering RESOLVE_INTENT, try to resolve automatically
        using cross-phase memory hints so we don't re-ask the caller."""

        context = ctx_assembler.assemble(state, memory, emotion_tracker)
        system_prompt = build_system_prompt(state.current_phase.value, **context)
        tools = get_tools_for_phase(state.current_phase.value)

        result = self.llm_client.chat(
            system_prompt=system_prompt,
            messages=state.message_history,
            tools=tools,
        )

        for tc in result.get("tool_calls", []):
            self._dispatch_tool(tc, state, memory)

        # Check if we can skip to PROCESS_CASE
        advanced, reason = self.state_machine.check_and_advance(state)
        if advanced:
            memory.add_conversation_summary_point(reason)

            # Now get a PROCESS_CASE response
            if state.current_phase == Phase.PROCESS_CASE:
                return self._get_process_case_response(
                    state, memory, emotion_tracker, ctx_assembler,
                    result.get("tool_calls", []),
                )

        return result.get("content") or existing_text

    def _get_process_case_response(
        self, state, memory, emotion_tracker, ctx_assembler, prev_tool_calls
    ) -> str:
        """Generate the first PROCESS_CASE response after auto-resolve."""

        # Build tool result messages from the resolve phase
        tool_messages = []
        for tc in prev_tool_calls:
            tool_messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": json.dumps(tc["arguments"]),
                    },
                }],
            })
            tool_messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": json.dumps({"status": "resolved", "phase_advanced": True}),
            })

        context = ctx_assembler.assemble(state, memory, emotion_tracker)
        system_prompt = build_system_prompt(state.current_phase.value, **context)
        tools = get_tools_for_phase(state.current_phase.value)

        result = self.llm_client.chat(
            system_prompt=system_prompt,
            messages=state.message_history + tool_messages,
            tools=tools,
        )

        for tc in result.get("tool_calls", []):
            self._dispatch_tool(tc, state, memory)

        return result.get("content", "")

    def _get_post_process_offer(
        self, state, memory, emotion_tracker, ctx_assembler
    ) -> str:
        """Generate the POST_PROCESS email-offer message."""
        context = ctx_assembler.assemble(state, memory, emotion_tracker)
        system_prompt = build_system_prompt(state.current_phase.value, **context)
        tools = get_tools_for_phase(state.current_phase.value)

        result = self.llm_client.chat(
            system_prompt=system_prompt,
            messages=state.message_history,
            tools=tools,
        )
        return result.get("content", "")

    # ──────────────────────────────────────────────── Signal detectors

    @staticmethod
    def _detect_wrap_up(user_message: str) -> bool:
        """Detect conversational wrap-up signals in PROCESS_CASE."""
        msg = user_message.lower()
        signals = [
            "that's all", "that is all", "nothing else",
            "no more questions", "i think that's it", "i'm good",
            "thank you", "thanks, that", "no other questions",
            "that answers my question", "that's everything",
            "i understand now", "got it, thanks",
            "no, that's it", "nope, that's all",
            "that's what i needed", "that helps",
        ]
        return any(s in msg for s in signals)

    @staticmethod
    def _detect_email_decision(user_message: str) -> str | None:
        """Detect accept / decline of the email summary offer."""
        msg = user_message.lower()

        decline = [
            "no thanks", "no thank you", "skip", "don't need",
            "not necessary", "nah", "nope", "i'm good", "pass",
            "don't send", "no need", "no, thanks", "no,",
        ]
        accept = [
            "yes", "sure", "please", "send it", "send me",
            "that would be great", "go ahead", "okay", "yeah",
            "yep", "absolutely", "i'd like that", "please do",
        ]

        for s in decline:
            if s in msg:
                return "skip"
        for s in accept:
            if s in msg:
                return "send"
        return None

    # ──────────────────────────────────────────────── Response builder

    @staticmethod
    def _build_response(text: str, state: SessionState, **kwargs) -> dict:
        resp = {
            "response": text,
            "phase": state.current_phase.value,
            "state": state.to_dict(),
        }
        if state.email_content:
            resp["email_preview"] = state.email_content
        resp.update(kwargs)
        return resp
