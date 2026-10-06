"""
Function-Calling Tool Schemas — phase-gated tool availability.

Each phase only exposes the tools the LLM is allowed to use. This is
a STRUCTURAL enforcement mechanism: the LLM literally cannot call
tools from other phases because they aren't in its schema.
"""

# ------------------------------------------------------------------ Tools

EXTRACT_IDENTITY_INFO = {
    "type": "function",
    "function": {
        "name": "extract_identity_info",
        "description": (
            "Extract any identity / PII fields the caller mentioned in their "
            "message.  Call this whenever the caller provides personal "
            "identifying information such as name, date of birth, phone, "
            "email, SSN last-4, or policy number."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Caller's full name if mentioned",
                },
                "dob": {
                    "type": "string",
                    "description": "Date of birth in YYYY-MM-DD format if mentioned",
                },
                "phone": {
                    "type": "string",
                    "description": "Phone number if mentioned",
                },
                "email": {
                    "type": "string",
                    "description": "Email address if mentioned",
                },
                "ssn_last4": {
                    "type": "string",
                    "description": "Last 4 digits of SSN or national ID if mentioned",
                },
                "policy_number": {
                    "type": "string",
                    "description": "Policy number (e.g. POL-XXXX) if mentioned",
                },
            },
        },
    },
}

STORE_MEMORY_HINT = {
    "type": "function",
    "function": {
        "name": "store_memory_hint",
        "description": (
            "Store information the caller mentioned that may be useful in a "
            "later workflow phase.  Call this whenever the caller mentions "
            "their reason for calling, references a specific claim, or shows "
            "an emotional state worth tracking."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "intent_hint": {
                    "type": "string",
                    "description": (
                        "What the caller seems to want, e.g. 'asking about "
                        "denied claim', 'wants claim status'"
                    ),
                },
                "case_hint": {
                    "type": "string",
                    "description": (
                        "Any reference to a specific claim, e.g. 'healthcare "
                        "claim from January', 'denied claim'"
                    ),
                },
                "emotion_signal": {
                    "type": "string",
                    "enum": [
                        "neutral",
                        "frustrated",
                        "angry",
                        "anxious",
                        "confused",
                        "refusing",
                    ],
                    "description": "Caller's emotional state detected from their message",
                },
            },
        },
    },
}

RESOLVE_INTENT = {
    "type": "function",
    "function": {
        "name": "resolve_intent",
        "description": (
            "Resolve the caller's intent and the specific claim they want to "
            "discuss.  Call this when you have enough information to determine "
            "the intent category and target claim ID."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "intent": {
                    "type": "string",
                    "enum": [
                        "status_inquiry",
                        "denial_question",
                        "document_submission",
                        "appeal_question",
                        "general_claim_question",
                        "next_steps",
                    ],
                    "description": "The resolved intent category",
                },
                "case_id": {
                    "type": "string",
                    "description": "The specific claim ID, e.g. CL-2048",
                },
                "confidence": {
                    "type": "string",
                    "enum": ["high", "medium", "low"],
                    "description": "Confidence in this resolution",
                },
            },
            "required": ["intent", "case_id", "confidence"],
        },
    },
}

GENERATE_EMAIL_SUMMARY = {
    "type": "function",
    "function": {
        "name": "generate_email_summary",
        "description": (
            "Generate a structured email summary of the conversation to send "
            "to the customer."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "subject": {"type": "string", "description": "Email subject line"},
                "greeting": {"type": "string", "description": "Email greeting"},
                "discussion_summary": {
                    "type": "string",
                    "description": "Summary of what was discussed during the call",
                },
                "claim_status": {
                    "type": "string",
                    "description": "Current status and outcome of the claim",
                },
                "next_steps": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of follow-up items and next steps",
                },
                "closing": {"type": "string", "description": "Closing message"},
            },
            "required": [
                "subject",
                "greeting",
                "discussion_summary",
                "claim_status",
                "next_steps",
                "closing",
            ],
        },
    },
}


# ---------------------------------------------- Phase-gated availability

def get_tools_for_phase(phase: str) -> list[dict]:
    """Return only the tools available in the given SOP phase.

    This is a key harness mechanism: the LLM only sees tools relevant to
    its current phase.  It structurally cannot call tools from other phases
    because they are not in its schema.
    """
    phase_tools = {
        "VERIFY_ID": [EXTRACT_IDENTITY_INFO, STORE_MEMORY_HINT],
        "RESOLVE_INTENT": [STORE_MEMORY_HINT, RESOLVE_INTENT],
        "PROCESS_CASE": [STORE_MEMORY_HINT],
        "POST_PROCESS": [STORE_MEMORY_HINT, GENERATE_EMAIL_SUMMARY],
    }
    return phase_tools.get(phase, [])
