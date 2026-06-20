from typing import Any

PIPELINE_DEFAULTS: dict[str, Any] = {
    "current_state":  "init",
    "proposed_next":  "fetch",
    "guardrail_ok":   True,
    "retry_count":    0,
    "error_message":  None,
    "audit_trail":    [],
    "document_id":    "",
    "raw_data":       None,
    "validated_data": None,
    "enriched_data":  None,
    "pipeline_runs":  [],
}

_TRIAGE_DEFAULTS: dict[str, Any] = {
    "turns":         [],
    "turn_count":    0,
    "resolved":      False,
    "last_intent":   None,
    "last_category": None,
    "entities":      {},
}

def init_session_defaults(
    session_state: dict[str, Any],
    mode: str = "triage",
) -> None:
    """
    Ensure all required session_state keys exist.

    Args:
        session_state: The workflow's mutable session dict.
        mode:          "triage" (conversation) | "pipeline" (doc processing).
    """
    defaults = PIPELINE_DEFAULTS if mode == "pipeline" else _TRIAGE_DEFAULTS
    for k, v in defaults.items():
        session_state.setdefault(k, v)