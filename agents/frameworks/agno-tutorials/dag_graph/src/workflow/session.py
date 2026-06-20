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

def init_session_defaults(
    session_state: dict[str, Any],
) -> None:
    """
    Ensure all required session_state keys exist.

    Args:
        session_state: The workflow's mutable session dict.
        mode:          "triage" (conversation) | "pipeline" (doc processing).
    """
    defaults = PIPELINE_DEFAULTS 
    for k, v in defaults.items():
        session_state.setdefault(k, v)