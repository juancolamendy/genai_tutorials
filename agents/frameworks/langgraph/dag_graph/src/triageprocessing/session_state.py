"""Task-run session state for the triage harness (harness step 2).

One EngineGraph thread_id == one task run. Artifact fields hold PATHS, not
blobs (step 4). command_log / approvals are reducer-backed like audit_trail.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Dict, List, Literal, Optional

from src.engine.engine_session_state import EngineSessionState, new_engine_session_state

from .state_transitions import State

RunStatus = Literal[
    "queued",
    "running",
    "awaiting_approval",
    "published",
    "rejected",
    "failed",
]


class TriageState(EngineSessionState):
    """Sentry triage-and-fix task run."""

    run_id: str
    run_status: RunStatus
    trigger: Dict[str, Any]
    inputs: Dict[str, Any]

    artifacts: Dict[str, str]
    command_log: Annotated[List[Dict[str, Any]], operator.add]
    approvals: Annotated[List[Dict[str, Any]], operator.add]

    worktree_path: Optional[str]
    diff_path: Optional[str]
    report_path: Optional[str]
    published: bool
    approval_decision: Optional[Literal["approve", "reject"]]
    approval_by: Optional[str]
    approval_note: Optional[str]

    # Gate payload fields (system aemit_event merges these into state)
    by: Optional[str]
    note: Optional[str]
    approved: Optional[bool]

    artifacts_dir: str


def new_triage_session_state(
    *,
    run_id: str = "",
    artifacts_dir: str = ".triage_artifacts",
) -> TriageState:
    engine = new_engine_session_state()
    return {
        **engine,
        "current_state": State.INIT.value,
        "proposed_next": State.INTAKE.value,
        "run_id": run_id,
        "run_status": "queued",
        "trigger": {},
        "inputs": {},
        "artifacts": {},
        "command_log": [],
        "approvals": [],
        "worktree_path": None,
        "diff_path": None,
        "report_path": None,
        "published": False,
        "approval_decision": None,
        "approval_by": None,
        "approval_note": None,
        "by": None,
        "note": None,
        "approved": None,
        "artifacts_dir": artifacts_dir,
    }
