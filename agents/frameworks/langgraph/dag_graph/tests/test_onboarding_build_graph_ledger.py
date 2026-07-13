"""build_graph must wire EventLedger next to sessions_dir (not CWD .event_ledger)."""

from uuid import uuid4

from src.onboarding.graph import build_graph


def test_build_graph_wires_ledger_colocated_with_sessions_dir():
    sessions_dir = f"/tmp/test_onboarding_ledger_wire_{uuid4()}"
    graph = build_graph(sessions_dir=sessions_dir)

    assert graph._ledger is not None
    assert str(graph._ledger.ledger_dir) == f"{sessions_dir}_ledger"
