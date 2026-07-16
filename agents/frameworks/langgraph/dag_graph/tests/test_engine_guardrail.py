"""Tests for engine guardrail factories."""

from src.engine.guardrail import make_handler_status_guardrail


def test_make_handler_status_guardrail_passes_when_ok():
    check = make_handler_status_guardrail("error")
    result = check({"handler_status": "ok"})
    assert result.passed is True


def test_make_handler_status_guardrail_fails_when_error():
    check = make_handler_status_guardrail("error")
    result = check({"handler_status": "error", "error_message": "boom"})
    assert result.passed is False
    assert result.fallback == "error"
    assert result.reason == "boom"


def test_make_handler_status_guardrail_default_reason():
    check = make_handler_status_guardrail("error")
    result = check({"handler_status": "error"})
    assert result.reason == "handler failed"
