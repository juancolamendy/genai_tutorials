"""Tests for handler registry and @handler decorator."""

import pytest

from src.engine.handler_registry import (
    handler,
    get_handler_metadata,
    does_state_wait_for_input,
    HANDLER_MAP_METADATA,
    clear_metadata,
)


def test_handler_decorator_registers_metadata():
    """Test that @handler decorator registers metadata."""
    clear_metadata()

    @handler(state="test_state", waits_for_input=True, description="Test handler")
    def dummy_handler(state):
        return state

    meta = get_handler_metadata("test_state")
    assert meta is not None
    assert meta.state == "test_state"
    assert meta.waits_for_input is True
    assert meta.description == "Test handler"


def test_handler_decorator_default_values():
    """Test that @handler uses default values for optional params."""
    clear_metadata()

    @handler(state="test_state")
    def dummy_handler(state):
        return state

    meta = get_handler_metadata("test_state")
    assert meta is not None
    assert meta.state == "test_state"
    assert meta.waits_for_input is False
    assert meta.description is None


def test_does_state_wait_for_input_true():
    """Test does_state_wait_for_input returns True for blocking states."""
    clear_metadata()

    @handler(state="blocking", waits_for_input=True)
    def blocking_handler(state):
        return state

    assert does_state_wait_for_input("blocking") is True


def test_does_state_wait_for_input_false():
    """Test does_state_wait_for_input returns False for non-blocking states."""
    clear_metadata()

    @handler(state="non_blocking", waits_for_input=False)
    def non_blocking_handler(state):
        return state

    assert does_state_wait_for_input("non_blocking") is False


def test_does_state_wait_for_input_unknown():
    """Test that unregistered states return False for waits_for_input."""
    clear_metadata()
    assert does_state_wait_for_input("nonexistent_state") is False


def test_get_handler_metadata_not_found():
    """Test that get_handler_metadata returns None for unregistered state."""
    clear_metadata()
    assert get_handler_metadata("unknown_state") is None


def test_multiple_handlers_registered():
    """Test that multiple handlers can be registered."""
    clear_metadata()

    @handler(state="state1", waits_for_input=False)
    def handler1(state):
        return state

    @handler(state="state2", waits_for_input=True)
    def handler2(state):
        return state

    @handler(state="state3", waits_for_input=False)
    def handler3(state):
        return state

    assert len(HANDLER_MAP_METADATA) == 3
    assert does_state_wait_for_input("state1") is False
    assert does_state_wait_for_input("state2") is True
    assert does_state_wait_for_input("state3") is False


def test_clear_metadata():
    """Test that clear_metadata clears the registry."""
    # First populate
    clear_metadata()

    @handler(state="test1", waits_for_input=False)
    def h1(state):
        return state

    @handler(state="test2", waits_for_input=True)
    def h2(state):
        return state

    assert len(HANDLER_MAP_METADATA) == 2

    # Clear
    clear_metadata()
    assert len(HANDLER_MAP_METADATA) == 0
    assert get_handler_metadata("test1") is None
    assert get_handler_metadata("test2") is None


def test_handler_decorator_returns_original_function():
    """Test that @handler decorator returns the original function unchanged."""
    clear_metadata()

    def my_handler(state):
        return state

    decorated = handler(state="test", waits_for_input=False)(my_handler)
    assert decorated is my_handler
    assert decorated({"test": "data"}) == {"test": "data"}
