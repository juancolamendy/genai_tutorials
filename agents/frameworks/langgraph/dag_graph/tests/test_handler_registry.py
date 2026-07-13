"""Tests for handler registry and @handler decorator."""


from src.engine.handler_registry import (
    clear_metadata,
    does_state_wait_for_input,
    get_handler_metadata,
    handler,
    handler_metadata_map,
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

    assert len(handler_metadata_map) == 3
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

    assert len(handler_metadata_map) == 2

    # Clear
    clear_metadata()
    assert len(handler_metadata_map) == 0
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


def test_handler_decorator_wait_kind_and_expected_events_default():
    """No-kwarg @handler call sites (e.g. docprocessing/handlers.py) must keep
    working unchanged: wait_kind defaults to "either", expected_events to None."""
    clear_metadata()

    @handler(state="legacy_state", waits_for_input=True, description="Legacy call site")
    def legacy_handler(state):
        return state

    meta = get_handler_metadata("legacy_state")
    assert meta is not None
    assert meta.wait_kind == "either"
    assert meta.expected_events is None


def test_handler_decorator_registers_wait_kind_system_event():
    """@handler can declare wait_kind="system_event" with expected_events."""
    clear_metadata()

    @handler(
        state="await_thing",
        waits_for_input=True,
        wait_kind="system_event",
        expected_events=["thing_happened", "timeout_escalation"],
    )
    def await_handler(state):
        return state

    meta = get_handler_metadata("await_thing")
    assert meta is not None
    assert meta.wait_kind == "system_event"
    assert meta.expected_events == ["thing_happened", "timeout_escalation"]


def test_handler_decorator_registers_wait_kind_human():
    """@handler can declare wait_kind="human" explicitly."""
    clear_metadata()

    @handler(state="collect", waits_for_input=True, wait_kind="human")
    def collect_handler(state):
        return state

    meta = get_handler_metadata("collect")
    assert meta is not None
    assert meta.wait_kind == "human"
    assert meta.expected_events is None
