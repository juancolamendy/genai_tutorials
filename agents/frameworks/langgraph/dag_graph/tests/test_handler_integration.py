"""Tests for handler decorator integration with workflow."""


from src.engine.handler_registry import (
    clear_metadata,
    does_state_wait_for_input,
    get_handler_metadata,
    handler,
)


def test_handlers_registered_with_decorators():
    """Test that handlers are properly registered via decorators."""
    clear_metadata()

    @handler(state="fetch", waits_for_input=False, description="Fetch document")
    def handle_fetch(state):
        return state

    @handler(state="validate", waits_for_input=True, description="Validate document")
    def handle_validate(state):
        return state

    # Verify both are registered
    assert get_handler_metadata("fetch") is not None
    assert get_handler_metadata("validate") is not None
    assert does_state_wait_for_input("fetch") is False
    assert does_state_wait_for_input("validate") is True


def test_handler_processes_state_correctly():
    """Test that decorated handler functions still work as expected."""
    clear_metadata()

    @handler(state="test", waits_for_input=False)
    def handle_test(state):
        state["current_state"] = "test"
        state["processed"] = True
        return state

    # Handler should work normally despite decorator
    input_state = {"current_state": "init", "processed": False}
    output_state = handle_test(input_state)

    assert output_state["current_state"] == "test"
    assert output_state["processed"] is True


def test_blocking_handler_metadata():
    """Test that blocking handlers are correctly identified."""
    clear_metadata()

    @handler(state="human_review", waits_for_input=True)
    def handle_human_review(state):
        return state

    # This state should be blocking
    assert does_state_wait_for_input("human_review") is True
    meta = get_handler_metadata("human_review")
    assert meta.waits_for_input is True


def test_non_blocking_handler_metadata():
    """Test that non-blocking handlers are correctly identified."""
    clear_metadata()

    @handler(state="enrich", waits_for_input=False)
    def handle_enrich(state):
        return state

    # This state should not be blocking
    assert does_state_wait_for_input("enrich") is False
    meta = get_handler_metadata("enrich")
    assert meta.waits_for_input is False


def test_handler_description_metadata():
    """Test that handler descriptions are captured."""
    clear_metadata()

    @handler(state="store", description="Persist enriched document")
    def handle_store(state):
        return state

    meta = get_handler_metadata("store")
    assert meta.description == "Persist enriched document"


def test_multiple_handlers_independent_state():
    """Test that multiple handlers don't interfere with each other."""
    clear_metadata()

    @handler(state="state1", waits_for_input=False)
    def h1(state):
        state["h1_ran"] = True
        return state

    @handler(state="state2", waits_for_input=True)
    def h2(state):
        state["h2_ran"] = True
        return state

    # Run handlers independently
    s1 = h1({"h1_ran": False})
    s2 = h2({"h2_ran": False})

    assert s1["h1_ran"] is True
    assert s2["h2_ran"] is True
    assert "h1_ran" not in s2  # h1 didn't run on s2
    assert "h2_ran" not in s1  # h2 didn't run on s1


def test_handler_state_immutability():
    """Test that handlers properly return new state dicts."""
    clear_metadata()

    @handler(state="test", waits_for_input=False)
    def handle_test(state):
        # Return updated dict with spread
        return {**state, "updated": True}

    input_state = {"current_state": "init", "updated": False}
    output_state = handle_test(input_state)

    # Input should be unchanged
    assert input_state["updated"] is False
    # Output should be updated
    assert output_state["updated"] is True


def test_handler_with_no_description():
    """Test that handler works without description."""
    clear_metadata()

    @handler(state="simple")
    def handle_simple(state):
        return state

    meta = get_handler_metadata("simple")
    assert meta.description is None


def test_handler_metadata_persists():
    """Test that metadata persists across calls."""
    clear_metadata()

    @handler(state="persist", waits_for_input=True, description="Test persist")
    def handle_persist(state):
        return state

    # Call multiple times
    handle_persist({})
    handle_persist({})
    handle_persist({})

    # Metadata should still be there
    meta = get_handler_metadata("persist")
    assert meta.waits_for_input is True
    assert meta.description == "Test persist"
