"""
Tests for engine/handler_registry.py — @handler decorator and metadata.
"""



class TestHandlerMetadata:
    """Tests for HandlerMetadata dataclass."""

    def test_handler_metadata_has_state(self):
        """HandlerMetadata has state field."""
        from engine.handler_registry import HandlerMetadata
        meta = HandlerMetadata(state="validate", waits_for_input=False)
        assert meta.state == "validate"

    def test_handler_metadata_has_waits_for_input(self):
        """HandlerMetadata has waits_for_input field (default False)."""
        from engine.handler_registry import HandlerMetadata
        meta = HandlerMetadata(state="validate")
        assert meta.waits_for_input is False

    def test_handler_metadata_has_description(self):
        """HandlerMetadata has description field (optional)."""
        from engine.handler_registry import HandlerMetadata
        meta = HandlerMetadata(state="validate", description="Validate doc")
        assert meta.description == "Validate doc"

    def test_handler_metadata_description_defaults_none(self):
        """HandlerMetadata description defaults to None."""
        from engine.handler_registry import HandlerMetadata
        meta = HandlerMetadata(state="validate")
        assert meta.description is None

    def test_handler_metadata_waits_for_input_true(self):
        """HandlerMetadata supports waits_for_input=True."""
        from engine.handler_registry import HandlerMetadata
        meta = HandlerMetadata(state="wait_docs", waits_for_input=True)
        assert meta.waits_for_input is True


class TestHandlerDecorator:
    """Tests for @handler decorator."""

    def test_handler_decorator_registers_metadata(self):
        """@handler decorator registers state and metadata."""
        from engine.handler_registry import HANDLER_MAP_METADATA, handler

        # Clear registry
        HANDLER_MAP_METADATA.clear()

        @handler(state="test_state", waits_for_input=False, description="Test")
        def test_handler(state):
            return state

        assert "test_state" in HANDLER_MAP_METADATA
        meta = HANDLER_MAP_METADATA["test_state"]
        assert meta.state == "test_state"
        assert meta.waits_for_input is False
        assert meta.description == "Test"

    def test_handler_decorator_returns_function(self):
        """@handler decorator returns the original function."""
        from engine.handler_registry import handler

        @handler(state="test", waits_for_input=False)
        def my_func(state):
            return "result"

        assert callable(my_func)
        assert my_func({}) == "result"

    def test_handler_decorator_multiple_handlers(self):
        """Multiple @handler decorators register independently."""
        from engine.handler_registry import HANDLER_MAP_METADATA, handler

        HANDLER_MAP_METADATA.clear()

        @handler(state="fetch", waits_for_input=False)
        def handle_fetch(state):
            return state

        @handler(state="validate", waits_for_input=False)
        def handle_validate(state):
            return state

        @handler(state="wait", waits_for_input=True)
        def handle_wait(state):
            return state

        assert len(HANDLER_MAP_METADATA) == 3
        assert HANDLER_MAP_METADATA["fetch"].waits_for_input is False
        assert HANDLER_MAP_METADATA["validate"].waits_for_input is False
        assert HANDLER_MAP_METADATA["wait"].waits_for_input is True

    def test_handler_decorator_overwrites_previous(self):
        """Registering same state twice overwrites."""
        from engine.handler_registry import HANDLER_MAP_METADATA, handler

        HANDLER_MAP_METADATA.clear()

        @handler(state="test", waits_for_input=False, description="First")
        def handler1(state):
            return state

        @handler(state="test", waits_for_input=True, description="Second")
        def handler2(state):
            return state

        # Should have latest registration
        assert HANDLER_MAP_METADATA["test"].waits_for_input is True
        assert HANDLER_MAP_METADATA["test"].description == "Second"


class TestGetHandlerMetadata:
    """Tests for get_handler_metadata() function."""

    def test_get_handler_metadata_existing_state(self):
        """get_handler_metadata() returns metadata for registered state."""
        from engine.handler_registry import HANDLER_MAP_METADATA, handler, get_handler_metadata

        HANDLER_MAP_METADATA.clear()

        @handler(state="test", waits_for_input=True, description="Test")
        def test_handler(state):
            return state

        meta = get_handler_metadata("test")
        assert meta is not None
        assert meta.state == "test"
        assert meta.waits_for_input is True

    def test_get_handler_metadata_missing_state(self):
        """get_handler_metadata() returns None for unregistered state."""
        from engine.handler_registry import HANDLER_MAP_METADATA, get_handler_metadata

        HANDLER_MAP_METADATA.clear()

        meta = get_handler_metadata("nonexistent")
        assert meta is None

    def test_get_handler_metadata_empty_registry(self):
        """get_handler_metadata() returns None from empty registry."""
        from engine.handler_registry import HANDLER_MAP_METADATA, get_handler_metadata

        HANDLER_MAP_METADATA.clear()

        meta = get_handler_metadata("any")
        assert meta is None


class TestDoesStateWaitForInput:
    """Tests for does_state_wait_for_input() function."""

    def test_does_state_wait_for_input_true(self):
        """does_state_wait_for_input() returns True for waiting states."""
        from engine.handler_registry import HANDLER_MAP_METADATA, handler, does_state_wait_for_input

        HANDLER_MAP_METADATA.clear()

        @handler(state="wait", waits_for_input=True)
        def handle_wait(state):
            return state

        assert does_state_wait_for_input("wait") is True

    def test_does_state_wait_for_input_false(self):
        """does_state_wait_for_input() returns False for non-waiting states."""
        from engine.handler_registry import HANDLER_MAP_METADATA, handler, does_state_wait_for_input

        HANDLER_MAP_METADATA.clear()

        @handler(state="fetch", waits_for_input=False)
        def handle_fetch(state):
            return state

        assert does_state_wait_for_input("fetch") is False

    def test_does_state_wait_for_input_unregistered(self):
        """does_state_wait_for_input() returns False for unregistered states."""
        from engine.handler_registry import HANDLER_MAP_METADATA, does_state_wait_for_input

        HANDLER_MAP_METADATA.clear()

        assert does_state_wait_for_input("unknown") is False
