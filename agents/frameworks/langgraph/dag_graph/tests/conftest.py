"""Pytest fixtures for LangGraph state machine tests."""

import pytest


class MockLLMAgent:
    """Mock LLM agent for testing without real API calls."""

    def run(self, prompt: str) -> str:
        """Return fixed result for testing."""
        return "mock_result"


@pytest.fixture
def sample_state():
    """Fixture: sample PipelineState for testing."""
    return {
        "current_state": "init",
        "proposed_next": "fetch",
        "retry_count": 0,
        "error_message": None,
        "error_type": None,
        "audit_trail": ["init"],
        "fallback_depth": 0,
        "started_at": 0.0,
        "node_timeout_seconds": 60,
        "document_id": "TEST-001",
        "raw_data": None,
        "validated_data": None,
        "enriched_data": None,
    }


@pytest.fixture
def mock_validate_agent():
    """Fixture: mock VALIDATE_AGENT for testing."""
    return MockLLMAgent()


@pytest.fixture
def mock_enrich_agent():
    """Fixture: mock ENRICH_AGENT for testing."""
    return MockLLMAgent()


@pytest.fixture
def mock_review_agent():
    """Fixture: mock REVIEW_AGENT for testing."""
    return MockLLMAgent()
