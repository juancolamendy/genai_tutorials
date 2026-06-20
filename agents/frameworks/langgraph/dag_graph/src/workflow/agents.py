"""Mock LLM agents for testing and development."""


class MockLLMAgent:
    """Mock LLM agent that returns fixed results for testing."""

    def run(self, prompt: str) -> str:
        """Return fixed result for testing without real API calls."""
        return "mock_result"


# Mock agents for document processing
VALIDATE_AGENT = MockLLMAgent()
ENRICH_AGENT = MockLLMAgent()
REVIEW_AGENT = MockLLMAgent()
