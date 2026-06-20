"""Tests to verify Phase 1 project setup."""

import sys
from pathlib import Path


def test_imports_resolve():
    """Test that langgraph and pytest can be imported."""
    try:
        import langgraph
        import pytest
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")


def test_directory_structure_exists():
    """Test that required directories exist."""
    root = Path(__file__).parent.parent
    assert (root / "src").exists(), "src/ directory missing"
    assert (root / "src" / "engine").exists(), "src/engine/ directory missing"
    assert (root / "src" / "pipeline").exists(), "src/pipeline/ directory missing"
    assert (root / "tests").exists(), "tests/ directory missing"


def test_pytest_ini_exists():
    """Test that pytest.ini configuration exists."""
    root = Path(__file__).parent.parent
    assert (root / "pytest.ini").exists(), "pytest.ini missing"


def test_pyproject_toml_exists():
    """Test that pyproject.toml exists."""
    root = Path(__file__).parent.parent
    assert (root / "pyproject.toml").exists(), "pyproject.toml missing"


def test_conftest_fixtures_available(sample_state, mock_validate_agent):
    """Test that conftest fixtures are available."""
    assert sample_state["document_id"] == "TEST-001"
    assert mock_validate_agent.run("test") == "mock_result"
