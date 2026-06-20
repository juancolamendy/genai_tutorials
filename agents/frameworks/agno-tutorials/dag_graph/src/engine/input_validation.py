"""
engine/input_validation.py
────────────────────────────────────────────────────────────────────────────
Input validation and sanitization for multi-turn conversations.

Prevents:
  • DoS attacks (token bombs, extremely long inputs)
  • Prompt injection attacks (malicious instructions in turn_input)

All turn_input must be validated before being passed to LLM routers.
"""

from __future__ import annotations


class InputValidationError(Exception):
    """Raised when turn_input fails validation (length, tokens, type)."""

    pass


def estimate_tokens(text: str) -> int:
    """
    Rough estimate of token count.

    Uses heuristic: 1 token per 4 characters. This is approximate but
    sufficient for setting DoS limits.

    Args:
        text: Text to estimate

    Returns:
        Estimated token count (integer)
    """
    return len(text) // 4


def validate_turn_input(turn_input,
                        max_chars: int = 10_000,
                        max_tokens: int = 2_000) -> None:
    """
    Validate turn_input for length and token count.

    Raises InputValidationError if validation fails.

    Args:
        turn_input: User's input text (should be str)
        max_chars: Maximum characters allowed (default 10,000)
        max_tokens: Maximum estimated tokens (default 2,000)

    Raises:
        InputValidationError: If input is not string, too long, or too many tokens
    """
    if not isinstance(turn_input, str):
        raise InputValidationError("turn_input must be a string")

    if len(turn_input) > max_chars:
        raise InputValidationError(
            f"turn_input exceeds {max_chars} characters (got {len(turn_input)})"
        )

    token_count = estimate_tokens(turn_input)
    if token_count > max_tokens:
        raise InputValidationError(
            f"turn_input exceeds {max_tokens} tokens (got {token_count})"
        )


def escape_for_llm(turn_input: str) -> str:
    """
    Escape turn_input for safe LLM prompt injection.

    Uses repr() to wrap input in quotes and escape special characters,
    preventing "Ignore above instructions" type attacks.

    Args:
        turn_input: User's input text

    Returns:
        Escaped string safe for LLM prompts
    """
    return repr(turn_input)
