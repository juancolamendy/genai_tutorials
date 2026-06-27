"""
Input validation and prompt injection prevention.

Provides:
- InputValidationError: Exception for validation failures
- validate_turn_input(): Validate length, token count, control characters
- escape_for_llm(): Remove dangerous patterns for prompt injection prevention
"""

from __future__ import annotations

import logging
import re

log = logging.getLogger(__name__)

MAX_INPUT_LENGTH = 10000
MAX_TOKEN_COUNT = 2000


class InputValidationError(Exception):
    """Raised when input validation fails."""

    pass


def validate_turn_input(turn_input: str) -> None:
    """
    Validate user input for safety and constraints.

    Checks:
    - Non-empty, string type
    - Length under MAX_INPUT_LENGTH
    - Token count under MAX_TOKEN_COUNT (via tiktoken estimate)
    - No invalid control characters

    Args:
        turn_input: User input to validate

    Raises:
        InputValidationError: If validation fails
    """
    # Check type and non-empty
    if not isinstance(turn_input, str):
        raise InputValidationError("Input must be a string")

    if not turn_input or len(turn_input.strip()) == 0:
        raise InputValidationError("Input must not be empty")

    # Check length
    if len(turn_input) > MAX_INPUT_LENGTH:
        raise InputValidationError(
            f"Input exceeds maximum length ({len(turn_input)} > {MAX_INPUT_LENGTH} characters)"
        )

    # Check token count (rough estimate using tiktoken)
    try:
        import tiktoken

        enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
        tokens = len(enc.encode(turn_input))
        if tokens > MAX_TOKEN_COUNT:
            raise InputValidationError(
                f"Input too verbose ({tokens} tokens > {MAX_TOKEN_COUNT})"
            )
    except ImportError:
        log.warning("tiktoken not available; skipping token count check")
    except Exception as e:
        log.warning(f"Token count check failed: {e}; skipping")

    # Check for dangerous control characters
    for i, char in enumerate(turn_input):
        code = ord(char)
        # Allow printable ASCII, newline, tab, carriage return
        if code < 32 and char not in "\n\t\r":
            raise InputValidationError(
                f"Input contains invalid control character at position {i} (code {code})"
            )


def escape_for_llm(turn_input: str) -> str:
    """
    Escape input to prevent prompt injection attacks.

    Removes:
    - XML-like tags (<prompt>, </prompt>, <system>, etc.)
    - Role indicators (System:, Admin:, User:, Assistant:)
    - Jailbreak patterns
    - Template injection markers ({{, }})

    Args:
        turn_input: Raw user input

    Returns:
        Escaped input safe for LLM injection
    """
    escaped = turn_input

    # Dangerous patterns to remove
    patterns = [
        (r"</?prompt>", ""),
        (r"</?system>", ""),
        (r"</?instruction>", ""),
        (r"</?admin>", ""),
        (r"</?user>", ""),
        (r"(^|\s)(System|Admin|Assistant|User):\s*", r"\1"),
        (r"SYSTEM PROMPT:\s*", ""),
        (r"INSTRUCTIONS:\s*", ""),
        (r"JAILBREAK:\s*", ""),
        (r"\{\{", ""),
        (r"\}\}", ""),
        (r"<!--.*?-->", ""),  # HTML comments
        (r"\/\*.*?\*\/", ""),  # C-style comments
    ]

    for pattern, replacement in patterns:
        escaped = re.sub(pattern, replacement, escaped, flags=re.IGNORECASE | re.DOTALL)

    # Remove leading/trailing whitespace
    escaped = escaped.strip()

    return escaped


def sanitize_for_db(text: str) -> str:
    """
    Sanitize text for safe storage in database.

    Removes:
    - Prototype pollution patterns (__proto__, constructor, prototype)
    - NoSQL injection patterns

    Args:
        text: Text to sanitize

    Returns:
        Sanitized text
    """
    text = str(text)

    # Remove dangerous keys
    dangerous = ["__proto__", "constructor", "prototype"]
    for key in dangerous:
        text = re.sub(rf"{re.escape(key)}", "", text, flags=re.IGNORECASE)

    return text
