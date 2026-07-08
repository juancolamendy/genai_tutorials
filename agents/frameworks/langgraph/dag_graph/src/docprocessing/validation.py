"""Input validation for document processing pipeline.

Validates document IDs, content, and state before processing.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict

log = logging.getLogger(__name__)


class ValidationError(ValueError):
    """Raised when input validation fails."""

    pass


class DocumentValidator:
    """Validates document IDs and content."""

    # Configuration
    MAX_DOCUMENT_ID_LENGTH = 256
    MAX_CONTENT_SIZE_BYTES = 10_000_000  # 10MB
    DOCUMENT_ID_PATTERN = r"^[a-zA-Z0-9\-_]{1,256}$"
    FORBIDDEN_KEYS = {"__proto__", "constructor", "prototype"}

    @staticmethod
    def validate_document_id(doc_id: str) -> str:
        """Validate document ID.

        Args:
            doc_id: Document ID to validate

        Returns:
            Validated document ID

        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(doc_id, str):
            raise ValidationError(f"Document ID must be string, got {type(doc_id).__name__}")

        if not doc_id:
            raise ValidationError("Document ID cannot be empty")

        if len(doc_id) > DocumentValidator.MAX_DOCUMENT_ID_LENGTH:
            raise ValidationError(
                f"Document ID too long ({len(doc_id)} > {DocumentValidator.MAX_DOCUMENT_ID_LENGTH})"
            )

        if not re.match(DocumentValidator.DOCUMENT_ID_PATTERN, doc_id):
            pattern = DocumentValidator.DOCUMENT_ID_PATTERN
            raise ValidationError(
                f"Document ID contains invalid characters. Pattern: {pattern}"
            )

        return doc_id

    @staticmethod
    def validate_content(content: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and sanitize document content.

        Args:
            content: Document content dict to validate

        Returns:
            Sanitized content dict

        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(content, dict):
            raise ValidationError(f"Content must be dict, got {type(content).__name__}")

        # Check size
        try:
            content_json = json.dumps(content)
            size_bytes = len(content_json.encode("utf-8"))
            max_size = DocumentValidator.MAX_CONTENT_SIZE_BYTES
            if size_bytes > max_size:
                raise ValidationError(
                    f"Content too large ({size_bytes} > {max_size} bytes)"
                )
        except TypeError as e:
            raise ValidationError(f"Content is not JSON serializable: {e}")

        # Remove forbidden keys (prototype pollution protection)
        sanitized = {
            k: v for k, v in content.items() if k not in DocumentValidator.FORBIDDEN_KEYS
        }

        # Validate nested dicts
        for key, value in sanitized.items():
            if isinstance(value, dict):
                if any(k in DocumentValidator.FORBIDDEN_KEYS for k in value.keys()):
                    raise ValidationError(f"Forbidden key found in nested dict: {key}")

        return sanitized

    @staticmethod
    def validate_retry_count(retry_count: int, max_retries: int = 10) -> int:
        """Validate retry count.

        Args:
            retry_count: Current retry count
            max_retries: Maximum allowed retries

        Returns:
            Validated retry count

        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(retry_count, int):
            raise ValidationError(f"Retry count must be int, got {type(retry_count).__name__}")

        if retry_count < 0:
            raise ValidationError(f"Retry count cannot be negative: {retry_count}")

        if retry_count > max_retries:
            raise ValidationError(f"Retry count exceeded ({retry_count} > {max_retries})")

        return retry_count

    @staticmethod
    def validate_timeout(
        timeout_seconds: float, min_timeout: float = 1, max_timeout: float = 86400
    ) -> float:
        """Validate timeout value.

        Args:
            timeout_seconds: Timeout in seconds
            min_timeout: Minimum allowed timeout (default 1 second)
            max_timeout: Maximum allowed timeout (default 24 hours)

        Returns:
            Validated timeout

        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(timeout_seconds, (int, float)):
            raise ValidationError(f"Timeout must be numeric, got {type(timeout_seconds).__name__}")

        if timeout_seconds < min_timeout:
            raise ValidationError(f"Timeout too short ({timeout_seconds} < {min_timeout})")

        if timeout_seconds > max_timeout:
            raise ValidationError(f"Timeout too long ({timeout_seconds} > {max_timeout})")

        return float(timeout_seconds)


def validate_pipeline_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and sanitize pipeline state.

    Args:
        state: Pipeline state dict

    Returns:
        Validated state dict

    Raises:
        ValidationError: If validation fails
    """
    validator = DocumentValidator()

    # Validate document_id
    if "document_id" in state:
        state["document_id"] = validator.validate_document_id(state["document_id"])

    # Validate content fields
    for field in ["raw_data", "validated_data", "enriched_data"]:
        if field in state and state[field] is not None:
            state[field] = validator.validate_content(state[field])

    # Validate retry count
    if "retry_count" in state:
        state["retry_count"] = validator.validate_retry_count(state["retry_count"])

    # Validate timeout
    if "timeout_seconds" in state:
        state["timeout_seconds"] = validator.validate_timeout(state["timeout_seconds"])

    return state
