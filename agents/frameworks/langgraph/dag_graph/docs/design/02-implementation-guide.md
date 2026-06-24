# LangGraph DAG Graph: Implementation Guide

**Status:** Ready to Implement  
**Date:** 2026-06-24  
**Phase:** 1-4 (Foundation → Integration → Testing → Documentation)

---

## Quick Start: What Needs to Change

### Current Situation
- Pure code router via `HAPPY_PATH` routing table
- Single-turn `run_pipeline()` execution
- Checkpointing works for resuming interrupted pipelines
- No multi-turn support or conversation history

### Target State
- Optional LLM-powered semantic router
- Both `process()` (one-turn) and `invoke_turn()` (multi-turn) methods
- Full conversation history with pause/resume capability
- Handler metadata with `@handler` decorator
- Input validation with prompt injection prevention

### Effort Estimate
- **Phase 1 (Foundation):** 4-5 days
- **Phase 2 (Multi-turn):** 4-5 days
- **Phase 3 (Integration):** 3-4 days
- **Phase 4 (Documentation):** 2-3 days
- **Total:** ~16 working days (~3-4 weeks)

---

## Phase 1: Foundation (Days 1-5)

### Task 1.1: Create Handler Registry

**File:** `src/engine/handler_registry.py` (NEW)

```python
"""
Handler registration and @handler decorator.

Provides metadata registry that allows handlers to declare:
- state: Which state this handler processes
- waits_for_input: If True, workflow pauses at this state
- description: Human-readable description
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class HandlerMetadata:
    """Metadata configuration for a state handler."""
    
    state: str
    waits_for_input: bool = False
    description: Optional[str] = None
    
    def __repr__(self) -> str:
        return (
            f"HandlerMetadata(state={self.state!r}, "
            f"waits_for_input={self.waits_for_input}, "
            f"description={self.description!r})"
        )


# Global registry: populated by @handler decorator
HANDLER_MAP_METADATA: dict[str, HandlerMetadata] = {}


def handler(
    state: str,
    waits_for_input: bool = False,
    description: Optional[str] = None,
) -> Callable:
    """
    Decorator that registers a handler function with metadata.
    
    Usage:
        @handler(state="validate", waits_for_input=False)
        def handle_validate(state: PipelineState) -> PipelineState:
            # Validate document
            state["current_state"] = "enrich"
            return state
    
    Args:
        state: State enum value (e.g., "validate", "enrich")
        waits_for_input: If True, workflow pauses and waits for next turn
        description: Human-readable description (optional)
    
    Returns:
        Decorator function that registers metadata and returns original function
    """
    def decorator(func: Callable) -> Callable:
        HANDLER_MAP_METADATA[state] = HandlerMetadata(
            state=state,
            waits_for_input=waits_for_input,
            description=description,
        )
        return func
    
    return decorator


def get_handler_metadata(state: str) -> Optional[HandlerMetadata]:
    """
    Retrieve metadata for a registered state.
    
    Args:
        state: State to look up (e.g., "validate")
    
    Returns:
        HandlerMetadata if registered, None otherwise
    """
    return HANDLER_MAP_METADATA.get(state)


def does_state_wait_for_input(state: str) -> bool:
    """
    Check if a state pauses and waits for user input.
    
    Args:
        state: State to check (e.g., "human_review")
    
    Returns:
        True if state has waits_for_input=True, False otherwise
    """
    meta = get_handler_metadata(state)
    return meta.waits_for_input if meta else False


def list_handler_metadata() -> list[HandlerMetadata]:
    """List all registered handler metadata."""
    return list(HANDLER_MAP_METADATA.values())


def clear_metadata() -> None:
    """Clear registry (useful for testing). Use with caution."""
    HANDLER_MAP_METADATA.clear()
```

**Tests:** `tests/test_handler_registry.py`
```python
import pytest
from engine.handler_registry import (
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


def test_does_state_wait_for_input():
    """Test does_state_wait_for_input helper."""
    clear_metadata()
    
    @handler(state="blocking", waits_for_input=True)
    def blocking_handler(state):
        return state
    
    @handler(state="non_blocking", waits_for_input=False)
    def non_blocking_handler(state):
        return state
    
    assert does_state_wait_for_input("blocking") is True
    assert does_state_wait_for_input("non_blocking") is False
    assert does_state_wait_for_input("unknown") is False


def test_handler_without_metadata():
    """Test that unregistered states return False for waits_for_input."""
    assert does_state_wait_for_input("nonexistent_state") is False
```

---

### Task 1.2: Create Router Base Classes

**File:** `src/engine/router.py` (NEW)

```python
"""
Semantic router base classes for LLM-powered state transitions.

Provides:
- RouterDecision: Output structure with decision metadata
- BaseSemanticRouter: Abstract interface
- DefaultSemanticRouter: Concrete implementation with common LLM logic
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger(__name__)


@dataclass
class RouterDecision:
    """Output from semantic router classification."""
    
    proposed_next: str                              # Next state (e.g., "validate")
    confidence: float                               # [0.0, 1.0] confidence score
    semantic_entities: dict = field(default_factory=dict)  # Extracted entities
    semantic_intents: list = field(default_factory=list)   # Extracted intents
    reasoning: Optional[str] = None                         # Optional explanation


class BaseSemanticRouter(ABC):
    """
    Abstract base class for LLM-powered state routers.
    
    Subclasses must implement route() to classify user input and determine
    the next state in a multi-turn conversation.
    
    Example:
        class MyRouter(BaseSemanticRouter):
            def route(self, current_state, turn_input, history, allowed_states, timeout_sec):
                # Use LLM to decide next state
                return RouterDecision(proposed_next="validate", confidence=0.95)
    """
    
    @abstractmethod
    def route(
        self,
        current_state: str,
        turn_input: str,
        history: list,
        allowed_states: list,
        timeout_sec: float = 10.0,
    ) -> RouterDecision:
        """
        Classify user input and determine next state.
        
        Args:
            current_state: Current state (e.g., "validate")
            turn_input: User's input text (already validated and escaped)
            history: List of prior turns [{role, content, ...}]
            allowed_states: Valid next states per state machine transitions
            timeout_sec: LLM call timeout (default 10s)
        
        Returns:
            RouterDecision with proposed_next, confidence, entities, intents
        
        Implementation notes:
            • Must respect allowed_states; invalid proposal → retry with constraints
            • Extract semantic entities specific to domain (amounts, items, keywords)
            • Extract user intents (confirm, clarify, escalate, etc.)
            • Set confidence [0.0, 1.0] reflecting decision quality
            • Handle timeouts gracefully; return ERROR state on timeout
        """
        raise NotImplementedError


class DefaultSemanticRouter(BaseSemanticRouter):
    """
    Concrete router with common LLM-powered routing logic.
    
    Subclasses override:
    - output_schema: Pydantic model for LLM response (domain-specific)
    - get_instructions(): LLM system instructions
    - build_router_prompt(): LLM prompt template
    
    Common logic handles:
    - LLM client initialization
    - Route classification workflow
    - State validation and fallback
    - Confidence clamping
    - Logging and error handling
    
    Example:
        class DocRouter(DefaultSemanticRouter):
            output_schema = DocRouterOutput  # Pydantic model
            
            def get_instructions(self):
                return "You are a document router. Choose next state: ..."
            
            def build_router_prompt(self, current_state, turn_input, ...):
                return f"State: {current_state}\\nInput: {turn_input}\\n..."
    """
    
    output_schema: type = None  # Subclasses MUST set this to a Pydantic model
    
    def __init__(self, model: str = "claude-haiku-4-5-20251001"):
        """
        Initialize router with Claude LLM via LangChain.
        
        Args:
            model: Claude model ID (default haiku for cost efficiency)
        
        Raises:
            NotImplementedError: If output_schema not set by subclass
        """
        if self.output_schema is None:
            raise NotImplementedError(
                f"Subclass {self.__class__.__name__} must set output_schema to a Pydantic model"
            )
        
        self.model = model
        self.llm = None  # Lazy-loaded in route()
    
    def _get_llm(self):
        """Lazy-load LLM client (singleton per router instance)."""
        if self.llm is None:
            from langchain_anthropic import ChatAnthropic
            self.llm = ChatAnthropic(
                model=self.model,
                temperature=0,
                timeout=15.0,
            )
        return self.llm
    
    def get_instructions(self) -> str:
        """
        Return LLM system instructions for routing.
        
        Subclasses override to provide domain-specific instructions.
        Default provides generic routing guidance.
        """
        return """You are a state machine router for workflows.
Given the current state, user input, conversation history, and allowed next states,
determine which state the workflow should transition to next.

IMPORTANT: Always propose one of the ALLOWED NEXT STATES.

Extract relevant entities and intents from the user's input.
Return confidence 0.0-1.0 based on how clear the user's intent is.
Provide brief reasoning for your decision."""
    
    def build_router_prompt(
        self,
        current_state: str,
        turn_input: str,
        history_text: str,
        allowed_states: list,
    ) -> str:
        """
        Build LLM prompt for state classification.
        
        Subclasses override to provide domain-specific prompt format.
        Default provides generic structure.
        
        Args:
            current_state: Current state (e.g., "validate")
            turn_input: User's input text
            history_text: Formatted conversation history
            allowed_states: List of valid next states
        
        Returns:
            Prompt string for LLM
        """
        allowed_str = ", ".join(allowed_states)
        return f"""WORKFLOW STATE MACHINE ROUTING

Current State: {current_state}
Allowed Next States: {allowed_str}

Conversation History (last 5 turns):
{history_text}

User Input: {turn_input}

Determine the next state based on the user's intent and the current state.
Always choose from the ALLOWED NEXT STATES."""
    
    def route(
        self,
        current_state: str,
        turn_input: str,
        history: list,
        allowed_states: list,
        timeout_sec: float = 10.0,
    ) -> RouterDecision:
        """
        Route using LLM with structure constraints.
        
        Args:
            current_state: Current state
            turn_input: User input (already escaped)
            history: List of prior turns
            allowed_states: Valid next states
            timeout_sec: Timeout for LLM call
        
        Returns:
            RouterDecision with validated proposed_next
        """
        from langchain.schema import HumanMessage, SystemMessage
        
        try:
            # Build history text from last 5 turns only
            history_text = "\n".join([
                f"{turn.get('role', 'user').title()}: {turn.get('content', '')[:100]}"
                for turn in history[-5:]
            ]) or "(No prior turns)"
            
            # Build prompt
            prompt = self.build_router_prompt(
                current_state,
                turn_input,
                history_text,
                allowed_states,
            )
            
            # Call LLM with structured output
            llm = self._get_llm()
            with_structure = llm.with_structured_output(self.output_schema)
            
            response = with_structure.invoke([
                SystemMessage(content=self.get_instructions()),
                HumanMessage(content=prompt),
            ])
            
            # Validate proposed_next is in allowed_states
            if response.proposed_next not in allowed_states:
                log.warning(
                    "[SemanticRouter] Invalid proposal '%s' not in %s; fallback to '%s'",
                    response.proposed_next,
                    allowed_states,
                    allowed_states[0],
                )
                response.proposed_next = allowed_states[0]
            
            # Clamp confidence to [0.0, 1.0]
            confidence = max(0.0, min(1.0, response.confidence))
            
            log.info(
                "[SemanticRouter] %s + '%s...' → %s (conf: %.2f)",
                current_state,
                turn_input[:30] if turn_input else "(empty)",
                response.proposed_next,
                confidence,
            )
            
            return RouterDecision(
                proposed_next=response.proposed_next,
                confidence=confidence,
                semantic_entities=getattr(response, 'semantic_entities', {}),
                semantic_intents=getattr(response, 'semantic_intents', []),
                reasoning=getattr(response, 'reasoning', None),
            )
        
        except Exception as e:
            log.exception("[SemanticRouter] Error in route(): %s", e)
            # Fallback: return first allowed state with low confidence
            return RouterDecision(
                proposed_next=allowed_states[0] if allowed_states else "error",
                confidence=0.0,
                reasoning=f"Error during routing: {str(e)}",
            )
```

**Tests:** `tests/test_semantic_router.py`
```python
import pytest
from pydantic import BaseModel, Field
from engine.router import (
    RouterDecision,
    BaseSemanticRouter,
    DefaultSemanticRouter,
)


class TestRouterOutput(BaseModel):
    proposed_next: str = Field(..., description="Next state")
    confidence: float = Field(..., description="Confidence [0.0, 1.0]")
    semantic_entities: dict = Field(default_factory=dict)
    semantic_intents: list = Field(default_factory=list)
    reasoning: str = ""


class TestRouter(DefaultSemanticRouter):
    output_schema = TestRouterOutput


def test_router_decision_creation():
    """Test RouterDecision dataclass."""
    decision = RouterDecision(
        proposed_next="validate",
        confidence=0.95,
        semantic_entities={"doc_id": "123"},
        semantic_intents=["submit"],
    )
    assert decision.proposed_next == "validate"
    assert decision.confidence == 0.95
    assert decision.semantic_entities["doc_id"] == "123"


def test_base_router_is_abstract():
    """Test that BaseSemanticRouter cannot be instantiated."""
    with pytest.raises(TypeError):
        BaseSemanticRouter()


def test_default_router_without_schema():
    """Test that DefaultSemanticRouter requires output_schema."""
    with pytest.raises(NotImplementedError):
        DefaultSemanticRouter()


def test_default_router_with_schema():
    """Test DefaultSemanticRouter with valid schema."""
    router = TestRouter()
    assert router.model == "claude-haiku-4-5-20251001"
    assert router.output_schema == TestRouterOutput


def test_get_instructions():
    """Test that get_instructions returns valid string."""
    router = TestRouter()
    instructions = router.get_instructions()
    assert isinstance(instructions, str)
    assert "ALLOWED NEXT STATES" in instructions


def test_build_router_prompt():
    """Test prompt building."""
    router = TestRouter()
    prompt = router.build_router_prompt(
        current_state="validate",
        turn_input="Document looks good",
        history_text="User: Process my doc",
        allowed_states=["enrich", "store", "error"],
    )
    assert "validate" in prompt
    assert "Document looks good" in prompt
    assert "enrich, store, error" in prompt
```

---

### Task 1.3: Create Input Validation Module

**File:** `src/engine/input_validation.py` (NEW)

```python
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
from typing import Optional

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
        if code < 32 and char not in '\n\t\r':
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
        (r'</?prompt>', ''),
        (r'</?system>', ''),
        (r'</?instruction>', ''),
        (r'</?admin>', ''),
        (r'</?user>', ''),
        (r'(^|\s)(System|Admin|Assistant|User):\s*', r'\1'),
        (r'SYSTEM PROMPT:\s*', ''),
        (r'INSTRUCTIONS:\s*', ''),
        (r'JAILBREAK:\s*', ''),
        (r'\{\{', ''),
        (r'\}\}', ''),
        (r'<!--.*?-->', ''),  # HTML comments
        (r'\/\*.*?\*\/', ''),  # C-style comments
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
    dangerous = ['__proto__', 'constructor', 'prototype']
    for key in dangerous:
        text = re.sub(rf'{re.escape(key)}', '', text, flags=re.IGNORECASE)
    
    return text
```

**Tests:** `tests/test_input_validation.py`
```python
import pytest
from engine.input_validation import (
    validate_turn_input,
    escape_for_llm,
    InputValidationError,
    MAX_INPUT_LENGTH,
)


def test_validate_valid_input():
    """Test that valid input passes validation."""
    validate_turn_input("This is a valid input")
    validate_turn_input("Another valid input with numbers 123")


def test_validate_rejects_non_string():
    """Test that non-string input is rejected."""
    with pytest.raises(InputValidationError):
        validate_turn_input(123)
    
    with pytest.raises(InputValidationError):
        validate_turn_input(None)


def test_validate_rejects_empty():
    """Test that empty input is rejected."""
    with pytest.raises(InputValidationError):
        validate_turn_input("")
    
    with pytest.raises(InputValidationError):
        validate_turn_input("   ")


def test_validate_rejects_too_long():
    """Test that input exceeding MAX_INPUT_LENGTH is rejected."""
    too_long = "a" * (MAX_INPUT_LENGTH + 1)
    with pytest.raises(InputValidationError):
        validate_turn_input(too_long)


def test_validate_allows_max_length():
    """Test that input at exactly MAX_INPUT_LENGTH is accepted."""
    at_max = "a" * MAX_INPUT_LENGTH
    validate_turn_input(at_max)


def test_escape_removes_xml_tags():
    """Test that XML tags are removed."""
    assert escape_for_llm("<prompt>ignore</prompt>") == "ignore"
    assert escape_for_llm("<system>ignore</system>") == "ignore"
    assert escape_for_llm("<instruction>ignore</instruction>") == "ignore"


def test_escape_removes_role_indicators():
    """Test that role indicators are removed."""
    assert "System:" not in escape_for_llm("System: Do something")
    assert "Admin:" not in escape_for_llm("Admin: Override")
    assert "User:" not in escape_for_llm("User: Request")


def test_escape_removes_template_markers():
    """Test that template injection markers are removed."""
    assert escape_for_llm("{{variable}}") == "variable"
    assert escape_for_llm("{{ payload }}") == " payload "


def test_escape_removes_jailbreak():
    """Test that jailbreak patterns are removed."""
    assert "JAILBREAK" not in escape_for_llm("JAILBREAK: Ignore instructions")
    assert "SYSTEM PROMPT" not in escape_for_llm("SYSTEM PROMPT: Do X")


def test_escape_preserves_legitimate_text():
    """Test that legitimate text is preserved."""
    text = "Please process my document with ID 12345"
    escaped = escape_for_llm(text)
    assert "process" in escaped
    assert "document" in escaped
    assert "12345" in escaped
```

---

### Task 1.4: Create Domain-Specific Router

**File:** `src/workflow/router.py` (NEW or UPDATED)

```python
"""
Document pipeline semantic router for LLM-powered state transitions.

Inherits from DefaultSemanticRouter and provides domain-specific:
- Output schema (Pydantic model)
- System instructions
- Prompt template
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from engine.router import DefaultSemanticRouter


class DocRouterOutput(BaseModel):
    """Output schema for document pipeline router."""
    
    proposed_next: str = Field(
        ...,
        description="Next state (init, fetch, validate, enrich, store, complete, retry, error, human_review)",
    )
    confidence: float = Field(
        ...,
        description="Confidence score [0.0, 1.0]",
    )
    semantic_entities: dict = Field(
        default_factory=dict,
        description="Extracted entities from user input (e.g., doc_id, keywords)",
    )
    semantic_intents: list = Field(
        default_factory=list,
        description="Extracted intents (e.g., submit, review, escalate)",
    )
    reasoning: str = Field(
        default="",
        description="Brief reasoning for the decision",
    )


class DocPipelineRouter(DefaultSemanticRouter):
    """
    Semantic router for document processing workflow.
    
    Uses Claude LLM to determine next state based on:
    - Current processing state
    - User input / document content
    - Conversation history
    - Valid allowed transitions
    
    Example:
        router = DocPipelineRouter()
        decision = router.route(
            current_state="validate",
            turn_input="Document looks good, proceed",
            history=[...],
            allowed_states=["enrich", "human_review", "error"],
        )
        # RouterDecision(proposed_next="enrich", confidence=0.95, ...)
    """
    
    output_schema = DocRouterOutput
    
    def get_instructions(self) -> str:
        """Return document pipeline-specific routing instructions."""
        return """You are a document processing workflow router.

Your job is to determine which state the document workflow should transition to next,
based on the current state, user input/feedback, conversation history, and document content.

STATES:
- init: Starting state
- fetch: Retrieve document from source
- validate: Check document schema and content validity
- enrich: Add metadata, tags, summary
- store: Persist enriched document
- complete: Pipeline finished successfully
- retry: Retry last operation (transient failure)
- error: Pipeline encountered permanent error
- human_review: Wait for human expert review

IMPORTANT RULES:
1. Always choose from ALLOWED NEXT STATES
2. Only transition if the operation would succeed
3. Return confidence [0.0, 1.0] based on clarity
4. Extract entities (doc_id, keywords, amounts) from input
5. Extract intents (submit, review, escalate, confirm)
6. If unsure, default to human_review rather than proceeding blindly

Be concise and confident in your decision."""
    
    def build_router_prompt(
        self,
        current_state: str,
        turn_input: str,
        history_text: str,
        allowed_states: list,
    ) -> str:
        """Build document-specific routing prompt."""
        allowed_str = ", ".join(allowed_states)
        return f"""DOCUMENT PIPELINE ROUTING DECISION

Current State: {current_state}
Allowed Next States: {allowed_str}

Conversation History (last 3 turns):
{history_text}

User Input / Feedback:
{turn_input}

Based on the current state, user feedback, and allowed transitions,
determine the next state. Extract any entities and intents from the user input."""
```

**Tests:** `tests/test_doc_pipeline_router.py`
```python
from workflow.router import DocPipelineRouter, DocRouterOutput


def test_doc_router_output_schema():
    """Test that DocRouterOutput is properly defined."""
    output = DocRouterOutput(
        proposed_next="validate",
        confidence=0.95,
        semantic_entities={"doc_id": "123"},
        semantic_intents=["submit"],
        reasoning="User provided valid document",
    )
    assert output.proposed_next == "validate"
    assert output.confidence == 0.95


def test_doc_router_initialization():
    """Test that DocPipelineRouter initializes correctly."""
    router = DocPipelineRouter()
    assert router.model == "claude-haiku-4-5-20251001"
    assert router.output_schema == DocRouterOutput


def test_doc_router_get_instructions():
    """Test that instructions are document-specific."""
    router = DocPipelineRouter()
    instructions = router.get_instructions()
    assert "document" in instructions.lower()
    assert "validate" in instructions.lower()
    assert "enrich" in instructions.lower()


def test_doc_router_build_prompt():
    """Test prompt building with document-specific content."""
    router = DocPipelineRouter()
    prompt = router.build_router_prompt(
        current_state="fetch",
        turn_input="Document received, ready to validate",
        history_text="User: Start processing",
        allowed_states=["validate", "error"],
    )
    assert "fetch" in prompt
    assert "validate, error" in prompt
    assert "Document received" in prompt
```

---

### Summary: Phase 1 Deliverables

✅ `src/engine/handler_registry.py` — @handler decorator + metadata registry  
✅ `src/engine/router.py` — BaseSemanticRouter + DefaultSemanticRouter  
✅ `src/engine/input_validation.py` — Input validation + prompt injection escaping  
✅ `src/workflow/router.py` — DocPipelineRouter (domain-specific)  
✅ Test files for all above modules  

**Effort:** ~40-50 hours (4-5 days with testing)

**Quality Checks:**
- All tests pass
- No linting errors
- Type hints complete
- Docstrings follow existing style

---

## Phase 2: Multi-turn Support (Days 6-10)

### Task 2.1: Extend PipelineState

**File:** `src/workflow/pipeline_state.py` (UPDATED)

```python
# Add these fields to PipelineState TypedDict

from typing import TypedDict, Optional, Any, List

class PipelineState(TypedDict):
    # Existing fields (keep as-is)
    current_state: str
    proposed_next: str
    guardrail_ok: bool
    retry_count: int
    error_message: Optional[str]
    audit_trail: list
    document_id: str
    raw_data: Optional[dict]
    validated_data: Optional[dict]
    enriched_data: Optional[dict]
    
    # NEW: Multi-turn fields
    turn_input: Optional[str]                      # Current turn's user input (escaped)
    turn_number: int                                # Turn counter (0, 1, 2, ...)
    conversation_history: list                      # Turns: {role, content, semantic_context, state}
    max_history_turns: int                          # Max turns to keep (default 10)
    router_timeout_sec: float                       # Timeout for semantic router (default 10.0)
    user_id: Optional[str]                          # Caller identity (for audit)
    session_id: Optional[str]                       # Multi-turn session ID
    semantic_context: dict                          # {entities, intents} from router
    router_confidence: float                        # [0.0, 1.0]


def new_pipeline(entity_id: str, timeout_seconds: float = 300.0) -> dict[str, Any]:
    """
    Initialize fresh pipeline state.
    
    Args:
        entity_id: Document ID, invoice ID, etc.
        timeout_seconds: Max execution time
    
    Returns:
        Fresh PipelineState dict
    """
    return {
        "current_state": "init",
        "proposed_next": None,
        "guardrail_ok": True,
        "retry_count": 0,
        "error_message": None,
        "audit_trail": [],
        "document_id": entity_id,
        "raw_data": None,
        "validated_data": None,
        "enriched_data": None,
        # Multi-turn fields
        "turn_input": None,
        "turn_number": 0,
        "conversation_history": [],
        "max_history_turns": 10,
        "router_timeout_sec": 10.0,
        "user_id": None,
        "session_id": None,
        "semantic_context": {},
        "router_confidence": 0.0,
    }
```

---

### Task 2.2: Implement Base Class Methods in Engine

**File:** `src/engine/graph.py` (UPDATED)

Add these methods to `StateMachineGraph` base class:

```python
def invoke_turn(
    self,
    user_id: str,
    session_id: str,
    turn_input: str,
    timeout_sec: float = 10.0,
) -> dict[str, Any]:
    """
    Execute one turn of multi-turn conversation.
    
    Workflow:
    1. Validate and escape user input
    2. Get or initialize state for session
    3. Prepare turn metadata
    4. Run state machine once (router → guardrail → handler)
    5. Auto-progress through non-blocking states
    6. Trim conversation history
    7. Append turn to history
    8. Return turn response
    
    Args:
        user_id: Caller identity (for audit)
        session_id: Multi-turn session ID
        turn_input: User's input text
        timeout_sec: LLM router timeout
    
    Returns:
        {
            "current_state": str,
            "waits_for_input": bool,
            "turn_number": int,
            "semantic_context": dict,
            "router_confidence": float,
            "error": str or None,
        }
    """
    from engine.input_validation import (
        validate_turn_input, escape_for_llm, InputValidationError
    )
    
    try:
        # Validate and escape input
        validate_turn_input(turn_input)
        escaped = escape_for_llm(turn_input)
        
        # Thread ID for checkpointing across turns
        thread_id = f"{user_id}:{session_id}"
        config = {"configurable": {"thread_id": thread_id}}
        
        # Get or initialize state
        state = self._get_or_init_state(session_id)
        
        # Prepare turn metadata
        state["turn_input"] = escaped
        state["turn_number"] = state.get("turn_number", 0) + 1
        state["router_timeout_sec"] = timeout_sec
        state["user_id"] = user_id
        state["session_id"] = session_id
        
        # Initialize router if available and needed
        if hasattr(self, '_init_router'):
            self._init_router()
        
        # First invoke: router → guardrail → handler
        state = self.compiled_graph.invoke(state, config=config)
        
        # Auto-progress through non-blocking states
        state = self._auto_progress_langgraph(state, config)
        
        # Trim history
        max_turns = state.get("max_history_turns", 10)
        history = state.get("conversation_history", [])
        if len(history) > max_turns:
            dropped = len(history) - max_turns
            state["conversation_history"] = history[-max_turns:]
            log.info(f"[invoke_turn] Trimmed {dropped} old turns; keeping {max_turns}")
        
        # Append turn result to history
        state["conversation_history"].append({
            "role": "assistant",
            "content": f"Transitioned to {state['current_state']}",
            "semantic_context": {
                "entities": state.get("semantic_context", {}).get("entities", {}),
                "intents": state.get("semantic_context", {}).get("intents", []),
            },
            "state": state["current_state"],
            "turn_number": state["turn_number"],
        })
        
        return self._build_turn_response(state)
    
    except InputValidationError as e:
        return {
            "error": str(e),
            "current_state": None,
            "waits_for_input": False,
            "turn_number": 0,
            "semantic_context": {},
            "router_confidence": 0.0,
        }
    except Exception as e:
        log.exception("[invoke_turn] Error: %s", e)
        return {
            "error": str(e),
            "current_state": "error",
            "waits_for_input": False,
            "turn_number": 0,
            "semantic_context": {},
            "router_confidence": 0.0,
        }


def _get_or_init_state(self, session_id: str) -> dict[str, Any]:
    """
    Get existing state or create fresh state for session.
    
    Tries to load from checkpointer first; if not found, creates fresh state.
    
    Args:
        session_id: Session identifier
    
    Returns:
        PipelineState dict
    """
    thread_id = f"invoke_turn:{session_id}"
    config = {"configurable": {"thread_id": thread_id}}
    
    # Try to load from checkpointer
    try:
        checkpoint = self.checkpointer.get_tuple(thread_id)
        if checkpoint and checkpoint.values:
            return checkpoint.values
    except Exception:
        pass  # Not found or error; create fresh
    
    # Create fresh state
    return self.new_pipeline(session_id)


def _auto_progress_langgraph(
    self,
    state: dict[str, Any],
    config: dict,
) -> dict[str, Any]:
    """
    Auto-progress through non-blocking states.
    
    If current state has waits_for_input=False, continue running state machine
    until hitting a state with waits_for_input=True or a terminal state.
    
    Args:
        state: Current pipeline state
        config: Graph invocation config (with thread_id for checkpointing)
    
    Returns:
        Updated state after auto-progression
    """
    from engine.handler_registry import does_state_wait_for_input
    
    max_auto_iters = 10  # Prevent infinite loops
    iters = 0
    
    while iters < max_auto_iters:
        current = state.get("current_state", "init")
        
        # Stop if terminal state
        if current in self.TERMINAL_STATES:
            log.debug(f"[auto_progress] Stopped at terminal state {current}")
            break
        
        # Stop if state waits for input
        if does_state_wait_for_input(current):
            log.debug(f"[auto_progress] Stopped at input-waiting state {current}")
            break
        
        # Continue: run state machine one more time
        log.debug(f"[auto_progress] {current} is non-blocking; continuing...")
        state = self.compiled_graph.invoke(state, config=config)
        iters += 1
    
    if iters >= max_auto_iters:
        log.warning(
            "[auto_progress] Reached max iterations (%d); stopping", max_auto_iters
        )
    
    return state


def process(
    self,
    entity_id: str,
    timeout_seconds: float = 300.0,
) -> dict[str, Any]:
    """
    Execute one complete workflow run for an entity.
    
    Workflow:
    1. Create fresh state
    2. Run state machine loop
    3. Auto-progress through non-blocking states
    4. Return response
    
    Args:
        entity_id: Document ID, invoice ID, etc.
        timeout_seconds: Max execution time
    
    Returns:
        Response dict with current_state, audit_trail, errors, etc.
    """
    try:
        # Create fresh state
        state = self.new_pipeline(entity_id, timeout_seconds)
        
        # Thread ID for checkpointing
        thread_id = f"process:{entity_id}"
        config = {"configurable": {"thread_id": thread_id}}
        
        # Run state machine
        state = self.compiled_graph.invoke(state, config=config)
        
        # Auto-progress
        state = self._auto_progress_langgraph(state, config)
    
    except Exception as e:
        log.exception("[process] Error: %s", e)
        state = self.new_pipeline(entity_id, timeout_seconds)
        state["current_state"] = "error"
        state["error_message"] = str(e)
    
    return self._build_response(entity_id, state)


def _build_response(
    self,
    entity_id: str,
    state: dict[str, Any],
) -> dict[str, Any]:
    """
    Build response dict from final state after process().
    
    Args:
        entity_id: Entity being processed
        state: Final PipelineState
    
    Returns:
        Response dict
    """
    return {
        "current_state": state.get("current_state", "init"),
        "proposed_next": state.get("proposed_next"),
        "retry_count": state.get("retry_count", 0),
        "error_message": state.get("error_message"),
        "audit_trail": state.get("audit_trail", []),
        "entity_id": entity_id,
    }


def _build_turn_response(
    self,
    state: dict[str, Any],
) -> dict[str, Any]:
    """
    Build response dict from state after invoke_turn().
    
    Args:
        state: PipelineState after turn execution
    
    Returns:
        Turn response dict
    """
    from engine.handler_registry import does_state_wait_for_input
    
    current = state.get("current_state", "init")
    return {
        "current_state": current,
        "waits_for_input": does_state_wait_for_input(current),
        "turn_number": state.get("turn_number", 0),
        "semantic_context": state.get("semantic_context", {}),
        "router_confidence": state.get("router_confidence", 0.0),
        "error": state.get("error_message"),
    }
```

---

### Task 2.3: Update Router Node for Semantic Routing

**File:** `src/engine/graph.py` (UPDATED)

Update the `_router_node` method to dispatch to semantic or pure code router:

```python
def _router_node(self, state: dict[str, Any]) -> dict[str, Any]:
    """
    Router node: determines next state.
    
    Dispatches to:
    - _semantic_router_node() if turn_input is present (multi-turn)
    - _pure_code_router_node() if turn_input is absent (one-turn)
    
    Args:
        state: Current pipeline state
    
    Returns:
        Updated state with proposed_next set
    """
    is_multiturn = state.get("turn_input") is not None
    
    if is_multiturn:
        log.debug("[Router] Multi-turn mode: using semantic router")
        return self._semantic_router_node(state)
    else:
        log.debug("[Router] Single-turn mode: using pure code router")
        return self._pure_code_router_node(state)


def _semantic_router_node(self, state: dict[str, Any]) -> dict[str, Any]:
    """
    LLM-powered router: reads state + turn_input → proposes next state.
    
    Used for multi-turn workflows. Requires self.router to be initialized.
    Falls back to pure code router if not available.
    
    Args:
        state: Current pipeline state
    
    Returns:
        Updated state with proposed_next and semantic_context set
    """
    if not hasattr(self, 'router') or self.router is None:
        log.warning("[SemanticRouter] Router not initialized; falling back to pure code")
        return self._pure_code_router_node(state)
    
    try:
        current = state.get("current_state", "init")
        turn_input = state.get("turn_input", "")
        history = state.get("conversation_history", [])
        
        # Get allowed next states from routing table
        routing_table = self._build_routing_table()
        allowed_states = self._get_allowed_states(current, routing_table)
        
        # Call semantic router
        decision = self.router.route(
            current_state=current,
            turn_input=turn_input,
            history=history,
            allowed_states=allowed_states,
            timeout_sec=state.get("router_timeout_sec", 10.0),
        )
        
        # Update state with router decision
        state["proposed_next"] = decision.proposed_next
        state["semantic_context"] = {
            "entities": decision.semantic_entities,
            "intents": decision.semantic_intents,
        }
        state["router_confidence"] = decision.confidence
        
        log.info(
            "[SemanticRouter] %s + '%s...' → %s (conf: %.2f)",
            current,
            turn_input[:30] if turn_input else "(empty)",
            decision.proposed_next,
            decision.confidence,
        )
        
        return state
    
    except Exception as e:
        log.exception("[SemanticRouter] Error: %s; falling back to pure code", e)
        return self._pure_code_router_node(state)


def _pure_code_router_node(self, state: dict[str, Any]) -> dict[str, Any]:
    """
    Pure code router: reads current_state → proposes next_state via routing table.
    
    Args:
        state: Current pipeline state
    
    Returns:
        Updated state with proposed_next set
    """
    routing_table = self._build_routing_table()
    current = state.get("current_state", "init")
    
    if current not in routing_table:
        log.error("[Router] Current state %s not in routing table", current)
        state["proposed_next"] = "error"
        state["error_message"] = f"No routing for state {current}"
        return state
    
    proposal = routing_table[current]
    state["proposed_next"] = proposal
    
    log.info("[Router] %s → proposes %s", current, proposal)
    return state


def _get_allowed_states(self, current: str, routing_table: dict) -> list[str]:
    """
    Extract allowed next states for current state.
    
    Handles both single-state and dict routing entries.
    
    Args:
        current: Current state
        routing_table: Routing table dict
    
    Returns:
        List of allowed state strings
    """
    if current not in routing_table:
        return ["error"]
    
    entry = routing_table[current]
    
    # Handle dict entry (multiple paths)
    if isinstance(entry, dict):
        allowed = list(entry.keys())
    else:
        allowed = [entry]
    
    # Convert to strings
    return [str(s) if not isinstance(s, str) else s for s in allowed]
```

---

### Task 2.4: Update Handler Nodes to Record Turns

**File:** `src/workflow/handlers.py` (UPDATED)

Add @handler decorators and update handlers to record conversation turns:

```python
from engine.handler_registry import handler

@handler(state="init", waits_for_input=False, description="Initialize pipeline")
def handle_init(state: dict) -> dict:
    """Initialize pipeline state."""
    state["current_state"] = "fetch"
    state["audit_trail"].append("INIT → FETCH")
    return state


@handler(state="fetch", waits_for_input=False, description="Fetch document")
def handle_fetch(state: dict) -> dict:
    """Fetch document from source (simulated)."""
    try:
        # Simulate 30% fetch failure
        import random
        if random.random() < 0.3:
            raise Exception("Simulated fetch failure (network)")
        
        state["raw_data"] = {
            "id": state["document_id"],
            "content": f"Document {state['document_id']} content",
            "created_at": "2024-01-01",
        }
        state["current_state"] = "validate"
        state["audit_trail"].append("FETCH succeeded")
    except Exception as e:
        state["error_message"] = str(e)
        state["retry_count"] += 1
        state["current_state"] = "retry"
        state["audit_trail"].append(f"FETCH failed: {e}")
    
    return state


@handler(state="validate", waits_for_input=True, description="Validate document")
def handle_validate(state: dict) -> dict:
    """Validate document schema and content."""
    # This handler waits for input (human review)
    state["current_state"] = "validate"
    state["audit_trail"].append("Waiting for validation review...")
    return state


@handler(state="enrich", waits_for_input=False, description="Enrich document")
def handle_enrich(state: dict) -> dict:
    """Enrich document with metadata."""
    if not state.get("validated_data"):
        state["error_message"] = "No validated_data; cannot enrich"
        state["current_state"] = "error"
        return state
    
    state["enriched_data"] = {
        "tags": ["important", "processed"],
        "summary": "Document summary",
        "word_count": 150,
        "language": "en",
    }
    state["current_state"] = "store"
    state["audit_trail"].append("ENRICH completed")
    return state


@handler(state="store", waits_for_input=False, description="Store document")
def handle_store(state: dict) -> dict:
    """Persist enriched document."""
    state["current_state"] = "complete"
    state["audit_trail"].append("STORE completed")
    return state


@handler(state="complete", waits_for_input=False, description="Mark complete")
def handle_complete(state: dict) -> dict:
    """Mark pipeline as complete."""
    # Terminal state
    state["audit_trail"].append("Pipeline COMPLETE")
    return state


@handler(state="retry", waits_for_input=False, description="Retry operation")
def handle_retry(state: dict) -> dict:
    """Retry last failed operation."""
    if state.get("retry_count", 0) >= 3:
        state["current_state"] = "error"
        state["error_message"] = "Max retries exceeded"
        state["audit_trail"].append("Max retries reached")
    else:
        state["current_state"] = "fetch"
        state["audit_trail"].append(f"Retry {state['retry_count']}/3")
    return state


@handler(state="error", waits_for_input=False, description="Handle error")
def handle_error(state: dict) -> dict:
    """Handle error state."""
    # Terminal state
    state["audit_trail"].append(f"ERROR: {state.get('error_message', 'Unknown')}")
    return state


@handler(state="human_review", waits_for_input=True, description="Await human review")
def handle_human_review(state: dict) -> dict:
    """Wait for human expert review."""
    # This handler waits for input (blocks until next turn)
    state["audit_trail"].append("Awaiting human review...")
    return state


# Export handler map (used in graph builder)
HANDLER_MAP = {
    state: func
    for state, func in {
        "init": handle_init,
        "fetch": handle_fetch,
        "validate": handle_validate,
        "enrich": handle_enrich,
        "store": handle_store,
        "complete": handle_complete,
        "retry": handle_retry,
        "error": handle_error,
        "human_review": handle_human_review,
    }.items()
}
```

---

### Summary: Phase 2 Deliverables

✅ Extended `PipelineState` with multi-turn fields  
✅ Implemented `invoke_turn()` method  
✅ Implemented `_auto_progress_langgraph()` method  
✅ Implemented `process()` method  
✅ Updated router node with semantic routing dispatch  
✅ Updated handler nodes with @handler decorators  
✅ Tests for all new methods  

**Effort:** ~40-50 hours (4-5 days)

**Quality Checks:**
- All tests pass
- Multi-turn flow works end-to-end
- Semantic router integrated correctly
- Auto-progression through non-blocking states works
- Handler metadata registry used consistently

---

## Remaining Phases (Summary)

### Phase 3: Integration & Testing
- Update `src/main.py` with examples of `process()` and `invoke_turn()`
- Add comprehensive integration tests
- Test multi-turn conversation flows
- Test pause/resume via checkpointing
- Test error handling

**Effort:** ~30-40 hours (3-4 days)

### Phase 4: Documentation
- Update README with multi-turn examples
- Add semantic router customization guide
- Create handler metadata guide
- Document input validation best practices
- Update API documentation

**Effort:** ~15-20 hours (2-3 days)

---

## Running Tests During Implementation

```bash
# Install test dependencies
pip install pytest pytest-cov pytest-asyncio

# Run all tests with coverage
pytest tests/ -v --cov=src --cov-report=term-missing

# Run specific test file
pytest tests/test_handler_registry.py -v

# Run with markers
pytest -m "not slow" -v
```

---

## Quick Reference: File Checklist

### Phase 1
- [ ] `src/engine/handler_registry.py` — NEW
- [ ] `src/engine/router.py` — NEW
- [ ] `src/engine/input_validation.py` — NEW
- [ ] `src/workflow/router.py` — NEW (or UPDATED)
- [ ] Tests for all above

### Phase 2
- [ ] `src/workflow/pipeline_state.py` — UPDATED
- [ ] `src/engine/graph.py` — UPDATED (add methods)
- [ ] `src/workflow/handlers.py` — UPDATED (add @handler decorators)
- [ ] Tests for multi-turn flows

### Phase 3
- [ ] `src/main.py` — UPDATED (add examples)
- [ ] Integration tests

### Phase 4
- [ ] `README.md` — UPDATED
- [ ] `docs/` — NEW guides

---

**Status:** Ready for Phase 1 implementation  
**Next Step:** Start with Task 1.1 (Handler Registry)
