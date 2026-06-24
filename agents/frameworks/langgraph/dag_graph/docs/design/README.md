# LangGraph DAG Graph Design Documents

**Current Version:** 1.0  
**Date:** 2026-06-24  
**Status:** Design Phase Complete, Ready for Implementation

---

## Overview

This directory contains design specifications for extending the **LangGraph DAG Graph** implementation with features from the **Agno DAG Graph** architecture (36 recent commits).

The design maps Agno's proven patterns onto LangGraph's native `StateGraph` and `checkpointer` foundation, providing:

1. **Multi-turn workflows** with conversation history and pause/resume
2. **Semantic routing** via LLM-powered state transitions
3. **Handler metadata registry** with `@handler` decorator
4. **Input validation** with prompt injection prevention
5. **Auto-progression** through non-blocking states
6. **Engine layer base class** with `process()` and `invoke_turn()` methods

---

## Document Guide

### 1. [Feature Mapping & Design Specification](01-feature-mapping-langgraph.md)

**Audience:** Architects, Tech Leads, Decision Makers

**Contents:**
- Executive summary of features
- Detailed feature-by-feature design
- Comparison of Agno vs LangGraph approaches
- File structure changes
- Design rationale and trade-offs
- Known limitations and mitigations
- Testing strategy

**Read this if you want to:**
- Understand what features are being added
- See design rationale and architectural decisions
- Review API contracts before implementation
- Understand how Agno patterns map to LangGraph

**Key Takeaways:**
- 8 major features from 36 Agno commits
- ~4 implementation phases over 3-4 weeks
- LangGraph's checkpointer + StateGraph are better foundation than Agno's custom workflow loop
- Backward compatible with existing `run_pipeline()` calls
- Semantic router is opt-in (only activated if `self.router` exists)

---

### 2. [Implementation Guide](02-implementation-guide.md)

**Audience:** Developers, Implementation Engineers

**Contents:**
- Quick start: what needs to change
- Phase-by-phase implementation with code examples
- Detailed task breakdowns with effort estimates
- Complete code samples for all new modules
- Unit and integration test examples
- Running tests and quality checks
- File checklist for each phase

**Read this if you want to:**
- Implement the features step-by-step
- See complete code examples
- Understand exact file changes needed
- Know how to test each component

**Quick Ref:**
- **Phase 1 (Days 1-5):** Handler registry, routers, input validation
- **Phase 2 (Days 6-10):** Multi-turn support, semantic routing, auto-progression
- **Phase 3 (Days 11-14):** Integration, end-to-end tests
- **Phase 4 (Days 15-18):** Documentation and guides

---

## Architecture Diagram

```
CURRENT STATE (LangGraph v1)
───────────────────────────

    main.py
        ↓
    run_pipeline(doc_id, timeout, thread_id)
        ↓
    StateMachineGraph.invoke()
        ↓
    LangGraph StateGraph
        ↓
    [Router] → [Guardrail] → [Handler] → loop
        ↓
    pure code routing only
    single-turn execution


TARGET STATE (LangGraph v2)
──────────────────────────

    main.py
        ├─ run_pipeline() [backward compat]
        │   ↓
        │   process(entity_id)
        │
        └─ invoke_turn() [new]
            ↓
        StateMachineGraph.invoke_turn()
            ↓
        LangGraph StateGraph
            ↓
        [SemanticRouter] or [PureCodeRouter]
            ↓
        [Guardrail] → [Handler]
            ↓
        [AutoProgress] → loop
            ↓
        multi-turn with conversation_history
        conversation_history tracked across turns
        pause/resume via checkpointer + thread_id
        LLM-powered state transitions (optional)
```

---

## Key Changes by Module

### New Modules (Phase 1)

| Module | Purpose | Lines |
|--------|---------|-------|
| `engine/handler_registry.py` | @handler decorator + metadata registry | ~100 |
| `engine/router.py` | BaseSemanticRouter + DefaultSemanticRouter | ~250 |
| `engine/input_validation.py` | Input validation + prompt injection prevention | ~150 |
| `workflow/router.py` | DocPipelineRouter (domain-specific) | ~80 |

### Updated Modules (Phase 2)

| Module | Changes | Impact |
|--------|---------|--------|
| `engine/graph.py` | Add `invoke_turn()`, `process()`, `_auto_progress_langgraph()` | +200 LOC |
| `workflow/pipeline_state.py` | Add multi-turn fields (turn_input, conversation_history, etc.) | +10 fields |
| `workflow/handlers.py` | Add @handler decorators to existing handlers | Minimal |
| `workflow/state_machine.py` | No changes | — |
| `workflow/guardrails.py` | No changes | — |

---

## API Changes

### New Entry Points

```python
# One-turn execution (replaces run_pipeline)
result = graph.process(entity_id="doc-001")
# → {"current_state": "complete", "audit_trail": [...], ...}

# Multi-turn execution (new)
response1 = graph.invoke_turn("user1", "session1", "Start processing")
# → {"current_state": "validate", "waits_for_input": True, "turn_number": 1, ...}

response2 = graph.invoke_turn("user1", "session1", "Looks good, proceed")
# → {"current_state": "store", "waits_for_input": False, "turn_number": 2, ...}

response3 = graph.invoke_turn("user1", "session1", "")
# → {"current_state": "complete", "waits_for_input": False, "turn_number": 3, ...}
```

### Handler Declaration (Before vs After)

**Before:**
```python
def handle_validate(state: PipelineState) -> PipelineState:
    ...
```

**After:**
```python
@handler(state="validate", waits_for_input=False)
def handle_validate(state: PipelineState) -> PipelineState:
    ...
```

### Custom Router (Before vs After)

**Before:**
```python
# No semantic routing; pure code only
routing_table = {"init": "fetch", "fetch": "validate", ...}
```

**After:**
```python
class MyRouter(DefaultSemanticRouter):
    output_schema = MyRouterOutput
    
    def get_instructions(self):
        return "You are a router..."

workflow.router = MyRouter()
```

---

## Implementation Timeline

| Phase | Focus | Duration | Status |
|-------|-------|----------|--------|
| 1 | Handler registry, routers, input validation | 4-5 days | Ready |
| 2 | Multi-turn support, semantic routing, auto-progression | 4-5 days | Ready |
| 3 | Integration, end-to-end tests, examples | 3-4 days | Ready |
| 4 | Documentation, guides, API examples | 2-3 days | Ready |
| **Total** | — | **13-17 days** | — |

---

## Feature Comparison: Agno vs LangGraph

| Feature | Agno | LangGraph v2 |
|---------|------|-------------|
| **State Machine** | Custom Loop class | LangGraph StateGraph |
| **Persistence** | JsonDb custom format | LangGraph SqliteCheckpointer |
| **Multi-turn** | `process_turn()` method | `invoke_turn()` + `checkpointer` |
| **Semantic Routing** | Agno agent (via agent.py) | LangChain ChatAnthropic + Pydantic |
| **Handler Metadata** | @handler decorator | @handler decorator (same) |
| **Input Validation** | Engine layer | Engine layer (same) |
| **Auto-progression** | `_auto_progress()` loop | `_auto_progress_langgraph()` loop |
| **Response Building** | `_build_response()` + `_build_turn_response()` | Same pattern |
| **Integration** | Agno-specific | LangChain ecosystem |
| **Visualization** | Manual logging | Native `.get_graph().draw_*()` |

---

## Risk Assessment & Mitigations

### Risk 1: Semantic Router Latency
**Problem:** LLM call adds 1-3s per turn  
**Impact:** Multi-turn UX degradation  
**Mitigation:** Use Claude Haiku (cheaper), optional caching, fallback to pure code

### Risk 2: Token Counting Accuracy
**Problem:** tiktoken differs from Claude's tokenizer  
**Impact:** Validation may pass/fail inconsistently  
**Mitigation:** Conservative estimates, monitor actual usage, threshold adjustment

### Risk 3: Conversation History Growth
**Problem:** Unbounded history → OOM risk  
**Impact:** Long conversations become slow  
**Mitigation:** Auto-trim to max_history_turns, optional summarization

### Risk 4: Prompt Injection Escaping
**Problem:** No 100% safe escaping  
**Impact:** Potential security vulnerabilities  
**Mitigation:** Multi-layer defense, validation on output, audit logging

### Risk 5: Breaking Changes
**Problem:** Existing code using old API fails  
**Impact:** Backward compatibility broken  
**Mitigation:** `process()` wraps existing `run_pipeline()`, optional features

---

## Success Criteria

### Functional Requirements
- [ ] Multi-turn conversations work end-to-end
- [ ] Semantic router makes valid state decisions
- [ ] Input validation prevents injection attacks
- [ ] Auto-progression skips through non-blocking states
- [ ] Pause/resume via checkpointer works across turns
- [ ] Handler metadata decorators functional
- [ ] Response builders return correct dicts

### Non-Functional Requirements
- [ ] All tests pass (unit + integration)
- [ ] 80%+ code coverage
- [ ] No linting errors (ruff)
- [ ] Type hints complete
- [ ] Docstrings follow style
- [ ] Performance: < 5s per turn (including LLM)
- [ ] Memory: < 100MB for 100-turn session

### Documentation Requirements
- [ ] README updated with multi-turn examples
- [ ] API documentation complete
- [ ] Handler customization guide
- [ ] Semantic router setup guide
- [ ] Input validation best practices

---

## Getting Started

### For Reviewers
1. Read the Executive Summary in [01-feature-mapping-langgraph.md](01-feature-mapping-langgraph.md)
2. Review the Architecture Diagram above
3. Check the Risk Assessment section

### For Implementers
1. Start with Phase 1 in [02-implementation-guide.md](02-implementation-guide.md)
2. Implement tasks 1.1 → 1.4 with tests
3. Verify all Phase 1 tests pass
4. Move to Phase 2 tasks

### For QA/Testing
1. Review testing strategy in [01-feature-mapping-langgraph.md](01-feature-mapping-langgraph.md#part-9-testing-strategy)
2. Review test examples in [02-implementation-guide.md](02-implementation-guide.md)
3. Create test plan covering all phases

---

## Glossary

| Term | Definition |
|------|-----------|
| **Multi-turn** | Workflow that pauses and waits for user input between state transitions |
| **Semantic routing** | LLM-powered decision making for next state based on user input |
| **Handler metadata** | Configuration attached to handler functions via @decorator |
| **Auto-progression** | Automatic continuation through non-blocking states without user input |
| **Prompt injection** | Attack where user input tricks LLM into ignoring instructions |
| **Waits for input** | Handler/state that pauses workflow and waits for next turn |
| **Non-blocking** | State that completes automatically without waiting (waits_for_input=False) |
| **Checkpointer** | LangGraph's persistence mechanism for saving/resuming execution state |
| **Thread ID** | Unique identifier for a conversation; used by checkpointer to resume |
| **Semantic context** | Entities and intents extracted from user input by router |

---

## References

### Agno Implementation
- Agno dag_graph project: `/Users/jcolamendy/ai_ml/genai_tutorials/agents/frameworks/agno-tutorials/dag_graph/`
- Key commits: Last 36 commits in Agno repo
- Commit summary: 02-implementation-guide.md Table 1

### LangGraph Documentation
- LangGraph: https://python.langchain.com/docs/langgraph/
- StateGraph: https://python.langchain.com/docs/langgraph/concepts/low_level/
- Checkpointer: https://python.langchain.com/docs/langgraph/concepts/persistence/

### Current Implementation
- LangGraph dag_graph: `/Users/jcolamendy/ai_ml/genai_tutorials/agents/frameworks/langgraph/dag_graph/`
- Current codebase: ~2000 lines (src/), ~500 lines (tests/)

---

## Notes & Considerations

### Design Philosophy
- **Leverage LangGraph's Strengths:** Use StateGraph and checkpointer natively
- **Adopt Agno Patterns:** Handler metadata, semantic routing, auto-progression
- **Keep It Simple:** Minimize custom code; prefer LangChain/LangGraph primitives
- **Backward Compatible:** Existing code continues to work
- **Production Ready:** Security, error handling, logging from day one

### Open Questions (For Discussion)
1. Should semantic router be default or opt-in? → **Opt-in** (better for one-turn flows)
2. What's max_history_turns default? → **10 turns** (balance memory vs context)
3. Should input validation be strict or lenient? → **Strict** (better for security)
4. How to handle semantic context in guardrails? → **Pass as state field** (already in design)
5. Should auto-progression be max iterations or timeout? → **Max iterations** (simpler, safer)

### Future Enhancements (Not In Scope)
- Summarization of old turns before trimming
- Caching of semantic router decisions
- Branching workflows (multiple valid next states)
- Conditional routing based on extracted entities
- Role-based access control for handlers

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-06-24 | Claude Code | Initial design documents |

---

## Questions or Feedback?

- Feature scope: See [01-feature-mapping-langgraph.md](01-feature-mapping-langgraph.md)
- Implementation details: See [02-implementation-guide.md](02-implementation-guide.md)
- Specific modules: Check individual file headers in implementation guide

---

**Status:** Ready for stakeholder review and implementation planning
