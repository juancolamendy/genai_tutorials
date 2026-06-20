# DOC-3 Implementation: Human Review Path

## Overview

DOC-003 demonstrates the **human review recovery path** - when validation fails, the pipeline routes to human review before continuing to enrichment.

## Setup (in main.py)

```python
# Patch handle_fetch to return malformed data
_original_fetch = handlers.handle_fetch
def _bad_fetch(p):
    bad = {"id": p["document_id"], "content": "Important report text.", "schema_version": ""}
    return audit({**p, "current_state": State.FETCH.value, "raw_data": bad},
                 "fetch OK  (malformed schema_version)")
handlers.handle_fetch = _bad_fetch
handlers.HANDLER_MAP[State.FETCH] = _bad_fetch

# Run DOC-003
wf = build_doc_pipeline(resume_session)
wf.process("DOC-20240619-003")

# Restore original handler
handlers.handle_fetch = _original_fetch
handlers.HANDLER_MAP[State.FETCH] = _original_fetch
```

## State Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                  HAPPY PATH DIVERGES                            │
└─────────────────────────────────────────────────────────────────┘

    INIT
     ↓
    [Router: INIT → FETCH]
     ↓
  [Guardrail: OK → FETCH]
     ↓
  ┌──────────────────────────────┐
  │  FETCH (Patched)             │
  │  raw_data = {                │
  │    "schema_version": ""       │  ← EMPTY! Intentionally malformed
  │  }                           │
  │  └──── FETCH OK ──────       │
  └──────────────────────────────┘
     ↓
  [Router: FETCH → VALIDATE]
     ↓
  [Guardrail: OK → VALIDATE]
     ↓
  ┌──────────────────────────────┐
  │  VALIDATE (LLM)              │
  │  Checks raw_data schema      │
  │  └──── VALIDATION FAILED ──  │
  │  "schema_version is empty"   │
  │  validated_data = None       │  ← FAILS!
  └──────────────────────────────┘
     ↓
  [Router: VALIDATE → ENRICH]
     ↓
  [Guardrail: check_validated_data_present]
  ┌──────────────────────────────┐
  │  ❌ Validation FAILS         │
  │  "validated_data is absent"  │
  │  fallback → HUMAN_REVIEW     │  ← Key divergence!
  └──────────────────────────────┘
     ↓
  ┌──────────────────────────────┐
  │  HUMAN_REVIEW (LLM)          │
  │  Simulates human reviewer    │
  │  Approves & fixes data       │
  │  validated_data = {fixed}    │
  │  └──── APPROVED ──────       │
  └──────────────────────────────┘
     ↓
  [Router: HUMAN_REVIEW → ENRICH]
     ↓
  [Guardrail: OK → ENRICH]
     ↓
  ENRICH → STORE → COMPLETE ✅
```

## Detailed Step Breakdown

### Step 1-2: INIT & FETCH (Malformed)
```
INIT → FETCH
  raw_data = {
    "id": "DOC-20240619-003",
    "content": "Important report text.",
    "schema_version": ""  ← EMPTY!
  }
```

### Step 3: VALIDATE - Detects Schema Error
```python
# In handle_validate()
result = VALIDATE_AGENT.run(f"<raw_data>{raw_json}</raw_data>").content

if result.is_valid:  # FALSE - schema_version is empty
    # ...pass
else:
    # Log validation failure
    log.warning("[VALIDATE] FAILED  issues=%s", issues_str)
    # validated_data remains None
    return audit({**p, "current_state": State.VALIDATE.value, "validated_data": None},
                 f"validate FAILED  issues={issues_str}")

# Result:
current_state = "validate"
validated_data = None  ← KEY: Validation failed
```

### Step 4: Guardrail - Detects Missing validated_data
```python
# In guardrails.py
def check_validated_data_present(state: PipelineState) -> GuardrailResult:
    if state.get("validated_data"):  # None! Fails check
        return GUARDRAIL_PASS
    return GuardrailResult(
        passed=False,
        reason="validated_data is absent; document needs review.",
        fallback=State.HUMAN_REVIEW,  ← Routes to HUMAN_REVIEW
    )

# Result:
proposed_next = "enrich" (from router)
         ↓
proposed_next = "human_review" (from guardrail override)
guardrail_ok = False
audit_trail += ["guardrail FAIL → enrich (validated_data is absent) → fallback human_review"]
```

### Step 5: HUMAN_REVIEW Handler
```python
# In handle_human_review()
log.warning("[REVIEW] 🔍 doc_id=%s routing to human review", p["document_id"])
raw_json = str(p.get("raw_data", {}))

try:
    decision = REVIEW_AGENT.run(f"<raw_data>{raw_json}</raw_data>").content
    
    if decision.approved:  # LLM simulates human approval
        validated = {
            **decision.fixed_data,  # LLM "fixed" the data
            "_human_approved": True,
            "_validated": True,
        }
        return audit({**p, "current_state": State.HUMAN_REVIEW.value,
                      "validated_data": validated},
                     f"human_review: APPROVED  note='{decision.reviewer_note}'")
except Exception as exc:
    # Handle error
    log.error("[REVIEW] exception: %s", exc)
    # Route to ERROR state

# Result:
current_state = "human_review"
validated_data = {fixed data with human approval}  ← NOW HAS DATA
_human_approved = True
_validated = True
```

### Steps 6-9: Happy Path Resumes (ENRICH → STORE → COMPLETE)
```
HUMAN_REVIEW → ENRICH (LLM enriches approved data)
             → STORE (Persist enriched data)
             → COMPLETE ✅
```

## Audit Trail Analysis

```
Audit Trail (13 entries):
  • init  doc_id=DOC-20240619-003
  • guardrail PASS → fetch                                    ← Route to FETCH
  • fetch OK  (malformed schema_version)                      ← Fetch succeeds (but data is bad)
  • guardrail PASS → validate                                 ← Route to VALIDATE
  ┌─── VALIDATION FAILS ──────────────────────────────────────┐
  │ • validate FAILED  issues=schema_version field is empty  │
  │ • guardrail FAIL → enrich (validated_data is absent)     │
  │                  → fallback human_review                 │
  └──────────────────────────────────────────────────────────┘
  ┌─── HUMAN REVIEW APPROVES ─────────────────────────────────┐
  │ • human_review: APPROVED                                 │
  └──────────────────────────────────────────────────────────┘
  ┌─── HAPPY PATH RESUMES ────────────────────────────────────┐
  │ • guardrail PASS → enrich                                │
  │ • enrich OK  tags=[...]  lang=en                         │
  │ • guardrail PASS → store                                 │
  │ • store OK  record_id=rec-DOC-20240619-003-...          │
  │ • guardrail PASS → complete                             │
  │ • COMPLETE ✅                                            │
  └──────────────────────────────────────────────────────────┘
```

## Key Mechanisms

### 1. Validation Failure Detection
```python
# LLM detects schema violations
if result.is_valid:  # LLM determines if valid
    # Pass
else:
    # Fail - validated_data remains None
```

**Why?** Ensures data quality before proceeding. Can't enrich invalid data!

### 2. Guardrail-Driven Human Review Route
```
Validation fails → validated_data is None
    ↓
Guardrail check_validated_data_present() fails
    ↓
Guardrail's fallback state = HUMAN_REVIEW
    ↓
Router dispatches to HUMAN_REVIEW handler (not ENRICH)
```

**Why?** Lets humans decide what to do with invalid data instead of blocking.

### 3. Human Review Approval (LLM-Simulated)
```python
# In handle_human_review()
decision = REVIEW_AGENT.run(...)  # LLM simulates reviewer

if decision.approved:
    # Use fixed_data from reviewer
    validated = {**decision.fixed_data, "_human_approved": True}
    # Continue to ENRICH with approved data
```

**Why?** Enables recovery path. Human (or LLM) can fix data and approve continuation.

### 4. Resume Happy Path
After human review approves:
```
HUMAN_REVIEW ✅
    ↓
[Router: HUMAN_REVIEW → ENRICH]
    ↓
ENRICH → STORE → COMPLETE
```

**Why?** Once data is approved, process continues normally with enrichment.

## Comparison: All Three Paths

| Path | Trigger | Resolution | Final State |
|------|---------|-----------|-------------|
| **DOC-001** (Happy) | N/A | Direct VALIDATE → ENRICH | COMPLETE ✅ |
| **DOC-002** (Retry) | Fetch fails | RETRY with clean state → Fetch again | COMPLETE ✅ |
| **DOC-003** (Review) | Validation fails | HUMAN_REVIEW approves → Continue | COMPLETE ✅ |

## Summary

**DOC-3 Workflow:** Demonstrates human review recovery path

| Phase | What Happens |
|-------|--------------|
| **Setup** | Patch FETCH to return malformed data |
| **FETCH** | Returns data with empty schema_version |
| **VALIDATE** | LLM validation fails, returns None |
| **Guardrail** | Detects missing validated_data, routes to HUMAN_REVIEW |
| **HUMAN_REVIEW** | LLM approves & fixes data |
| **Happy Path** | ENRICH → STORE → COMPLETE ✅ |

**Key Insight:** The system handles validation failures gracefully by routing to human review, allowing data to be fixed and approved before continuing processing.

## All Three Scenarios Implemented

✅ **DOC-001**: Happy path (no issues)  
✅ **DOC-002**: Retry path (transient failure)  
✅ **DOC-003**: Human review path (validation failure)  

All three demonstrate different error recovery mechanisms within the same state machine!
