# DOC-2 Workflow: Retry Path Explained

## Overview

DOC-002 demonstrates the **retry recovery path** - when a document fetch fails transiently, the pipeline automatically retries without manual intervention.

## Setup

```python
run_doc("DOC-20240619-002", seed=0)   # seed=0 → random() < 0.25 (FORCES FAILURE)
```

The `seed=0` causes `random.random() < 0.25` to be TRUE on the first attempt, triggering a simulated transient fetch failure.

## State Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    ATTEMPT #1 (FAILS)                           │
└─────────────────────────────────────────────────────────────────┘

    INIT
     ↓
    [Router: INIT → FETCH]
     ↓
  [Guardrail: OK → FETCH]
     ↓
  ┌──────────────────────────────┐
  │  FETCH (Attempt #1)          │
  │  • retry_count = 0           │
  │  • random.random() < 0.25    │  ← TRUE! Simulated failure
  │  • raw_data = None           │  ← NOT fetched
  │  └──── FETCH FAILED ──────   │
  └──────────────────────────────┘
     ↓
  [Router: FETCH → VALIDATE]
     ↓
  [Guardrail: check_raw_data_present]
  ┌──────────────────────────────┐
  │  ❌ Validation FAILS         │
  │  "raw_data is absent"        │
  │  fallback → RETRY            │
  └──────────────────────────────┘
     ↓
  ┌──────────────────────────────┐
  │  RETRY Handler               │
  │  • retry_count = 0 → 1       │
  │  • raw_data = None (clear)   │
  │  └────── RETRY #1 ──────     │
  └──────────────────────────────┘
     ↓
  [Router: RETRY → FETCH]
     ↓
  [Guardrail: OK → FETCH]

┌─────────────────────────────────────────────────────────────────┐
│                    ATTEMPT #2 (SUCCEEDS)                        │
└─────────────────────────────────────────────────────────────────┘

  ┌──────────────────────────────┐
  │  FETCH (Attempt #2)          │
  │  • retry_count = 1           │
  │  • Condition: retry_count==0 │  ← FALSE! (count is now 1)
  │  • raw_data = {...fetched}   │  ← Successfully fetched
  │  └────── FETCH OK ──────     │
  └──────────────────────────────┘
     ↓
  [Router: FETCH → VALIDATE]
     ↓
  [Guardrail: check_raw_data_present]
  ┌──────────────────────────────┐
  │  ✅ Validation PASSES        │
  │  "raw_data is present"       │
  └──────────────────────────────┘
     ↓
  VALIDATE (LLM validates schema)
     ↓
  ENRICH (LLM enriches with tags, summary, metadata)
     ↓
  STORE (Persists to document store)
     ↓
  COMPLETE ✅
```

## Detailed Step Breakdown

### Step 1: INIT
```
State: INIT
Action: Initialize fresh pipeline state
Next: FETCH (via happy path router)
```

### Step 2: FETCH - Attempt #1 (FAILS)
```python
# In handle_fetch()
if random.random() < 0.25 and p["retry_count"] == 0:  # TRUE!
    log.warning("[FETCH] transient failure — will retry")
    return audit({**p, "current_state": State.FETCH.value, "raw_data": None},
                 "fetch FAILED (simulated transient error)")

# Result:
current_state = "fetch"
raw_data = None  ← KEY: No data fetched
retry_count = 0
audit_trail = ["init", "guardrail PASS", "fetch FAILED (simulated transient error)"]
```

### Step 3: Guardrail - check_raw_data_present
```python
# In guardrails.py
def check_raw_data_present(state: PipelineState) -> GuardrailResult:
    if state.get("raw_data"):  # None! Fails check
        return GUARDRAIL_PASS
    return GuardrailResult(
        passed=False,
        reason="raw_data is absent; fetch may have failed.",
        fallback=State.RETRY,  ← Redirects to RETRY instead of VALIDATE
    )

# Result:
proposed_next = "fetch" (from router)
         ↓
proposed_next = "retry" (from guardrail override)
guardrail_ok = False
```

### Step 4: RETRY Handler
```python
# In handle_retry()
new_count = p["retry_count"] + 1  # 0 → 1
return audit({
    **p,
    "current_state": State.RETRY.value,
    "retry_count": new_count,  ← INCREMENTED
    "raw_data": None,  ← CLEARED (fresh attempt)
}, f"retry #1 — clearing stale payload")

# Result:
current_state = "retry"
retry_count = 1  ← Important: Now retry_count > 0
raw_data = None  ← Cleared for fresh fetch
```

### Step 5: FETCH - Attempt #2 (SUCCEEDS)
```python
# In handle_fetch()
if random.random() < 0.25 and p["retry_count"] == 0:  # FALSE!
    # retry_count is now 1, so condition fails
    # No simulated failure this time!

# So we proceed to fetch successfully:
raw = {
    "id": "DOC-20240619-002",
    "content": "Full text of document DOC-20240619-002. Lorem ipsum dolor sit amet.",
    "schema_version": "2.1",
    "source": "document-store-v2",
}
return audit({**p, "current_state": State.FETCH.value, "raw_data": raw},
             f"fetch OK  schema_version=2.1")

# Result:
current_state = "fetch"
raw_data = {...complete document}  ← SUCCESS!
retry_count = 1
```

### Steps 6-9: Happy Path (VALIDATE → ENRICH → STORE → COMPLETE)
```
VALIDATE: LLM validates raw_data → validated_data ✅
    ↓
ENRICH: LLM enriches with tags, summary, metadata → enriched_data ✅
    ↓
STORE: Persist enriched_data to document store ✅
    ↓
COMPLETE: Terminal success state ✅
```

## Key Mechanisms

### 1. Transient Failure Simulation
```python
# Only fails on first attempt
if random.random() < 0.25 and p["retry_count"] == 0:
    # Fail
else:
    # Succeed
```

**Why?** Models transient failures that might happen on first attempt but succeed on retry.

### 2. Guardrail-Driven Retry
```
Fetch fails → raw_data is None
    ↓
Guardrail check_raw_data_present() fails
    ↓
Guardrail's fallback state = RETRY
    ↓
Router dispatches to RETRY handler (not VALIDATE)
```

**Why?** Validates data consistency before proceeding. Can't validate without raw data!

### 3. Retry Counter Reset
```python
# In RETRY handler
"raw_data": None,   # CLEAR stale data

# In FETCH handler condition
if random.random() < 0.25 and p["retry_count"] == 0:
                                  ↑
                    Only fails if retry_count == 0
```

**Why?** Ensures:
- Only first attempt can trigger simulated failure
- Second attempt (retry_count=1) always succeeds
- Prevents infinite retry loops

## Audit Trail

```
  Audit Trail (13 entries):
    • init  doc_id=DOC-20240619-002
    • guardrail PASS → fetch
    • fetch FAILED (simulated transient error)          ← Attempt #1 FAILS
    • guardrail FAIL → fetch → fallback retry
    • retry #1 — clearing stale payload
    • guardrail PASS → fetch
    • fetch OK  schema_version=2.1                      ← Attempt #2 SUCCEEDS
    • guardrail PASS → validate
    • validate OK  issues=[]
    • guardrail PASS → enrich
    • enrich OK  tags=[...]  lang=...
    • guardrail PASS → store
    • store OK  record_id=...
    • guardrail PASS → complete
    • COMPLETE ✅
```

## Summary

**DOC-2 Workflow:** Demonstrates automatic retry recovery

| Phase | What Happens |
|-------|--------------|
| **Setup** | Seed random(0) to force failure |
| **Attempt #1** | FETCH fails with simulated error, raw_data is None |
| **Guardrail** | Detects missing raw_data, redirects to RETRY |
| **RETRY** | Increments retry_count (0→1), clears stale data |
| **Attempt #2** | FETCH succeeds (retry_count≠0 bypasses failure condition) |
| **Happy Path** | VALIDATE → ENRICH → STORE → COMPLETE ✅ |

**Key Insight:** The system gracefully recovers from transient failures through guardrail validation and automatic retry, without human intervention.
