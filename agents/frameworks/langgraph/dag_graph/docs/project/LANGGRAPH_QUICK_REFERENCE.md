# LangGraph Quick Reference: Checkpointing & Multi-Turn

## 1. Core Concepts at a Glance

| Concept | Purpose | Scope | Use Case |
|---------|---------|-------|----------|
| **Checkpointer** | Save graph state snapshots | Single thread | Conversation continuity, fault recovery |
| **Store** | Save application key-value data | Cross-thread | User preferences, shared facts |
| **thread_id** | Conversation identifier | Graph-scoped | Multi-turn conversation tracking |
| **interrupt()** | Pause execution for input | Node-level | Human-in-the-loop workflows |

---

## 2. Quick Start: Multi-Turn Conversation

```python
from langgraph.graph import MessagesState, StateGraph
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import HumanMessage

# Define state
class State(MessagesState):
    pass

# Define node
def chat(state):
    messages = state["messages"]
    response = model.invoke(messages)
    return {"messages": [response]}

# Build graph with persistence
builder = StateGraph(State)
builder.add_node("chat", chat)
builder.set_entry_point("chat")
builder.set_finish_point("chat")

graph = builder.compile(checkpointer=InMemorySaver())

# Use same thread_id to maintain conversation
config = {"configurable": {"thread_id": "user_123"}}

# Turn 1
graph.invoke({"messages": [HumanMessage(content="Hi")]}, config)

# Turn 2 (remembers Turn 1)
graph.invoke({"messages": [HumanMessage(content="Who am I?")]}, config)
```

---

## 3. Storage Backends Comparison

### Development
```python
from langgraph.checkpoint.memory import InMemorySaver
checkpointer = InMemorySaver()
```

### SQLite (Local Development)
```python
from langgraph_checkpoint_sqlite import SqliteSaver
import sqlite3
checkpointer = SqliteSaver(conn=sqlite3.connect("checkpoints.db"))
```

### PostgreSQL (Production)
```python
from langgraph_checkpoint_postgres import PostgresSaver
import psycopg
conn = await psycopg.AsyncConnection.connect("postgresql://localhost/langgraph")
checkpointer = PostgresSaver(async_connection=conn)
```

### Azure Cosmos DB (Azure Deployments)
```python
from langchain_azure_cosmosdb import CosmosDBSaver
checkpointer = CosmosDBSaver(
    cosmos_endpoint="https://your-account.documents.azure.com",
    cosmos_key="your-key",
    database_name="langgraph",
    container_name="checkpoints"
)
```

---

## 4. Thread ID Patterns

```python
# User-specific
thread_id = f"user_{user_id}"

# Conversation-specific
thread_id = f"user_{user_id}_conv_{conversation_id}"

# Hierarchical (for organization)
thread_id = f"{org_id}:{user_id}:{conversation_id}"

# UUID-based
import uuid
thread_id = str(uuid.uuid4())
```

---

## 5. State Retrieval Across Turns

```python
# Get current state
state = graph.get_state(
    config={"configurable": {"thread_id": "user_123"}}
)

# List all checkpoints (history)
history = graph.list_state(
    config={"configurable": {"thread_id": "user_123"}}
)

for checkpoint in history:
    print(f"Step: {checkpoint.step}, Time: {checkpoint.timestamp}")
```

---

## 6. Human-in-the-Loop Pattern

```python
from langgraph.types import Interrupt, Command

def approval_node(state):
    # Pause and wait for human input
    decision = interrupt(
        value={
            "question": "Approve this action?",
            "action": state["pending_action"]
        }
    )
    
    # Resumed with decision
    return {"approved": decision == "yes"}

# Build graph
builder = StateGraph(State)
builder.add_node("approval", approval_node)
# ... add more nodes
graph = builder.compile(checkpointer=PostgresSaver(...))

# Initial execution (pauses)
config = {"configurable": {"thread_id": "task_123"}}
try:
    result = graph.invoke(input_data, config)
except Interrupt as e:
    pause_data = e.value

# Later: Resume with decision
result = graph.invoke(
    Command(resume="yes"),
    config  # Same thread_id!
)
```

---

## 7. Streaming for Multi-Turn Interactions

```python
config = {"configurable": {"thread_id": "user_123"}}

# Stream with checkpoints
async for chunk in graph.astream_events(user_input, config, version="v3"):
    if chunk.event == "on_messages":
        # Token-by-token LLM output
        print(chunk.data["messages"][-1].content, end="", flush=True)
    
    elif chunk.event == "on_checkpoint":
        # State persisted here
        current_state = chunk.data
    
    elif chunk.event == "on_interrupts":
        # Pause point reached
        interrupts = chunk.data["interrupts"]
```

---

## 8. Store for Cross-Thread Data

```python
from langgraph.store.memory import InMemoryStore

store = InMemoryStore()
graph = builder.compile(checkpointer=checkpointer, store=store)

# In a node
def my_node(state, *, store):
    # Read user preferences (shared across conversations)
    prefs = store.get("user_123", ("preferences",))
    
    # Write shared facts
    store.put(("user_123", "facts"), {"favorite_color": "blue"})
    
    return {"result": ...}
```

---

## 9. Critical Rules for Interrupts

```python
# ✓ DO: Interrupt at top level
def good_node(state):
    decision = interrupt(value="What do you think?")
    return {"decision": decision}

# ✗ DON'T: Wrap interrupt in try/except
def bad_node(state):
    try:
        decision = interrupt(value="...")
    except Exception:
        decision = None

# ✓ DO: Idempotent operations before interrupt
def safe_node(state):
    data = read_only_operation(state)
    response = interrupt(value=data)
    # Side effects AFTER resumption
    if response == "approved":
        db.create_record(data)

# ✓ DO: Use edges for validation loops
# ✗ DON'T: Loop interrupt() inside node
```

---

## 10. Production Checklist

- [ ] Use **PostgresSaver** for production (not InMemory)
- [ ] Deploy checkpointer and store to different backends if scale requires
- [ ] Implement **thread_id** strategy for your domain
- [ ] Set up **database backups and replication**
- [ ] Design **node pre-interrupt operations to be idempotent**
- [ ] Place **side effects after interrupts** or in separate nodes
- [ ] Monitor checkpoint operation **latency and throughput**
- [ ] Test **fault recovery procedures**
- [ ] Configure **streaming events** for observability
- [ ] Document **thread_id naming conventions**

---

## 11. Common Patterns

### Simple Chatbot
```python
graph.invoke(
    {"messages": [HumanMessage(content=user_input)]},
    {"configurable": {"thread_id": user_id}}
)
```

### Multi-Turn Agent
```python
config = {"configurable": {"thread_id": f"{user_id}:{task_id}"}}
for turn in range(max_turns):
    result = graph.invoke(
        {"messages": [HumanMessage(content=next_input)]},
        config  # Automatic state restoration
    )
```

### Fault-Tolerant Execution
```python
config = {"configurable": {"thread_id": "task_123"}}
try:
    result = graph.invoke(input_data, config)
except Exception:
    # State preserved, safe to retry
    result = graph.invoke(input_data, config)  # Resumes from saved state
```

### Human Review Loop
```python
config = {"configurable": {"thread_id": request_id}}
# Execute until interrupt
result = graph.invoke(initial_data, config)
# Get human decision
decision = await get_human_input()
# Resume with decision
result = graph.invoke(Command(resume=decision), config)
```

---

## 12. API Methods Reference

### Graph Execution
```python
# Synchronous
result = graph.invoke(input_data, config)

# Asynchronous
result = await graph.ainvoke(input_data, config)

# Streaming
for chunk in graph.stream(input_data, config):
    ...

# Event streaming (recommended)
for event in graph.stream_events(input_data, config, version="v3"):
    if event.event == "on_checkpoint":
        ...
```

### State Management
```python
# Get current thread state
state = graph.get_state(config)

# List checkpoints
history = graph.list_state(config)

# Update state (if supported)
graph.update_state(config, new_state)
```

### Checkpointer Interface
```python
# Methods all checkpointers implement
checkpointer.put(checkpoint)           # Save state
checkpointer.put_writes(writes)        # Save intermediate outputs
checkpointer.get_tuple(config)         # Retrieve checkpoint
checkpointer.list(config)              # List checkpoints

# Async variants (non-blocking)
await checkpointer.aput(checkpoint)
await checkpointer.aget_tuple(config)
```

---

## 13. Debugging Tips

```python
# Get execution history
history = graph.list_state(config)
print(f"Total turns: {len(history)}")

# Inspect current state
state = graph.get_state(config)
print(f"Messages: {len(state.values['messages'])}")
print(f"Next nodes: {state.next}")

# Check last checkpoint
last = history[-1]
print(f"Last step: {last.step}, Time: {last.timestamp}")

# Stream with debug mode
for event in graph.stream_events(
    input_data,
    config,
    version="v3"
):
    if event.event == "on_debug":
        print(f"Debug: {event.data}")
```

---

## 14. Performance Tips

```python
# Use async checkpointer for high throughput
from langgraph_checkpoint_postgres import AsyncPostgresSaver

# Use streaming to avoid blocking on full execution
async for chunk in graph.astream_events(...):
    # Process tokens as they arrive
    pass

# Batch operations where possible
# Use connection pooling for database backends

# Monitor checkpoint latency
import time
start = time.time()
graph.invoke(input_data, config)
checkpoint_time = time.time() - start
```

---

## 15. Version & Resources

**Current Version:** LangGraph 1.2.6+

**Documentation:** https://docs.langchain.com/oss/python/langgraph/  
**Persistence Guide:** `/oss/python/langgraph/persistence`  
**Interrupts Guide:** `/oss/python/langgraph/interrupts`  
**GitHub:** https://github.com/langchain-ai/langgraph  
**Community Forum:** https://discourse.langchain.com
