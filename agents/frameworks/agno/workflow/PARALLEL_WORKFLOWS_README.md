# Parallel Workflow Examples

This directory contains examples demonstrating parallel task execution using the Agno framework.

## 📁 Files

### 1. `parallel_workflow.py` (Basic Example)
**Purpose**: Simple introduction to parallel execution

**Structure**:
- 2 agents run in parallel (HackerNews + Web research)
- 1 synthesizer combines results

**Use Case**: Quick parallel execution with minimal complexity

```bash
python parallel_workflow.py
```

---

### 2. `parallel_workflow_advanced.py` (Multi-Stage Example)
**Purpose**: Demonstrates complex multi-stage parallel orchestration

**Structure**:
- **Stage 1**: 3 researchers run in parallel
  - Tech News Researcher (HackerNews)
  - Market Researcher (Web)
  - Financial Analyst (Stock data)
- **Stage 2**: 2 analysts run in parallel
  - Technical Analyst
  - Business Analyst
- **Stage 3**: 1 executive synthesizer

**Use Case**: Complex workflows with multiple parallel stages

```bash
python parallel_workflow_advanced.py
```

---

### 3. `parallel_vs_sequential.py` (Performance Comparison)
**Purpose**: Shows performance benefits of parallel execution

**Structure**:
- Runs the same workflow in two modes:
  - Sequential: Tasks run one after another
  - Parallel: Tasks run simultaneously
- Measures and compares execution time

**Use Case**: Understanding when to use parallel vs sequential execution

```bash
python parallel_vs_sequential.py
```

---

## 🎯 Key Concepts

### When to Use Parallel Execution

✅ **Use `Parallel()` when:**
- Tasks are **independent** (don't need each other's output)
- Tasks involve I/O operations (API calls, web requests, database queries)
- You want to **reduce total execution time**
- Multiple data sources need to be queried simultaneously

❌ **Don't use `Parallel()` when:**
- Tasks **depend on previous results**
- Tasks must execute in a **specific order**
- Tasks share mutable state that could cause race conditions

### Parallel Execution Pattern

```python
from agno.workflow import Parallel, Step, Workflow

workflow = Workflow(
    steps=[
        # These run simultaneously
        Parallel(
            Step(name="Task 1", agent=agent1),
            Step(name="Task 2", agent=agent2),
            Step(name="Task 3", agent=agent3),
            name="Parallel Phase"
        ),
        # This runs after all parallel tasks complete
        Step(name="Synthesis", agent=synthesizer),
    ]
)
```

### Multi-Stage Parallel Pattern

```python
workflow = Workflow(
    steps=[
        # Stage 1: Parallel Research
        Parallel(
            Step(name="Research A", agent=researcher_a),
            Step(name="Research B", agent=researcher_b),
            name="Research Phase"
        ),
        # Stage 2: Parallel Analysis
        Parallel(
            Step(name="Analysis A", agent=analyst_a),
            Step(name="Analysis B", agent=analyst_b),
            name="Analysis Phase"
        ),
        # Stage 3: Synthesis
        Step(name="Final Report", agent=synthesizer),
    ]
)
```

---

## 🚀 Performance Benefits

**Example Scenario**: 3 independent API calls, each taking 10 seconds

| Execution Mode | Total Time | Calculation |
|---------------|------------|-------------|
| **Sequential** | 30 seconds | 10s + 10s + 10s |
| **Parallel** | ~10 seconds | max(10s, 10s, 10s) |

**Speedup**: 3x faster with parallel execution!

---

## 🛠️ Common Use Cases

### 1. Multi-Source Research
```python
Parallel(
    Step("Academic Papers", agent=academic_researcher),
    Step("News Articles", agent=news_researcher),
    Step("Social Media", agent=social_researcher),
)
```

### 2. Data Collection from Multiple APIs
```python
Parallel(
    Step("Stock Data", agent=finance_agent),
    Step("Weather Data", agent=weather_agent),
    Step("News Data", agent=news_agent),
)
```

### 3. Parallel Analysis
```python
Parallel(
    Step("Sentiment Analysis", agent=sentiment_analyzer),
    Step("Trend Analysis", agent=trend_analyzer),
    Step("Risk Analysis", agent=risk_analyzer),
)
```

### 4. A/B Testing Different Approaches
```python
Parallel(
    Step("Approach A", agent=agent_approach_a),
    Step("Approach B", agent=agent_approach_b),
)
```

---

## 📊 Execution Flow Visualization

### Sequential Workflow
```
Start → Agent1 → Agent2 → Agent3 → Synthesizer → End
        (10s)    (10s)    (10s)    (5s)
        Total: 35 seconds
```

### Parallel Workflow
```
        ┌─ Agent1 (10s) ─┐
Start ──┼─ Agent2 (10s) ─┼→ Synthesizer (5s) → End
        └─ Agent3 (10s) ─┘
        Total: ~15 seconds
```

---

## 💡 Best Practices

1. **Group Independent Tasks**: Identify tasks that don't depend on each other's output
2. **Optimize Stages**: Balance the number of parallel tasks per stage
3. **Consider Resource Limits**: API rate limits, system resources
4. **Error Handling**: Ensure failures in one parallel task don't block others
5. **Debug Visibility**: Use `debug_mode=True` during development

---

## 🔧 Running the Examples

### Prerequisites
```bash
# Install Agno framework
pip install agno

# Set up API keys (if needed)
export OPENAI_API_KEY="your-key-here"
```

### Run Examples
```bash
# Basic example
python parallel_workflow.py

# Advanced multi-stage example
python parallel_workflow_advanced.py

# Performance comparison
python parallel_vs_sequential.py
```

---

## 📝 Notes

- All examples use `gpt-4o-mini` for cost efficiency
- Debug mode can be toggled with `debug_mode` parameter
- Parallel execution is especially beneficial for I/O-bound tasks
- The framework handles the async execution internally

---

## 🎓 Learning Path

1. Start with `parallel_workflow.py` - Learn basic parallel execution
2. Explore `parallel_vs_sequential.py` - Understand performance benefits
3. Study `parallel_workflow_advanced.py` - Master complex orchestration

---

## 🤝 Contributing

Feel free to add more examples or improve existing ones!
