---
name: running-processes
description: List all running processes and summarize resource usage. Use when the user asks to list processes, check what is running, find which process is using the most CPU or memory, or inspect system resource consumption.
allowed-tools: Bash, Read
---

# Running Processes

## Goal
Retrieve all running processes on the machine and present a summary that includes total process count by status, the top CPU consumers, and the top memory consumers.

## Scripts
- `./scripts/list_processes.py` - Lists running processes and highlights top resource consumers using psutil

## Process

### 1. Run the Script
Execute with `uv run` so the `psutil` dependency is resolved automatically:

```bash
uv run ./scripts/list_processes.py
```

### 2. Present the Output
Display the results in three sections:

**Summary**
- Total number of running processes
- Breakdown by process status (running, sleeping, idle, zombie, etc.)

**Top 5 by CPU Usage**
- Ranked list showing PID, process name, and CPU percentage

**Top 5 by Memory Usage**
- Ranked list showing PID, process name, and RSS memory in MB

### 3. Highlight Notable Findings
After presenting the raw output, call out:
- The single process consuming the most CPU and its percentage
- The single process consuming the most memory and its MB usage
- Any zombie or unusual-status processes if present

## Output
A structured process report with a status summary and the heaviest CPU and memory consumers identified by name and PID.
