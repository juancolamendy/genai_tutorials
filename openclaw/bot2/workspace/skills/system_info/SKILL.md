---
name: system-info
description: Print or check system information including OS, CPU, memory, and disk. Use when the user asks to check system info, print system specs, show hardware details, or inspect machine resources.
allowed-tools: Bash, Read
---

# System Info

## Goal
Display a summary of the current machine's system information: OS, CPU, memory, and disk usage.

## Scripts
- `./scripts/printout_sysinfo.py` - Prints system info using psutil and py-cpuinfo

## Process

### 1. Run the Script
Execute the script with `uv run` so dependencies are resolved automatically from the inline metadata:

```bash
uv run ./scripts/printout_sysinfo.py
```

### 2. Present the Output
Display the script output to the user in a readable format, labeling each field clearly:

- **System** – OS name
- **Node** – Hostname
- **Release** – OS version / kernel release
- **Processor** – CPU brand name
- **CPU Cores** – Logical core count
- **Memory** – Total RAM in GB
- **Disk Total** – Total disk capacity in GB

## Output
A printed summary of the machine's system specifications.
