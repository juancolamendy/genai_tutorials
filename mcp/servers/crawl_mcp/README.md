# Crawl MCP Server (Python)

A Model Context Protocol (MCP) server template for crawling tasks. This Python implementation currently serves as a placeholder and prints a greeting message when run.

## Features

- Prints "Hello from crawl-mcp!" to the console
- Project structure ready for future MCP tool development

## Prerequisites

- Python 3.12 or higher

## Installation

**Navigate to the project directory:**
```bash
cd crawl_mcp
```

**(Optional) Set up a virtual environment:**
```bash
uv sync
```

**Install dependencies:**
```bash
# No dependencies required at this time
uv run playwright install chromium
```

**Create a `.env` file with your configuration:**
```bash
GOOGLE_API_KEY=XXX
```

**Add the server configuration:**
```json
{
    "mcpServers": {
        "crawl_mcp": {
            "command": "uv",
            "args": [
                "--directory",
                "<path>/crawl_mcp",
                "run",
                "main.py"
            ],
            "env": {
                "GOOGLE_API_KEY": "<key>",
            }
        }
    }
}

## Use

```md
Use the crawl mcp to crawl the website: {url}
```