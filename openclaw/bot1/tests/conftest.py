import os
# Set a placeholder key before main.py is imported.
# Does not override a real key from .env if already set.
os.environ.setdefault("ANTHROPIC_API_KEY", "test-placeholder")
