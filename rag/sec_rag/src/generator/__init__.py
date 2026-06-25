"""SEC RAG Generator package."""

# Generator is defined in retriever.generator module
# Re-export here for namespace consistency
import sys
from pathlib import Path

# Ensure retriever package is importable
_src = str(Path(__file__).parent.parent)
if _src not in sys.path:
    sys.path.insert(0, _src)

from retriever.generator import Generator, GeneratorResponse

__all__ = ["Generator", "GeneratorResponse"]
