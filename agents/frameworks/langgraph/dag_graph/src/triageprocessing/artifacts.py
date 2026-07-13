"""STEP 4 — Structured artifacts: state holds paths, raw content on disk."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ArtifactStore:
    def __init__(self, root: str = ".triage_artifacts") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def run_dir(self, run_id: str) -> Path:
        path = self.root / run_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def write(self, run_id: str, name: str, content: str) -> str:
        path = self.run_dir(run_id) / name
        path.write_text(content, encoding="utf-8")
        return str(path)

    def write_json(self, run_id: str, name: str, data: Any) -> str:
        return self.write(run_id, name, json.dumps(data, indent=2, default=str))

    def read(self, path: str) -> str:
        return Path(path).read_text(encoding="utf-8")
