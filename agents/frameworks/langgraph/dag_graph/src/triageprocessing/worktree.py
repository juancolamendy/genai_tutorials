"""STEP 7 — Isolated patch mode (mock worktree + diff + publish)."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path


def create_worktree(repo: str, run_id: str, worktrees_root: str = ".triage_worktrees") -> str:
    """Create an isolated work directory for this run.

    Mock: copies repo if it exists, otherwise creates an empty sandbox with
    a placeholder culprit file so FakeWorker has somewhere to write.
    """
    root = Path(worktrees_root)
    root.mkdir(parents=True, exist_ok=True)
    dest = root / run_id
    if dest.exists():
        return str(dest)

    src = Path(repo)
    if src.is_dir():
        shutil.copytree(src, dest, dirs_exist_ok=True, ignore=shutil.ignore_patterns(".git"))
    else:
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "src").mkdir(exist_ok=True)
        (dest / "src" / "app.py").write_text(
            "def handle_request():\n    return current_user()['id']\n",
            encoding="utf-8",
        )
    return str(dest)


def collect_diff(worktree_path: str) -> str:
    """Return a verbatim diff-like artifact (mock: list new/changed files)."""
    root = Path(worktree_path)
    lines = [f"# diff for {worktree_path}", ""]
    for path in sorted(root.rglob("*")):
        if path.is_file():
            rel = path.relative_to(root)
            digest = hashlib.sha1(path.read_bytes()).hexdigest()[:8]
            lines.append(f"+ {rel}  sha1={digest}")
    return "\n".join(lines) + "\n"


def publish(worktree_path: str, run_id: str, message: str) -> str:
    """Mock publish: write a PUBLISHED marker and return a fake commit sha."""
    root = Path(worktree_path)
    marker = root / ".published"
    payload = f"{run_id}\n{message}\n"
    marker.write_text(payload, encoding="utf-8")
    return hashlib.sha1(payload.encode()).hexdigest()[:12]
