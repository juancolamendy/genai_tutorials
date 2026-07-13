"""STEP 3 — Wrap external systems as clients (read-only mocks by default).

Named clients.py (not adapters.py) per domain convention. Swap mock bodies
for real HTTP without touching the graph.
"""

from __future__ import annotations

from typing import Any, Protocol

from .artifacts import ArtifactStore


class ContextClient(Protocol):
    name: str

    def fetch(self, inputs: dict[str, Any]) -> Any: ...


class SentryClient:
    name = "sentry"

    def fetch(self, inputs: dict[str, Any]) -> Any:
        return {
            "issue": inputs.get("issue", "PROJ-142"),
            "title": "TypeError: cannot read 'user' of None",
            "culprit": "src/app.py in handle_request",
            "stacktrace": [
                "src/app.py:42 handle_request",
                "src/auth.py:17 current_user",
            ],
            "count_24h": 87,
        }


class GitHubClient:
    name = "github"

    def fetch(self, inputs: dict[str, Any]) -> Any:
        return {
            "recent_commits": [
                {
                    "sha": "a1b2c3",
                    "msg": "refactor auth middleware",
                    "files": ["src/app.py"],
                }
            ]
        }


class LinearClient:
    name = "linear"

    def fetch(self, inputs: dict[str, Any]) -> Any:
        return {
            "ticket": inputs.get("issue"),
            "priority": "high",
            "assignee": None,
        }


DEFAULT_CLIENTS: list[ContextClient] = [SentryClient(), GitHubClient(), LinearClient()]


def gather_context(
    run_id: str,
    inputs: dict[str, Any],
    store: ArtifactStore,
    clients: list[ContextClient] | None = None,
) -> dict[str, str]:
    """Run every client; persist raw output; return {name: artifact_path}."""
    paths: dict[str, str] = {}
    for client in clients or DEFAULT_CLIENTS:
        raw = client.fetch(inputs)
        paths[client.name] = store.write_json(run_id, f"{client.name}.raw.json", raw)
    return paths
