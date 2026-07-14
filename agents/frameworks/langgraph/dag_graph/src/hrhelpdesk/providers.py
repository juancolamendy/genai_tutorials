"""In-memory fake providers for policy KB, tickets, and desk booking."""

from __future__ import annotations

import itertools
from typing import Any

_POLICY_SNIPPETS: list[dict[str, str]] = [
    {
        "id": "pto-001",
        "text": "Full-time employees accrue 15 days of PTO per year. "
        "Submit requests at least two weeks in advance via the HR portal.",
    },
    {
        "id": "remote-002",
        "text": "Remote work is permitted up to two days per week with manager approval. "
        "Desk booking is required for in-office days.",
    },
    {
        "id": "benefits-003",
        "text": "Open enrollment runs each November. Medical, dental, and vision plans "
        "are available; changes outside enrollment require a qualifying life event.",
    },
]

_ticket_store: list[dict[str, str]] = []
_booking_store: list[dict[str, Any]] = []
_ticket_id_gen = itertools.count(1)
_booking_id_gen = itertools.count(1)


def retrieve_policy(query: str) -> list[dict[str, str]]:
    """Return hardcoded FAQ snippets loosely matching the query."""
    q = query.lower()
    hits = [
        s
        for s in _POLICY_SNIPPETS
        if any(word in s["text"].lower() for word in q.split() if len(word) > 3)
    ]
    return hits or _POLICY_SNIPPETS[:2]


def create_ticket(subject: str, body: str) -> str:
    """Create a ticket and return its id."""
    ticket_id = f"TICKET-{next(_ticket_id_gen)}"
    _ticket_store.append({"id": ticket_id, "subject": subject, "body": body})
    return ticket_id


def check_desk_availability(date: str, location: str) -> bool:
    """Fake availability — always true except empty inputs."""
    return bool(date and location)


def confirm_booking(date: str, location: str, seat_pref: str) -> str:
    """Confirm a desk booking and return booking id."""
    booking_id = f"BOOK-{next(_booking_id_gen)}"
    _booking_store.append(
        {
            "booking_id": booking_id,
            "date": date,
            "location": location,
            "seat_pref": seat_pref,
            "status": "confirmed",
        }
    )
    return booking_id


def reset_providers() -> None:
    """Clear module stores (for tests)."""
    _ticket_store.clear()
    _booking_store.clear()


__all__ = [
    "retrieve_policy",
    "create_ticket",
    "check_desk_availability",
    "confirm_booking",
    "reset_providers",
    "_ticket_store",
    "_booking_store",
]
