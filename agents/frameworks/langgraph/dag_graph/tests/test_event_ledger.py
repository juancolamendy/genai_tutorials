"""Tests for the file-backed event dedupe store (design spec §2.2/§14).

Must be file-backed, not in-memory: the CLI is a fresh process per one-shot
command, so an in-memory ledger would never actually dedupe anything across
separate invocations. The decisive test here simulates that by creating two
separate EventLedger instances pointed at the same directory.
"""

import uuid

import pytest

from src.engine.event_ledger import EventLedger


def _ledger_dir():
    return f"/tmp/test_event_ledger_{uuid.uuid4()}"


@pytest.mark.asyncio
async def test_unprocessed_event_id_is_not_processed():
    ledger = EventLedger(ledger_dir=_ledger_dir())
    assert await ledger.is_processed("event-1") is False


@pytest.mark.asyncio
async def test_marking_processed_makes_is_processed_true():
    ledger = EventLedger(ledger_dir=_ledger_dir())
    await ledger.mark_processed("event-1")
    assert await ledger.is_processed("event-1") is True


@pytest.mark.asyncio
async def test_marking_one_event_id_does_not_affect_another():
    ledger = EventLedger(ledger_dir=_ledger_dir())
    await ledger.mark_processed("event-1")
    assert await ledger.is_processed("event-2") is False


@pytest.mark.asyncio
async def test_dedupe_persists_across_separate_instances_same_directory():
    """The scenario an in-memory ledger would fail: two separate instances
    (simulating two separate CLI process invocations) sharing a directory
    must agree on what's already been processed."""
    shared_dir = _ledger_dir()
    ledger_process_1 = EventLedger(ledger_dir=shared_dir)
    await ledger_process_1.mark_processed("sweep:thread-1:await_documents_signed")

    ledger_process_2 = EventLedger(ledger_dir=shared_dir)
    assert await ledger_process_2.is_processed("sweep:thread-1:await_documents_signed") is True


@pytest.mark.asyncio
async def test_event_id_with_colons_and_slashes_is_safe_for_filesystem():
    """event_ids are often composite (e.g. "sweep:{thread_id}:{state}") and
    thread_ids can contain characters that aren't safe as bare filenames."""
    ledger = EventLedger(ledger_dir=_ledger_dir())
    event_id = "sweep:user/1:await_documents_signed"
    await ledger.mark_processed(event_id)
    assert await ledger.is_processed(event_id) is True
