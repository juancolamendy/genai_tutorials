import pytest

from pipeline_example.models import ExecState


@pytest.mark.asyncio
async def test_exec_state_async_locks():
    state = ExecState()
    await state.set_callsid("CA123")
    await state.set_called("+15551234567")
    await state.set_last_activity(123.0)
    await state.increment_total_duration()

    assert await state.get_callsid() == "CA123"
    assert await state.get_called() == "+15551234567"
    assert await state.get_last_activity() == 123.0
    assert await state.get_total_duration() == 1
