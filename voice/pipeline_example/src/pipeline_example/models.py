import asyncio
from dataclasses import dataclass, field


@dataclass
class ExecState:
    total_secs: int = 0
    last_activity_time: float | None = None
    callsid: str | None = None
    called: str | None = None
    _lock: asyncio.Lock = field(init=False, repr=False)

    def __post_init__(self):
        self._lock = asyncio.Lock()

    async def get_total_duration(self) -> int:
        async with self._lock:
            return self.total_secs

    async def set_total_duration(self, value: int):
        async with self._lock:
            self.total_secs = value

    async def increment_total_duration(self):
        async with self._lock:
            self.total_secs += 1

    async def get_last_activity(self) -> float | None:
        async with self._lock:
            return self.last_activity_time

    async def set_last_activity(self, value: float | None):
        async with self._lock:
            self.last_activity_time = value

    async def get_callsid(self) -> str | None:
        async with self._lock:
            return self.callsid

    async def set_callsid(self, value: str | None):
        async with self._lock:
            self.callsid = value

    async def get_called(self) -> str | None:
        async with self._lock:
            return self.called

    async def set_called(self, value: str | None):
        async with self._lock:
            self.called = value
