from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import suppress

from backend.frame_provider import FrameProvider

from .detector import OrbDetector
from .models import VisionResult

logger = logging.getLogger(__name__)


class VisionPipeline:
    def __init__(
        self, frame_provider: FrameProvider, detector: OrbDetector | None = None
    ) -> None:
        self.frame_provider = frame_provider
        self.detector = detector or OrbDetector()
        self.latest_result: VisionResult | None = None
        self._subscribers: set[asyncio.Queue[VisionResult]] = set()
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        if self._task and not self._task.done():
            return
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if not self._task:
            return
        self._task.cancel()
        with suppress(asyncio.CancelledError):
            await self._task
        self._task = None

    async def subscribe(self) -> AsyncIterator[VisionResult]:
        queue: asyncio.Queue[VisionResult] = asyncio.Queue(maxsize=1)
        self._subscribers.add(queue)
        if self.latest_result:
            queue.put_nowait(self.latest_result)
        try:
            while True:
                yield await queue.get()
        finally:
            self._subscribers.discard(queue)

    async def _run(self) -> None:
        async for frame in self.frame_provider.subscribe():
            try:
                result = await asyncio.to_thread(self.detector.detect, frame)
            except Exception:
                logger.exception("Vision stage failed for frame %s", frame.frame_id)
                continue
            self.latest_result = result
            for queue in self._subscribers:
                if queue.full():
                    queue.get_nowait()
                queue.put_nowait(result)

