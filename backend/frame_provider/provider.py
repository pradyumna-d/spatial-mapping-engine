from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from collections import deque
from collections.abc import AsyncIterator

import cv2
from numpy.typing import NDArray

from .models import ConnectionStatus, Frame

logger = logging.getLogger(__name__)


class FrameProvider:
    """The sole owner of the RTSP connection and H.264 decoder."""

    def __init__(
        self,
        url: str,
        *,
        reconnect_delay: float = 1.0,
        open_timeout_ms: int = 5_000,
        read_timeout_ms: int = 5_000,
    ) -> None:
        self.url = url
        self.reconnect_delay = reconnect_delay
        self.open_timeout_ms = open_timeout_ms
        self.read_timeout_ms = read_timeout_ms

        self.latest_frame: Frame | None = None
        self.fps = 0.0
        self.status = ConnectionStatus("stopped")

        self._frame_id = 0
        self._frame_times: deque[float] = deque()
        self._subscribers: set[asyncio.Queue[Frame]] = set()
        self._status_subscribers: set[asyncio.Queue[ConnectionStatus]] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._worker: threading.Thread | None = None
        self._stop_event = threading.Event()

    async def start(self) -> None:
        if self._worker and self._worker.is_alive():
            return
        self._loop = asyncio.get_running_loop()
        self._stop_event.clear()
        self._worker = threading.Thread(
            target=self._capture_loop, name="frame-provider"
        )
        self._worker.start()

    async def stop(self) -> None:
        self._stop_event.set()
        if self._worker:
            await asyncio.to_thread(self._worker.join)
            self._worker = None
        self._set_status("stopped")

    async def subscribe(self) -> AsyncIterator[Frame]:
        queue: asyncio.Queue[Frame] = asyncio.Queue(maxsize=1)
        self._subscribers.add(queue)
        if self.latest_frame:
            queue.put_nowait(self.latest_frame)
        try:
            while True:
                yield await queue.get()
        finally:
            self._subscribers.discard(queue)

    async def subscribe_status(self) -> AsyncIterator[ConnectionStatus]:
        queue: asyncio.Queue[ConnectionStatus] = asyncio.Queue(maxsize=1)
        self._status_subscribers.add(queue)
        queue.put_nowait(self.status)
        try:
            while True:
                yield await queue.get()
        finally:
            self._status_subscribers.discard(queue)

    def _open_capture(self) -> cv2.VideoCapture:
        os.environ.setdefault(
            "OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp"
        )
        return cv2.VideoCapture(
            self.url,
            cv2.CAP_FFMPEG,
            [
                cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
                self.open_timeout_ms,
                cv2.CAP_PROP_READ_TIMEOUT_MSEC,
                self.read_timeout_ms,
            ],
        )

    def _capture_loop(self) -> None:
        assert self._loop is not None
        while not self._stop_event.is_set():
            capture: cv2.VideoCapture | None = None
            try:
                self._notify_status("connecting")
                capture = self._open_capture()
                if not capture.isOpened():
                    raise RuntimeError("RTSP connection failed")

                self._notify_status("connected")
                while not self._stop_event.is_set():
                    decoded, image = capture.read()
                    if not decoded:
                        raise RuntimeError("RTSP stream interrupted")
                    self._loop.call_soon_threadsafe(
                        self._publish, image, time.time(), time.monotonic()
                    )
            except Exception as error:
                if not self._stop_event.is_set():
                    logger.warning("%s; reconnecting", error)
                    self._notify_status("disconnected", str(error))
            finally:
                if capture is not None:
                    capture.release()

            self._stop_event.wait(self.reconnect_delay)

    def _notify_status(self, state: str, message: str = "") -> None:
        assert self._loop is not None
        self._loop.call_soon_threadsafe(self._set_status, state, message)

    def _set_status(self, state: str, message: str = "") -> None:
        status = ConnectionStatus(state, message)
        if status == self.status:
            return
        self.status = status
        if state != "connected":
            self.fps = 0.0
            self._frame_times.clear()
        for queue in self._status_subscribers:
            self._replace_latest(queue, status)

    def _publish(
        self, image: NDArray, timestamp: float, monotonic_time: float
    ) -> None:
        self._frame_id += 1
        image.setflags(write=False)
        height, width = image.shape[:2]
        self._frame_times.append(monotonic_time)
        while monotonic_time - self._frame_times[0] > 2.0:
            self._frame_times.popleft()
        if len(self._frame_times) > 1:
            self.fps = (len(self._frame_times) - 1) / (
                max(self._frame_times[-1] - self._frame_times[0], 1e-9)
            )
        frame = Frame(self._frame_id, timestamp, image, width, height)
        self.latest_frame = frame
        for queue in self._subscribers:
            self._replace_latest(queue, frame)

    @staticmethod
    def _replace_latest(queue: asyncio.Queue, item: object) -> None:
        if queue.full():
            queue.get_nowait()
        queue.put_nowait(item)
