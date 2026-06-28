import asyncio
import math
import threading
import time

import numpy as np
from fastapi import WebSocketDisconnect

import backend.app as backend_app
from backend.frame_provider import FrameProvider


def test_multiple_subscribers_receive_the_same_frames() -> None:
    async def check() -> None:
        provider = FrameProvider("unused")
        first = provider.subscribe()
        second = provider.subscribe()
        first_frame = asyncio.create_task(anext(first))
        second_frame = asyncio.create_task(anext(second))
        await asyncio.sleep(0)

        image = np.zeros((12, 20, 3), dtype=np.uint8)
        provider._publish(image, 123.0, 1.0)

        a, b = await asyncio.gather(first_frame, second_frame)
        assert a is b
        assert (a.frame_id, a.timestamp, a.width, a.height) == (1, 123.0, 20, 12)
        assert not a.image.flags.writeable
        await first.aclose()
        await second.aclose()

    asyncio.run(check())


def test_slow_subscriber_gets_latest_frame() -> None:
    async def check() -> None:
        provider = FrameProvider("unused")
        stream = provider.subscribe()
        waiting = asyncio.create_task(anext(stream))
        await asyncio.sleep(0)
        image = np.zeros((1, 1, 3), dtype=np.uint8)

        provider._publish(image, 1.0, 1.0)
        assert (await waiting).frame_id == 1
        provider._publish(image, 2.0, 2.0)
        provider._publish(image, 3.0, 3.0)
        assert (await anext(stream)).frame_id == 3
        assert provider.fps == 1.0
        await stream.aclose()

    asyncio.run(check())


def test_reconnect_releases_capture_before_opening_another() -> None:
    async def check() -> None:
        provider = FrameProvider("unused", reconnect_delay=0.005)
        lock = threading.Lock()
        live = opens = max_live = 0

        class FakeCapture:
            def __init__(self) -> None:
                nonlocal live, opens, max_live
                with lock:
                    opens += 1
                    self.open_number = opens
                    live += 1
                    if live > max_live:
                        max_live = live

            def isOpened(self) -> bool:
                return True

            def read(self):
                time.sleep(0.002)
                if self.open_number == 1:
                    return False, None
                return True, np.zeros((1, 1, 3), dtype=np.uint8)

            def release(self) -> None:
                nonlocal live
                with lock:
                    live -= 1

        provider._open_capture = FakeCapture
        frames = provider.subscribe()
        waiting = asyncio.create_task(anext(frames))
        await asyncio.sleep(0)
        await provider.start()
        assert (await asyncio.wait_for(waiting, 1)).frame_id == 1
        await provider.stop()
        await frames.aclose()

        assert opens > 1
        assert max_live == 1
        assert live == 0

    asyncio.run(check())


def test_capture_generates_ordered_ids_and_timestamps() -> None:
    async def check() -> None:
        provider = FrameProvider("unused")
        frames = provider.subscribe()
        capture_reads = 0

        class FakeCapture:
            def isOpened(self) -> bool:
                return True

            def read(self):
                nonlocal capture_reads
                capture_reads += 1
                time.sleep(0.005)
                return True, np.zeros((2, 3, 3), dtype=np.uint8)

            def release(self) -> None:
                pass

        provider._open_capture = FakeCapture
        before = time.time()
        await provider.start()
        received = [await asyncio.wait_for(anext(frames), 1) for _ in range(5)]
        await provider.stop()
        after = time.time()
        await frames.aclose()

        assert [frame.frame_id for frame in received] == sorted(
            {frame.frame_id for frame in received}
        )
        assert all(
            math.isfinite(frame.timestamp) and before <= frame.timestamp <= after
            for frame in received
        )
        assert capture_reads >= len(received)

    asyncio.run(check())


def test_subscriber_churn_does_not_reopen_capture() -> None:
    async def check() -> None:
        provider = FrameProvider("unused")
        opened = released = 0

        class FakeCapture:
            def __init__(self) -> None:
                nonlocal opened
                opened += 1

            def isOpened(self) -> bool:
                return True

            def read(self):
                time.sleep(0.005)
                return True, np.zeros((1, 1, 3), dtype=np.uint8)

            def release(self) -> None:
                nonlocal released
                released += 1

        provider._open_capture = FakeCapture
        await provider.start()
        for _ in range(3):
            first, second = provider.subscribe(), provider.subscribe()
            a, b = await asyncio.gather(anext(first), anext(second))
            assert a is b
            await first.aclose()
            await second.aclose()
        assert opened == 1
        await provider.stop()
        assert released == 1

    asyncio.run(check())


def test_stop_does_not_block_event_loop_or_leave_worker() -> None:
    async def check() -> None:
        provider = FrameProvider("unused")
        unblock_read = threading.Event()

        class FakeCapture:
            def isOpened(self) -> bool:
                return True

            def read(self):
                unblock_read.wait()
                return False, None

            def release(self) -> None:
                pass

        provider._open_capture = FakeCapture
        await provider.start()
        await asyncio.sleep(0.01)
        threading.Timer(0.1, unblock_read.set).start()
        stopping = asyncio.create_task(provider.stop())
        await asyncio.sleep(0.01)
        assert not stopping.done()
        await asyncio.wait_for(stopping, 1)
        assert provider._worker is None
        assert not any(
            thread.name == "frame-provider" and thread.is_alive()
            for thread in threading.enumerate()
        )

    asyncio.run(check())


def test_browser_disconnect_cleans_subscriptions_and_tasks() -> None:
    async def check() -> None:
        class DisconnectedBrowser:
            async def accept(self) -> None:
                pass

            async def send_json(self, _: object) -> None:
                raise WebSocketDisconnect()

            async def send_bytes(self, _: object) -> None:
                pass

        original = backend_app.provider
        provider = FrameProvider("unused")
        backend_app.provider = provider
        baseline = set(asyncio.all_tasks())
        try:
            await backend_app.stream(DisconnectedBrowser())
            await asyncio.sleep(0)
            assert not provider._subscribers
            assert not provider._status_subscribers
            assert not backend_app.vision_pipeline._subscribers
            assert set(asyncio.all_tasks()) == baseline
        finally:
            backend_app.provider = original

    asyncio.run(check())
