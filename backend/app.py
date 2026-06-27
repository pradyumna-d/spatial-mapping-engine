from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager

import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from backend.frame_provider import FrameProvider


provider = FrameProvider(
    os.getenv("RTSP_URL", "rtsp://192.168.1.78:8080/h264.sdp")
)


@asynccontextmanager
async def lifespan(_: FastAPI):
    await provider.start()
    try:
        yield
    finally:
        await provider.stop()


app = FastAPI(title="Mapping Engine", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, object]:
    return {
        "status": provider.status.state,
        "frame_id": provider.latest_frame.frame_id if provider.latest_frame else None,
        "fps": provider.fps,
    }


@app.websocket("/ws")
async def stream(websocket: WebSocket) -> None:
    await websocket.accept()
    frames = provider.subscribe()
    statuses = provider.subscribe_status()
    frame_task = asyncio.create_task(anext(frames))
    status_task = asyncio.create_task(anext(statuses))

    try:
        while True:
            done, _ = await asyncio.wait(
                (frame_task, status_task), return_when=asyncio.FIRST_COMPLETED
            )
            if status_task in done:
                status = status_task.result()
                await websocket.send_json(
                    {
                        "type": "status",
                        "status": status.state,
                        "message": status.message,
                    }
                )
                status_task = asyncio.create_task(anext(statuses))

            if frame_task in done:
                frame = frame_task.result()
                encoded, jpeg = await asyncio.to_thread(
                    cv2.imencode, ".jpg", frame.image
                )
                if encoded:
                    await websocket.send_json(
                        {
                            "type": "frame",
                            "frame_id": frame.frame_id,
                            "timestamp": frame.timestamp,
                            "width": frame.width,
                            "height": frame.height,
                            "fps": provider.fps,
                        }
                    )
                    await websocket.send_bytes(memoryview(jpeg))
                frame_task = asyncio.create_task(anext(frames))
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        frame_task.cancel()
        status_task.cancel()
        await asyncio.gather(frame_task, status_task, return_exceptions=True)
        await frames.aclose()
        await statuses.aclose()
