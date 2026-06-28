from __future__ import annotations

import asyncio
import base64
import os
from contextlib import asynccontextmanager

import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from backend.frame_provider import FrameProvider
from backend.vision import VisionPipeline, VisionResult


provider = FrameProvider(
    os.getenv(
        "RTSP_URL",
        "rtsp://admin:zaq1xsw2@192.168.1.81:5543/live/channel0",
    )
)
vision_pipeline = VisionPipeline(provider)


@asynccontextmanager
async def lifespan(_: FastAPI):
    await provider.start()
    await vision_pipeline.start()
    try:
        yield
    finally:
        await vision_pipeline.stop()
        await provider.stop()


app = FastAPI(title="Mapping Engine", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, object]:
    return {
        "status": provider.status.state,
        "frame_id": provider.latest_frame.frame_id if provider.latest_frame else None,
        "fps": provider.fps,
        "feature_count": (
            vision_pipeline.latest_result.feature_count
            if vision_pipeline.latest_result
            else None
        ),
    }


def vision_payload(result: VisionResult) -> dict[str, object]:
    return {
        "type": "vision",
        "frame_id": result.frame_id,
        "timestamp": result.timestamp,
        "feature_count": result.feature_count,
        "keypoints": [
            [
                point.x,
                point.y,
                point.size,
                point.angle,
                point.response,
                point.octave,
            ]
            for point in result.keypoints
        ],
        "descriptors": {
            "data": base64.b64encode(result.descriptors).decode("ascii"),
            "rows": result.descriptors.shape[0],
            "cols": result.descriptors.shape[1],
            "dtype": str(result.descriptors.dtype),
        },
        "detector_fps": result.detector_fps,
        "detection_time_ms": result.detection_time_ms,
    }


@app.websocket("/ws")
async def stream(websocket: WebSocket) -> None:
    await websocket.accept()
    frames = provider.subscribe()
    statuses = provider.subscribe_status()
    results = vision_pipeline.subscribe()
    frame_task = asyncio.create_task(anext(frames))
    status_task = asyncio.create_task(anext(statuses))
    vision_task = asyncio.create_task(anext(results))

    try:
        while True:
            done, _ = await asyncio.wait(
                (frame_task, status_task, vision_task),
                return_when=asyncio.FIRST_COMPLETED,
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

            if vision_task in done:
                await websocket.send_json(vision_payload(vision_task.result()))
                vision_task = asyncio.create_task(anext(results))
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        frame_task.cancel()
        status_task.cancel()
        vision_task.cancel()
        await asyncio.gather(
            frame_task, status_task, vision_task, return_exceptions=True
        )
        await frames.aclose()
        await statuses.aclose()
        await results.aclose()
