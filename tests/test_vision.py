import asyncio
import base64

import numpy as np

from backend.app import vision_payload
from backend.frame_provider import Frame
from backend.vision import OrbDetector, VisionPipeline


def frame(image: np.ndarray, frame_id: int = 1) -> Frame:
    image.setflags(write=False)
    return Frame(frame_id, 123.0, image, image.shape[1], image.shape[0])


def test_orb_detects_features_without_modifying_frame() -> None:
    detector = OrbDetector()
    random = np.random.default_rng(1)
    textured = random.integers(0, 256, (480, 640, 3), dtype=np.uint8)
    original = textured.copy()

    rich = detector.detect(frame(textured))
    grayscale_id = id(detector._grayscale)
    poor = detector.detect(frame(np.zeros((480, 640, 3), dtype=np.uint8), 2))

    assert rich.feature_count > 1_500
    assert poor.feature_count < 500
    assert rich.descriptors.shape == (rich.feature_count, 32)
    assert rich.descriptors.dtype == np.uint8
    assert not rich.descriptors.flags.writeable
    assert rich.detector_fps > 0
    assert rich.detection_time_ms > 0
    assert np.array_equal(textured, original)
    assert id(detector._grayscale) == grayscale_id

    payload = vision_payload(rich)
    encoded = payload["descriptors"]
    assert base64.b64decode(encoded["data"]) == rich.descriptors.tobytes()
    assert encoded["rows"] == rich.feature_count


def test_pipeline_subscribes_and_publishes_structured_results() -> None:
    async def check() -> None:
        source_frame = frame(
            np.random.default_rng(2).integers(
                0, 256, (240, 320, 3), dtype=np.uint8
            ),
            42,
        )

        class Source:
            async def subscribe(self):
                yield source_frame
                await asyncio.Future()

        pipeline = VisionPipeline(Source())
        results = pipeline.subscribe()
        waiting = asyncio.create_task(anext(results))
        await asyncio.sleep(0)
        await pipeline.start()
        result = await asyncio.wait_for(waiting, 2)
        await pipeline.stop()
        await results.aclose()

        assert result.frame_id == 42
        assert result.timestamp == 123.0
        assert result.feature_count == len(result.keypoints)
        assert pipeline.latest_result is result
        assert pipeline._task is None

    asyncio.run(check())
