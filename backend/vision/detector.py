from __future__ import annotations

import time

import cv2
import numpy as np
from numpy.typing import NDArray

from backend.frame_provider import Frame

from .models import Keypoint, VisionResult


class OrbDetector:
    def __init__(self, max_features: int = 2_000) -> None:
        self._orb = cv2.ORB_create(nfeatures=max_features)
        self._grayscale: NDArray[np.uint8] | None = None

    def detect(self, frame: Frame) -> VisionResult:
        started = time.perf_counter()
        if self._grayscale is None or self._grayscale.shape != frame.image.shape[:2]:
            self._grayscale = np.empty(frame.image.shape[:2], dtype=np.uint8)
        cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY, dst=self._grayscale)
        detected, descriptors = self._orb.detectAndCompute(self._grayscale, None)
        detection_time_ms = (time.perf_counter() - started) * 1_000

        if descriptors is None:
            descriptors = np.empty((0, 32), dtype=np.uint8)
        descriptors.setflags(write=False)
        keypoints = tuple(
            Keypoint(
                x=float(point.pt[0]),
                y=float(point.pt[1]),
                size=float(point.size),
                angle=float(point.angle),
                response=float(point.response),
                octave=int(point.octave),
            )
            for point in detected
        )
        return VisionResult(
            frame_id=frame.frame_id,
            timestamp=frame.timestamp,
            keypoints=keypoints,
            descriptors=descriptors,
            detector_fps=1_000 / max(detection_time_ms, 1e-9),
            detection_time_ms=detection_time_ms,
        )
