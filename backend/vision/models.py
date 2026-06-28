from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class Keypoint:
    x: float
    y: float
    size: float
    angle: float
    response: float
    octave: int


@dataclass(frozen=True, slots=True)
class VisionResult:
    frame_id: int
    timestamp: float
    keypoints: tuple[Keypoint, ...]
    descriptors: NDArray[np.uint8]
    detector_fps: float
    detection_time_ms: float

    @property
    def feature_count(self) -> int:
        return len(self.keypoints)

