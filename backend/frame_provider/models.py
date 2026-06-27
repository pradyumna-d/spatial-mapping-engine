from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class Frame:
    frame_id: int
    timestamp: float
    image: NDArray[np.uint8]
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class ConnectionStatus:
    state: str
    message: str = ""

