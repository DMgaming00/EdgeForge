"""Shared detection representation."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Detections:
    """Detections in ORIGINAL image pixel coordinates, xyxy."""
    boxes: np.ndarray            # (N,4) float32 x1,y1,x2,y2
    scores: np.ndarray           # (N,) float32
    class_ids: np.ndarray        # (N,) int64, 0..79 contiguous
    image_size: tuple[int, int]  # (height, width)
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.scores.shape[0])

    @staticmethod
    def empty(image_size: tuple[int, int], meta: dict | None = None) -> Detections:
        return Detections(np.zeros((0, 4), np.float32), np.zeros((0,), np.float32),
                          np.zeros((0,), np.int64), image_size, meta or {})

    def validate(self) -> None:
        n = len(self)
        if self.boxes.shape != (n, 4) or self.class_ids.shape != (n,):
            raise ValueError("inconsistent detection array shapes")
        if n and not np.isfinite(self.boxes).all():
            raise ValueError("non-finite box coordinates")
        if n and (self.scores.min() < 0 or self.scores.max() > 1.0 + 1e-6):
            raise ValueError("scores outside [0,1]")
