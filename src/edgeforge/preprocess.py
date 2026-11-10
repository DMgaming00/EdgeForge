"""Deterministic YOLOX letterbox preprocessing (shared by every backend)."""
from __future__ import annotations

import cv2
import numpy as np

PAD_VALUE = 114


def letterbox(img_bgr: np.ndarray, size: int) -> tuple[np.ndarray, float]:
    """Resize keeping aspect ratio, pad bottom/right with 114. Returns (CHW float32 0-255, ratio).

    Matches YOLOX's ``preproc`` (BGR, no mean/std normalization, values 0..255).
    """
    if img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
        raise ValueError(f"expected HxWx3 image, got {img_bgr.shape}")
    h, w = img_bgr.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(h * r), int(w * r)
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), PAD_VALUE, dtype=np.uint8)
    canvas[:nh, :nw] = resized
    chw = np.ascontiguousarray(canvas.transpose(2, 0, 1), dtype=np.float32)
    return chw, r
