"""Decoding and NMS, shared by every backend (operates on raw YOLOX head output)."""
from __future__ import annotations

import numpy as np

from .types import Detections

STRIDES = (8, 16, 32)


def make_grids(size: int, strides=STRIDES) -> tuple[np.ndarray, np.ndarray]:
    """Return (grid (N,2), stride (N,1)) matching the head's flattening order."""
    grids, ss = [], []
    for s in strides:
        n = size // s
        ys, xs = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        grids.append(np.stack((xs, ys), 2).reshape(-1, 2))
        ss.append(np.full((n * n, 1), s))
    return np.concatenate(grids).astype(np.float32), np.concatenate(ss).astype(np.float32)


_GRID_CACHE: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def decode_boxes(raw: np.ndarray, size: int) -> np.ndarray:
    """raw (N, 5+C) with undecoded xywh -> copy with cx,cy,w,h in input-pixel units."""
    if size not in _GRID_CACHE:
        _GRID_CACHE[size] = make_grids(size)
    grid, stride = _GRID_CACHE[size]
    if raw.shape[0] != grid.shape[0]:
        raise ValueError(f"output has {raw.shape[0]} anchors, expected {grid.shape[0]} for {size}")
    out = raw.copy()
    out[:, :2] = (raw[:, :2] + grid) * stride
    out[:, 2:4] = np.exp(raw[:, 2:4]) * stride
    return out


def iou_one_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0]); y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2]); y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    a = (box[2] - box[0]) * (box[3] - box[1])
    b = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / np.maximum(a + b - inter, 1e-12)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> np.ndarray:
    """Greedy NMS; returns kept indices sorted by descending score (stable on ties)."""
    order = np.argsort(-scores, kind="stable")
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        order = rest[iou_one_to_many(boxes[i], boxes[rest]) <= iou_thr]
    return np.asarray(keep, dtype=np.int64)


def batched_nms(boxes, scores, class_ids, iou_thr):
    """Class-aware NMS via coordinate offset trick."""
    if boxes.shape[0] == 0:
        return np.zeros((0,), np.int64)
    offset = class_ids.astype(np.float32)[:, None] * (boxes.max() + 1.0)
    return nms(boxes + offset, scores, iou_thr)


def postprocess(raw: np.ndarray, size: int, ratio: float, image_size: tuple[int, int],
                conf_thr: float = 0.01, nms_thr: float = 0.65, max_det: int = 100) -> Detections:
    """Raw head output (1,N,85) or (N,85) -> Detections in original image coordinates.

    Score = objectness * class probability (YOLOX convention). Each anchor keeps its
    best class (as in the reference YOLOX demo / eval path).
    """
    raw = raw[0] if raw.ndim == 3 else raw
    if not np.isfinite(raw).all():
        raise ValueError("backend produced non-finite outputs")
    dec = decode_boxes(raw, size)
    cls_scores = dec[:, 5:]
    cls_id = cls_scores.argmax(1)
    scores = dec[:, 4] * cls_scores[np.arange(len(dec)), cls_id]
    m = scores >= conf_thr
    if not m.any():
        return Detections.empty(image_size)
    d, s, c = dec[m], scores[m], cls_id[m]
    xyxy = np.empty((len(d), 4), np.float32)
    xyxy[:, 0] = d[:, 0] - d[:, 2] / 2; xyxy[:, 1] = d[:, 1] - d[:, 3] / 2
    xyxy[:, 2] = d[:, 0] + d[:, 2] / 2; xyxy[:, 3] = d[:, 1] + d[:, 3] / 2
    keep = batched_nms(xyxy, s, c, nms_thr)[:max_det]
    xyxy, s, c = xyxy[keep] / ratio, s[keep], c[keep]
    h, w = image_size
    xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], 0, w)
    xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], 0, h)
    return Detections(xyxy.astype(np.float32), s.astype(np.float32), c.astype(np.int64), image_size)
