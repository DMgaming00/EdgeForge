"""Backend abstraction: raw forward + shared pre/post-processing."""
from __future__ import annotations

import time
from abc import ABC, abstractmethod

import numpy as np

from ..config import RunConfig
from ..postprocess import postprocess
from ..preprocess import letterbox
from ..types import Detections


class Backend(ABC):
    def __init__(self, cfg: RunConfig):
        self.cfg = cfg

    @abstractmethod
    def infer(self, x: np.ndarray) -> np.ndarray:
        """x: (1,3,S,S) float32 -> raw head output (1,N,85) float32 numpy."""

    @property
    @abstractmethod
    def provider(self) -> str: ...

    def validate_output(self, out: np.ndarray) -> None:
        n = sum((self.cfg.resolution // s) ** 2 for s in (8, 16, 32))
        if out.shape != (1, n, 85):
            raise ValueError(f"unexpected output shape {out.shape}, expected (1,{n},85)")
        if not np.isfinite(out).all():
            raise ValueError("non-finite values in model output")

    def detect(self, img_bgr: np.ndarray, conf_thr: float = 0.01, nms_thr: float = 0.65
               ) -> Detections:
        """Full pipeline with per-stage timings (ns) stored in Detections.meta['timing_ns']."""
        t0 = time.perf_counter_ns()
        x, ratio = letterbox(img_bgr, self.cfg.resolution)
        x = x[None]
        t1 = time.perf_counter_ns()
        out = self.infer(x)
        t2 = time.perf_counter_ns()
        self.validate_output(out)
        det = postprocess(out, self.cfg.resolution, ratio, img_bgr.shape[:2], conf_thr, nms_thr)
        t3 = time.perf_counter_ns()
        det.meta["timing_ns"] = {"preprocess": t1 - t0, "inference": t2 - t1,
                                 "postprocess": t3 - t2, "total": t3 - t0}
        return det


def build_backend(cfg: RunConfig, onnx_path=None) -> Backend:
    if cfg.backend == "pytorch":
        from .pytorch_backend import PyTorchBackend
        return PyTorchBackend(cfg)
    from .ort_backend import ONNXRuntimeBackend
    return ONNXRuntimeBackend(cfg, onnx_path)
