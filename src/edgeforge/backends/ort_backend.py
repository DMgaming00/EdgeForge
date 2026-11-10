from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort

from ..config import RunConfig
from .base import Backend

_OPT = {
    "none": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
    "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
    "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
}


class ONNXRuntimeBackend(Backend):
    def __init__(self, cfg: RunConfig, onnx_path: Path | str | None = None):
        super().__init__(cfg)
        from ..export.onnx_export import onnx_path_for
        path = Path(onnx_path) if onnx_path else onnx_path_for(cfg.model, cfg.resolution, cfg.precision)
        if not path.exists():
            raise FileNotFoundError(f"missing ONNX model {path}. Run `edgeforge export`.")
        self.path = path
        so = ort.SessionOptions()
        so.intra_op_num_threads = cfg.threads
        so.inter_op_num_threads = 1
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.graph_optimization_level = _OPT[cfg.graph_opt]
        avail = ort.get_available_providers()
        if "CPUExecutionProvider" not in avail:
            raise RuntimeError(f"CPUExecutionProvider unavailable; have {avail}")
        self.sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name

    @property
    def provider(self) -> str:
        return self.sess.get_providers()[0]

    def infer(self, x: np.ndarray) -> np.ndarray:
        return self.sess.run(None, {self.inp: x})[0]
