"""Static INT8 (QDQ) quantization with ONNX Runtime, calibrated on a DISJOINT image set."""
from __future__ import annotations

from pathlib import Path

import cv2
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)

from ..config import MANIFEST_DIR
from ..evaluate.dataset import image_path, load_manifest
from ..preprocess import letterbox
from .onnx_export import export_onnx, onnx_path_for


class _Reader(CalibrationDataReader):
    def __init__(self, ids: list[int], size: int):
        self.it = iter(ids)
        self.size = size

    def get_next(self):
        i = next(self.it, None)
        if i is None:
            return None
        return {"images": letterbox(cv2.imread(str(image_path(i))), self.size)[0][None]}


def quantize_model(model: str, resolution: int, method: str = "minmax", force: bool = False
                   ) -> tuple[Path, dict]:
    """Returns (int8 onnx path, info). Calibration images come from calib_*.json only."""
    fp32 = export_onnx(model, resolution)
    out = onnx_path_for(model, resolution, "int8")
    calib = load_manifest(next(MANIFEST_DIR.glob("calib_*.json")))
    ev = load_manifest(next(MANIFEST_DIR.glob("eval_*.json")))
    if set(calib["image_ids"]) & set(ev["image_ids"]):
        raise RuntimeError("calibration and evaluation sets overlap")
    info = {"calibration_manifest": f"calib_{calib['n']}.json", "n_calibration": calib["n"],
            "method": method, "format": "QDQ", "weights": "int8 per-channel",
            "activations": "uint8"}
    if out.exists() and not force:
        return out, info
    m = {"minmax": CalibrationMethod.MinMax, "entropy": CalibrationMethod.Entropy,
         "percentile": CalibrationMethod.Percentile}[method]
    quantize_static(str(fp32), str(out), _Reader(calib["image_ids"], resolution),
                    quant_format=QuantFormat.QDQ, per_channel=True,
                    weight_type=QuantType.QInt8, activation_type=QuantType.QUInt8,
                    calibrate_method=m)
    return out, info
