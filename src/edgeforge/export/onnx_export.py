from __future__ import annotations

from pathlib import Path

from ..config import MODELS_DIR


def onnx_path_for(model: str, resolution: int, precision: str = "fp32") -> Path:
    suffix = "" if precision == "fp32" else f"-{precision}"
    return MODELS_DIR / "onnx" / f"{model}-{resolution}{suffix}.onnx"


def export_onnx(model: str, resolution: int, opset: int = 17, force: bool = False) -> Path:
    """Export raw-head FP32 graph with static 1x3xSxS input (legacy TorchScript exporter)."""
    path = onnx_path_for(model, resolution)
    if path.exists() and not force:
        return path
    import torch

    from ..models.yolox_loader import load_yolox
    path.parent.mkdir(parents=True, exist_ok=True)
    net = load_yolox(model)
    dummy = torch.zeros(1, 3, resolution, resolution)
    torch.onnx.export(net, dummy, str(path), input_names=["images"], output_names=["output"],
                      opset_version=opset, dynamo=False)
    import onnx
    onnx.checker.check_model(onnx.load(str(path)))
    return path
