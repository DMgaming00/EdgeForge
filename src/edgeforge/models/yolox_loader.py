"""Build YOLOX with the official Apache-2.0 `yolox` network code and load checkpoints."""
from __future__ import annotations

import torch
from yolox.models import YOLOPAFPN, YOLOX, YOLOXHead

from ..config import MODEL_SPECS, MODELS_DIR, NUM_CLASSES, sha256_file


def weights_path(model: str):
    return MODELS_DIR / MODEL_SPECS[model]["weights"]


def weights_hash(model: str) -> str:
    return sha256_file(weights_path(model))


def load_yolox(model: str) -> torch.nn.Module:
    """FP32 eval-mode model returning RAW head output (1,N,85): boxes NOT decoded."""
    spec = MODEL_SPECS[model]
    path = weights_path(model)
    if not path.exists():
        raise FileNotFoundError(
            f"missing weights {path}. Run `edgeforge prepare --weights` to download them.")
    backbone = YOLOPAFPN(spec["depth"], spec["width"], depthwise=spec["depthwise"], act="silu")
    head = YOLOXHead(NUM_CLASSES, spec["width"], depthwise=spec["depthwise"], act="silu")
    net = YOLOX(backbone, head)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    net.load_state_dict(ckpt["model"])
    net.eval()
    net.head.decode_in_inference = False
    for p in net.parameters():
        p.requires_grad_(False)
    return net
