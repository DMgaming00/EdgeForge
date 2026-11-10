from __future__ import annotations

import numpy as np
import torch

from ..config import RunConfig
from ..models.yolox_loader import load_yolox
from .base import Backend


class PyTorchBackend(Backend):
    def __init__(self, cfg: RunConfig):
        super().__init__(cfg)
        torch.set_num_threads(cfg.threads)
        self.net = load_yolox(cfg.model)

    @property
    def provider(self) -> str:
        return "PyTorch-eager-CPU"

    def infer(self, x: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            return self.net(torch.from_numpy(x)).numpy()
