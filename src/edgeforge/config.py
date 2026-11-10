"""Paths, model specs, and the experiment configuration dataclass."""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"
IMAGES_DIR = DATA_DIR / "images"
MANIFEST_DIR = DATA_DIR / "manifests"
RAW_DIR = DATA_DIR / "raw"
ANN_FILE = RAW_DIR / "annotations" / "instances_val2017.json"
RESULTS_DIR = ROOT / "results"
COCO_IMG_URL = "http://images.cocodataset.org/val2017/{file}"

# YOLOX checkpoints: Apache-2.0 repo (Megvii-BaseDetection/YOLOX) release 0.1.1rc0.
MODEL_SPECS = {
    "yolox-tiny": dict(depth=0.33, width=0.375, depthwise=False, weights="yolox_tiny.pth",
                       url="https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.pth"),
    "yolox-s": {"depth": 0.33, "width": 0.5, "depthwise": False, "weights": "yolox_s.pth",
                "url": "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth"},
    "yolox-nano": dict(depth=0.33, width=0.25, depthwise=True, weights="yolox_nano.pth",
                       url="https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.pth"),
}
NUM_CLASSES = 80
VALID_BACKENDS = ("pytorch", "onnxruntime")
VALID_GRAPH_OPT = ("none", "basic", "all")

# 80-class contiguous index -> COCO category id
COCO_CATEGORY_IDS = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28,
    31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
    56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
    85, 86, 87, 88, 89, 90,
]
COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush",
]
STREET_CLASSES = ["person", "bicycle", "car", "motorcycle", "bus", "truck", "traffic light",
                  "stop sign"]


@dataclass(frozen=True)
class RunConfig:
    """One experimental configuration (a point in the sweep)."""
    model: str = "yolox-tiny"
    backend: str = "pytorch"
    resolution: int = 416
    threads: int = 4
    graph_opt: str = "all"      # onnxruntime only
    precision: str = "fp32"
    batch_size: int = 1

    def __post_init__(self):
        if self.model not in MODEL_SPECS:
            raise ValueError(f"unknown model {self.model!r}; choose from {list(MODEL_SPECS)}")
        if self.backend not in VALID_BACKENDS:
            raise ValueError(f"unknown backend {self.backend!r}; choose from {VALID_BACKENDS}")
        if self.resolution <= 0 or self.resolution % 32:
            raise ValueError("resolution must be a positive multiple of 32 (YOLOX strides)")
        if self.threads < 1:
            raise ValueError("threads must be >= 1")
        if self.graph_opt not in VALID_GRAPH_OPT:
            raise ValueError(f"graph_opt must be one of {VALID_GRAPH_OPT}")
        if self.batch_size != 1:
            raise ValueError("only batch size 1 is supported")
        if self.precision not in ("fp32", "int8"):
            raise ValueError("precision must be fp32 or int8")
        if self.precision == "int8" and self.backend != "onnxruntime":
            raise ValueError("int8 is only available with the onnxruntime backend")

    @property
    def id(self) -> str:
        if self.backend == "pytorch":
            return f"pt-{self.model}-{self.resolution}-t{self.threads}"
        base = f"ort-{self.model}-{self.resolution}-t{self.threads}-go{self.graph_opt}"
        return base + ("-int8" if self.precision == "int8" else "")

    def to_dict(self) -> dict:
        return asdict(self) | {"id": self.id}


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()
