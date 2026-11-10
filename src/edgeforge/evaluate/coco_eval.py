"""COCO mAP via pycocotools against ground-truth annotations."""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from ..config import ANN_FILE, COCO_CATEGORY_IDS, COCO_NAMES, STREET_CLASSES

_GT_CACHE: dict[str, COCO] = {}


def load_gt(ann_file: Path = ANN_FILE) -> COCO:
    key = str(ann_file)
    if key not in _GT_CACHE:
        with contextlib.redirect_stdout(io.StringIO()):
            _GT_CACHE[key] = COCO(key)
    return _GT_CACHE[key]


def to_coco_results(image_id: int, boxes, scores, class_ids) -> list[dict]:
    out = []
    for b, s, c in zip(boxes, scores, class_ids, strict=True):
        out.append({"image_id": int(image_id), "category_id": COCO_CATEGORY_IDS[int(c)],
                    "bbox": [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])],
                    "score": float(s)})
    return out


def evaluate_coco(results: list[dict], image_ids: list[int], gt: COCO | None = None) -> dict:
    """mAP@[.5:.95], AP50, AP75 and per-class AP restricted to `image_ids`."""
    gt = gt or load_gt()
    cat_ids = COCO_CATEGORY_IDS
    ann_counts = {}
    for cid, name in zip(cat_ids, COCO_NAMES, strict=True):
        ann_counts[name] = len(gt.getAnnIds(imgIds=image_ids, catIds=[cid], iscrowd=False))
    if not results:
        return {"mAP": 0.0, "AP50": 0.0, "AP75": 0.0, "per_class": {}, "n_images": len(image_ids),
                "n_detections": 0}
    with contextlib.redirect_stdout(io.StringIO()):
        dt = gt.loadRes(results)
        ev = COCOeval(gt, dt, "bbox")
        ev.params.imgIds = list(image_ids)
        ev.evaluate(); ev.accumulate(); ev.summarize()
    prec = ev.eval["precision"]  # [T,R,K,A,M]
    per_class = {}
    for k, name in enumerate(COCO_NAMES):
        p = prec[:, :, k, 0, 2]
        p50 = prec[0, :, k, 0, 2]
        ap = float(np.mean(p[p > -1])) if (p > -1).any() else None
        ap50 = float(np.mean(p50[p50 > -1])) if (p50 > -1).any() else None
        per_class[name] = {"AP": ap, "AP50": ap50, "n_annotations": ann_counts[name]}
    return {"mAP": float(ev.stats[0]), "AP50": float(ev.stats[1]), "AP75": float(ev.stats[2]),
            "APs": float(ev.stats[3]), "APm": float(ev.stats[4]), "APl": float(ev.stats[5]),
            "per_class": per_class, "n_images": len(image_ids),
            "n_detections": len(results), "street_classes": STREET_CLASSES}


def dump_results(results: list[dict], path: Path) -> None:
    path.write_text(json.dumps(results))
