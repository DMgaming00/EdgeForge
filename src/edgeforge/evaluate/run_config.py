"""Run one configuration end to end: bench -> COCO eval -> parity. Writes one JSON."""
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2

from ..backends import build_backend
from ..bench.runner import current_rss_mb, load_bench_images, peak_rss_mb, run_benchmark
from ..config import MANIFEST_DIR, RunConfig, sha256_file
from ..env_info import collect_env
from ..preprocess import letterbox
from .coco_eval import evaluate_coco, to_coco_results
from .dataset import image_path, load_manifest, manifest_hash
from .parity import parity_metrics

PARITY_ATOL, PARITY_RTOL = 1e-3, 1e-3   # raw head outputs span ~1e-6..~4e2 (pixel units)
PARITY_IMAGES = 20


def _model_file(cfg: RunConfig) -> Path:
    from ..export.onnx_export import onnx_path_for
    from ..models.yolox_loader import weights_path
    return onnx_path_for(cfg.model, cfg.resolution, cfg.precision) if cfg.backend == "onnxruntime" \
        else weights_path(cfg.model)


def run_parity(cfg: RunConfig, image_ids: list[int]) -> dict | None:
    """ORT FP32 vs PyTorch FP32 raw outputs at the same resolution/weights."""
    if cfg.backend != "onnxruntime":
        return None
    from dataclasses import replace
    pt = build_backend(replace(cfg, backend="pytorch", graph_opt="all", precision="fp32"))
    ort_b = build_backend(cfg)
    agg = {"max_abs_err": 0.0, "max_rel_err": 0.0, "mean_abs_err": 0.0, "passed": True}
    for iid in image_ids[:PARITY_IMAGES]:
        x = letterbox(cv2.imread(str(image_path(iid))), cfg.resolution)[0][None]
        m = parity_metrics(pt.infer(x), ort_b.infer(x), PARITY_ATOL, PARITY_RTOL)
        agg["max_abs_err"] = max(agg["max_abs_err"], m["max_abs_err"])
        agg["max_rel_err"] = max(agg["max_rel_err"], m["max_rel_err"])
        agg["mean_abs_err"] += m["mean_abs_err"] / min(PARITY_IMAGES, len(image_ids))
        agg["passed"] &= m["passed"]
    agg["gating"] = cfg.precision == "fp32"   # int8: reported as quantization error, not gated
    agg.update(atol=PARITY_ATOL, rtol=PARITY_RTOL, n_images=min(PARITY_IMAGES, len(image_ids)),
               reference="PyTorch FP32 eager, same weights/resolution/preprocessing")
    return agg


def run_config(cfg: RunConfig, out_dir: Path, mode: str = "standard", do_eval: bool = True,
               manifest: Path | None = None, label: str | None = None) -> dict:
    manifest = manifest or next(MANIFEST_DIR.glob("eval_*.json"))
    manifest = Path(manifest)
    man = load_manifest(manifest)
    ids = man["image_ids"]
    cid = label or cfg.id
    failures: list[str] = []
    quant_info = None
    if cfg.precision == "int8":
        from ..export.quantize import quantize_model
        _, quant_info = quantize_model(cfg.model, cfg.resolution)
    out_dir.mkdir(parents=True, exist_ok=True)

    t_load = time.perf_counter()
    backend = build_backend(cfg)
    load_s = time.perf_counter() - t_load
    rss_loaded = current_rss_mb()

    # 1) benchmark first so peak RSS reflects only this backend (+ bench images)
    bench = run_benchmark(backend, load_bench_images(ids), mode)
    peak = peak_rss_mb()

    result = {
        "id": cid, "config": cfg.to_dict(), "env": collect_env(),
        "model_file": {"path": _model_file(cfg).name, "size_mb": _model_file(cfg).stat().st_size / 2**20,
                       "sha256": sha256_file(_model_file(cfg))},
        "provider": backend.provider, "execution_device": "cpu",
        "model_load_s": load_s,
        "memory": {"rss_after_load_mb": rss_loaded, "peak_rss_mb": peak,
                   "note": "ru_maxrss of a fresh process incl. interpreter + libraries"},
        "benchmark": bench,
        "dataset": {"manifest": manifest.name, "manifest_sha256": manifest_hash(man),
                    "n_images": len(ids)},
        "accuracy": None, "parity": None, "failures": failures,
    }

    if do_eval:
        res, dets_dump = [], []
        for iid in ids:
            img = cv2.imread(str(image_path(iid)))
            if img is None:
                failures.append(f"unreadable image {iid}")
                continue
            try:
                d = backend.detect(img)
                d.validate()
            except Exception as e:  # never silently drop images
                failures.append(f"image {iid}: {e}")
                continue
            res += to_coco_results(iid, d.boxes, d.scores, d.class_ids)
            dets_dump.append({"image_id": iid, "boxes": d.boxes.round(2).tolist(),
                              "scores": d.scores.round(4).tolist(),
                              "class_ids": d.class_ids.tolist()})
        if failures:
            result["accuracy"] = None
        else:
            result["accuracy"] = evaluate_coco(res, ids)
        (out_dir / "detections").mkdir(exist_ok=True)
        (out_dir / "detections" / f"{cid}.json").write_text(json.dumps(dets_dump))
        result["parity"] = run_parity(cfg, ids)

    reasons = list(failures)
    if do_eval and result["accuracy"] is None:
        reasons.append("accuracy not computed")
    if result["parity"] is not None and result["parity"]["gating"] and not result["parity"]["passed"]:
        reasons.append("FP32 parity vs PyTorch failed")
    result["valid"] = bool(do_eval and not reasons)
    if cfg.precision == "int8":
        result["quantization"] = quant_info
    result["invalid_reasons"] = reasons
    (out_dir / f"{cid}.json").write_text(json.dumps(result, indent=1))
    return result
