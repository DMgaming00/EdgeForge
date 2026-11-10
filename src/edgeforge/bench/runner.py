"""Per-configuration benchmark: warmup, timed loop, stage separation, memory."""
from __future__ import annotations

import resource
import sys

import cv2
import numpy as np
import psutil

from ..backends.base import Backend
from ..evaluate.dataset import image_path
from .stats import MODES, fps_from_latency_ms, summarize

STAGES = ("preprocess", "inference", "postprocess", "total")


def peak_rss_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 2**20 if sys.platform == "darwin" else r / 1024  # darwin: bytes, linux: KiB


def current_rss_mb() -> float:
    return psutil.Process().memory_info().rss / 2**20


def load_bench_images(image_ids: list[int], n: int = 20) -> list[np.ndarray]:
    imgs = [cv2.imread(str(image_path(i))) for i in image_ids[:n]]
    if any(i is None for i in imgs):
        raise FileNotFoundError("missing benchmark images; run `edgeforge prepare`")
    return imgs


def run_benchmark(backend: Backend, images: list[np.ndarray], mode: str = "standard") -> dict:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {list(MODES)}")
    warmup, n, repeats = MODES[mode]
    for i in range(warmup):                       # excluded from stats
        backend.detect(images[i % len(images)])
    runs = []
    for _ in range(repeats):
        samples = {s: [] for s in STAGES}
        for i in range(n):
            t = backend.detect(images[i % len(images)]).meta["timing_ns"]
            for s in STAGES:
                samples[s].append(t[s] / 1e6)
        runs.append({"samples_ms": samples,
                     "summary": {s: summarize(samples[s]) for s in STAGES}})
    # headline = median over repeats of each repeat's p50 / p95 / p99 (model-only inference)
    def med(stage, key):
        return float(np.median([r["summary"][stage][key] for r in runs]))
    headline = {s: {k: med(s, k) for k in ("mean", "p50", "p95", "p99", "min", "max")}
                for s in STAGES}
    total_p50 = headline["total"]["p50"]
    return {"mode": mode, "warmup": warmup, "n_per_repeat": n, "repeats": repeats,
            "metric_definitions": {
                "primary_latency": "inference.p50 = model forward only (ms), batch 1, "
                                   "median over repeats of per-repeat p50",
                "pipeline_latency": "total.p50 = preprocess + inference + decode/NMS (ms)"},
            "headline": headline,
            "throughput_fps_pipeline": fps_from_latency_ms(total_p50),
            "throughput_fps_model_only": fps_from_latency_ms(headline["inference"]["p50"]),
            "n_distinct_images": len(images), "runs": runs}
