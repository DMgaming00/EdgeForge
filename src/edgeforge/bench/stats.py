"""Latency statistics (pure functions, unit-tested)."""
from __future__ import annotations

import numpy as np

MODES = {
    # name: (warmup, timed measurements per repeat, repeats)
    "smoke": (3, 10, 1),
    "standard": (10, 100, 1),
    "rigorous": (20, 200, 3),
}


def percentile(samples_ms, q: float) -> float:
    if len(samples_ms) == 0:
        raise ValueError("no samples")
    return float(np.percentile(np.asarray(samples_ms, dtype=np.float64), q, method="linear"))


def summarize(samples_ms) -> dict:
    a = np.asarray(samples_ms, dtype=np.float64)
    if a.size == 0:
        raise ValueError("no samples")
    return {"n": int(a.size), "mean": float(a.mean()), "median": float(np.median(a)),
            "p50": percentile(a, 50), "p95": percentile(a, 95), "p99": percentile(a, 99),
            "min": float(a.min()), "max": float(a.max()), "std": float(a.std(ddof=1)) if a.size > 1
            else 0.0}


def fps_from_latency_ms(total_ms_per_frame: float) -> float:
    if total_ms_per_frame <= 0:
        raise ValueError("latency must be positive")
    return 1000.0 / total_ms_per_frame


def drift_pct(first: float, last: float) -> float:
    return (last - first) / first * 100.0
