"""Pareto frontier: minimise latency, maximise mAP. Only eligible points participate."""
from __future__ import annotations

import math


def dominates(a: dict, b: dict) -> bool:
    """a dominates b: no worse in both objectives and strictly better in at least one."""
    return (a["latency"] <= b["latency"] and a["map"] >= b["map"]
            and (a["latency"] < b["latency"] or a["map"] > b["map"]))


def _complete(p: dict) -> bool:
    for k in ("latency", "map"):
        v = p.get(k)
        if v is None or isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            return False
    return True


def eligible(p: dict) -> bool:
    return bool(p.get("valid", False)) and _complete(p)


def pareto_front(points: list[dict]) -> list[str]:
    """Return ids of non-dominated eligible points. Exact ties are all kept."""
    el = [p for p in points if eligible(p)]
    return [p["id"] for p in el if not any(dominates(q, p) for q in el if q is not p)]
