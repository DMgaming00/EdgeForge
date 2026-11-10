"""Numerical parity metrics and detection agreement."""
from __future__ import annotations

import numpy as np

from ..postprocess import iou_one_to_many


def parity_metrics(ref: np.ndarray, test: np.ndarray, atol: float, rtol: float) -> dict:
    if ref.shape != test.shape:
        raise ValueError(f"shape mismatch {ref.shape} vs {test.shape}")
    diff = np.abs(ref.astype(np.float64) - test.astype(np.float64))
    denom = np.abs(ref.astype(np.float64))
    meaningful = denom > 1e-3          # relative error is meaningless near zero
    rel = diff[meaningful] / denom[meaningful] if meaningful.any() else np.zeros(1)
    passed = bool(np.allclose(test, ref, atol=atol, rtol=rtol))
    return {"max_abs_err": float(diff.max()), "mean_abs_err": float(diff.mean()),
            "max_rel_err": float(rel.max()), "atol": atol, "rtol": rtol, "passed": passed}


def detection_agreement(ref_dets: list[dict], test_dets: list[dict], iou_thr: float = 0.5,
                        score_thr: float = 0.3) -> dict:
    """Per-image greedy matching of same-class boxes between two configs (above score_thr).

    ref_dets/test_dets: lists (aligned per image) of {"boxes","scores","class_ids"} lists.
    Returns precision/recall/F1 of `test` w.r.t. `ref` treated as reference (NOT accuracy).
    """
    tp = n_ref = n_test = 0
    for r, t in zip(ref_dets, test_dets, strict=True):
        rb = np.asarray(r["boxes"], np.float32).reshape(-1, 4)
        rs = np.asarray(r["scores"]); rc = np.asarray(r["class_ids"])
        tb = np.asarray(t["boxes"], np.float32).reshape(-1, 4)
        ts = np.asarray(t["scores"]); tc = np.asarray(t["class_ids"])
        rm, tm = rs >= score_thr, ts >= score_thr
        rb, rc = rb[rm], rc[rm]
        tb, tc, ts = tb[tm], tc[tm], ts[tm]
        n_ref += len(rb); n_test += len(tb)
        used = np.zeros(len(rb), bool)
        for j in np.argsort(-ts):
            if not len(rb):
                break
            ious = iou_one_to_many(tb[j], rb)
            ious[used | (rc != tc[j])] = -1
            k = int(ious.argmax())
            if ious[k] >= iou_thr:
                used[k] = True; tp += 1
    p = tp / n_test if n_test else 1.0 if not n_ref else 0.0
    r = tp / n_ref if n_ref else 1.0 if not n_test else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f1, "matched": tp, "n_ref": n_ref, "n_test": n_test,
            "iou_thr": iou_thr, "score_thr": score_thr}
