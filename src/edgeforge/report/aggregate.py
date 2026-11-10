"""Merge a latency run and an accuracy run into the single file the website reads.

Latency, memory and drift come from `bench_dir` (clean, quiet-machine sweep);
accuracy, parity, detections and validity come from `acc_dir` (full-val sweep).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..evaluate.parity import detection_agreement
from ..pareto import pareto_front

MIN_MAP_FRACTION = 0.5       # accuracy-collapse gate relative to the reference
INT8_MIN_AGREEMENT_F1 = 0.5   # sanity gate: a quantized model must still broadly agree with FP32


def bootstrap_p50_ci(samples, n_boot: int = 2000, seed: int = 0) -> list[float]:
    a = np.asarray(samples, dtype=np.float64)
    rng = np.random.default_rng(seed)
    meds = np.median(rng.choice(a, size=(n_boot, a.size), replace=True), axis=1)
    return [float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))]


def _load(d: Path) -> dict[str, dict]:
    return {p.stem: json.loads(p.read_text()) for p in sorted(d.glob("*.json"))}


def build_report(bench_dir: Path, acc_dir: Path, reference_id: str, out_file: Path,
                 extra: dict | None = None) -> dict:
    bench, acc = _load(bench_dir), _load(acc_dir)
    drift = [c for c in bench.values() if c.get("is_drift_check")]
    bench = {k: v for k, v in bench.items() if not v.get("is_drift_check")}
    if reference_id not in bench or reference_id not in acc:
        raise ValueError(f"reference {reference_id} missing from bench or accuracy run")
    missing = set(bench) ^ set(acc)
    if missing:
        raise ValueError(f"configs present in only one run: {sorted(missing)}")
    hashes = {c["dataset"]["manifest_sha256"] for c in acc.values()}

    def dets(cid):
        p = acc_dir / "detections" / f"{cid}.json"
        return json.loads(p.read_text()) if p.exists() else None
    ref_d = dets(reference_id)

    rows = []
    for cid, b in bench.items():
        a = acc[cid]
        cfg = a["config"]
        agree = detection_agreement(ref_d, dets(cid)) if (ref_d and cid != reference_id
                                                          and dets(cid)) else None
        reasons = list(a["invalid_reasons"])
        if len(hashes) > 1:
            reasons.append("evaluated on a different dataset manifest")
        if cfg["precision"] == "int8" and agree and agree["f1"] < INT8_MIN_AGREEMENT_F1:
            reasons.append(f"int8 agreement F1 {agree['f1']:.2f} below {INT8_MIN_AGREEMENT_F1}")
        ref_map = acc[reference_id]["accuracy"]["mAP"]
        if a["accuracy"] and a["accuracy"]["mAP"] < MIN_MAP_FRACTION * ref_map:
            reasons.append(f"accuracy collapse: mAP {a['accuracy']['mAP']:.3f} < "
                           f"{MIN_MAP_FRACTION:.0%} of reference {ref_map:.3f}")
        if b.get("noisy"):
            reasons.append("latency repeats disagreed by "
                           f"{b.get('repeat_spread_pct') or 0:.0f}% after retries (noisy machine)")
        valid = a["valid"] and not reasons
        lat = b["benchmark"]["headline"]["inference"]["p50"]
        samples = [x for r in b["benchmark"]["runs"] for x in r["samples_ms"]["inference"]]
        acc_ = a["accuracy"]
        rows.append({
            "id": cid, "config": cfg, "provider": b["provider"], "valid": valid,
            "invalid_reasons": reasons, "latency_p50_ms": lat,
            "latency_p50_ci95": bootstrap_p50_ci(samples),
            "repeat_spread_pct": b.get("repeat_spread_pct"),
            "benchmark": b["benchmark"], "memory": b["memory"], "model_file": b["model_file"],
            "accuracy": acc_, "parity": a["parity"], "quantization": a.get("quantization"),
            "agreement_vs_reference": agree, "model_load_s": b["model_load_s"],
        })
    front = pareto_front([{"id": r["id"], "valid": r["valid"], "latency": r["latency_p50_ms"],
                           "map": r["accuracy"]["mAP"] if r["accuracy"] else None} for r in rows])
    for r in rows:
        r["pareto_optimal"] = r["id"] in front
    ref_b = bench[reference_id]
    report = {"schema_version": 2, "reference_id": reference_id, "env": ref_b["env"],
              "accuracy_env": acc[reference_id]["env"], "dataset": acc[reference_id]["dataset"],
              "pareto_ids": front, "configs": rows, "benchmark_mode": ref_b["benchmark"]["mode"],
              "drift_checks": [{"id": c["id"], "p50_inference_ms":
                                c["benchmark"]["headline"]["inference"]["p50"]} for c in drift]}
    report.update(extra or {})
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(report))
    return report
