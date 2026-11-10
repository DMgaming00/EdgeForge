import json

import pytest

from edgeforge.config import RunConfig
from edgeforge.report.aggregate import bootstrap_p50_ci, build_report


def test_int8_requires_onnxruntime_and_id():
    with pytest.raises(ValueError):
        RunConfig(backend="pytorch", precision="int8")
    c = RunConfig(backend="onnxruntime", precision="int8")
    assert c.id.endswith("-int8") and c.to_dict()["precision"] == "int8"


def test_bootstrap_ci_brackets_median_and_is_deterministic():
    x = list(range(1, 102))
    lo, hi = bootstrap_p50_ci(x)
    assert lo <= 51 <= hi and (lo, hi) == tuple(bootstrap_p50_ci(x))


def _res(cid, lat, mAP, precision="fp32", agree_ok=True, noisy=False, valid=True):
    summ = {k: {"p50": lat, "p95": lat, "p99": lat, "mean": lat, "min": lat, "max": lat}
            for k in ("preprocess", "inference", "postprocess", "total")}
    base = {"id": cid, "config": {"id": cid, "precision": precision, "backend": "onnxruntime"},
            "provider": "CPU", "model_load_s": 0.1, "env": {}, "memory": {"peak_rss_mb": 1},
            "model_file": {}, "dataset": {"manifest_sha256": "h"}, "parity": None,
            "benchmark": {"mode": "smoke", "headline": summ,
                          "runs": [{"samples_ms": {"inference": [lat] * 5}}]},
            "accuracy": {"mAP": mAP}, "invalid_reasons": [], "valid": valid, "noisy": noisy}
    return base


def _write(d, objs, dets):
    d.mkdir()
    (d / "detections").mkdir()
    for o in objs:
        (d / f"{o['id']}.json").write_text(json.dumps(o))
        (d / "detections" / f"{o['id']}.json").write_text(json.dumps(dets[o["id"]]))


def test_report_merges_and_gates_invalid(tmp_path):
    one = [{"boxes": [[0, 0, 10, 10]], "scores": [0.9], "class_ids": [1]}]
    none = [{"boxes": [], "scores": [], "class_ids": []}]
    objs = [_res("ref", 40, 0.34), _res("q", 10, 0.33, "int8"), _res("noisy", 5, 0.34, noisy=True)]
    dets = {"ref": one, "q": none, "noisy": one}    # q disagrees completely -> invalid
    _write(tmp_path / "b", objs, dets); _write(tmp_path / "a", objs, dets)
    r = build_report(tmp_path / "b", tmp_path / "a", "ref", tmp_path / "out.json")
    by = {c["id"]: c for c in r["configs"]}
    assert not by["q"]["valid"] and "int8 agreement" in by["q"]["invalid_reasons"][0]
    assert not by["noisy"]["valid"]          # noisy latency never reaches the frontier
    assert r["pareto_ids"] == ["ref"]


def test_report_rejects_mismatched_runs(tmp_path):
    objs = [_res("ref", 40, 0.34)]
    _write(tmp_path / "b", objs, {"ref": []}); _write(tmp_path / "a", objs, {"ref": []})
    (tmp_path / "a" / "extra.json").write_text(json.dumps(_res("extra", 1, 0.1)))
    with pytest.raises(ValueError):
        build_report(tmp_path / "b", tmp_path / "a", "ref", tmp_path / "o.json")


def test_accuracy_collapse_gate(tmp_path):
    objs = [_res("ref", 40, 0.34), _res("broken", 5, 0.01)]
    d = {"ref": [], "broken": []}
    _write(tmp_path / "b", objs, d); _write(tmp_path / "a", objs, d)
    r = build_report(tmp_path / "b", tmp_path / "a", "ref", tmp_path / "o.json")
    by = {c["id"]: c for c in r["configs"]}
    assert not by["broken"]["valid"] and "accuracy collapse" in by["broken"]["invalid_reasons"][0]
    assert r["pareto_ids"] == ["ref"]
