import pytest

from edgeforge.bench.stats import drift_pct, fps_from_latency_ms, percentile, summarize
from edgeforge.pareto import dominates, pareto_front


def test_percentiles_match_numpy_known():
    x = list(range(1, 101))
    assert percentile(x, 50) == pytest.approx(50.5)
    assert percentile(x, 95) == pytest.approx(95.05)
    s = summarize(x)
    assert s["min"] == 1 and s["max"] == 100 and s["n"] == 100 and s["p99"] == pytest.approx(99.01)


def test_summary_empty_raises():
    with pytest.raises(ValueError):
        summarize([])


def test_fps_and_drift():
    assert fps_from_latency_ms(20.0) == 50.0
    with pytest.raises(ValueError):
        fps_from_latency_ms(0)
    assert drift_pct(10, 11) == pytest.approx(10.0)


def P(i, lat, m, valid=True):
    return {"id": i, "latency": lat, "map": m, "valid": valid}


def test_dominance_rules():
    assert dominates(P("a", 1, 0.5), P("b", 2, 0.4))
    assert dominates(P("a", 1, 0.5), P("b", 1, 0.4))        # equal latency, better mAP
    assert not dominates(P("a", 1, 0.5), P("b", 1, 0.5))    # exact tie: neither dominates
    assert not dominates(P("a", 1, 0.3), P("b", 2, 0.4))    # trade-off


def test_front_basic_and_ties():
    pts = [P("fast", 10, 0.30), P("mid", 20, 0.35), P("slow", 40, 0.36), P("bad", 30, 0.30),
           P("tie", 10, 0.30)]
    assert set(pareto_front(pts)) == {"fast", "tie", "mid", "slow"}


def test_front_excludes_invalid_and_incomplete():
    pts = [P("ok", 20, 0.3), P("invalid_fast", 5, 0.9, valid=False), P("nomap", 1, None),
           P("nan", float("nan"), 0.5), {"id": "missing", "valid": True}]
    assert pareto_front(pts) == ["ok"]


def test_front_empty():
    assert pareto_front([]) == []
