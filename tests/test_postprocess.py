import numpy as np
import pytest

from edgeforge.postprocess import batched_nms, decode_boxes, make_grids, nms, postprocess


def test_grid_layout_and_count():
    g, s = make_grids(64)
    assert g.shape == (64 + 16 + 4, 2) and s.shape == (84, 1)
    assert tuple(g[1]) == (1, 0)              # x varies fastest
    assert s[0, 0] == 8 and s[-1, 0] == 32


def test_decode_boxes_known_value():
    g, s = make_grids(64)
    raw = np.zeros((84, 85), np.float32)
    raw[1, :2] = (0.5, 0.25); raw[1, 2:4] = (0.0, np.log(2.0))
    d = decode_boxes(raw, 64)
    assert d[1, 0] == pytest.approx((0.5 + 1) * 8) and d[1, 1] == pytest.approx(0.25 * 8)
    assert d[1, 2] == pytest.approx(8) and d[1, 3] == pytest.approx(16)


def test_decode_wrong_anchor_count():
    with pytest.raises(ValueError):
        decode_boxes(np.zeros((10, 85), np.float32), 64)


def test_nms_suppresses_overlap_keeps_disjoint_and_order():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], np.float32)
    scores = np.array([0.8, 0.9, 0.7], np.float32)
    assert nms(boxes, scores, 0.5).tolist() == [1, 2]


def test_nms_is_class_aware():
    boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 10]], np.float32)
    keep = batched_nms(boxes, np.array([0.9, 0.8], np.float32), np.array([0, 1]), 0.5)
    assert sorted(keep.tolist()) == [0, 1]
    keep = batched_nms(boxes, np.array([0.9, 0.8], np.float32), np.array([0, 0]), 0.5)
    assert keep.tolist() == [0]


def test_nms_tie_is_stable_and_empty_ok():
    boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 10]], np.float32)
    assert nms(boxes, np.array([0.5, 0.5], np.float32), 0.5).tolist() == [0]
    assert batched_nms(np.zeros((0, 4), np.float32), np.zeros(0, np.float32),
                       np.zeros(0, np.int64), 0.5).size == 0


def test_postprocess_end_to_end_coordinates_and_rescale():
    size = 64
    raw = np.zeros((84, 85), np.float32)
    # anchor 10 is on stride-8 grid cell (x=2,y=1): box centre (2+0)*8=16,(1+0)*8=8, w=h=16 (exp(log2)*8)
    raw[10, 2:4] = np.log(2.0); raw[10, 4] = 0.9; raw[10, 5 + 2] = 0.9
    det = postprocess(raw[None], size, ratio=0.5, image_size=(100, 100), conf_thr=0.5)
    assert len(det) == 1 and det.class_ids[0] == 2
    assert det.scores[0] == pytest.approx(0.81)
    # xyxy in input px = (8,0,24,16) -> /0.5 = (16,0,48,32)
    assert det.boxes[0].tolist() == pytest.approx([16, 0, 48, 32])
    det.validate()


def test_postprocess_rejects_nan_and_handles_no_detections():
    raw = np.zeros((84, 85), np.float32)
    assert len(postprocess(raw, 64, 1.0, (64, 64))) == 0
    raw[0, 0] = np.nan
    with pytest.raises(ValueError):
        postprocess(raw, 64, 1.0, (64, 64))
