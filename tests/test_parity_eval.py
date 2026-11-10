import numpy as np
import pytest

from edgeforge.config import COCO_CATEGORY_IDS, RunConfig
from edgeforge.evaluate.coco_eval import evaluate_coco, to_coco_results
from edgeforge.evaluate.parity import detection_agreement, parity_metrics
from edgeforge.types import Detections


def test_parity_pass_and_fail():
    a = np.random.default_rng(0).random((1, 10, 5)).astype(np.float32) + 1
    ok = parity_metrics(a, a + 1e-6, 1e-4, 1e-4)
    assert ok["passed"] and ok["max_abs_err"] == pytest.approx(1e-6, rel=0.1)
    bad = parity_metrics(a, a + 1e-2, 1e-4, 1e-4)
    assert not bad["passed"]
    with pytest.raises(ValueError):
        parity_metrics(a, a[:, :5], 1e-4, 1e-4)


def test_agreement_perfect_and_disjoint():
    d = [{"boxes": [[0, 0, 10, 10]], "scores": [0.9], "class_ids": [1]}]
    assert detection_agreement(d, d)["f1"] == 1.0
    other = [{"boxes": [[0, 0, 10, 10]], "scores": [0.9], "class_ids": [2]}]   # wrong class
    assert detection_agreement(d, other)["f1"] == 0.0
    both_empty = [{"boxes": [], "scores": [], "class_ids": []}]
    assert detection_agreement(both_empty, both_empty)["f1"] == 1.0


def test_to_coco_results_xywh_and_category_mapping():
    r = to_coco_results(7, [[10, 20, 30, 60]], [0.5], [11])   # class 11 -> category id 13
    assert r[0]["bbox"] == [10, 20, 20, 40] and r[0]["category_id"] == 13


def _tiny_gt():
    from pycocotools.coco import COCO
    gt = COCO()
    gt.dataset = {"images": [{"id": 1, "width": 100, "height": 100}],
                  "categories": [{"id": c, "name": str(c)} for c in COCO_CATEGORY_IDS],
                  "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 40, 40],
                                   "area": 1600, "iscrowd": 0}]}
    gt.createIndex()
    return gt


def test_coco_eval_perfect_and_wrong_detection():
    gt = _tiny_gt()
    perfect = evaluate_coco(to_coco_results(1, [[10, 10, 50, 50]], [0.9], [0]), [1], gt)
    assert perfect["mAP"] == pytest.approx(1.0, abs=1e-6)
    assert perfect["per_class"]["person"]["n_annotations"] == 1
    miss = evaluate_coco(to_coco_results(1, [[60, 60, 90, 90]], [0.9], [0]), [1], gt)
    assert miss["mAP"] == pytest.approx(0.0, abs=1e-6)
    assert evaluate_coco([], [1], gt)["mAP"] == 0.0


def test_run_config_validation():
    with pytest.raises(ValueError):
        RunConfig(resolution=100)
    with pytest.raises(ValueError):
        RunConfig(backend="tensorrt")
    with pytest.raises(ValueError):
        RunConfig(threads=0)
    with pytest.raises(ValueError):
        RunConfig(graph_opt="max")
    assert RunConfig(backend="onnxruntime", threads=2, graph_opt="none").id.endswith("t2-gonone")


def test_detections_validate():
    d = Detections(np.array([[0, 0, 1, 1]], np.float32), np.array([1.5], np.float32),
                   np.array([0]), (10, 10))
    with pytest.raises(ValueError):
        d.validate()
