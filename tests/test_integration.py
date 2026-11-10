"""Real weights + COCO data. Run: pytest -m integration"""
import cv2
import pytest

from edgeforge.config import MANIFEST_DIR, MODELS_DIR, RunConfig

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def img():
    from edgeforge.evaluate.dataset import image_path, load_manifest
    iid = load_manifest(next(MANIFEST_DIR.glob("eval_*.json")))["image_ids"][2]
    return cv2.imread(str(image_path(iid)))


def test_pytorch_and_ort_agree_on_real_image(img):
    from edgeforge.backends import build_backend
    from edgeforge.export.onnx_export import export_onnx
    assert (MODELS_DIR / "yolox_tiny.pth").exists()
    export_onnx("yolox-tiny", 416)
    a = build_backend(RunConfig(backend="pytorch")).detect(img)
    b = build_backend(RunConfig(backend="onnxruntime")).detect(img)
    assert len(a) > 0 and len(a) == len(b)
    assert abs(a.scores[0] - b.scores[0]) < 1e-3


def test_parity_passes():
    from edgeforge.evaluate.dataset import load_manifest
    from edgeforge.evaluate.run_config import run_parity
    ids = load_manifest(next(MANIFEST_DIR.glob("eval_*.json")))["image_ids"]
    assert run_parity(RunConfig(backend="onnxruntime"), ids)["passed"]
