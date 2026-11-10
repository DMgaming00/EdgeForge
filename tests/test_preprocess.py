import numpy as np
import pytest

from edgeforge.preprocess import PAD_VALUE, letterbox


def test_letterbox_shape_ratio_and_padding():
    img = np.random.default_rng(1).integers(0, 255, (200, 400, 3), dtype=np.uint8)
    out, r = letterbox(img, 416)
    assert out.shape == (3, 416, 416) and out.dtype == np.float32
    assert r == pytest.approx(1.04)
    assert (out[:, 209:, :] == PAD_VALUE).all()          # bottom padding
    assert not (out[:, :208, :] == PAD_VALUE).all()


def test_letterbox_deterministic():
    img = np.random.default_rng(2).integers(0, 255, (123, 321, 3), dtype=np.uint8)
    a, _ = letterbox(img, 320); b, _ = letterbox(img, 320)
    assert np.array_equal(a, b)


def test_letterbox_rejects_bad_input():
    with pytest.raises(ValueError):
        letterbox(np.zeros((10, 10), np.uint8), 320)
