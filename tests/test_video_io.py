import cv2
import numpy as np
import pytest

from edgeforge.config import RunConfig
from edgeforge.types import Detections
from edgeforge.video.process import Hud, draw_detections, draw_hud, read_frames


def _write(path, n=5):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    for i in range(n):
        w.write(np.full((48, 64, 3), i * 20, np.uint8))
    w.release()


def test_read_frames_roundtrip_and_limit(tmp_path):
    p = tmp_path / "v.mp4"; _write(p, 5)
    assert len(list(read_frames(str(p), 100))) == 5
    assert len(list(read_frames(str(p), 3))) == 3


def test_read_frames_missing_file():
    with pytest.raises(FileNotFoundError):
        list(read_frames("/nonexistent.mp4", 1))


def test_draw_respects_threshold_and_hud_runs():
    f = np.zeros((100, 100, 3), np.uint8)
    d = Detections(np.array([[10, 10, 50, 50], [60, 60, 90, 90]], np.float32),
                   np.array([0.9, 0.1], np.float32), np.array([0, 2]), (100, 100))
    assert draw_detections(f, d, 0.35) == 1
    hud = Hud(RunConfig()); hud.update(10.0, 20.0)
    assert hud.rolling_fps == pytest.approx(50.0) and hud.rolling_infer_ms == 10.0
    draw_hud(f, hud, 1, "CPUExecutionProvider")
