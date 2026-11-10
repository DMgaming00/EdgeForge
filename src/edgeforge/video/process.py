"""Frame-by-frame video annotation with a HUD of MEASURED values only."""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from ..backends import build_backend
from ..config import COCO_NAMES, STREET_CLASSES, RunConfig
from ..types import Detections

# design tokens (BGR for OpenCV)
BG, PANEL, TEXT, DIM, ACCENT = (13, 15, 14), (21, 25, 23), (220, 231, 236), (132, 143, 138), (0, 169, 242)
TEAL = (184, 168, 111)


def _color(cid: int) -> tuple[int, int, int]:
    return ACCENT if cid in (0, 1, 3) else TEAL if cid in (2, 5, 7) else (119, 180, 79)


@dataclass
class Hud:
    cfg: RunConfig
    window: int = 30
    lat: deque = field(default_factory=lambda: deque(maxlen=30))
    frame_t: deque = field(default_factory=lambda: deque(maxlen=30))

    def update(self, infer_ms: float, frame_ms: float):
        self.lat.append(infer_ms); self.frame_t.append(frame_ms)

    @property
    def rolling_infer_ms(self): return float(np.mean(self.lat)) if self.lat else 0.0

    @property
    def rolling_fps(self): return 1000.0 / float(np.mean(self.frame_t)) if self.frame_t else 0.0


STREET_IDS = frozenset(COCO_NAMES.index(n) for n in STREET_CLASSES)


def draw_detections(frame: np.ndarray, det: Detections, conf: float,
                    only: frozenset[int] | None = None) -> int:
    """Draw boxes >= conf (optionally only the given class ids: a DISPLAY filter)."""
    n = 0
    for b, s, c in zip(det.boxes, det.scores, det.class_ids, strict=True):
        if s < conf or (only is not None and int(c) not in only):
            continue
        n += 1
        x1, y1, x2, y2 = (int(v) for v in b)
        col = _color(int(c))
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
        label = f"{COCO_NAMES[int(c)]} {s:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, y1), col, -1)
        cv2.putText(frame, label, (x1 + 3, max(th, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BG, 1,
                    cv2.LINE_AA)
    return n


def draw_hud(frame: np.ndarray, hud: Hud, n_det: int, provider: str) -> None:
    c = hud.cfg
    lines = [f"{c.model}  {c.backend}  {c.precision.upper()}  {c.resolution}px  CPU t{c.threads}",
             f"inference {hud.rolling_infer_ms:6.1f} ms (rolling {hud.window}f)",
             f"pipeline  {hud.rolling_fps:5.1f} fps  OFFLINE",
             f"detections {n_det}   [{provider}]"]
    pad, lh = 8, 18
    w = 330
    ov = frame.copy()
    cv2.rectangle(ov, (8, 8), (8 + w, 8 + pad * 2 + lh * len(lines)), PANEL, -1)
    cv2.addWeighted(ov, 0.85, frame, 0.15, 0, frame)
    cv2.rectangle(frame, (8, 8), (8 + w, 8 + pad * 2 + lh * len(lines)), ACCENT, 1)
    for i, t in enumerate(lines):
        cv2.putText(frame, t, (8 + pad, 8 + pad + lh * (i + 1) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    TEXT if i else ACCENT, 1, cv2.LINE_AA)


def draw_latency_strip(width: int, lat: list[float], ymax: float, color, h: int = 64) -> np.ndarray:
    """Rolling per-frame inference latency (ms), last 120 frames. Measured values only."""
    strip = np.full((h, width, 3), BG, np.uint8)
    cv2.line(strip, (0, 0), (width, 0), PANEL, 1)
    data = lat[-120:]
    if len(data) > 1:
        pts = np.array([[int(i * (width - 90) / 119 + 8), int(h - 8 - min(v, ymax) / ymax * (h - 20))]
                        for i, v in enumerate(data)], np.int32)
        cv2.polylines(strip, [pts], False, color, 1, cv2.LINE_AA)
    cv2.putText(strip, f"inference ms (0-{ymax:.0f})  now {data[-1] if data else 0:.1f}",
                (width - 250, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, DIM, 1, cv2.LINE_AA)
    return strip


def read_frames(src: str, max_frames: int):
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video {src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    try:
        i = 0
        while i < max_frames:
            ok, f = cap.read()
            if not ok:
                break
            yield f, fps
            i += 1
    finally:
        cap.release()


def process_video(src: str, dst: str, cfg: RunConfig, cfg2: RunConfig | None = None,
                  max_frames: int = 300, conf: float = 0.35,
                  street_only: bool = True) -> dict:
    """Annotate up to max_frames. With cfg2, writes a side-by-side comparison.

    Returned FPS is OFFLINE processing throughput (decode excluded, draw/encode excluded
    from inference latency but included in none); it is not a live-camera claim.
    """
    backends = [build_backend(cfg)] + ([build_backend(cfg2)] if cfg2 else [])
    huds = [Hud(b.cfg) for b in backends]
    stats = [{"infer_ms": [], "pipeline_ms": [], "detections": 0} for _ in backends]
    writer = None
    n = 0
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    tmp = str(Path(dst).with_suffix(".raw.mp4"))
    out_fps = 30.0
    t_wall = time.perf_counter()
    for frame, fps in read_frames(src, max_frames):
        out_fps = fps
        panels = []
        for k, b in enumerate(backends):
            t0 = time.perf_counter_ns()
            det = b.detect(frame)
            ms = (time.perf_counter_ns() - t0) / 1e6
            tm = det.meta["timing_ns"]
            huds[k].update(tm["inference"] / 1e6, ms)
            stats[k]["infer_ms"].append(tm["inference"] / 1e6)
            stats[k]["pipeline_ms"].append(ms)
            img = frame.copy()
            nd = draw_detections(img, det, conf, STREET_IDS if street_only else None)
            stats[k]["detections"] += nd
            draw_hud(img, huds[k], nd, b.provider)
            panels.append(img)
        ymax = max(max(st["infer_ms"][-120:]) for st in stats) * 1.15
        panels = [np.vstack([p, draw_latency_strip(p.shape[1], stats[k]["infer_ms"], ymax,
                                                   (ACCENT, TEAL)[k % 2])])
                  for k, p in enumerate(panels)]
        out = np.hstack(panels) if len(panels) > 1 else panels[0]
        if writer is None:
            h, w = out.shape[:2]
            writer = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (w, h))
            if not writer.isOpened():
                raise RuntimeError("cannot open video writer")
        writer.write(out)
        n += 1
    if writer is None:
        raise ValueError("no frames decoded from input video")
    writer.release()
    _reencode_h264(tmp, dst)
    summary = {"frames": n, "wall_s": time.perf_counter() - t_wall, "output": str(dst),
               "street_classes_only": street_only, "conf_threshold": conf,
               "note": "fps values are offline processing throughput, not live inference claims",
               "runs": []}
    for b, s in zip(backends, stats, strict=True):
        inf, pipe = np.array(s["infer_ms"]), np.array(s["pipeline_ms"])
        summary["runs"].append({
            "config": b.cfg.to_dict(), "provider": b.provider,
            "inference_ms_p50": float(np.percentile(inf, 50)),
            "inference_ms_p95": float(np.percentile(inf, 95)),
            "pipeline_ms_p50": float(np.percentile(pipe, 50)),
            "offline_fps": float(1000.0 / pipe.mean()), "detections_total": s["detections"]})
    return summary


def _reencode_h264(tmp: str, dst: str) -> None:
    """Re-encode to H.264 for browser playback if ffmpeg exists; else keep mp4v."""
    import shutil
    import subprocess
    if shutil.which("ffmpeg"):
        r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-c:v", "libx264",
                            "-pix_fmt", "yuv420p", "-crf", "23", "-movflags", "+faststart", dst])
        if r.returncode == 0:
            Path(tmp).unlink()
            return
    Path(tmp).replace(dst)
