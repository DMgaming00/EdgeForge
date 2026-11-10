"""Hardware / software provenance."""
from __future__ import annotations

import datetime as dt
import os
import platform
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version

from .config import ROOT


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return ""


def _pkg(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def cpu_model() -> str:
    if sys.platform == "darwin":
        return _run(["sysctl", "-n", "machdep.cpu.brand_string"]) or platform.processor()
    return platform.processor() or platform.machine()


def git_state() -> dict:
    h = _run(["git", "-C", str(ROOT), "rev-parse", "HEAD"])
    if not h:
        return {"commit": None, "dirty": None}
    return {"commit": h, "dirty": bool(_run(["git", "-C", str(ROOT), "status", "--porcelain"]))}


def power_state() -> str:
    if sys.platform == "darwin":
        line = _run(["pmset", "-g", "batt"]).splitlines()
        return line[0].replace("Now drawing from ", "").strip("'") if line else "unknown"
    return "unknown"


def collect_env() -> dict:
    import psutil
    mem = psutil.virtual_memory().total
    return {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "git": git_state(),
        "os": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "cpu_model": cpu_model(),
        "cpu_cores_logical": os.cpu_count(),
        "cpu_cores_physical": psutil.cpu_count(logical=False),
        "ram_gb": round(mem / 2**30, 1),
        "python": platform.python_version(),
        "torch": _pkg("torch"),
        "onnxruntime": _pkg("onnxruntime"),
        "onnx": _pkg("onnx"),
        "numpy": _pkg("numpy"),
        "opencv": _pkg("opencv-python-headless") or _pkg("opencv-python"),
        "pycocotools": _pkg("pycocotools"),
        "yolox": _pkg("yolox"),
        "power_source": power_state(),
        "load_avg_1m_at_start": round(os.getloadavg()[0], 2),
        "notes": ("Laptop CPU; no thermal-state control. Power source recorded at run start. "
                  "Background load not controlled."),
    }
