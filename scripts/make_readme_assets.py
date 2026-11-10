"""Regenerate the README figures and results table from the recorded results (no numbers are typed in).

  pip install matplotlib
  python scripts/make_readme_assets.py     -> docs/img/frontier.png, docs/img/levers.png, docs/img/latency.png, docs/results_table.md
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/img"
OUT.mkdir(parents=True, exist_ok=True)
BG, PANEL, BORDER, TEXT, SEC = "#0E0F0D", "#171915", "#23261F", "#ECE7DC", "#8A8F84"
AMBER, OK, BAD, TEAL = "#F2A900", "#4FB477", "#E5484D", "#6FA8B8"

for p in (ROOT / "tools/fonts").glob("ibm-plex-*.ttf"):  # optional: brand fonts if present locally
    font_manager.fontManager.addfont(str(p))
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": PANEL, "savefig.facecolor": BG, "axes.edgecolor": BORDER, "axes.labelcolor": SEC,
    "text.color": TEXT, "xtick.color": SEC, "ytick.color": SEC, "grid.color": BORDER, "axes.grid": True, "grid.linewidth": 0.8,
    "font.family": ["IBM Plex Sans", "DejaVu Sans"], "font.size": 12,
})

RES = json.loads((ROOT / "site/public/data/results.json").read_text())
C = {c["id"]: c for c in RES["configs"]}
REF = RES["reference_id"]
ms = lambda c: c["latency_p50_ms"]  # noqa: E731
mp = lambda c: c["accuracy"]["mAP"] * 100  # noqa: E731


def label(c):
    cf = c["config"]
    rt = "PyTorch" if cf["backend"] == "pytorch" else "ORT"
    m = "S" if cf["model"] == "yolox-s" else "Tiny"
    extra = ""
    if cf["threads"] != 4:
        extra += f" {cf['threads']}t"
    if cf["backend"] == "onnxruntime" and cf["graph_opt"] != "all":
        extra += f" opt:{cf['graph_opt']}"
    return f"{rt} {m} {cf['resolution']}{' INT8' if cf.get('precision') == 'int8' else ''}{extra}"


# ---------------------------------------------------------------- frontier
fig, ax = plt.subplots(figsize=(12, 6.6), dpi=150)
par = sorted([c for c in RES["configs"] if c["pareto_optimal"]], key=ms)
for c in RES["configs"]:
    ort = c["config"]["backend"] == "onnxruntime"
    isp = c["pareto_optimal"]
    ax.scatter(ms(c), mp(c), s=120 if isp else 70, marker="o" if ort else "s", facecolor=AMBER if isp else "none",
               edgecolor=AMBER if isp else TEAL, linewidth=2, zorder=3, alpha=1 if isp else 0.8)
ax.plot([ms(c) for c in par], [mp(c) for c in par], color=AMBER, lw=2, ls=(0, (5, 4)), zorder=2)
offs = {0: (14, -4), 1: (-6, 18), 2: (12, -26), 3: (16, -6)}
for j, c in enumerate(par):
    dx, dy = offs[min(j, 3)]
    ax.annotate(f"{label(c)}\n{ms(c):.1f} ms · {mp(c):.2f} mAP", (ms(c), mp(c)), xytext=(dx, dy), textcoords="offset points",
                color=TEXT, fontsize=10.5, ha="left" if j != 1 else "center", va="bottom" if dy > 0 else "top")
ax.set_ylim(28.3, 36.2)
r = C[REF]
ax.scatter(ms(r), mp(r), s=330, facecolor="none", edgecolor=SEC, linewidth=1.2, zorder=1)
ax.annotate("reference\nPyTorch FP32", (ms(r), mp(r)), xytext=(0, -34), textcoords="offset points", color=SEC, fontsize=10, ha="center")
ax.set_xlabel("p50 model-only inference latency, ms (lower is better)")
ax.set_ylabel("COCO mAP@[.5:.95], % (higher is better)")
ax.set_title("Accuracy vs latency — 18 configurations, YOLOX on Apple M1 (CPU only)", color=TEXT, fontsize=14, loc="left", pad=14)
ax.text(0.99, 0.02, "○ ONNX Runtime   □ PyTorch   ● Pareto-optimal (amber)", transform=ax.transAxes, ha="right", color=SEC, fontsize=10)
fig.tight_layout()
fig.savefig(OUT / "frontier.png")
plt.close(fig)

# ---------------------------------------------------------------- one-change-at-a-time levers
ref_ms = ms(r)
rows = [
    ("Static INT8 quantization (ORT)", C["ort-yolox-tiny-416-t4-goall-int8"], True),
    ("Resolution 416 to 320", C["pt-yolox-tiny-320-t4"], False),
    ("PyTorch to ONNX Runtime", C["ort-yolox-tiny-416-t4-goall"], False),
    ("Threads 4 to 8", C["pt-yolox-tiny-416-t8"], False),
    ("Model Tiny to S", C["pt-yolox-s-416-t4"], False),
    ("Resolution 416 to 512", C["pt-yolox-tiny-512-t4"], False),
    ("Threads 4 to 1", C["pt-yolox-tiny-416-t1"], False),
]
n, g = C["ort-yolox-tiny-416-t4-gonone"], C["ort-yolox-tiny-416-t4-goall"]
rows.insert(3, ("ORT graph optimization none to all", None, False))
fig, ax = plt.subplots(figsize=(12, 5.8), dpi=150)
vals = []
for name, c, hi in rows:
    d = (ms(g) - ms(n)) / ms(n) * 100 if c is None else (ms(c) - ref_ms) / ref_ms * 100
    dm = mp(g) - mp(n) if c is None else mp(c) - mp(r)
    vals.append((name, d, dm, hi, c is None))
for i, (_name, d, dm, hi, noisy) in enumerate(vals[::-1]):
    col = SEC if noisy else AMBER if hi else (OK if d < 0 else BAD)
    ax.barh(i, d, color=col, height=0.62, zorder=3)
    ax.text(d + (-2 if d < 0 else 2), i, f"{d:+.1f}%", ha="right" if d < 0 else "left", va="center", color=col, fontsize=11, fontweight="bold")
    ax.text(138, i, f"{dm:+.2f} mAP", ha="right", va="center", color=OK if dm > 0.05 else BAD if dm < -0.05 else SEC, fontsize=11)
ax.set_yticks(range(len(vals)))
ax.set_yticklabels([v[0] for v in vals[::-1]], color=TEXT)
ax.axvline(0, color=SEC, lw=1.2)
ax.set_xlim(-75, 140)
ax.set_xlabel(f"change in p50 latency vs PyTorch Tiny 416, 4 threads ({ref_ms:.1f} ms) — left is faster")
ax.set_title("What actually moved the needle (one change at a time)", color=TEXT, fontsize=14, loc="left", pad=14)
ax.grid(axis="y", visible=False)
fig.tight_layout()
fig.savefig(OUT / "levers.png")
plt.close(fig)


# ---------------------------------------------------------------- latency distributions (raw samples)
def samples(cid):
    d = json.loads((ROOT / f"results/lat-clean/{cid}.json").read_text())
    return np.concatenate([np.array(r["samples_ms"]["inference"]) for r in d["benchmark"]["runs"]])


fig, ax = plt.subplots(figsize=(12, 4.6), dpi=150)
for cid, col, nm in ((REF, TEAL, "PyTorch FP32"), ("ort-yolox-tiny-416-t4-goall", SEC, "ONNX Runtime FP32"),
                     ("ort-yolox-tiny-416-t4-goall-int8", AMBER, "ONNX Runtime INT8")):
    s = samples(cid)
    ax.hist(np.clip(s, 8, 42), bins=np.linspace(8, 42, 85), color=col, alpha=0.8, label=f"{nm}  (p50 {np.median(s):.1f} ms)", zorder=3)
ax.set_xlabel("model-only inference time per image, ms (600 timed runs per configuration)")
ax.set_ylabel("runs")
ax.set_title("Latency distributions, YOLOX-Tiny 416, 4 threads", color=TEXT, fontsize=14, loc="left", pad=14)
ax.legend(facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT, loc="upper right")
fig.tight_layout()
fig.savefig(OUT / "latency.png")
plt.close(fig)

# ---------------------------------------------------------------- markdown table
lines = ["| Configuration | p50 ms | repeat spread | FPS (model-only) | mAP % | AP50 % | Peak RSS | Pareto |", "|---|---|---|---|---|---|---|---|"]
for c in sorted(RES["configs"], key=ms):
    lines.append(f"| {label(c)} | **{ms(c):.2f}** | {c['repeat_spread_pct']:.1f}% | {c['benchmark']['throughput_fps_model_only']:.1f} | "
                 f"{mp(c):.2f} | {c['accuracy']['AP50'] * 100:.2f} | {c['memory']['peak_rss_mb']:.0f} MB | {'●' if c['pareto_optimal'] else ''} |")
(ROOT / "docs/results_table.md").write_text("\n".join(lines) + "\n")
print("wrote figures and table:", ", ".join(sorted(p.name for p in OUT.iterdir())))
