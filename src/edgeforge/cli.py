"""EdgeForge command line interface."""
from __future__ import annotations

import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .config import MANIFEST_DIR, MODEL_SPECS, RESULTS_DIR, ROOT, RunConfig

app = typer.Typer(help="EdgeForge: CPU inference optimization & benchmarking", no_args_is_help=True)
console = Console()


def _cfg(model, backend, resolution, threads, graph_opt, precision="fp32") -> RunConfig:
    try:
        return RunConfig(model=model, backend=backend, resolution=resolution, threads=threads,
                         graph_opt=graph_opt, precision=precision)
    except ValueError as e:
        console.print(f"[red]invalid configuration:[/red] {e}")
        raise typer.Exit(2) from e


@app.command()
def prepare(n_eval: int = 500, n_calib: int = 100, weights: bool = True,
            annotations: bool = True, images: bool = True):
    """Download weights/annotations, create manifests, download manifest images (cached)."""
    import urllib.request
    import zipfile

    from .config import ANN_FILE, MODELS_DIR, RAW_DIR
    from .evaluate import dataset as ds
    if weights:
        MODELS_DIR.mkdir(exist_ok=True)
        for name, spec in MODEL_SPECS.items():
            p = MODELS_DIR / spec["weights"]
            if not p.exists() or p.stat().st_size < 1_000_000:
                console.print(f"downloading {name} weights")
                urllib.request.urlretrieve(spec["url"], p)
    if annotations and not ANN_FILE.exists():
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        z = RAW_DIR / "ann.zip"
        console.print("downloading COCO annotations (~250MB)")
        urllib.request.urlretrieve(
            "http://images.cocodataset.org/annotations/annotations_trainval2017.zip", z)
        with zipfile.ZipFile(z) as zf:
            zf.extract("annotations/instances_val2017.json", RAW_DIR)
        z.unlink()
    ev, ca = MANIFEST_DIR / f"eval_{n_eval}.json", MANIFEST_DIR / f"calib_{n_calib}.json"
    if not (ev.exists() and ca.exists()):
        ds.make_manifests(n_eval, n_calib)
    if images:
        ids = ds.load_manifest(ev)["image_ids"] + ds.load_manifest(ca)["image_ids"]
        ds.download_images(ids, lambda i, n: i % 50 == 0 and console.print(f"images {i}/{n}"))
    console.print("[green]prepare complete[/green]")


@app.command()
def export(model: str = "yolox-tiny", resolution: int = 416, force: bool = False):
    """Export a model to ONNX (static input shape)."""
    from .export.onnx_export import export_onnx
    p = export_onnx(model, resolution, force=force)
    console.print(f"[green]{p}[/green] ({p.stat().st_size / 2**20:.1f} MB)")


@app.command()
def validate(model: str = "yolox-tiny", resolution: int = 416, threads: int = 4,
             graph_opt: str = "all"):
    """FP32 parity: ONNX Runtime vs PyTorch raw outputs on real images."""
    from .evaluate.dataset import load_manifest
    from .evaluate.run_config import run_parity
    cfg = _cfg(model, "onnxruntime", resolution, threads, graph_opt)
    ids = load_manifest(next(MANIFEST_DIR.glob("eval_*.json")))["image_ids"]
    m = run_parity(cfg, ids)
    console.print_json(json.dumps(m))
    raise typer.Exit(0 if m["passed"] else 1)


def _single(cfg, mode, do_eval, out, label=None, manifest=None):
    from .evaluate.run_config import run_config
    out = Path(out) if out else RESULTS_DIR / ("adhoc-" + dt.datetime.now().strftime("%Y%m%d-%H%M%S"))
    r = run_config(cfg, out, mode, do_eval, manifest=Path(manifest) if manifest else None, label=label)
    h = r["benchmark"]["headline"]
    t = Table(title=f"{r['id']}  [{r['provider']}]  mode={mode}")
    for c in ("stage", "p50 ms", "p95 ms", "p99 ms"):
        t.add_column(c)
    for s in ("preprocess", "inference", "postprocess", "total"):
        t.add_row(s, *(f"{h[s][k]:.2f}" for k in ("p50", "p95", "p99")))
    console.print(t)
    if r["accuracy"]:
        console.print(f"mAP@[.5:.95]={r['accuracy']['mAP']:.4f}  AP50={r['accuracy']['AP50']:.4f}"
                      f"  ({r['accuracy']['n_images']} images)")
    if r["parity"]:
        console.print(f"parity vs PyTorch: {r['parity']}")
    console.print(f"valid={r['valid']} {r['invalid_reasons']}  -> {out}")
    return r


@app.command()
def run(model: str = "yolox-tiny", backend: str = "pytorch", resolution: int = 416,
        threads: int = 4, graph_opt: str = "all", mode: str = "standard",
        no_eval: bool = False, out: str = "", label: str = "", precision: str = "fp32",
        manifest: str = ""):
    """Benchmark + COCO eval + parity for ONE configuration (fresh process = clean memory)."""
    _single(_cfg(model, backend, resolution, threads, graph_opt, precision), mode, not no_eval,
            out or None, label or None, manifest or None)


@app.command()
def evaluate(model: str = "yolox-tiny", backend: str = "pytorch", resolution: int = 416,
             threads: int = 4, graph_opt: str = "all", out: str = ""):
    """Accuracy (+ smoke timing) for one configuration."""
    _single(_cfg(model, backend, resolution, threads, graph_opt), "smoke", True, out or None)


@app.command()
def bench(model: str = "yolox-tiny", backend: str = "pytorch", resolution: int = 416,
          threads: int = 4, graph_opt: str = "all", mode: str = "standard", out: str = ""):
    """Latency/throughput/memory only (no accuracy)."""
    _single(_cfg(model, backend, resolution, threads, graph_opt), mode, False, out or None)


SWEEP = [  # structured design (not a Cartesian product); first entry is the reference
    # (model, backend, resolution, threads, graph_opt, precision)
    ("yolox-tiny", "pytorch", 416, 4, "all", "fp32"),
    ("yolox-tiny", "onnxruntime", 416, 4, "all", "fp32"),
    ("yolox-tiny", "onnxruntime", 416, 4, "none", "fp32"),
    ("yolox-tiny", "onnxruntime", 416, 4, "basic", "fp32"),
    ("yolox-tiny", "pytorch", 320, 4, "all", "fp32"), ("yolox-tiny", "pytorch", 512, 4, "all", "fp32"),
    ("yolox-tiny", "onnxruntime", 320, 4, "all", "fp32"),
    ("yolox-tiny", "onnxruntime", 512, 4, "all", "fp32"),
    *[("yolox-tiny", "pytorch", 416, t, "all", "fp32") for t in (1, 2, 8)],
    *[("yolox-tiny", "onnxruntime", 416, t, "all", "fp32") for t in (1, 2, 8)],
    ("yolox-s", "pytorch", 416, 4, "all", "fp32"),
    ("yolox-s", "onnxruntime", 416, 4, "all", "fp32"),
    ("yolox-s", "onnxruntime", 416, 4, "all", "int8"),
    ("yolox-tiny", "onnxruntime", 416, 4, "all", "int8"),
]


def _spread_pct(path: Path) -> float:
    r = json.loads(path.read_text())["benchmark"]["runs"]
    p50 = [x["summary"]["inference"]["p50"] for x in r]
    return (max(p50) - min(p50)) / (sum(p50) / len(p50)) * 100 if len(p50) > 1 else 0.0


@app.command()
def sweep(mode: str = "standard", run_id: str = "", resume: bool = True, no_eval: bool = False,
          manifest: str = "", max_load: float = 3.0, force: bool = False,
          max_spread_pct: float = 10.0, retries: int = 2):
    """Run the structured experiment design sequentially, one fresh process per config.

    Latency sweep (--no-eval): refuses to start if the 1-min load average exceeds --max-load
    (override with --force), re-runs a config up to --retries times when its repeats disagree
    by more than --max-spread-pct, and re-benchmarks the reference at the end (drift check).
    Accuracy sweep (--manifest fullval_*.json, no load requirement) evaluates every config.
    """
    import os

    from .export.onnx_export import export_onnx
    from .export.quantize import quantize_model
    run_id = run_id or dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = RESULTS_DIR / run_id
    out.mkdir(parents=True, exist_ok=True)
    if no_eval and not force and os.getloadavg()[0] > max_load:
        console.print(f"[red]load average {os.getloadavg()[0]:.1f} > {max_load}: close other apps "
                      "(or pass --force; results will be flagged noisy).[/red]")
        raise typer.Exit(3)
    for m, _, r, _, _, prec in SWEEP:
        export_onnx(m, r)
        if prec == "int8":
            quantize_model(m, r)
    jobs = [(RunConfig(model=m, backend=b, resolution=r, threads=t, graph_opt=g, precision=p),
             None, False) for m, b, r, t, g, p in SWEEP]
    ref = jobs[0][0]
    if no_eval:
        jobs.insert(0, (ref, "drift-start-" + ref.id, True))
        jobs.append((ref, "drift-end-" + ref.id, True))
    for i, (cfg, label, drift) in enumerate(jobs, 1):
        cid = label or cfg.id
        if resume and (out / f"{cid}.json").exists():
            console.print(f"[dim]skip {cid} (done)[/dim]")
            continue
        for attempt in range(retries + 1):
            console.print(f"[bold]({i}/{len(jobs)}) {cid}[/bold] attempt {attempt + 1}")
            cmd = [sys.executable, "-m", "edgeforge.cli", "run", "--model", cfg.model, "--backend",
                   cfg.backend, "--resolution", str(cfg.resolution), "--threads", str(cfg.threads),
                   "--graph-opt", cfg.graph_opt, "--precision", cfg.precision, "--mode", mode,
                   "--out", str(out), "--label", cid]
            if no_eval or drift:
                cmd += ["--no-eval"]
            if manifest:
                cmd += ["--manifest", manifest]
            rc = subprocess.run(cmd, check=False, cwd=ROOT).returncode
            if rc:
                console.print(f"[red]{cid} failed (exit {rc}); sweep aborted[/red]")
                raise typer.Exit(rc)
            sp = _spread_pct(out / f"{cid}.json")
            d = json.loads((out / f"{cid}.json").read_text())
            d["repeat_spread_pct"] = sp
            d["attempts"] = attempt + 1
            d["is_drift_check"] = drift
            d["noisy"] = bool(no_eval and sp > max_spread_pct)
            (out / f"{cid}.json").write_text(json.dumps(d, indent=1))
            if not (no_eval and sp > max_spread_pct) or attempt == retries:
                break
            console.print(f"[yellow]repeat spread {sp:.1f}% > {max_spread_pct}%: re-running[/yellow]")
    console.print(f"[green]sweep done -> {out}[/green]")


@app.command()
def report(bench_run: str = typer.Option(..., help="latency run id (sweep --no-eval)"),
           acc_run: str = typer.Option(..., help="accuracy run id (sweep --manifest ...)"),
           reference: str = "", out: str = ""):
    """Merge latency + accuracy runs into site/public/data/results.json."""
    from .report.aggregate import build_report
    reference = reference or RunConfig().id
    out_f = Path(out) if out else ROOT / "site" / "public" / "data" / "results.json"
    extra = {}
    gp = RESULTS_DIR / "graph_opt_stats.json"
    if gp.exists():
        extra["graph_opt_stats"] = json.loads(gp.read_text())
    r = build_report(RESULTS_DIR / bench_run, RESULTS_DIR / acc_run, reference, out_f, extra)
    console.print(f"{len(r['configs'])} configs, Pareto: {r['pareto_ids']}\n-> {out_f}")


@app.command()
def explain(model: str = "yolox-tiny", resolution: int = 416):
    """Dump how ONNX Runtime graph optimization changes the graph (op counts per level)."""
    from collections import Counter

    import onnx
    import onnxruntime as ort

    from .export.onnx_export import export_onnx
    src = export_onnx(model, resolution)
    out = {"model": model, "resolution": resolution, "levels": {}}
    out["levels"]["original"] = dict(Counter(n.op_type for n in onnx.load(str(src)).graph.node))
    for lvl, enum in (("basic", ort.GraphOptimizationLevel.ORT_ENABLE_BASIC),
                      ("all", ort.GraphOptimizationLevel.ORT_ENABLE_ALL)):
        so = ort.SessionOptions(); so.graph_optimization_level = enum
        dst = RESULTS_DIR / f"_opt_{lvl}.onnx"
        so.optimized_model_filepath = str(dst)
        ort.InferenceSession(str(src), so, providers=["CPUExecutionProvider"])
        out["levels"][lvl] = dict(Counter(n.op_type for n in onnx.load(str(dst)).graph.node))
        dst.unlink()
    for k, v in out["levels"].items():
        out["levels"][k] = dict(sorted(v.items(), key=lambda kv: -kv[1]))
        out[f"total_nodes_{k}"] = sum(v.values())
    (RESULTS_DIR / "graph_opt_stats.json").write_text(json.dumps(out, indent=1))
    console.print_json(json.dumps({k: v for k, v in out.items() if k.startswith("total")}))


@app.command()
def video(src: str, dst: str = "", model: str = "yolox-tiny", backend: str = "onnxruntime",
          resolution: int = 416, threads: int = 4, graph_opt: str = "all",
          compare_backend: str = "", compare_resolution: int = 0, max_frames: int = 300,
          conf: float = 0.35, precision: str = "fp32", compare_precision: str = "",
          compare_model: str = "", all_classes: bool = False):
    """Annotate a short video with detections and a measured-performance HUD."""
    from .video.process import process_video
    cfg = _cfg(model, backend, resolution, threads, graph_opt, precision)
    cfg2 = None
    if compare_backend or compare_resolution or compare_precision or compare_model:
        cfg2 = _cfg(compare_model or model, compare_backend or backend,
                    compare_resolution or resolution, threads, graph_opt,
                    compare_precision or precision)
    dst = dst or str(ROOT / "videos" / (Path(src).stem + "_annotated.mp4"))
    s = process_video(src, dst, cfg, cfg2, max_frames=max_frames, conf=conf,
                      street_only=not all_classes)
    Path(dst).with_suffix(".json").write_text(json.dumps(s, indent=1))
    console.print_json(json.dumps(s))


if __name__ == "__main__":
    app()
