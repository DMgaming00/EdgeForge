# Methodology

## Question
How much COCO detection accuracy is lost (or gained) per millisecond of CPU inference latency
changed, across runtime, resolution, thread count, and ONNX graph-optimization settings?

## Experiment design (structured, not Cartesian)
Reference: PyTorch FP32 eager, YOLOX-Tiny, 416 px, 4 threads. Then:
1. **Runtime** — ORT FP32 (graph opt `all`) vs PyTorch, otherwise identical.
2. **Graph optimization** — ORT `none` / `basic` / `all` at 416 px, 4 threads.
3. **Resolution** — 320 / 416 / 512 for both runtimes.
4. **Threads** — 1 / 2 / 4 / 8 for both runtimes at 416 px.
5. **Drift** — reference benchmarked again at the very start and very end of the sweep (latency only).
Plus: YOLOX-S 416 (PyTorch, ORT FP32, ORT INT8) and static INT8 on ORT Tiny 416. YOLOX-Nano failed the validity gate and is excluded (DECISIONS D8). Total 18 configs + 2 drift measurements.

## Accuracy
- Data: All 4,952 annotated COCO val2017 images (accuracy sweep); a 500-image subset (`eval_500.json`) is kept for smoke runs; chosen with `random.Random(0)` among images having at least one
  non-crowd annotation; manifest `data/manifests/eval_500.json` (+ SHA-256 of every image).
  A disjoint 100-image calibration manifest (`calib_100.json`) is used for static INT8 quantization.
- Scoring: `pycocotools` COCOeval (bbox), restricted to the manifest's image IDs. Detection threshold
  0.01, NMS IoU 0.65, max 100 detections per image (standard YOLOX evaluation settings).
- Per-class AP is read from the COCOeval precision tensor (IoU .50:.95, all areas, maxDets 100) and
  reported with the number of ground-truth boxes; classes with n < 30 are flagged.
- Any failing image invalidates a configuration (no silent exclusion).

## Latency
- Timer: `time.perf_counter_ns`; stages (preprocess, inference, postprocess) timed separately.
- Mode `rigorous`: 20 warmup passes (discarded), then 3 independent repeats × 200 timed frames.
  The headline value is the median over repeats of each repeat's percentile. Raw samples are stored.
- Inputs: 20 fixed val2017 images cycled; model load excluded; batch 1.
- Configurations run sequentially, one fresh process each. Power source and 1-min load average
  are recorded. The final latency sweep ran plugged in behind a load guard (refuses to start above a 1-min load of 3.0); the reference's
  start/end drift (+5.7%) is reported, so gaps under about 6% are within noise.
- Peak memory: `ru_maxrss` of the per-config process (includes interpreter + libraries).

## Correctness
- **Parity:** ORT raw outputs vs PyTorch raw outputs (same weights/resolution/preprocessing), 20 images.
- **Agreement:** greedy same-class matching (IoU ≥ 0.5, score ≥ 0.3) of each config's detections
  against the reference's; reported as precision/recall/F1. This is *agreement*, not accuracy.

## Pareto
Minimise p50 inference latency, maximise mAP. Only valid configs on the same manifest participate.
Implemented in Python (`pareto.py`, unit-tested) and TypeScript (`site/src/pareto.ts`).
