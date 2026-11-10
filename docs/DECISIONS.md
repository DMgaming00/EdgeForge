# Technical decisions

## D1 — Detector: YOLOX-Tiny (reference), YOLOX-S (larger size)
- **Why:** Anchor-free single-stage detector, trained on COCO (80 classes), small (5.1 M params
  for Tiny), PyTorch network code is Apache-2.0, ONNX-export-friendly (SiLU/Conv/Concat/Sigmoid only).
- **Verified in M0 (this machine, Apple M1 / 8 GB):**
  1. Weights load into the official `yolox` network (`YOLOPAFPN` + `YOLOXHead`) — yes.
  2. A real COCO image produces sensible detections (e.g. a bear image -> `bear` 0.96) — yes.
  3. `torch.onnx.export` (legacy TorchScript exporter, opset 17, static shape) succeeds and passes
     `onnx.checker` — yes.
  4. ONNX Runtime CPU executes the graph; raw-output max abs error vs PyTorch ~5e-5 — yes.
  5. Decoding + class-aware NMS implemented once in NumPy (`postprocess.py`), shared by all backends.
  6. Peak process RSS during inference ~0.4 GB — fits easily.
  7. COCO val2017 annotations + 600 images downloaded; checksums committed.
  8. Licenses: see `LICENSES.md`.
- **Export design:** the model is exported with `decode_in_inference=False`, i.e. the graph emits the
  *raw head output* `(1, N, 85)`. Box decoding, score computation and NMS are done outside the
  graph, identically for both backends. This makes the FP32 parity test a direct comparison of
  network outputs and keeps NMS out of the timed "inference" stage (it is timed separately).
- **Install note:** the `yolox` sdist imports torch in `setup.py`; it must be installed with
  `--no-deps --no-build-isolation`. We only import `yolox.models`.
- **Not chosen:** torchvision SSDLite (NMS/postprocessing is baked into the module, which defeats
  a shared postprocess), Ultralytics YOLO (AGPL-3.0 — incompatible with the permissive-license goal).
- YOLOX-S (same release) is also used as a larger model size. (An earlier 92-byte download was a transient server timeout, not a missing file.)

## D2 — Static input shapes, one ONNX file per resolution
Resolutions are model-level experiment variables; dynamic axes would add shape-handling overhead
to the ORT comparison and risk different kernels. 320/416/512 files are exported lazily.

## D3 — Metric definitions
- **Primary latency** = model forward pass only (batch 1), p50. This is the x-axis of the frontier.
- **Pipeline latency** = preprocess + forward + decode/NMS (also recorded and shown).
- p50/p95/p99 headline values are the median over repeats of each repeat's percentile.

## D4 — One process per configuration
Each configuration runs in a fresh Python process (sequentially, never concurrently), so
`ru_maxrss` is a clean per-config peak and allocator/thread-pool state does not leak between
configs. Benchmark runs before the accuracy pass so peak RSS is not inflated by COCO tooling.

## D5 — Parity tolerance
Raw output spans ~1e-6 to ~4e2 (pixel units). Tolerance `atol=1e-3, rtol=1e-3`
(numpy `allclose` semantics) on 20 real images per ORT config; max abs / mean abs / max relative
error are recorded. Observed errors are 1–2 orders of magnitude inside the tolerance.

## D6 — Validity gate for the Pareto frontier
A config is eligible only if: no image failed, accuracy was computed on the same manifest hash as
all others, and (for ORT) FP32 parity vs PyTorch passed. Invalid configs are shown in red and never
enter the frontier. The frontend recomputes the frontier from the raw rows.

## D7 — Two-phase measurement (latency vs accuracy)
Accuracy does not depend on machine load, latency does. So they are separate sweeps merged by
`edgeforge report`: a latency sweep (`--no-eval`, refuses to start above load average 3, retries a
config if its repeats differ by >10%, drift-checks the reference) and an accuracy sweep over all
4,952 annotated val2017 images (`--manifest fullval_4952.json`). A first combined run on a busy
laptop showed 77% reference drift, which motivated this design.

## D8 — YOLOX-Nano excluded by the validity gate
The Nano checkpoint loads into the official depthwise network and emits plausible scores, but its
boxes are spatially wrong (1.2% mAP on full val vs ~25% expected). Cause not found. It was removed from the
sweep (its accuracy run was kept locally but is not committed because of its size) and replaced by YOLOX-S. The
gate (mAP < 50% of reference => invalid) is still enforced and unit-tested. No conclusions about Nano.

## D9 — INT8
Static QDQ quantization (ORT `quantize_static`, min-max, per-channel int8 weights, uint8
activations) calibrated on 100 images from `calib_100.json`, checked disjoint from evaluation at
runtime. Gate: detection agreement F1 vs FP32 reference >= 0.5. Quantization error is reported,
not gated like FP32 parity. Only one calibration strategy was tried.

## D10 — Scope cuts
CoreML/MPS and other model families are deferred. INT8 (D9) and a second model size (YOLOX-S) are part of the
reported sweep. The Nano checkpoint is supported by the code path but is not part of the reported results.
