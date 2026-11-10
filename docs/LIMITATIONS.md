# Limitations

- **Accuracy set.** All 4,952 annotated COCO val2017 images (not test-dev); our reference mAP 32.15 % vs ~32.8 % officially. Per-class AP for rare classes is noisy.
- **Single machine.** One Apple M1 laptop, 8 GB, not thermally controlled; reference drift over the sweep was +5.7 %, so small latency gaps are within noise.
  Absolute latencies are specific to that state; use relative comparisons and the drift check.
- **Heterogeneous cores.** M1 has 4 performance + 4 efficiency cores; "threads=8" uses both kinds and
  is not a clean scaling test. macOS does not allow core pinning.
- **Memory** is process peak RSS, including interpreter/library overhead.
- **INT8** is min-max static QDQ calibrated on 100 images; other calibration methods untested. CoreML/MPS not measured. YOLOX-Nano excluded (accuracy gate).
- **Weights license** is inferred from the repository license (see `LICENSES.md`).
- **Not automotive-validated.** This is a performance-analysis project; nothing here is a safety
  claim or ADAS readiness statement.
- **`edgeforge video`** reports offline processing throughput on a clip; it is not live-camera performance.
- **Unexplained results:** ONNX Runtime FP32 was slower than PyTorch on YOLOX-S (52.0 vs 44.3 ms), INT8 latency is bimodal (about 13 and 16 ms), and the YOLOX-Nano checkpoint produced spatially wrong boxes (1.2% mAP). None was investigated.
- **Two latency statistics:** the headline p50 is the median of repeat p50s; the bootstrap CI in `results.json` is over pooled samples, so they can differ by about 0.1 ms.
