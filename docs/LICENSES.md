# Licenses

| Component | License | Notes |
|---|---|---|
| EdgeForge code | Apache-2.0 | `LICENSE` |
| YOLOX network code (`yolox` PyPI 0.3.0, Megvii-BaseDetection/YOLOX) | Apache-2.0 | Imported as a library, not vendored. Verified from the repository `LICENSE`. |
| YOLOX-Tiny / Nano checkpoints (release `0.1.1rc0`) | Distributed from the Apache-2.0 repository's GitHub release | **No separate weights license is stated** on the release; we treat them as covered by the repository license, but did not obtain an independent legal confirmation. Weights are *not* redistributed in this repo (downloaded by `edgeforge prepare`). The checkpoints were trained on COCO train2017 by the YOLOX authors; no training was done here. |
| COCO 2017 annotations | CC BY 4.0 | © COCO Consortium. |
| COCO 2017 images | Flickr terms; individual images carry their own licenses | Not redistributed; downloaded from images.cocodataset.org. Only image IDs and SHA-256 checksums are committed. |
| PyTorch (BSD-3), ONNX Runtime (MIT), ONNX (Apache-2.0), OpenCV (Apache-2.0), pycocotools (BSD-2), NumPy (BSD) | permissive | Not redistributed. |
| IBM Plex Sans / Mono (via `@fontsource`) | SIL OFL 1.1 | Bundled in the site build. |
