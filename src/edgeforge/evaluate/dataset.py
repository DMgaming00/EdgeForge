"""Deterministic COCO val2017 subsets: manifests, download, integrity."""
from __future__ import annotations

import hashlib
import json
import random
import urllib.request
from pathlib import Path

from ..config import ANN_FILE, COCO_IMG_URL, IMAGES_DIR, MANIFEST_DIR, sha256_file


def make_manifests(n_eval: int = 500, n_calib: int = 100, seed: int = 0,
                   ann_file: Path = ANN_FILE, out_dir: Path = MANIFEST_DIR) -> dict:
    """Disjoint eval/calibration image-ID manifests (images with >=1 non-crowd annotation)."""
    d = json.load(open(ann_file))
    ids_with_ann = sorted({a["image_id"] for a in d["annotations"] if not a.get("iscrowd", 0)})
    if n_eval + n_calib > len(ids_with_ann):
        raise ValueError("requested more images than available")
    rng = random.Random(seed)
    pick = rng.sample(ids_with_ann, n_eval + n_calib)
    ev, ca = sorted(pick[:n_eval]), sorted(pick[n_eval:])
    assert not set(ev) & set(ca)
    out_dir.mkdir(parents=True, exist_ok=True)
    info = {}
    for name, ids in (("eval", ev), ("calib", ca)):
        body = {"name": name, "seed": seed, "source": "COCO val2017", "n": len(ids),
                "image_ids": ids}
        p = out_dir / f"{name}_{len(ids)}.json"
        p.write_text(json.dumps(body, indent=1))
        info[name] = p
    return info


def make_full_manifest(ann_file: Path = ANN_FILE, out_dir: Path = MANIFEST_DIR) -> Path:
    """All val2017 images with >=1 non-crowd annotation (superset of eval_500/calib_100)."""
    d = json.load(open(ann_file))
    ids = sorted({a["image_id"] for a in d["annotations"] if not a.get("iscrowd", 0)})
    p = out_dir / f"fullval_{len(ids)}.json"
    p.write_text(json.dumps({"name": "fullval", "source": "COCO val2017", "n": len(ids),
                             "image_ids": ids}))
    return p


def load_manifest(path: Path) -> dict:
    m = json.loads(Path(path).read_text())
    if len(set(m["image_ids"])) != len(m["image_ids"]):
        raise ValueError("duplicate image ids in manifest")
    return m


def manifest_hash(m: dict) -> str:
    """Hash of the sorted ID list only (stable across formatting)."""
    return hashlib.sha256(json.dumps(sorted(m["image_ids"])).encode()).hexdigest()


def image_path(image_id: int) -> Path:
    return IMAGES_DIR / f"{image_id:012d}.jpg"


def download_images(ids: list[int], progress=None) -> dict[int, str]:
    """Download missing images; returns {id: sha256}. Fails loudly on any error."""
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    sums = {}
    for i, iid in enumerate(ids):
        p = image_path(iid)
        if not p.exists():
            tmp = p.with_suffix(".part")
            urllib.request.urlretrieve(COCO_IMG_URL.format(file=p.name), tmp)
            if tmp.stat().st_size < 1000:
                raise RuntimeError(f"suspiciously small download for {p.name}")
            tmp.rename(p)
        sums[iid] = sha256_file(p)
        if progress:
            progress(i + 1, len(ids))
    return sums


def write_checksums(ids: list[int], sums: dict[int, str], out: Path) -> None:
    out.write_text(json.dumps({str(i): sums[i] for i in ids}, indent=0, sort_keys=True))
