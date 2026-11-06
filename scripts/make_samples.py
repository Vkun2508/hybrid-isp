"""Build the demo samples in samples/ (run once; outputs are committed).

- Real smartphone RAW: noisy + ground-truth pairs from SIDD-CC (HF dataset MhdMohammadi/SIDD-CC),
  taken ONLY from the held-out test scenes (004, 008) of the training notebook's split, so the
  network never saw them. Centre-cropped to ~1.5 MP, aligned to RGGB, stored as float16 .npz.
- Photos: public-domain test images from scikit-image, with simulator presets (low light, WB cast, high ISO).

    python scripts/make_samples.py [--download-dir <dir>]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from isp.analysis import auto_exposure  # noqa: E402
from isp.classical.color import cam2rgb_from_xyz2cam  # noqa: E402
from isp.io import _to_rggb  # noqa: E402

REPO = "MhdMohammadi/SIDD-CC"
PREFIX = "Raw/train/Data"
SIDD = [
    dict(id="sidd-n6-iso3200-lowlight", instance="0076_004_N6_03200_00320_3200_L",
         name="Nexus 6 · ISO 3200 · low light",
         description="Real smartphone RAW from an unseen test scene, with its clean ground truth."),
    dict(id="sidd-ip-iso1600", instance="0191_008_IP_01600_01600_3200_N",
         name="iPhone 7 · ISO 1600 · indoor",
         description="Real smartphone RAW from an unseen test scene, with its clean ground truth."),
]
PHOTOS = [
    dict(id="photo-coffee-night", src="coffee", name="Coffee · simulated night shot",
         description="A photo turned into sensor RAW, then under-exposed by 3 stops at high ISO.",
         preset={"noise": 0.75, "exposure_ev": -3.0, "wb_shift": 0.0}),
    dict(id="photo-astronaut-tungsten", src="astronaut", name="Portrait · wrong white balance",
         description="A photo turned into sensor RAW, with the camera's white balance metadata badly off.",
         preset={"noise": 0.3, "exposure_ev": 0.0, "wb_shift": 0.8}),
    dict(id="photo-cat-highiso", src="chelsea", name="Cat · high ISO",
         description="A photo turned into sensor RAW with heavy high-ISO sensor noise.",
         preset={"noise": 0.7, "exposure_ev": 0.0, "wb_shift": 0.0}),
]
CROP = (1024, 1536)  # raw H, W (~1.57 MP)


def mat_str(v) -> str:
    a = np.asarray(v)
    if a.dtype.kind == "U":
        return "".join(a.ravel().tolist()).strip()
    return b"".join(a.ravel().tolist()).decode(errors="replace").strip()


def build_sidd(entry: dict, dl: Path) -> dict:
    import scipy.io
    from huggingface_hub import HfApi, hf_hub_download

    folder = f"{PREFIX}/{entry['instance']}"
    files = sorted(f.path for f in HfApi().list_repo_tree(REPO, path_in_repo=folder, repo_type="dataset"))
    noisy = next(f for f in files if "_NOISY_RAW_" in f)
    idx = noisy.rsplit("_", 1)[1]
    paths = {k: hf_hub_download(REPO, next(f for f in files if f"_{k}_RAW_" in f and f.endswith(idx)),
                                repo_type="dataset", local_dir=str(dl))
             for k in ("NOISY", "GT", "METADATA")}
    meta = scipy.io.loadmat(paths["METADATA"])
    cfa = mat_str(meta["cfa_str"]).upper()
    n = np.asarray(meta["AsShotNeutral"], np.float64).ravel()
    cm2 = np.asarray(meta["ColorMatrix2"], np.float64).ravel().reshape(3, 3)
    arr = {k: _to_rggb(scipy.io.loadmat(paths[k])["data"].astype(np.float32), cfa) for k in ("NOISY", "GT")}
    h, w = arr["GT"].shape
    ch, cw = min(CROP[0], h) // 2 * 2, min(CROP[1], w) // 2 * 2
    y, x = (h - ch) // 4 * 2, (w - cw) // 4 * 2
    noisy_c, gt_c = arr["NOISY"][y:y + ch, x:x + cw], arr["GT"][y:y + ch, x:x + cw]
    wb = (float(n[1] / n[0]), 1.0, float(n[1] / n[2]))
    cam2rgb = cam2rgb_from_xyz2cam(cm2)
    exposure = auto_exposure(gt_c)
    out = ROOT / "samples" / f"{entry['id']}.npz"
    np.savez_compressed(out, noisy=noisy_c.astype(np.float16), gt=gt_c.astype(np.float16),
                        wb_gains=np.asarray(wb, np.float32), cam2rgb=cam2rgb, exposure=np.float32(exposure))
    _, inst, scene, cam, iso, shutter, kelvin, bright = ["", *entry["instance"].split("_")]
    print(f"{entry['id']}: cfa={cfa} crop={cw}x{ch} wb={np.round(wb, 3)} exposure=x{exposure:.1f} "
          f"-> {out.name} ({out.stat().st_size / 1e6:.1f} MB)")
    return dict(id=entry["id"], kind="sidd", file=out.name, name=entry["name"], description=entry["description"],
                has_gt=True, preset={"noise": 0.0, "exposure_ev": 0.0, "wb_shift": 0.0},
                source={"dataset": "SIDD-CC (Abdelhamed et al., CVPR 2018)", "instance": entry["instance"],
                        "scene": scene, "camera": cam, "iso": int(iso), "split": "test (unseen)"})


def build_photo(entry: dict) -> dict:
    import skimage.data as d

    img = getattr(d, entry["src"])()
    out = ROOT / "samples" / f"{entry['id']}.jpg"
    cv2.imwrite(str(out), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"{entry['id']}: {img.shape[1]}x{img.shape[0]} -> {out.name}")
    return dict(id=entry["id"], kind="image", file=out.name, name=entry["name"], description=entry["description"],
                has_gt=True, preset=entry["preset"],
                source={"dataset": f"scikit-image test image '{entry['src']}' (public domain / CC0)"})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--download-dir", default=str(ROOT / ".cache" / "sidd"))
    ap.add_argument("--skip-sidd", action="store_true")
    args = ap.parse_args()
    (ROOT / "samples").mkdir(exist_ok=True)
    reg = []
    if not args.skip_sidd:
        reg += [build_sidd(e, Path(args.download_dir)) for e in SIDD]
    reg += [build_photo(e) for e in PHOTOS]
    (ROOT / "samples" / "samples.json").write_text(json.dumps(reg, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote samples/samples.json ({len(reg)} samples)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
