"""Shared evaluation helpers: SIDD-CC test pairs -> RawImage + rendered ground truth."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from isp.analysis import auto_exposure  # noqa: E402
from isp.classical.color import cam2rgb_from_xyz2cam  # noqa: E402
from isp.io import _to_rggb  # noqa: E402
from isp.raw import RawImage  # noqa: E402

REPO = "MhdMohammadi/SIDD-CC"
PREFIX = "Raw/train/Data"
TEST_SCENES = ("004", "008")  # held out by the training notebook's scene split (seed 42, 10 scenes)
VAL_SCENES = ("003",)  # the notebook's validation scene: router thresholds are tuned on it only
CACHE = ROOT / ".cache" / "eval"
CACHE_VAL = ROOT / ".cache" / "val"
CROP = (1024, 1536)


def _mat_str(v) -> str:
    a = np.asarray(v)
    if a.dtype.kind == "U":
        return "".join(a.ravel().tolist()).strip()
    return b"".join(a.ravel().tolist()).decode(errors="replace").strip()


def test_instances(scenes=TEST_SCENES) -> list[str]:
    from huggingface_hub import HfApi

    names = [f.path.split("/")[-1] for f in HfApi().list_repo_tree(REPO, path_in_repo=PREFIX, repo_type="dataset")]
    return sorted(n for n in names if n.split("_")[1] in scenes)


def fetch_crop(instance: str, dl_dir: Path, crop=CROP, keep_mat: bool = False, cache: Path | None = None) -> Path:
    """Download one noisy/GT/metadata triple, align to RGGB, centre-crop, cache as .npz (float16)."""
    import scipy.io
    from huggingface_hub import HfApi, hf_hub_download

    cache = cache or CACHE
    out = cache / f"{instance}.npz"
    if out.exists():
        return out
    cache.mkdir(parents=True, exist_ok=True)
    folder = f"{PREFIX}/{instance}"
    files = sorted(f.path for f in HfApi().list_repo_tree(REPO, path_in_repo=folder, repo_type="dataset"))
    noisy = next(f for f in files if "_NOISY_RAW_" in f)
    idx = noisy.rsplit("_", 1)[1]
    paths = {k: hf_hub_download(REPO, next(f for f in files if f"_{k}_RAW_" in f and f.endswith(idx)),
                                repo_type="dataset", local_dir=str(dl_dir))
             for k in ("NOISY", "GT", "METADATA")}
    meta = scipy.io.loadmat(paths["METADATA"])
    cfa = _mat_str(meta["cfa_str"]).upper()
    n = np.asarray(meta["AsShotNeutral"], np.float64).ravel()
    cm2 = np.asarray(meta["ColorMatrix2"], np.float64).ravel().reshape(3, 3)
    arr = {k: _to_rggb(scipy.io.loadmat(paths[k])["data"].astype(np.float32), cfa) for k in ("NOISY", "GT")}
    h, w = arr["GT"].shape
    ch, cw = min(crop[0], h) // 2 * 2, min(crop[1], w) // 2 * 2
    y, x = (h - ch) // 4 * 2, (w - cw) // 4 * 2
    gt = arr["GT"][y:y + ch, x:x + cw]
    np.savez_compressed(out, noisy=arr["NOISY"][y:y + ch, x:x + cw].astype(np.float16), gt=gt.astype(np.float16),
                        wb_gains=np.asarray((n[1] / n[0], 1.0, n[1] / n[2]), np.float32),
                        cam2rgb=cam2rgb_from_xyz2cam(cm2), exposure=np.float32(auto_exposure(gt)), cfa=cfa)
    if not keep_mat:
        for k in ("NOISY", "GT"):
            Path(paths[k]).unlink(missing_ok=True)
    return out


def load_pair(path: Path) -> tuple[RawImage, np.ndarray, dict]:
    """-> (noisy RawImage with .gt set to the rendered reference, GT bayer, info)."""
    from isp.pipelines import reference_render

    z = np.load(path)
    inst = path.stem
    _, scene, cam, iso, *_ = inst.split("_")
    raw = RawImage(bayer=z["noisy"].astype(np.float32), wb_gains=tuple(float(v) for v in z["wb_gains"]),
                   cam2rgb=z["cam2rgb"].astype(np.float32), exposure=1.0,
                   meta={"source": "sidd", "filename": inst})
    gt_bayer = z["gt"].astype(np.float32)
    raw.exposure = auto_exposure(gt_bayer)  # one AE definition everywhere (recomputed, not cached)
    raw.gt = reference_render(gt_bayer, raw)
    return raw, gt_bayer, {"instance": inst, "scene": scene, "camera": cam, "iso": int(iso)}
