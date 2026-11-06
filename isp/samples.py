"""Built-in demo samples (samples/samples.json): real SIDD smartphone RAW pairs and photos."""
from __future__ import annotations

import json
import threading
from functools import lru_cache
from pathlib import Path

import numpy as np

from isp.analysis import auto_exposure
from isp.io import InputError, load_image
from isp.raw import RawImage

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"
_lock = threading.Lock()


@lru_cache(maxsize=1)
def registry() -> list[dict]:
    p = SAMPLES_DIR / "samples.json"
    if not p.exists():
        return []
    return json.loads(p.read_text(encoding="utf-8"))


def get_entry(sample_id: str) -> dict:
    for e in registry():
        if e["id"] == sample_id:
            return e
    raise InputError("unknown_sample", f"There is no sample called '{sample_id}'.", "Pick one of the listed samples.", 404)


@lru_cache(maxsize=8)
def _load_cached(sample_id: str) -> RawImage:
    e = get_entry(sample_id)
    path = SAMPLES_DIR / e["file"]
    if e["kind"] == "sidd":
        from isp.pipelines import reference_render

        z = np.load(path)
        raw = RawImage(
            bayer=z["noisy"].astype(np.float32), wb_gains=tuple(float(v) for v in z["wb_gains"]),
            cam2rgb=z["cam2rgb"].astype(np.float32), exposure=auto_exposure(z["gt"].astype(np.float32)),
            meta={"source": "sidd", "filename": e["file"], "pattern": "RGGB (SIDD, aligned)",
                  "orig_size": [int(z["noisy"].shape[1]), int(z["noisy"].shape[0])],
                  "notes": [f"Real smartphone RAW ({e['source']['camera']}, ISO {e['source']['iso']}) from an unseen "
                            "test scene. Ground truth = the dataset's clean RAW, rendered with the same "
                            "classical colour/tone stages."],
                  "has_metadata_wb": True},
        )
        raw.gt = reference_render(z["gt"].astype(np.float32), raw)
        return raw
    raw = load_image(path.read_bytes(), e["file"])
    raw.meta["source"] = "sample"
    return raw


def load_sample(sample_id: str) -> RawImage:
    with _lock:
        raw = _load_cached(sample_id)
    return raw.copy()
