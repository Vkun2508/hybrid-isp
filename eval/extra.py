"""Extra evaluation sections: efficiency sweep, tiling seams, router value, per-phone breakdown.

Imported by run_eval.py; each function returns plain JSON-ready data for results/benchmarks.json.
"""
from __future__ import annotations

import json
import threading
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from common import load_pair
from isp import metrics as M
from isp import model_slot, pipelines
from isp.pipelines import HybridOptions
from isp.raw import RawImage, pack, unpack
from isp.simulate import SimParams, simulate

P3 = ("traditional", "learned", "hybrid")


class PeakRSS:
    """Peak resident memory of this process while the block runs (sampled every 2 ms), minus the start."""

    def __enter__(self):
        import psutil

        self.proc = psutil.Process()
        self.base = self.peak = self.proc.memory_info().rss
        self._stop = False
        self._t = threading.Thread(target=self._poll, daemon=True)
        self._t.start()
        return self

    def _poll(self):
        while not self._stop:
            self.peak = max(self.peak, self.proc.memory_info().rss)
            time.sleep(0.002)

    def __exit__(self, *exc):
        self._stop = True
        self._t.join()

    @property
    def delta_mb(self) -> float:
        return round((self.peak - self.base) / 1e6, 1)


def _median_ms(fn, runs: int) -> float:
    fn()  # warm-up
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000)
    return round(float(np.median(ts)), 1)


def _variant_models(manifest: Path) -> dict[str, model_slot.LearnedModel]:
    base = json.loads(manifest.read_text())
    out = {}
    for label, f in (("FP32", "isp_unet.onnx"), ("INT8", "isp_unet_int8.onnx")):
        if (manifest.parent / f).exists():
            tmp = manifest.parent / f"_eval_{f}.json"
            tmp.write_text(json.dumps({**base, "file": f}))
            out[label] = model_slot.load_model(tmp)
            tmp.unlink(missing_ok=True)
    return out


def _raw_at(raw: RawImage, mp: float) -> RawImage:
    """Resize a real RAW (per Bayer plane) to `mp` megapixels at the same aspect ratio."""
    h, w = raw.bayer.shape
    s = (mp * 1e6 / (h * w)) ** 0.5
    nh, nw = max(8, int(h * s / 2) // 4 * 4), max(8, int(w * s / 2) // 4 * 4)
    q = pack(raw.bayer)
    q2 = np.stack([cv2.resize(q[..., c], (nw, nh), interpolation=cv2.INTER_LINEAR) for c in range(4)], axis=-1)
    return raw.copy(bayer=unpack(q2), gt=None)


def efficiency_sweep(manifest: Path, sample: Path, sizes=(0.5, 1, 2, 4)) -> list[dict]:
    """Network latency + peak memory for FP32/INT8, and whole-pipeline latency, at several resolutions."""
    models = _variant_models(manifest)
    raw0 = load_pair(sample)[0]
    rows = []
    for mp in sizes:
        raw = _raw_at(raw0, mp)
        inp = pack(raw.bayer)
        row = {"megapixels": mp, "size": f"{raw.bayer.shape[1]}x{raw.bayer.shape[0]}"}
        for label, m in models.items():
            with PeakRSS() as mem:
                row[f"net_{label.lower()}_ms"] = _median_ms(lambda: m.run(inp), 3)
            row[f"net_{label.lower()}_mem_mb"] = mem.delta_mb
        fp32 = models.get("FP32") or next(iter(models.values()))
        for p in P3:  # hybrid timed with the network forced on: its worst case (the router may skip it)
            fp = "learned" if p == "hybrid" else None
            with PeakRSS() as mem:
                row[f"{p}_ms"] = _median_ms(lambda: pipelines.run(raw, names=[p], model=fp32, force_path=fp, with_metrics=False), 2)
            row[f"{p}_mem_mb"] = mem.delta_mb
        rows.append(row)
        print("  ", row, flush=True)
    return rows


def tiling_seams(manifest: Path, sample: Path, mp: float = 4.0) -> dict:
    """Same network, same 4 MP image: one pass vs overlapping tiles. Any seam shows up as a difference."""
    base = json.loads(manifest.read_text())
    raw = _raw_at(load_pair(sample)[0], mp)
    inp = pack(raw.bayer)
    outs = {}
    for name, tile, ov in (("single pass", 4096, 32), ("tiles 512, overlap 32", 512, 32), ("tiles 256, overlap 32", 256, 32),
                           ("tiles 256, overlap 64", 256, 64)):
        tmp = manifest.parent / "_eval_tiles.json"
        tmp.write_text(json.dumps({**base, "tile": tile, "overlap": ov}))
        m = model_slot.load_model(tmp)
        tmp.unlink(missing_ok=True)
        outs[name] = (m.run(inp), _median_ms(lambda: m.run(inp), 2))
    ref = outs["single pass"][0]
    res = {"image": f"{raw.bayer.shape[1]}x{raw.bayer.shape[0]} RAW", "variants": [],
           "note": "interior = excluding a 32-pixel border, where padding (not tiling) decides the result"}
    for name, (o, ms) in outs.items():
        d = np.abs(o - ref)
        di = d[32:-32, 32:-32]
        psnr = lambda x: 99.0 if x.max() == 0 else round(float(10 * np.log10(1 / np.mean(x ** 2))), 2)  # noqa: E731
        res["variants"].append({"variant": name, "latency_ms": ms, "max_abs_diff": float(d.max()), "psnr_vs_single_pass": psnr(d),
                                "interior_max_abs_diff": float(di.max()), "interior_psnr": psnr(di)})
    print("  ", res, flush=True)
    return res


def router_value(pairs: list[Path], model) -> dict:
    """Adaptive router vs always-network vs never-network: quality AND compute, on identical inputs."""
    conds = [("as captured", SimParams()), ("extra noise", SimParams(noise=0.5)),
             ("under-exposed 2 EV", SimParams(exposure_ev=-2.0)), ("white-balance error", SimParams(wb_shift=0.6))]
    modes = {"Adaptive router": (HybridOptions(), None), "Always network": (HybridOptions(router=False), None),
             "Never network (classical only)": (HybridOptions(use_model=False), None)}
    acc = {k: {"psnr": [], "de": [], "ms": [], "paths": defaultdict(int)} for k in modes}
    by_iso = defaultdict(lambda: defaultdict(int))
    for j, p in enumerate(pairs):
        raw, _, info = load_pair(p)
        for ci, (cname, sp) in enumerate(conds):
            deg = simulate(raw, replace(sp, seed=500 + j))
            for mname, (opts, _) in modes.items():
                t0 = time.perf_counter()
                o = pipelines.run(deg, names=["hybrid"], model=model, opts=opts, with_metrics=False)
                ms = (time.perf_counter() - t0) * 1000
                img = np.clip(o.results["hybrid"].image, 0, 1)
                a = acc[mname]
                a["psnr"].append(M.psnr(img, raw.gt))
                lab = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGB2Lab)
                gl = cv2.cvtColor(raw.gt.astype(np.float32), cv2.COLOR_RGB2Lab)
                from skimage.color import deltaE_ciede2000

                a["de"].append(float(np.mean(deltaE_ciede2000(gl[::2, ::2], lab[::2, ::2]))))
                a["ms"].append(ms)
                a["paths"][o.route.path] += 1
                if mname == "Adaptive router" and ci == 0:
                    bucket = "ISO <= 200" if info["iso"] <= 200 else ("ISO 400-800" if info["iso"] <= 800 else "ISO >= 1000")
                    by_iso[bucket][o.route.path] += 1
        print(f"  router [{j + 1}/{len(pairs)}]", flush=True)
    rows = []
    for mname, a in acc.items():
        n = sum(a["paths"].values())
        rows.append({"mode": mname, "psnr": round(float(np.mean(a["psnr"])), 3), "delta_e": round(float(np.mean(a["de"])), 3),
                     "latency_ms": round(float(np.mean(a["ms"])), 1),
                     "network_share": round(1 - a["paths"].get("fast", 0) / max(n, 1), 3),
                     "paths": dict(a["paths"])})
    return {"conditions": [c for c, _ in conds], "n_images": len(pairs), "modes": rows,
            "paths_by_iso_as_captured": {k: dict(v) for k, v in sorted(by_iso.items())}}


def breakdown(per_image: list[dict]) -> dict:
    """Mean PSNR / dE per phone and per ISO band (as captured), for every pipeline."""
    names = {"G4": "LG G4", "GP": "Google Pixel", "IP": "iPhone 7", "N6": "Nexus 6", "S6": "Galaxy S6 Edge"}

    def group(key):
        g = defaultdict(list)
        for r in per_image:
            g[key(r)].append(r)
        out = []
        for k, rs in sorted(g.items()):
            row = {"group": k, "n": len(rs)}
            for p in ("input", *P3):
                row[p] = {"psnr": round(float(np.mean([r[p]["psnr"] for r in rs])), 2),
                          "delta_e": round(float(np.mean([r[p]["delta_e"] for r in rs])), 2)}
            out.append(row)
        return out

    def iso_band(r):
        i = r["iso"]
        return "ISO 50-200" if i <= 200 else ("ISO 320-800" if i <= 800 else "ISO 1600-3200")

    return {"by_phone": group(lambda r: names.get(r["camera"], r["camera"])), "by_iso": group(iso_band)}
