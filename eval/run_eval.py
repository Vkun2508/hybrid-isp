"""Full evaluation on the held-out SIDD-CC test scenes -> results/benchmarks.json + eval_P2_<TeamID>.md.

Uses the SAME pipelines the website runs (isp.pipelines) and the model in models/model.json.

    python eval/prepare_testset.py                       # once: cache the test pairs
    python eval/run_eval.py --team-id T01 --usns "1AB21CS001,1AB21CS002,1AB21CS003,1AB21CS004"
    python eval/run_eval.py --quick                      # 6 images, smaller sweeps (smoke test)

Metrics, in the brief's order: (1) PSNR, SSIM  (2) colour accuracy dE2000  (3) LPIPS (AlexNet)
(4) parameters, model size  (5) inference latency, memory.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CACHE, ROOT, TEST_SCENES, load_pair  # noqa: E402

from isp import metrics as live_metrics  # noqa: E402
from isp import model_slot, pipelines  # noqa: E402
from isp.pipelines import HybridOptions  # noqa: E402
from isp.raw import pack  # noqa: E402
from isp.simulate import SimParams, iso_label, simulate  # noqa: E402
from skimage.color import deltaE_ciede2000  # noqa: E402

RESULTS = ROOT / "results"
P3 = ("traditional", "learned", "hybrid")


# ------------------------------------------------------------------ metrics (full image)
def de2000(a: np.ndarray, b: np.ndarray) -> float:
    la = cv2.cvtColor(np.clip(a, 0, 1).astype(np.float32), cv2.COLOR_RGB2Lab)
    lb = cv2.cvtColor(np.clip(b, 0, 1).astype(np.float32), cv2.COLOR_RGB2Lab)
    return float(np.mean(deltaE_ciede2000(lb, la)))


class Lpips:
    def __init__(self, enabled: bool):
        self.fn = None
        if enabled:
            try:
                import lpips
                import torch

                self.torch = torch
                self.fn = lpips.LPIPS(net="alex", verbose=False).eval()
            except Exception as e:  # LPIPS is optional (needs torch)
                print("LPIPS disabled:", e)

    def __call__(self, a: np.ndarray, b: np.ndarray) -> float | None:
        if self.fn is None:
            return None
        t = lambda x: self.torch.from_numpy(np.clip(x, 0, 1).transpose(2, 0, 1)[None].astype(np.float32) * 2 - 1)  # noqa: E731
        with self.torch.no_grad():
            return float(self.fn(t(a), t(b)).item())


def score(img: np.ndarray, gt: np.ndarray, lp: Lpips | None = None) -> dict:
    a, b = np.clip(img, 0, 1).astype(np.float32), np.clip(gt, 0, 1).astype(np.float32)
    d = {"psnr": live_metrics.psnr(a, b), "ssim": live_metrics.ssim(a, b), "delta_e": de2000(a, b)}
    if lp is not None:
        d["lpips"] = lp(a, b)
    return d


def mean_dicts(rows: list[dict]) -> dict:
    keys = rows[0].keys()
    out = {}
    for k in keys:
        vals = [r[k] for r in rows if r.get(k) is not None]
        out[k] = float(np.mean(vals)) if vals else None
    return out


def rss_mb() -> float | None:
    try:
        import psutil

        return psutil.Process().memory_info().rss / 1e6
    except Exception:
        return None


# ------------------------------------------------------------------ sections
def quality(pairs, model, lp) -> tuple[list[dict], list[dict], dict]:
    per_image, raw_dom = [], {"noisy": [], "denoised": []}
    peak = rss_mb() or 0.0
    times = {p: [] for p in P3}
    for i, path in enumerate(pairs, 1):
        raw, gt_bayer, info = load_pair(path)
        out = pipelines.run(raw, model=model, with_metrics=False)
        row = {**info, "route": out.route.path, "input": score(pipelines.naive_render(raw), raw.gt, lp)}
        for p in P3:
            row[p] = score(out.results[p].image, raw.gt, lp)
            times[p].append(out.results[p].latency_ms)
        # RAW-domain denoising quality (what the network itself is trained for)
        if model.m.output == "raw4":
            y = model.run(pack(raw.bayer))
            g = pack(gt_bayer)
            raw_dom["noisy"].append(live_metrics.psnr(np.clip(pack(raw.bayer), 0, 1), np.clip(g, 0, 1)))
            raw_dom["denoised"].append(live_metrics.psnr(np.clip(y, 0, 1), np.clip(g, 0, 1)))
        per_image.append(row)
        peak = max(peak, rss_mb() or 0.0)
        print(f"  [{i}/{len(pairs)}] {info['instance']}: " + "  ".join(
            f"{p[:4]} {row[p]['psnr']:.2f}" for p in ("input", *P3)) + f"  route={out.route.path}", flush=True)
    summary = []
    for p in ("input", *P3):
        m = mean_dicts([r[p] for r in per_image])
        summary.append({"pipeline": p, "psnr": round(m["psnr"], 3), "ssim": round(m["ssim"], 4),
                        "delta_e": round(m["delta_e"], 3), "lpips": round(m["lpips"], 4) if m.get("lpips") is not None else None})
    extra = {
        "latency_ms_per_image": {p: round(float(np.median(times[p])), 1) for p in P3},
        "peak_rss_mb": round(peak, 1) if peak else None,
        "raw_domain_psnr": {k: round(float(np.mean(v)), 3) for k, v in raw_dom.items() if v},
    }
    return summary, per_image, extra


def efficiency(manifest_path: Path) -> list[dict]:
    rows = []
    base = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    files = []
    if base and base.get("file"):
        for name in ("isp_unet.onnx", "isp_unet_int8.onnx", base["file"]):
            if (manifest_path.parent / name).exists() and name not in files:
                files.append(name)
    x = (np.random.default_rng(0).random((512, 512, 4)).astype(np.float32) * 0.3)  # 1 MP RAW = 512x512 packed
    for name in files:
        tmp = manifest_path.parent / f"_eval_{name}.json"
        tmp.write_text(json.dumps({**base, "file": name}))
        try:
            before = rss_mb()
            m = model_slot.load_model(tmp)
            m.run(x)
            ts = []
            for _ in range(20):
                t0 = time.perf_counter()
                m.run(x)
                ts.append((time.perf_counter() - t0) * 1000)
            after = rss_mb()
            rows.append({"variant": ("ONNX INT8" if "int8" in name else "ONNX FP32"), "file": name,
                         "params": m.info.params, "size_mb": m.info.size_mb,
                         "latency_ms_1mp": round(float(np.median(ts)), 1),
                         "peak_mem_mb": round(after - before, 1) if before and after else None})
        finally:
            tmp.unlink(missing_ok=True)
    return rows


def robustness(pairs, model) -> list[dict]:
    sweeps = [
        ("Added sensor noise", "approx. ISO", "noise", [0.0, 0.25, 0.5, 0.75, 1.0]),
        ("Under-exposure", "EV", "exposure_ev", [0.0, -1.0, -2.0, -3.0, -4.0]),
        ("White-balance error", "error (fraction of R/B gain)", "wb_shift", [0.0, 0.25, 0.5, 0.75, 1.0]),
    ]
    loaded = [load_pair(p)[0] for p in pairs]
    out = []
    for title, xlab, field, levels in sweeps:
        series = {p: [] for p in P3}
        series["hybrid_router_off"] = []
        for lv in levels:
            acc = {k: [] for k in series}
            for j, raw in enumerate(loaded):
                deg = simulate(raw, SimParams(**{field: lv}, seed=100 + j))
                r = pipelines.run(deg, model=model, with_metrics=False)
                for p in P3:
                    acc[p].append(live_metrics.psnr(np.clip(r.results[p].image, 0, 1), raw.gt))
                r2 = pipelines.run(deg, names=["hybrid"], model=model, opts=HybridOptions(router=False), with_metrics=False)
                acc["hybrid_router_off"].append(live_metrics.psnr(np.clip(r2.results["hybrid"].image, 0, 1), raw.gt))
            for k in series:
                series[k].append(round(float(np.mean(acc[k])), 3))
            print(f"  {title} {lv}: " + "  ".join(f"{k[:6]} {series[k][-1]:.2f}" for k in series), flush=True)
        x = [iso_label(v) for v in levels] if field == "noise" else levels
        out.append({"title": title, "x_label": xlab, "x": x,
                    "series": [{"pipeline": k, "psnr": v} for k, v in series.items() if k != "hybrid_router_off"],
                    "router_off": series["hybrid_router_off"]})
    return out


def ablation(pairs, model, int8_model) -> list[dict]:
    conds = [("as captured", SimParams()), ("noise + under-exposed", SimParams(noise=0.5, exposure_ev=-2.0)),
             ("white-balance error", SimParams(wb_shift=0.6))]
    variants = [
        ("Full hybrid", HybridOptions(), None),
        ("No learned core (classical denoise)", HybridOptions(use_model=False), None),
        ("No router (always network, fixed sharpening)", HybridOptions(router=False), None),
        ("No white-balance correction", HybridOptions(wb_correction=False), None),
        ("No exposure normalisation", HybridOptions(exposure_norm=False), None),
        ("No chroma cleanup", HybridOptions(chroma_cleanup=False), None),
        ("No sharpening", HybridOptions(sharpen=False), None),
        ("Bilinear instead of Malvar demosaic", HybridOptions(demosaic="bilinear"), None),
        ("No noise-model bound on the network", HybridOptions(bound_residual=False), None),
    ]
    if int8_model is not None:
        variants.append(("INT8 learned core (instead of FP32)", HybridOptions(), int8_model))
    loaded = [load_pair(p)[0] for p in pairs]
    rows = []
    for name, opts, alt in variants:
        acc = []
        for j, raw in enumerate(loaded):
            for _, sp in conds:
                deg = simulate(raw, replace(sp, seed=200 + j))
                r = pipelines.run(deg, names=["hybrid"], model=alt or model, opts=opts, with_metrics=False)
                acc.append(score(r.results["hybrid"].image, raw.gt))
        m = mean_dicts(acc)
        rows.append({"variant": name, "psnr": round(m["psnr"], 3), "ssim": round(m["ssim"], 4), "delta_e": round(m["delta_e"], 3)})
        print(f"  {name}: {m['psnr']:.2f} dB  SSIM {m['ssim']:.4f}  dE {m['delta_e']:.2f}", flush=True)
    return rows


def failures(per_image: list[dict], pairs: list[Path], model, n: int = 3) -> list[dict]:
    """The worst hybrid results (lowest PSNR), saved as input | hybrid | ground truth crops."""
    out_dir = RESULTS / "failures"
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.jpg"):
        old.unlink()
    worst = sorted(per_image, key=lambda r: r["hybrid"]["psnr"])[:n]
    by_name = {p.stem: p for p in pairs}
    items = []
    for r in worst:
        raw, _, info = load_pair(by_name[r["instance"]])
        res = pipelines.run(raw, names=["hybrid"], model=model, with_metrics=False)
        crops = []
        for img in (pipelines.naive_render(raw), res.results["hybrid"].image, raw.gt):
            h, w = img.shape[:2]
            c = img[h // 2 - 192:h // 2 + 192, w // 2 - 256:w // 2 + 256]
            crops.append(c)
        strip = np.concatenate(crops, axis=1)
        f = out_dir / f"{info['instance']}.jpg"
        cv2.imwrite(str(f), cv2.cvtColor((np.clip(strip, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
        items.append({
            "title": f"{info['camera']} · ISO {info['iso']} · scene {info['scene']}",
            "description": (f"Input | Hybrid | Ground truth (centre crop). Hybrid {r['hybrid']['psnr']:.2f} dB vs traditional "
                            f"{r['traditional']['psnr']:.2f} dB, dE {r['hybrid']['delta_e']:.2f}; router chose '{r['route']}'."),
            "image_url": f"/api/results-files/failures/{f.name}",
            "instance": info["instance"],
        })
    return items


# ------------------------------------------------------------------ report
def write_md(path: Path, usns: str, b: dict) -> None:
    q = {r["pipeline"]: r for r in b["quality"]}
    L = [usns, "", f"# Evaluation: P2 Hybrid smartphone ISP ({b['updated']})", "",
         f"Test set: {b['dataset']['name']}, held-out scenes {', '.join(b['dataset']['test_scenes'])}, "
         f"{b['dataset']['n_images']} images ({b['dataset']['note']}). Model: {b['model']['name']} ({b['model']['variant']}).", "",
         "## 1. PSNR and SSIM (sRGB, higher is better)", "", "| Pipeline | PSNR (dB) | SSIM |", "|---|---:|---:|"]
    for p in ("input", *P3):
        L.append(f"| {p} | {q[p]['psnr']:.2f} | {q[p]['ssim']:.4f} |")
    L += ["", "## 2. Colour accuracy: mean CIEDE2000 (lower is better)", "", "| Pipeline | ΔE2000 |", "|---|---:|"]
    for p in ("input", *P3):
        L.append(f"| {p} | {q[p]['delta_e']:.3f} |")
    L += ["", "## 3. Perceptual: LPIPS, AlexNet (lower is better)", "", "| Pipeline | LPIPS |", "|---|---:|"]
    for p in ("input", *P3):
        L.append(f"| {p} | {q[p]['lpips']:.4f} |" if q[p].get("lpips") is not None else f"| {p} | n/a |")
    L += ["", "## 4. Model parameters and size", "", "| Variant | Parameters | Size (MB) |", "|---|---:|---:|"]
    for e in b.get("efficiency", []):
        L.append(f"| {e['variant']} | {e['params']:,} | {e['size_mb']:.3f} |")
    L += ["", "## 5. Inference latency and memory (CPU)", "",
          "| Variant | Latency, 1 MP RAW (ms) | Memory delta (MB) |", "|---|---:|---:|"]
    for e in b.get("efficiency", []):
        L.append(f"| {e['variant']} | {e['latency_ms_1mp']:.1f} | {e['peak_mem_mb'] if e['peak_mem_mb'] is not None else 'n/a'} |")
    x = b["extra"]
    L += ["", f"End-to-end per image ({b['dataset']['crop']}): " + ", ".join(f"{k} {v:.0f} ms" for k, v in x["latency_ms_per_image"].items())
          + (f"; peak process memory {x['peak_rss_mb']:.0f} MB." if x.get("peak_rss_mb") else "."),
          f"Machine: {b['machine']}."]
    if x.get("raw_domain_psnr"):
        L += ["", f"RAW-domain denoising (packed RGGB, what the network optimises): noisy {x['raw_domain_psnr'].get('noisy', 0):.2f} dB "
              f"-> denoised {x['raw_domain_psnr'].get('denoised', 0):.2f} dB."]
    if b.get("robustness"):
        L += ["", "## Robustness (PSNR dB as the degradation grows)", ""]
        for rb in b["robustness"]:
            L += [f"**{rb['title']}** ({rb['x_label']}: {', '.join(str(v) for v in rb['x'])})", ""]
            for s in rb["series"]:
                L.append(f"- {s['pipeline']}: " + ", ".join(f"{v:.2f}" for v in s["psnr"]))
            L += [f"- hybrid, router off: " + ", ".join(f"{v:.2f}" for v in rb["router_off"]), ""]
    if b.get("ablation"):
        L += ["## Ablation (hybrid; mean over as-captured, noise + -2 EV, and WB-error conditions)", "",
              "| Variant | PSNR | SSIM | ΔE |", "|---|---:|---:|---:|"]
        for a in b["ablation"]:
            L.append(f"| {a['variant']} | {a['psnr']:.2f} | {a['ssim']:.4f} | {a['delta_e']:.2f} |")
    if b.get("efficiency_sweep"):
        sw = b["efficiency_sweep"]
        L += ["", "## Latency and peak memory vs resolution (CPU)", "",
              "| RAW size | Network FP32 | Network INT8 | Traditional | Learned only | Hybrid | Hybrid peak memory |",
              "|---|---:|---:|---:|---:|---:|---:|"]
        for r in sw:
            L.append(f"| {r['megapixels']} MP ({r['size']}) | {r.get('net_fp32_ms', 0):.0f} ms | {r.get('net_int8_ms', 0):.0f} ms | "
                     f"{r['traditional_ms']:.0f} ms | {r['learned_ms']:.0f} ms | {r['hybrid_ms']:.0f} ms | {r['hybrid_mem_mb']:.0f} MB |")
    if b.get("tiling"):
        t = b["tiling"]
        L += ["", f"## Tiled inference ({t['image']}): seams", "", f"_{t.get('note', '')}_", "",
              "| Variant | Latency | PSNR vs one pass | Interior max abs diff | Interior PSNR |", "|---|---:|---:|---:|---:|"]
        for v in t["variants"]:
            L.append(f"| {v['variant']} | {v['latency_ms']:.0f} ms | {v['psnr_vs_single_pass']:.1f} dB | "
                     f"{v.get('interior_max_abs_diff', 0):.2e} | {v.get('interior_psnr', 0):.1f} dB |")
    if b.get("router"):
        r = b["router"]
        L += ["", f"## Adaptive router: quality vs compute ({r['n_images']} images x {', '.join(r['conditions'])})", "",
              "| Mode | PSNR | ΔE2000 | Mean latency | Images through the network |", "|---|---:|---:|---:|---:|"]
        for m in r["modes"]:
            L.append(f"| {m['mode']} | {m['psnr']:.2f} | {m['delta_e']:.2f} | {m['latency_ms']:.0f} ms | {m['network_share'] * 100:.0f}% |")
        L += ["", "Paths chosen by the adaptive router (as captured): " + "; ".join(
            f"{k}: " + ", ".join(f"{p} {n}" for p, n in v.items()) for k, v in r["paths_by_iso_as_captured"].items())]
    if b.get("breakdown"):
        for title, key in (("Per phone", "by_phone"), ("Per ISO band", "by_iso")):
            L += ["", f"## {title} (PSNR dB / ΔE2000, as captured)", "",
                  "| Group | n | Input | Traditional | Learned only | Hybrid |", "|---|---:|---:|---:|---:|---:|"]
            for g in b["breakdown"][key]:
                L.append(f"| {g['group']} | {g['n']} | " + " | ".join(
                    f"{g[p]['psnr']:.2f} / {g[p]['delta_e']:.2f}" for p in ("input", "traditional", "learned", "hybrid")) + " |")
    if b.get("failures"):
        L += ["", "## Failure cases", ""]
        for f in b["failures"]:
            L.append(f"- **{f['title']}**: {f['description']} (`results/failures/{f['instance']}.jpg`)")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(model_slot.MANIFEST_PATH))
    ap.add_argument("--limit", type=int, default=0, help="max test images for quality (0 = all cached)")
    ap.add_argument("--robust-n", type=int, default=8, help="images used for robustness and ablation")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-lpips", action="store_true")
    ap.add_argument("--skip", default="", help="comma list of sections to skip: robustness,ablation,efficiency,failures")
    ap.add_argument("--team-id", default="TeamID")
    ap.add_argument("--usns", default="USN1, USN2, USN3, USN4")
    ap.add_argument("--md-only", action="store_true", help="only (re)write eval_P2_<TeamID>.md from results/benchmarks.json")
    ap.add_argument("--allow-stub", action="store_true", help="allow evaluating the classical stand-in (smoke tests only)")
    args = ap.parse_args()
    skip = set(filter(None, args.skip.split(",")))
    if args.md_only:
        b = json.loads((RESULTS / "benchmarks.json").read_text(encoding="utf-8"))
        md = ROOT / f"eval_P2_{args.team_id}.md"
        write_md(md, ", ".join(u.strip() for u in args.usns.split(",")), b)
        print(f"wrote {md.name}")
        return 0

    pairs = sorted(CACHE.glob("*.npz"))
    if not pairs:
        print("No cached test pairs. Run: python eval/prepare_testset.py")
        return 1
    if args.quick:
        args.limit, args.robust_n = args.limit or 6, min(args.robust_n, 3)
    if args.limit:
        pairs = pairs[: args.limit]
    # robustness/ablation subset: spread over scenes and cameras
    sub = pairs[:: max(1, len(pairs) // args.robust_n)][: args.robust_n]

    mpath = Path(args.manifest)
    model = model_slot.load_model(mpath)
    print(f"model: {model.info.name} ({model.info.kind}, file={model.info.file}, stub={model.is_stub})")
    if model.is_stub and not args.allow_stub:
        print("REFUSING: models/model.json does not load a trained network (placeholder in use"
              + (f": {model.info.load_error}" if model.info.load_error else "") + "). "
              "Put the exported ONNX in models/, or pass --allow-stub for a smoke test.")
        return 2
    int8 = None
    if mpath.exists() and not model.is_stub:
        base = json.loads(mpath.read_text())
        alt = "isp_unet_int8.onnx" if base.get("file") != "isp_unet_int8.onnx" else "isp_unet.onnx"
        if (mpath.parent / alt).exists():
            tmp = mpath.parent / "_eval_alt.json"
            tmp.write_text(json.dumps({**base, "file": alt}))
            int8_candidate = model_slot.load_model(tmp)
            tmp.unlink(missing_ok=True)
            if "int8" in alt:
                int8 = int8_candidate
            else:  # the main model is INT8: compare against FP32 in the ablation instead
                int8, model = model, int8_candidate

    t0 = time.time()
    print(f"quality on {len(pairs)} images…")
    summary, per_image, extra = quality(pairs, model, Lpips(not args.no_lpips))
    b = {
        "status": "provisional" if model.is_stub else "final",
        "updated": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "model": {"name": model.info.name, "params": model.info.params or 0,
                  "variant": "classical stand-in (no trained weights)" if model.is_stub else (model.info.file or "")},
        "dataset": {"name": "SIDD-CC smartphone RAW", "test_scenes": list(TEST_SCENES), "n_images": len(pairs),
                    "note": "scene-level hold-out, 5 phones, one noisy/clean pair per test instance",
                    "crop": f"centre {load_pair(pairs[0])[0].bayer.shape[1]}x{load_pair(pairs[0])[0].bayer.shape[0]}"},
        "machine": f"{os.cpu_count()} logical CPUs, ONNX Runtime CPU",
        "quality": summary,
        "extra": extra,
    }
    if "efficiency" not in skip:
        print("efficiency…")
        b["efficiency"] = efficiency(mpath)
    if "robustness" not in skip:
        print(f"robustness on {len(sub)} images…")
        b["robustness"] = robustness(sub, model)
    if "ablation" not in skip:
        print(f"ablation on {len(sub)} images…")
        b["ablation"] = ablation(sub, model, int8)
    if "failures" not in skip:
        b["failures"] = failures(per_image, pairs, model)
    from extra import breakdown, efficiency_sweep, router_value, tiling_seams

    b["breakdown"] = breakdown(per_image)
    if "router" not in skip:
        print(f"router value on {len(pairs)} images…")
        b["router"] = router_value(pairs, model)
    if "sweep" not in skip and not model.is_stub:
        print("efficiency sweep…")
        b["efficiency_sweep"] = efficiency_sweep(mpath, pairs[0])
        b["tiling"] = tiling_seams(mpath, pairs[0])
    b["meta"] = {"subset_n": len(sub),
                 "ablation_conditions": ["as captured", "noise + 2 EV under", "white-balance error"],
                 "robustness_note": f"mean PSNR over {len(sub)} test images per level"}
    b["per_image"] = per_image

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "benchmarks.json").write_text(json.dumps(b, indent=2), encoding="utf-8")
    md = ROOT / f"eval_P2_{args.team_id}.md"
    write_md(md, ", ".join(u.strip() for u in args.usns.split(",")), b)
    print(f"wrote results/benchmarks.json and {md.name} in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
