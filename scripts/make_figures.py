"""Figures for the design document (docs/figures/), generated from the real pipeline and trained model.

    python scripts/make_figures.py            # low-light, backlit and intermediate-state figures
    python scripts/make_figures.py --ablation # ablation chart from results/benchmarks.json

1. fig_lowlight.jpg      extreme low light: RAW input as captured | traditional | hybrid | ground truth
2. fig_backlit.jpg       high-contrast scene: highlight-aware vs mean-only auto-exposure, with clipped pixels marked
   fig_bright_regions.jpg bright-region detail: the network alone vs the network inside the noise-model bound
                         (+ results/bright_regions.json: RAW PSNR on all 26 test images, whole image and bright regions)
3. fig_intermediates.jpg the hybrid's intermediate states: Bayer mosaic, noise-level map, what the network removed,
                         where the noise-model bound acted, denoised RAW, final
4. fig_ablation.png      PSNR change when each hybrid module is removed (from the evaluation)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from isp import analysis, metrics, pipelines  # noqa: E402
from isp.classical import denoise, wb  # noqa: E402
from isp.classical.demosaic import bilinear  # noqa: E402
from isp.io import load_any  # noqa: E402
from isp.model_slot import get_model  # noqa: E402
from isp.raw import pack, unpack  # noqa: E402
from isp.samples import load_sample  # noqa: E402
from isp.simulate import SimParams, simulate  # noqa: E402

OUT = ROOT / "docs" / "figures"
BG = (23, 23, 25)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def to_u8(img: np.ndarray) -> np.ndarray:
    return (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)


def tile(img: np.ndarray, title: str, sub: str = "", h: int = 300) -> np.ndarray:
    """Resize to height h and add a label band (title + subtitle) above."""
    im = to_u8(img) if img.dtype != np.uint8 else img
    w = int(im.shape[1] * h / im.shape[0])
    im = cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)
    band = np.full((56, w, 3), BG, np.uint8)
    cv2.putText(band, title, (10, 24), FONT, 0.62, (245, 245, 245), 1, cv2.LINE_AA)
    if sub:
        cv2.putText(band, sub, (10, 46), FONT, 0.5, (150, 152, 158), 1, cv2.LINE_AA)
    return np.concatenate([band, im], axis=0)


def row(tiles: list[np.ndarray], gap: int = 8) -> np.ndarray:
    h = max(t.shape[0] for t in tiles)
    parts = []
    for t in tiles:
        if t.shape[0] < h:
            t = np.concatenate([t, np.full((h - t.shape[0], t.shape[1], 3), BG, np.uint8)], axis=0)
        parts += [t, np.full((h, gap, 3), BG, np.uint8)]
    return np.concatenate(parts[:-1], axis=1)


def stack(rows: list[np.ndarray], gap: int = 14) -> np.ndarray:
    w = max(r.shape[1] for r in rows)
    parts = []
    for r in rows:
        if r.shape[1] < w:
            r = np.concatenate([r, np.full((r.shape[0], w - r.shape[1], 3), BG, np.uint8)], axis=1)
        parts += [r, np.full((gap, w, 3), BG, np.uint8)]
    return np.concatenate(parts[:-1], axis=0)


def save(img: np.ndarray, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT / name), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
    print("wrote", OUT / name, img.shape[1], "x", img.shape[0])


def as_captured(raw) -> np.ndarray:
    """What the sensor recorded, with no exposure lift and no processing beyond a bilinear demosaic + camera colour."""
    x = wb.apply_gains(raw.bayer, raw.wb_gains or wb.gray_world(raw.bayer))
    return pipelines.render_linear_camera(bilinear(x), raw.cam2rgb)


def heat(m: np.ndarray, vmax: float | None = None) -> np.ndarray:
    v = m / (vmax if vmax else max(float(np.percentile(m, 99.5)), 1e-9))
    return cv2.applyColorMap(to_u8(np.clip(v, 0, 1)), cv2.COLORMAP_INFERNO)[..., ::-1].copy()


def psnr_label(img, gt) -> str:
    return f"PSNR {metrics.psnr(np.clip(img, 0, 1), gt):.1f} dB" if gt is not None else ""


# ----------------------------------------------------------------------------------------------- 1. low light
def fig_lowlight():
    rows = []
    cases = [("sidd-n6-iso3200-lowlight", None, "Real phone RAW: Nexus 6, ISO 3200, low light (unseen test scene)"),
             ("photo-coffee-night", SimParams(noise=0.75, exposure_ev=-3.0, seed=1), "Night shot: 3 stops under-exposed, high ISO (simulated sensor)")]
    for sid, sim, caption in cases:
        raw = load_sample(sid)
        if sim:
            raw = simulate(raw, sim)
        out = pipelines.run(raw, names=["traditional", "hybrid"], with_metrics=False)
        trad, hyb = out.results["traditional"].image, out.results["hybrid"].image
        tiles = [tile(as_captured(raw), "RAW input, as captured", "no processing"),
                 tile(trad, "Traditional ISP", psnr_label(trad, raw.gt)),
                 tile(hyb, "Hybrid (ours)", psnr_label(hyb, raw.gt) + f"  route: {out.route.label}"),
                 tile(raw.gt, "Ground truth", "clean reference")]
        r = row(tiles)
        band = np.full((34, r.shape[1], 3), BG, np.uint8)
        cv2.putText(band, caption, (10, 24), FONT, 0.66, (245, 245, 245), 1, cv2.LINE_AA)
        rows.append(np.concatenate([band, r], axis=0))
    save(stack(rows), "fig_lowlight.jpg")


# ----------------------------------------------------------------------------------------------- 2. backlit
def clip_map(img: np.ndarray) -> tuple[np.ndarray, float]:
    clipped = (img >= 0.99).any(axis=-1)
    vis = to_u8(img * 0.45)
    vis[clipped] = (229, 86, 111)
    return vis, float(clipped.mean() * 100)


def fig_backlit():
    """Same scene, same target brightness (mean green 0.14): the only difference is the highlight cap in
    analysis.auto_exposure (gain <= 1 / 99th-percentile green)."""
    from server.main import _orient
    f = "pixel4a.dng"
    p = ROOT / ".cache" / "phone_raw" / f
    raw = load_any(p.read_bytes(), f)
    q = pack(raw.bayer)
    mean_g = float(np.maximum(0.5 * (q[..., 1] + q[..., 2]), 0).mean())
    naive = raw.copy()
    naive.exposure = float(np.clip(0.14 / max(mean_g, 1e-5), 1, 16))
    hyb = pipelines.run(raw, names=["hybrid"], with_metrics=False).results["hybrid"].image
    nai = pipelines.run(naive, names=["hybrid"], with_metrics=False).results["hybrid"].image
    cm_h, pct_h = clip_map(hyb)
    cm_n, pct_n = clip_map(nai)
    o = lambda im: _orient(im, raw.flip)
    tiles = [tile(o(as_captured(raw)), "RAW input, as captured", "Google Pixel 4a DNG"),
             tile(o(hyb), "Hybrid: highlight-aware exposure", f"x{raw.exposure:.2f} gain (capped by the brightest 1%)"),
             tile(o(cm_h), "Clipped pixels (red)", f"{pct_h:.2f}% of the image"),
             tile(o(nai), "Mean-brightness exposure only", f"x{naive.exposure:.2f} gain, same target, no cap"),
             tile(o(cm_n), "Clipped pixels (red)", f"{pct_n:.2f}% of the image")]
    save(row(tiles, gap=6), "fig_backlit.jpg")
    print(f"  {f}: clipped {pct_h:.2f}% (highlight-aware x{raw.exposure:.2f}) vs {pct_n:.2f}% (mean-only x{naive.exposure:.2f})")
    return {"file": f, "gain_aware": round(raw.exposure, 3), "gain_mean_only": round(naive.exposure, 3),
            "clipped_pct_aware": round(pct_h, 2), "clipped_pct_mean_only": round(pct_n, 2)}


def fig_bright_regions(backlit: dict | None = None):
    """RAW PSNR of the network alone vs inside the noise-model bound, whole image and bright regions (clean RAW
    above 20% of white), on all 26 held-out test images; then a picture of the image where the bound matters most."""
    sys.path.insert(0, str(ROOT / "eval"))
    from common import CACHE, load_pair
    model = get_model()
    k = analysis.CONFIG.residual_k or 2.0
    rows_, best = [], None
    for path in sorted(CACHE.glob("*.npz")):
        raw, gt_bayer, info = load_pair(path)
        x, g = pack(raw.bayer), pack(gt_bayer)
        y = model.run(x)
        yb = denoise.bound_residual(x, y, k)
        bright = (g.mean(axis=-1) > 0.2)
        def rp(a, m=None):
            a, gg = np.clip(a, 0, 1), np.clip(g, 0, 1)
            if m is None:
                return metrics.psnr(a, gg)
            mse = float(((a - gg) ** 2)[m].mean())
            return 10 * np.log10(1.0 / max(mse, 1e-12))
        r = {"instance": info["instance"], "bright_share": round(float(bright.mean()), 3),
             "noisy": round(rp(x), 2), "network": round(rp(y), 2), "bounded": round(rp(yb), 2)}
        if bright.mean() > 0.02:
            r.update(noisy_bright=round(rp(x, bright), 2), network_bright=round(rp(y, bright), 2),
                     bounded_bright=round(rp(yb, bright), 2))
            gain = r["bounded_bright"] - r["network_bright"]
            if best is None or gain > best[0]:
                best = (gain, path)
        rows_.append(r)
        print("  ", r)
    mean = lambda key: round(float(np.mean([r[key] for r in rows_ if key in r])), 2)
    summary = {"k_sigma": k, "n_images": len(rows_),
               "whole_image": {"noisy": mean("noisy"), "network": mean("network"), "bounded": mean("bounded")},
               "bright_regions": {"n_images": sum("noisy_bright" in r for r in rows_), "noisy": mean("noisy_bright"),
                                  "network": mean("network_bright"), "bounded": mean("bounded_bright")},
               "network_worse_than_input": {"network": sum(r["network"] < r["noisy"] for r in rows_),
                                            "bounded": sum(r["bounded"] < r["noisy"] for r in rows_)},
               "backlit_exposure": backlit, "per_image": rows_}
    (ROOT / "results" / "bright_regions.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print("summary", {k_: v for k_, v in summary.items() if k_ != "per_image"})

    # picture: the image where the bound helps bright regions most, at the bright window it changes most
    raw, gt_bayer, info = load_pair(best[1])
    on = pipelines.run(raw, names=["hybrid"], with_metrics=False).results["hybrid"].image
    off = pipelines.run(raw, names=["hybrid"], opts=pipelines.HybridOptions(bound_residual=False),
                        with_metrics=False).results["hybrid"].image
    gt, inp = raw.gt, pipelines.naive_render(raw)
    s_ = 160
    lum = gt.mean(-1)
    err = ((off - gt) ** 2).mean(-1) - ((on - gt) ** 2).mean(-1)
    best_w, bw = None, -1e9
    for yy in range(0, gt.shape[0] - s_, s_ // 2):
        for xx in range(0, gt.shape[1] - s_, s_ // 2):
            if lum[yy:yy + s_, xx:xx + s_].mean() < 0.45:
                continue
            v = err[yy:yy + s_, xx:xx + s_].mean()
            if v > bw:
                bw, best_w = v, (yy, xx)
    yy, xx = best_w or (0, 0)
    c = (slice(yy, yy + s_), slice(xx, xx + s_))
    up = lambda im: cv2.resize(to_u8(im[c]), (s_ * 2, s_ * 2), interpolation=cv2.INTER_NEAREST)
    ps = lambda im: f"PSNR {metrics.psnr(np.clip(im[c], 0, 1), gt[c]):.1f} dB (this crop)"
    r_ = next(r for r in rows_ if r["instance"] == info["instance"])
    tiles = [tile(up(inp), "Noisy input", ps(inp)),
             tile(up(off), "Hybrid, network unbounded", ps(off)),
             tile(up(on), "Hybrid, network within 2 sigma", ps(on)),
             tile(up(gt), "Ground truth", "clean reference")]
    r = row(tiles)
    band = np.full((34, r.shape[1], 3), BG, np.uint8)
    cv2.putText(band, f"{info['instance']}: bright regions, RAW PSNR network {r_['network_bright']:.1f} dB -> bounded "
                f"{r_['bounded_bright']:.1f} dB (input {r_['noisy_bright']:.1f})", (10, 24), FONT, 0.6, (245, 245, 245), 1, cv2.LINE_AA)
    save(np.concatenate([band, r], axis=0), "fig_bright_regions.jpg")


# ----------------------------------------------------------------------------------------------- 3. intermediates
def fig_intermediates():
    raw = load_sample("sidd-n6-iso3200-lowlight")
    model = get_model()
    inp = pack(raw.bayer)
    y = model.run(inp)
    a, b = denoise.estimate_nlf(inp)
    k = analysis.CONFIG.residual_k or 2.0
    sigma = np.sqrt(np.maximum(a * np.maximum(y, 0) + b, 1e-12))
    bounded = denoise.bound_residual(inp, y, k, (a, b))
    active = (np.abs(y - inp) > k * sigma).any(axis=-1)
    removed = np.abs(bounded - inp).mean(axis=-1)
    out = pipelines.run(raw, names=["hybrid"], with_metrics=False)
    # centre crop for the per-pixel maps so the structure is visible
    h, w = inp.shape[:2]
    cy, cx, s = h // 2, w // 2, min(h, w) // 2
    crop = (slice(cy - s // 2, cy + s // 2), slice(cx - s // 2, cx + s // 2))
    bay = pipelines.viz_mosaic(raw.bayer, crop=96)
    noisy_lin = pipelines.render_linear_camera(bilinear(wb.apply_gains(raw.bayer, raw.wb_gains)) * raw.exposure, raw.cam2rgb)
    den = unpack(bounded)
    den_lin = pipelines.render_linear_camera(bilinear(wb.apply_gains(den, raw.wb_gains)) * raw.exposure, raw.cam2rgb)
    fc = (slice(2 * crop[0].start, 2 * crop[0].stop), slice(2 * crop[1].start, 2 * crop[1].stop))
    act_vis = to_u8(np.clip(inp[..., 1][crop] * raw.exposure * 2, 0, 1))[..., None].repeat(3, -1)
    act_vis[active[crop]] = (229, 86, 111)
    r1 = row([tile(bay, "1. Bayer mosaic (RGGB)", "sensor colour-filter samples"),
              tile(heat(sigma[..., 1][crop]), "2. Noise-level map", f"sigma(x), sigma^2 = {a:.1e} x + {b:.1e}"),
              tile(heat(removed[crop]), "3. What the network removed", "|denoised - noisy|, packed RAW")])
    r2 = row([tile(act_vis, "4. Noise-model bound active (red)", f"{active.mean() * 100:.1f}% of pixels capped at {k:g} sigma"),
              tile(noisy_lin[fc], "5. Before the network", "noisy RAW, minimal render"),
              tile(den_lin[fc], "6. After network + bound", "denoised RAW, minimal render"),
              tile(out.results["hybrid"].image[fc], "7. Hybrid output", "after WB, demosaic, colour, tone, sharpening")])
    save(stack([r1, r2]), "fig_intermediates.jpg")


# ----------------------------------------------------------------------------------------------- 4. ablation
def fig_ablation():
    b = json.loads((ROOT / "results" / "benchmarks.json").read_text(encoding="utf-8"))
    rows = b.get("ablation") or []
    if not rows:
        print("no ablation in results/benchmarks.json")
        return
    full = next(r for r in rows if r["variant"] == "Full hybrid")
    items = [(r["variant"], r["psnr"] - full["psnr"], r["delta_e"] - full["delta_e"]) for r in rows if r["variant"] != "Full hybrid"]
    items.sort(key=lambda t: t[1])
    W, rowh, left = 1100, 40, 430
    H = 90 + rowh * len(items) + 40
    img = np.full((H, W, 3), BG, np.uint8)
    cv2.putText(img, "Ablation: PSNR change when one module is removed (vs full hybrid "
                f"{full['psnr']:.2f} dB)", (16, 32), FONT, 0.62, (245, 245, 245), 1, cv2.LINE_AA)
    lo = min(-0.2, min(d for _, d, _ in items) * 1.1)
    hi = max(0.2, max(d for _, d, _ in items) * 1.1)
    x0 = left + int((0 - lo) / (hi - lo) * (W - left - 140))
    cv2.line(img, (x0, 60), (x0, H - 40), (90, 92, 98), 1)
    for i, (name, d, de) in enumerate(items):
        y = 70 + i * rowh
        cv2.putText(img, name[:52], (16, y + 22), FONT, 0.5, (210, 210, 214), 1, cv2.LINE_AA)
        x1 = left + int((d - lo) / (hi - lo) * (W - left - 140))
        col = (229, 86, 111) if d < 0 else (68, 162, 130)
        cv2.rectangle(img, (min(x0, x1), y + 8), (max(x0, x1), y + 30), col, -1)
        cv2.putText(img, f"{d:+.2f} dB  (dE {de:+.2f})", (max(x0, x1) + 8 if d >= 0 else x0 + 8, y + 25), FONT, 0.45,
                    (190, 190, 194), 1, cv2.LINE_AA)
    cv2.putText(img, "red = removing the module hurts (the module helps); green = removing it helps", (16, H - 14), FONT, 0.45,
                (150, 152, 158), 1, cv2.LINE_AA)
    OUT.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT / "fig_ablation.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print("wrote", OUT / "fig_ablation.png")


if __name__ == "__main__":
    if "--ablation" in sys.argv:
        fig_ablation()
    else:
        only = sys.argv[1:]
        if not only or "lowlight" in only:
            fig_lowlight()
        if not only or "backlit" in only:
            fig_bright_regions(fig_backlit())
        if not only or "intermediates" in only:
            fig_intermediates()
