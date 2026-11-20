"""Export the real photosite data behind the Overview's 3D "sensor story".

    .venv/Scripts/python.exe scripts/make_sensor_story.py

Takes one unseen RAW (Nexus 6, ISO 3200) at -3 EV and follows the hybrid pipeline's real order
(isp/pipelines.py, _hybrid_raw_denoiser): the network denoises the native RAW first, then exact maths applies
white balance and the exposure gain, then demosaic/colour/tone give the photo. For a grid of single photosites
(every 8th pixel, keeping the RGGB pattern) it stores:

    hA  noisy RAW value            (what the sensor recorded)
    hB  the network's denoised RAW  (same exposure)
    hC  hB after WB + gain          (exact maths)
    hD  hA after the same WB + gain (what brightening the noisy RAW does: the classical order)
    rgb the hybrid's final colour   (mean of the output over the photosite's 8x8 cell)
    rgb2 the classical pipeline's colour, brightened by the same gain (a fair, exposure-matched baseline)

Heights use one display transform for all three, h = clamp(v)^0.45, so the stages compare honestly.
The exposure-matched classical image is the unchanged `traditional` pipeline run with the hybrid's gain folded into
the scene exposure (the textbook pipeline has no auto-exposure of its own, so at -3 EV it is otherwise near-black).
Output: web/public/story/field.bin (16-byte header + 10 bytes per photosite), web/public/story/meta.json and
web/public/hero/d/classical_gain_3.webp.
"""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from isp import analysis as _an  # noqa: E402
from isp.analysis import analyze, decide  # noqa: E402
from isp.classical import denoise, wb  # noqa: E402
from isp.model_slot import get_model  # noqa: E402
from isp import metrics  # noqa: E402
from isp.pipelines import HybridOptions, apply_options, noise_std_map, run, traditional  # noqa: E402
from isp.raw import pack, unpack  # noqa: E402
from isp.samples import load_sample  # noqa: E402
from isp.simulate import SimParams, simulate  # noqa: E402

SAMPLE = "sidd-n6-iso3200-lowlight"
EV = -3.0
STEP = 8  # one photosite every 8 pixels (pairs of 2x2 blocks every 16 px keep the RGGB order)
GAMMA = 0.45


def disp(v: np.ndarray) -> np.ndarray:
    return np.round(255 * np.clip(v, 0, 1) ** GAMMA).astype(np.uint8)


def main() -> int:
    raw = simulate(load_sample(SAMPLE), SimParams(exposure_ev=EV))
    a = analyze(raw)
    opts = HybridOptions()
    route = apply_options(decide(raw, a), raw, opts)
    model = get_model()
    m = model.m
    if m.output != "raw4" or route.path == "fast":
        print("expected the RAW-denoiser hybrid path", m.output, route.path)
        return 1

    # the network step, exactly as in pipelines._hybrid_raw_denoiser
    x = wb.apply_gains(raw.bayer, raw.wb_gains or route.wb_gains) if m.input_wb else raw.bayer
    inp = pack(x)
    nmap = noise_std_map(raw, inp.mean(axis=-1), a.sigma, 1.0) if m.needs_noise_map else None
    y = model.run(inp, nmap)
    k = _an.CONFIG.residual_k if opts.bound_residual else 0.0
    if k > 0:
        y = denoise.bound_residual(inp, y, k, denoise.estimate_nlf(inp))
    if m.input_wb:
        g = raw.wb_gains or route.wb_gains
        y = pack(wb.apply_gains(unpack(y), (1.0 / g[0], 1.0, 1.0 / g[2])))
    den = unpack(y)
    gained = wb.apply_gains(den, route.wb_gains) * (route.gain * raw.exposure)
    noisy_gained = wb.apply_gains(raw.bayer, route.wb_gains) * (route.gain * raw.exposure)
    # exposure-matched classical baseline: the same pipeline, with the hybrid's gain added to the scene exposure
    raw_eq = raw.copy(exposure=raw.exposure * route.gain)
    classical = traditional(raw_eq, a, route, model, opts).image
    cl_metrics = metrics.compute(classical, raw.gt)

    final = run(raw, ["hybrid"], model=model, with_metrics=False).results["hybrid"].image  # float sRGB, HxWx3
    H, W = raw.bayer.shape
    # photosite grid: keep 2x2 RGGB quads, one quad every 2*STEP pixels -> rows/cols at 0,1, 16,17, 32,33 ...
    ys = np.sort(np.concatenate([np.arange(0, H, 2 * STEP), np.arange(1, H, 2 * STEP)]))
    xs = np.sort(np.concatenate([np.arange(0, W, 2 * STEP), np.arange(1, W, 2 * STEP)]))
    gy, gx = np.meshgrid(ys, xs, indexing="ij")
    hA = disp(raw.bayer[gy, gx])
    hB = disp(den[gy, gx])
    hC = disp(gained[gy, gx])
    hD = disp(noisy_gained[gy, gx])
    # final colour: mean of the photo over each photosite's cell (STEP x STEP around its quad position)
    small = final[: (H // STEP) * STEP, : (W // STEP) * STEP].reshape(H // STEP, STEP, W // STEP, STEP, 3).mean(axis=(1, 3))
    rgb = np.round(255 * np.clip(small, 0, 1)).astype(np.uint8)  # (H/STEP, W/STEP, 3): one cell per photosite
    small2 = classical[: (H // STEP) * STEP, : (W // STEP) * STEP].reshape(H // STEP, STEP, W // STEP, STEP, 3).mean(axis=(1, 3))
    rgb2 = np.round(255 * np.clip(small2, 0, 1)).astype(np.uint8)
    gh, gw = hA.shape
    assert rgb.shape[:2] == (gh, gw), (rgb.shape, hA.shape)

    out = ROOT / "web" / "public" / "story"
    out.mkdir(parents=True, exist_ok=True)
    body = np.concatenate([hA[..., None], hB[..., None], hC[..., None], hD[..., None], rgb, rgb2], axis=-1).astype(np.uint8)  # (gh, gw, 10)
    header = struct.pack("<4sHHHHI", b"HISP", 2, gw, gh, STEP, 0)
    from PIL import Image
    Image.fromarray(np.round(255 * np.clip(classical, 0, 1)).astype(np.uint8)).save(
        ROOT / "web" / "public" / "hero" / "d" / "classical_gain_3.webp", "WEBP", quality=62, method=6)
    (out / "field.bin").write_bytes(header + body.tobytes())

    def jitter(h):  # mean absolute difference to the right-hand neighbour of the same colour: noise proxy
        h = h.astype(np.float32)
        return float(np.mean(np.abs(h[:, 2:] - h[:, :-2])))

    stats = {
        "noisy_jitter": round(jitter(hA), 2), "denoised_jitter": round(jitter(hB), 2), "gained_jitter": round(jitter(hC), 2),
        "mean_hA": round(float(hA.mean()), 1), "mean_hB": round(float(hB.mean()), 1), "mean_hC": round(float(hC.mean()), 1),
        "gained_noisy_jitter": round(jitter(hD), 2), "mean_hD": round(float(hD.mean()), 1),
    }
    hero = json.loads((ROOT / "web" / "public" / "hero" / "hero.json").read_text(encoding="utf-8"))
    li = hero["levels"].index(EV)
    meta = {
        "sample": SAMPLE, "camera": hero.get("camera"), "iso": hero.get("iso"), "ev": EV,
        "grid": [gw, gh], "step": STEP, "photosites": int(gw * gh), "gamma": GAMMA,
        "gain": round(float(route.gain * raw.exposure), 2), "wb": [round(float(v), 3) for v in route.wb_gains],
        "psnr": {p: hero["metrics"][p][li]["psnr"] for p in ("traditional", "learned", "hybrid")},
        "classical_gain": {k: round(float(v), 3) for k, v in cl_metrics.items() if k in ("psnr", "ssim", "delta_e")},
        "level": li, "stats": stats,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(json.dumps(meta, indent=1))
    print(f"field.bin {(out / 'field.bin').stat().st_size / 1e3:.0f} kB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
