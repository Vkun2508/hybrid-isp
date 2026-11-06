"""The three pipelines compared in the app, all fed the same RawImage.

traditional  all classical, metadata-driven (as-shot WB, Malvar, noise-aware bilateral, fixed tone + sharpening)
learned      the network alone + only the minimal rendering its output format needs to be viewable
             (for the team's RAW denoiser: network -> bilinear demosaic + as-shot WB + CCM + tone)
hybrid       classical front end -> router -> learned core (model slot) -> classical back end
             (for a RAW denoiser: the network denoises the native RAW *before* WB/gain, which is its
             training distribution; the classical stages then do colour, exposure, tone, sharpening)
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field, replace

import cv2
import numpy as np

from isp import metrics
from isp.analysis import Analysis, Route, analyze, decide
from isp.classical import demosaic, denoise, sharpen, tone, wb
from isp.classical.color import apply_ccm, linear_to_srgb
from isp.model_slot import LearnedModel, get_model
from isp.raw import RawImage, pack, unpack

PIPELINES = ("traditional", "learned", "hybrid")
THUMB_W = 720


@dataclass
class HybridOptions:
    """Ablation switches for the hybrid pipeline (all True = the full system)."""
    router: bool = True          # False: always the learned path, fixed sharpening, no chroma/local-tone decisions
    wb_correction: bool = True   # False: trust the as-shot white balance blindly
    exposure_norm: bool = True   # False: no digital gain for under-exposed input
    chroma_cleanup: bool = True
    sharpen: bool = True
    demosaic: str = "malvar"     # "malvar" | "bilinear"
    use_model: bool = True       # False: classical denoise instead of the learned core

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class Stage:
    name: str
    ms: float
    image: np.ndarray  # display-ready sRGB float (H, W, 3), already small
    note: str = ""


@dataclass
class PipelineResult:
    name: str
    image: np.ndarray
    latency_ms: float
    stages: list[Stage] = field(default_factory=list)
    metrics: dict | None = None
    info: dict = field(default_factory=dict)


class _Timer:
    def __init__(self):
        self.t = time.perf_counter()

    def lap(self) -> float:
        now = time.perf_counter()
        ms, self.t = (now - self.t) * 1000.0, now
        return round(ms, 2)


# --------------------------------------------------------------------------- display helpers
def _shrink(img: np.ndarray, width: int = THUMB_W) -> np.ndarray:
    h, w = img.shape[:2]
    if w <= width:
        return img
    return cv2.resize(img, (width, max(1, int(h * width / w))), interpolation=cv2.INTER_AREA)


def _auto_gain(x: np.ndarray) -> float:
    p = float(np.percentile(x[::4, ::4], 99))
    return float(np.clip(0.9 / max(p, 1e-4), 1.0, 32.0))


def viz_mosaic(bayer: np.ndarray, crop: int = 128) -> np.ndarray:
    """Centre crop of the Bayer mosaic, each pixel tinted by its filter colour, zoomed 4x (nearest)."""
    h, w = bayer.shape
    c = min(crop, h, w) // 2 * 2
    y, x = (h - c) // 2 // 2 * 2, (w - c) // 2 // 2 * 2
    b = np.clip(bayer[y:y + c, x:x + c] * _auto_gain(bayer), 0, 1)
    rgb = np.zeros((c, c, 3), np.float32)
    rgb[0::2, 0::2, 0] = b[0::2, 0::2]
    rgb[0::2, 1::2, 1] = b[0::2, 1::2]
    rgb[1::2, 0::2, 1] = b[1::2, 0::2]
    rgb[1::2, 1::2, 2] = b[1::2, 1::2]
    rgb = linear_to_srgb(rgb)
    return cv2.resize(rgb, (c * 4, c * 4), interpolation=cv2.INTER_NEAREST)


def viz_linear(rgb: np.ndarray, gain: float | None = None) -> np.ndarray:
    small = _shrink(rgb)
    g = _auto_gain(small) if gain is None else gain
    return linear_to_srgb(np.clip(small * g, 0, 1))


def viz_srgb(srgb: np.ndarray) -> np.ndarray:
    return np.clip(_shrink(srgb), 0, 1)


def viz_noise(nmap: np.ndarray) -> np.ndarray:
    m = _shrink(nmap)
    m = m / max(float(m.max()), 1e-6)
    return np.clip(cv2.applyColorMap((m * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)[..., ::-1] / 255.0, 0, 1).astype(np.float32)


def render_linear_camera(rgb_cam: np.ndarray, cam2rgb: np.ndarray) -> np.ndarray:
    return tone.global_tone(np.clip(apply_ccm(rgb_cam, cam2rgb), 0, 1))


def to_display(y: np.ndarray, output: str, cam2rgb: np.ndarray) -> np.ndarray:
    """Convert an RGB model output to display sRGB according to its declared format."""
    if output == "srgb":
        return np.clip(y, 0, 1)
    if output == "linear_camera":
        y = apply_ccm(y, cam2rgb)
    return tone.global_tone(np.clip(y, 0, 1))


def naive_render(raw: RawImage) -> np.ndarray:
    """The sensor data with no processing beyond bilinear demosaic + as-shot colour (the 'input' view)."""
    x = wb.apply_gains(raw.bayer, raw.wb_gains or wb.gray_world(raw.bayer))
    return render_linear_camera(demosaic.bilinear(x) * raw.exposure, raw.cam2rgb)


def reference_render(bayer: np.ndarray, raw: RawImage) -> np.ndarray:
    """How ground-truth RAW is turned into the sRGB reference for metrics: as-shot WB, Malvar,
    baseline exposure, CCM, tone. No denoise, no sharpening (those are what is being evaluated)."""
    x = wb.apply_gains(bayer, raw.wb_gains or wb.gray_world(bayer))
    return render_linear_camera(demosaic.malvar(x) * raw.exposure, raw.cam2rgb)


def noise_std_map(raw: RawImage, x: np.ndarray, sigma: float, gain: float) -> np.ndarray:
    """Per-pixel noise std after the digital gain: sqrt(a*x + b) * gain."""
    if raw.noise_profile is not None:
        a, b = raw.noise_profile
        b = max(b, sigma ** 2 - a * float(np.mean(x)))
    else:
        a, b = 0.0, sigma ** 2
    return (np.sqrt(np.maximum(a * np.maximum(x, 0) + b, 0.0)) * gain).astype(np.float32)


def _demosaic(x: np.ndarray, method: str) -> np.ndarray:
    return demosaic.bilinear(x) if method == "bilinear" else demosaic.malvar(x)


# --------------------------------------------------------------------------- pipelines
def traditional(raw: RawImage, a: Analysis, route: Route, model: LearnedModel, opts: HybridOptions) -> PipelineResult:
    st: list[Stage] = []
    t = _Timer()
    st.append(Stage("RAW mosaic", 0.0, viz_mosaic(raw.bayer), "Black level removed, RGGB, normalised"))
    gains = raw.wb_gains or wb.gray_world(raw.bayer)
    x = wb.apply_gains(raw.bayer, gains)
    st.append(Stage("White balance", t.lap(), viz_mosaic(x), "As-shot gains" if raw.wb_gains else "Gray-world gains"))
    rgb = demosaic.malvar(x)
    st.append(Stage("Demosaic", t.lap(), viz_linear(rgb), "Malvar-He-Cutler 5x5"))
    rgb = denoise.bilateral(rgb, a.sigma)
    st.append(Stage("Denoise", t.lap(), viz_linear(rgb), f"Bilateral, range sigma = 3 x {a.sigma:.4f}"))
    rgb = np.clip(apply_ccm(rgb * raw.exposure, raw.cam2rgb), 0, 1)
    st.append(Stage("Colour (CCM)", t.lap(), viz_linear(rgb), "Camera RGB -> linear sRGB"))
    out = tone.global_tone(rgb)
    st.append(Stage("Tone + gamma", t.lap(), viz_srgb(out), "sRGB gamma + S-curve"))
    out = sharpen.unsharp_luma(out, 0.5)
    st.append(Stage("Sharpen", t.lap(), viz_srgb(out), "Unsharp mask on luma, fixed 0.5"))
    return PipelineResult("traditional", out, round(sum(s.ms for s in st), 2), st)


def _rgb_model_input(raw: RawImage, x: np.ndarray, model: LearnedModel, sigma: float, gain: float):
    """Input tensor for RGB-input models (+ optional noise map)."""
    rgb = np.clip(apply_ccm(demosaic.malvar(x), raw.cam2rgb), 0, 1)
    if model.m.input_gamma:
        rgb = tone.global_tone(rgb)
    nmap = noise_std_map(raw, rgb.mean(axis=-1), sigma, gain) if model.m.needs_noise_map else None
    return rgb, nmap


def learned(raw: RawImage, a: Analysis, route: Route, model: LearnedModel, opts: HybridOptions) -> PipelineResult:
    st: list[Stage] = []
    t = _Timer()
    m = model.m
    x = raw.bayer if m.output == "raw4" else raw.bayer * raw.exposure
    wb_done = False
    if m.input_wb:
        x = wb.apply_gains(x, raw.wb_gains or wb.gray_world(raw.bayer))
        wb_done = True

    if m.output == "raw4":
        inp = pack(x)
        nmap = noise_std_map(raw, inp.mean(axis=-1), a.sigma, 1.0) if m.needs_noise_map else None
        st.append(Stage("Network input", t.lap(), viz_mosaic(x), f"packed RAW {inp.shape[1]}x{inp.shape[0]}x4"))
        y = unpack(model.run(inp, nmap))
        st.append(Stage("Network (RAW denoise)", t.lap(), viz_mosaic(y), model.info.name))
        if not wb_done:
            y = wb.apply_gains(y, raw.wb_gains or wb.gray_world(raw.bayer))
        out = render_linear_camera(demosaic.bilinear(y) * raw.exposure, raw.cam2rgb)
        st.append(Stage("Minimal render", t.lap(), viz_srgb(out), "Bilinear demosaic + as-shot WB + CCM + gamma"))
    else:
        if m.input == "raw4":
            inp = pack(x)
            nmap = noise_std_map(raw, inp.mean(axis=-1), a.sigma, 1.0) if m.needs_noise_map else None
        else:
            inp, nmap = _rgb_model_input(raw, x, model, a.sigma, 1.0)
        st.append(Stage("Network input", t.lap(), viz_mosaic(x), f"{m.input} tensor {tuple(inp.shape)}"))
        y = model.run(inp, nmap)
        out = to_display(y, m.output, raw.cam2rgb)
        st.append(Stage("Network", t.lap(), viz_srgb(out), model.info.name))
    return PipelineResult("learned", out, round(sum(s.ms for s in st), 2), st,
                          info={"model": model.info.name, "is_stub": model.is_stub})


def hybrid(raw: RawImage, a: Analysis, route: Route, model: LearnedModel, opts: HybridOptions) -> PipelineResult:
    if model.m.output == "raw4":
        return _hybrid_raw_denoiser(raw, a, route, model, opts)
    return _hybrid_rgb_model(raw, a, route, model, opts)


def _hybrid_raw_denoiser(raw, a, route, model, opts) -> PipelineResult:
    st: list[Stage] = []
    t = _Timer()
    m = model.m
    st.append(Stage("RAW mosaic", 0.0, viz_mosaic(raw.bayer), "Black level removed, RGGB, normalised"))

    # 1. Denoise the native RAW (learned core) - before WB/gain, i.e. the network's training distribution.
    if route.path == "fast":
        bayer = raw.bayer
        resid_sigma = a.sigma
        st.append(Stage("Learned core skipped", t.lap(), viz_mosaic(bayer), "Low noise: router chose the fast path"))
    else:
        x = wb.apply_gains(raw.bayer, raw.wb_gains or route.wb_gains) if m.input_wb else raw.bayer
        inp = pack(x)
        nmap = noise_std_map(raw, inp.mean(axis=-1), a.sigma, 1.0) if m.needs_noise_map else None
        if nmap is not None:
            st.append(Stage("Noise map", t.lap(), viz_noise(nmap), "Per-pixel sigma = sqrt(a*x+b)"))
        y = model.run(inp, nmap)
        if m.input_wb:  # undo the as-shot gains so the router's WB below is the only one applied
            g = raw.wb_gains or route.wb_gains
            y = pack(wb.apply_gains(unpack(y), (1.0 / g[0], 1.0, 1.0 / g[2])))
        bayer = unpack(y)
        resid_sigma = denoise.estimate_sigma(pack(bayer)[..., 1])  # noise left after the network
        st.append(Stage("Learned core (RAW denoise)", t.lap(), viz_mosaic(bayer),
                        model.info.name + (" (placeholder)" if model.is_stub else "")))

    # 2. Classical: router-chosen WB + exposure normalisation (after denoising, so noise isn't amplified first).
    x = wb.apply_gains(bayer, route.wb_gains) * (route.gain * raw.exposure)
    st.append(Stage("WB + exposure", t.lap(), viz_mosaic(np.clip(x, 0, 1)), f"{route.wb_method} WB, gain x{route.gain:.1f}"))

    # 3. Demosaic (+ light classical denoise on the fast path).
    rgb = _demosaic(np.minimum(x, 1.0), opts.demosaic)
    if route.path == "fast":
        rgb = denoise.guided_filter(rgb, radius=2, eps=(1.5 * route.sigma_eff) ** 2)
        st.append(Stage("Demosaic + fast denoise", t.lap(), viz_linear(rgb), f"{opts.demosaic} + guided filter"))
    else:
        st.append(Stage("Demosaic", t.lap(), viz_linear(rgb), "Malvar-He-Cutler 5x5" if opts.demosaic == "malvar" else "Bilinear"))

    # 4. Colour + tone.
    out = render_linear_camera(rgb, raw.cam2rgb)
    st.append(Stage("Colour + tone", t.lap(), viz_srgb(out), "CCM + sRGB gamma + S-curve"))
    if route.local_tone:
        out = tone.local_tone(out)
        st.append(Stage("Local tone", t.lap(), viz_srgb(out), "CLAHE on L*, 50% blend"))
    if route.chroma > 0:
        out = denoise.chroma_denoise(out, route.chroma)
        st.append(Stage("Chroma cleanup", t.lap(), viz_srgb(out), f"Bilateral on Cr/Cb, strength {route.chroma:.2f}"))

    # 5. Noise-aware sharpening on what's left after denoising (scaled by the gain).
    from isp import analysis as _an

    amount = route.sharpen if not opts.router else sharpen.noise_aware_amount(
        resid_sigma * route.gain * raw.exposure, _an.CONFIG.sharpen_k, _an.CONFIG.sharpen_sigma0)
    if opts.sharpen and amount > 0:
        out = sharpen.unsharp_luma(out, amount)
    st.append(Stage("Sharpen", t.lap(), viz_srgb(out), f"Noise-aware amount {amount if opts.sharpen else 0:.2f}"))
    return PipelineResult("hybrid", out, round(sum(s.ms for s in st), 2), st,
                          info={"path": route.path, "model": model.info.name, "is_stub": model.is_stub,
                                "residual_sigma": round(resid_sigma, 5), "options": opts.as_dict()})


def _hybrid_rgb_model(raw, a, route, model, opts) -> PipelineResult:
    """Hybrid for models that output RGB (kept so any RAW->RGB / RGB->RGB model also plugs in)."""
    st: list[Stage] = []
    t = _Timer()
    m = model.m
    st.append(Stage("RAW mosaic", 0.0, viz_mosaic(raw.bayer), "Black level removed, RGGB, normalised"))
    pre_wb = m.input_wb or route.path == "fast"
    x = wb.apply_gains(raw.bayer, route.wb_gains) if pre_wb else raw.bayer.copy()
    x = np.minimum(x * (route.gain * raw.exposure), 1.0)
    st.append(Stage("WB + exposure", t.lap(), viz_mosaic(np.clip(x, 0, 1)), f"{route.wb_method} WB, gain x{route.gain:.1f}"))
    if route.path == "fast":
        rgb = denoise.guided_filter(_demosaic(x, opts.demosaic), radius=2, eps=(1.5 * route.sigma_eff) ** 2)
        st.append(Stage("Fast core", t.lap(), viz_linear(rgb), "Demosaic + guided filter (network skipped)"))
        out = render_linear_camera(rgb, raw.cam2rgb)
        st.append(Stage("Colour + tone", t.lap(), viz_srgb(out), "CCM + sRGB gamma + S-curve"))
    else:
        if m.input == "raw4":
            inp = pack(x)
            nmap = noise_std_map(raw, inp.mean(axis=-1), a.sigma, route.gain) if m.needs_noise_map else None
        else:
            inp, nmap = _rgb_model_input(raw, x, model, a.sigma, route.gain)
        y = model.run(inp, nmap)
        if not pre_wb and m.output == "linear_camera":
            y = y * np.asarray(route.wb_gains, np.float32)
        out = to_display(y, m.output, raw.cam2rgb)
        st.append(Stage("Learned core", t.lap(), viz_srgb(out), model.info.name))
    if route.local_tone:
        out = tone.local_tone(out)
        st.append(Stage("Local tone", t.lap(), viz_srgb(out), "CLAHE on L*, 50% blend"))
    if route.chroma > 0:
        out = denoise.chroma_denoise(out, route.chroma)
        st.append(Stage("Chroma cleanup", t.lap(), viz_srgb(out), f"strength {route.chroma:.2f}"))
    if opts.sharpen:
        out = sharpen.unsharp_luma(out, route.sharpen)
    st.append(Stage("Sharpen", t.lap(), viz_srgb(out), f"amount {route.sharpen if opts.sharpen else 0:.2f}"))
    return PipelineResult("hybrid", out, round(sum(s.ms for s in st), 2), st,
                          info={"path": route.path, "model": model.info.name, "is_stub": model.is_stub,
                                "options": opts.as_dict()})


_RUNNERS = {"traditional": traditional, "learned": learned, "hybrid": hybrid}


def apply_options(route: Route, raw: RawImage, opts: HybridOptions) -> Route:
    r = replace(route, reasons=list(route.reasons))
    if not opts.exposure_norm and r.gain != 1.0:
        r.gain = 1.0
        r.reasons.append("Ablation: exposure normalisation off.")
    if not opts.wb_correction and raw.wb_gains is not None and r.wb_method != "as-shot":
        r.wb_method, r.wb_gains = "as-shot", raw.wb_gains
        r.reasons.append("Ablation: white-balance correction off (as-shot gains).")
    if not opts.router:
        r.path, r.chroma, r.local_tone, r.sharpen, r.forced = "learned", 0.0, False, 0.5, True
        r.reasons.append("Ablation: router off (always learned core, fixed sharpening 0.5).")
    if not opts.chroma_cleanup:
        r.chroma = 0.0
    if not opts.use_model and r.path != "fast":
        r.path = "fast"
        r.reasons.append("Ablation: learned core off (classical denoise instead).")
    r.sigma_eff = r.sigma_eff if r.gain == route.gain else r.sigma_eff / max(route.gain, 1e-6) * r.gain
    return r


@dataclass
class RunOutput:
    analysis: Analysis
    route: Route
    results: dict[str, PipelineResult]
    analysis_ms: float


def run(raw: RawImage, names: tuple[str, ...] | list[str] = PIPELINES, force_path: str | None = None,
        model: LearnedModel | None = None, opts: HybridOptions | None = None,
        with_metrics: bool = True) -> RunOutput:
    model = model or get_model()
    opts = opts or HybridOptions()
    t0 = time.perf_counter()
    a = analyze(raw)
    route = apply_options(decide(raw, a, force_path=force_path), raw, opts)
    analysis_ms = round((time.perf_counter() - t0) * 1000, 2)
    results: dict[str, PipelineResult] = {}
    for n in names:
        res = _RUNNERS[n](raw, a, route, model, opts)
        if with_metrics and raw.gt is not None and raw.gt.shape[:2] == res.image.shape[:2]:
            res.metrics = metrics.compute(res.image, raw.gt)
        results[n] = res
    return RunOutput(a, route, results, analysis_ms)
