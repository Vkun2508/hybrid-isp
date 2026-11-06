"""Image analyser + adaptive router: decides how much of the network each photo needs, and why."""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field

import cv2
import numpy as np

from isp.classical.denoise import estimate_sigma
from isp.classical.sharpen import noise_aware_amount
from isp.classical.wb import angular_error_deg, shades_of_gray
from isp.raw import RawImage, pack

TARGET_MEAN = 0.16                                                    # linear mean green of a well-exposed scene
FAST_SIGMA = float(os.environ.get("ISP_ROUTER_FAST_SIGMA", "0.004"))  # below: classical fast path
HIGH_SIGMA = float(os.environ.get("ISP_ROUTER_HIGH_SIGMA", "0.03"))   # above: network + chroma cleanup
MAX_GAIN = 16.0


@dataclass
class Analysis:
    sigma: float          # noise std (normalised RAW units, before any gain)
    ev: float             # exposure offset vs a well-exposed scene, in stops
    clipped: float        # fraction of clipped pixels
    wb_conf: float        # 0..1 trust in the as-shot white balance
    wb_angle: float | None
    dynamic_range: float  # stops (p99.5 / p0.5 of smoothed luma)
    mean_level: float

    def as_dict(self) -> dict:
        return {k: (round(v, 5) if isinstance(v, float) else v) for k, v in asdict(self).items()}


@dataclass
class Route:
    path: str                      # "fast" | "learned" | "learned_chroma"
    label: str
    gain: float
    wb_method: str
    wb_gains: tuple[float, float, float]
    local_tone: bool
    sharpen: float
    chroma: float
    sigma_eff: float
    forced: bool = False
    reasons: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["wb_gains"] = [round(g, 4) for g in self.wb_gains]
        for k in ("gain", "sharpen", "chroma", "sigma_eff"):
            d[k] = round(d[k], 5)
        return d


def analyze(raw: RawImage) -> Analysis:
    q = pack(raw.bayer)
    g = 0.5 * (q[..., 1] + q[..., 2])
    sigma = estimate_sigma(q[..., 1])
    mean_g = float(g.mean())
    ev = float(np.log2(max(mean_g, 1e-6) / TARGET_MEAN))
    clipped = float((q.max(axis=-1) >= 0.98).mean())

    sog = shades_of_gray(raw.bayer)
    if raw.wb_gains is not None:
        angle = angular_error_deg(raw.wb_gains, sog)
        wb_conf = float(np.clip(1.0 - (angle - 5.0) / 10.0, 0.0, 1.0))
    else:
        angle, wb_conf = None, 0.0

    small = cv2.resize(g, (max(1, g.shape[1] // 4), max(1, g.shape[0] // 4)), interpolation=cv2.INTER_AREA)
    lo, hi = np.percentile(small, [0.5, 99.5])
    dr = float(np.log2((hi + 1e-4) / (lo + 1e-4)))
    return Analysis(sigma=sigma, ev=ev, clipped=clipped, wb_conf=wb_conf, wb_angle=angle,
                    dynamic_range=dr, mean_level=mean_g)


def _blend_gains(a, b, w: float) -> tuple[float, float, float]:
    """Blend two WB estimates in illuminant space (w = weight of a)."""
    ia, ib = 1.0 / np.asarray(a, np.float64), 1.0 / np.asarray(b, np.float64)
    ia, ib = ia / ia[1], ib / ib[1]
    ill = w * ia + (1 - w) * ib
    return (float(ill[1] / ill[0]), 1.0, float(ill[1] / ill[2]))


def decide(raw: RawImage, a: Analysis, force_path: str | None = None) -> Route:
    reasons: list[str] = []

    # Exposure
    if a.ev < -0.6:
        gain = float(min(2.0 ** (-a.ev), MAX_GAIN))
        reasons.append(f"Under-exposed by {-a.ev:.1f} EV -> digital gain x{gain:.1f} before the network (noise map scaled to match).")
    else:
        gain = 1.0
        reasons.append(f"Exposure OK ({a.ev:+.1f} EV) -> no gain.")

    # White balance
    sog = shades_of_gray(raw.bayer)
    if raw.wb_gains is None:
        wb_method, wb_gains = "shades-of-gray", sog
        reasons.append("No white-balance metadata -> shades-of-gray estimate (p=6).")
    elif a.wb_conf >= 0.999:
        wb_method, wb_gains = "as-shot", raw.wb_gains
        reasons.append(f"As-shot white balance agrees with the scene ({a.wb_angle:.1f} deg) -> keep it.")
    elif a.wb_conf <= 0.001:
        wb_method, wb_gains = "shades-of-gray", sog
        reasons.append(f"As-shot white balance disagrees with the scene by {a.wb_angle:.1f} deg -> replaced by shades-of-gray.")
    else:
        wb_method, wb_gains = "blend", _blend_gains(raw.wb_gains, sog, a.wb_conf)
        reasons.append(f"As-shot white balance is uncertain ({a.wb_angle:.1f} deg off) -> blended {a.wb_conf:.0%} as-shot / shades-of-gray.")

    # Noise -> path
    sigma_eff = a.sigma * gain
    if force_path:
        path, forced = force_path, True
        reasons.append(f"Router override: path forced to '{force_path}'.")
    elif sigma_eff < FAST_SIGMA:
        path, forced = "fast", False
        reasons.append(f"Noise sigma={sigma_eff:.4f} < {FAST_SIGMA} -> fast classical path (network skipped).")
    elif sigma_eff < HIGH_SIGMA:
        path, forced = "learned", False
        reasons.append(f"Noise sigma={sigma_eff:.4f} -> learned core.")
    else:
        path, forced = "learned_chroma", False
        reasons.append(f"High noise sigma={sigma_eff:.4f} >= {HIGH_SIGMA} -> learned core + chroma-only cleanup, weaker sharpening.")

    chroma = float(np.clip((sigma_eff - HIGH_SIGMA) / HIGH_SIGMA + 0.5, 0.0, 1.0)) if path == "learned_chroma" else 0.0
    sharpen = noise_aware_amount(sigma_eff)
    reasons.append(f"Sharpening = 0.8 / (1 + sigma/0.01) = {sharpen:.2f}.")

    local = a.dynamic_range > 12.0 and gain < 4
    if local:
        reasons.append(f"High dynamic range ({a.dynamic_range:.1f} stops) -> local tone mapping.")

    labels = {"fast": "Fast classical path", "learned": "Learned core", "learned_chroma": "Learned core + chroma cleanup"}
    return Route(path=path, label=labels.get(path, path), gain=gain, wb_method=wb_method, wb_gains=wb_gains,
                 local_tone=local, sharpen=sharpen, chroma=chroma, sigma_eff=sigma_eff, forced=forced, reasons=reasons)
