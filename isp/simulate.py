"""Sensor simulator: lets judges create hard inputs live (noise / ISO, under-exposure, WB error)."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from isp.raw import RawImage

# Brooks et al. 2019 noise model: variance = shot * x + read, read tied to shot on a log scale.
_LOG_SHOT_MIN, _LOG_SHOT_MAX = np.log(1e-4), np.log(0.02)


@dataclass
class SimParams:
    noise: float = 0.0        # 0..1  (0 = sensor as captured, 1 = very high ISO)
    exposure_ev: float = 0.0  # -5..+2 stops applied to the linear RAW
    wb_shift: float = 0.0     # -1..1  error injected into the as-shot white balance (blue <-> amber)
    seed: int = 0

    def clamp(self) -> "SimParams":
        return SimParams(
            noise=float(np.clip(self.noise, 0, 1)),
            exposure_ev=float(np.clip(self.exposure_ev, -5, 2)),
            wb_shift=float(np.clip(self.wb_shift, -1, 1)),
            seed=int(self.seed),
        )

    def is_identity(self) -> bool:
        return self.noise == 0 and self.exposure_ev == 0 and self.wb_shift == 0

    def as_dict(self) -> dict:
        return asdict(self)


def noise_params(t: float) -> tuple[float, float]:
    """Map the 0..1 slider to (shot, read) variance coefficients."""
    if t <= 0:
        return 0.0, 0.0
    log_shot = _LOG_SHOT_MIN + t * (_LOG_SHOT_MAX - _LOG_SHOT_MIN)
    log_read = 2.18 * log_shot + 1.20
    return float(np.exp(log_shot)), float(np.exp(log_read))


def iso_label(t: float) -> int:
    """Rough ISO equivalent for the UI label."""
    return int(round(100 * 2 ** (t * 7))) if t > 0 else 100


def simulate(raw: RawImage, p: SimParams) -> RawImage:
    p = p.clamp()
    if p.is_identity():
        return raw
    rng = np.random.default_rng(p.seed)
    x = raw.bayer.astype(np.float32)
    if p.exposure_ev:
        x = np.minimum(x * (2.0 ** p.exposure_ev), 1.0)
    shot, read = noise_params(p.noise)
    if shot > 0:
        std = np.sqrt(np.maximum(shot * np.maximum(x, 0) + read, 0)).astype(np.float32)
        # real black-level-subtracted RAW keeps negative noise values; only saturation clips
        x = np.clip(x + rng.standard_normal(x.shape, dtype=np.float32) * std, -0.5, 1.0)
    out = raw.copy(bayer=x.astype(np.float32))
    if shot > 0:
        out.noise_profile = (shot, read)
    if p.wb_shift:
        s = p.wb_shift
        if raw.wb_gains is not None:
            r, g, b = raw.wb_gains
            out.wb_gains = (r * (1 + 0.45 * s), g, b * (1 - 0.45 * s))
        else:  # no metadata to corrupt: put a colour cast into the RAW itself
            x = out.bayer
            x[0::2, 0::2] *= 1 + 0.45 * s
            x[1::2, 1::2] *= 1 - 0.45 * s
            out.bayer = np.minimum(x, 1.0)
    out.meta["simulated"] = p.as_dict()
    return out
