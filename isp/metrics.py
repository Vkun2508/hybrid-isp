"""Full-reference quality metrics shown live in the app (when ground truth exists).

Live numbers are for the demo: PSNR + SSIM on the centre crop (<= 768x768), CIEDE2000 on that
crop at half scale. SSIM follows skimage's defaults exactly (7x7 uniform window, K1=0.01,
K2=0.03, sample covariance), re-implemented with OpenCV box filters for speed.
Official numbers (full images, LPIPS) come from the offline eval scripts -> Benchmarks page.
"""
from __future__ import annotations

import cv2
import numpy as np
from skimage.color import deltaE_ciede2000

CROP = 768
_WIN = 7


def _center_crop(a: np.ndarray, b: np.ndarray, side: int = CROP) -> tuple[np.ndarray, np.ndarray]:
    h, w = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1])
    a, b = a[:h, :w], b[:h, :w]
    ch, cw = min(h, side), min(w, side)
    y, x = (h - ch) // 2, (w - cw) // 2
    return a[y:y + ch, x:x + cw], b[y:y + ch, x:x + cw]


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 99.0 if mse <= 1e-12 else float(10 * np.log10(1.0 / mse))


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Mean SSIM over channels; matches skimage.metrics.structural_similarity(data_range=1, channel_axis=-1)."""
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    n = _WIN * _WIN
    cov_norm = n / (n - 1)
    pad = (_WIN - 1) // 2
    vals = []
    for c in range(a.shape[-1]):
        x = a[..., c].astype(np.float32)
        y = b[..., c].astype(np.float32)
        f = lambda z: cv2.boxFilter(z, -1, (_WIN, _WIN), normalize=True, borderType=cv2.BORDER_REFLECT)  # noqa: E731
        ux, uy = f(x), f(y)
        vx = cov_norm * (f(x * x) - ux * ux)
        vy = cov_norm * (f(y * y) - uy * uy)
        vxy = cov_norm * (f(x * y) - ux * uy)
        s = ((2 * ux * uy + c1) * (2 * vxy + c2)) / ((ux * ux + uy * uy + c1) * (vx + vy + c2))
        vals.append(float(s[pad:-pad, pad:-pad].mean()))
    return float(np.mean(vals))


def delta_e(a: np.ndarray, b: np.ndarray) -> float:
    a2 = cv2.resize(a.astype(np.float32), None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    b2 = cv2.resize(b.astype(np.float32), None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    la = cv2.cvtColor(a2, cv2.COLOR_RGB2Lab)  # float input: sRGB gamma decoded, D65
    lb = cv2.cvtColor(b2, cv2.COLOR_RGB2Lab)
    return float(np.mean(deltaE_ciede2000(lb, la)))


def compute(out: np.ndarray, gt: np.ndarray) -> dict:
    a, b = _center_crop(np.clip(out, 0, 1).astype(np.float32), np.clip(gt, 0, 1).astype(np.float32))
    return {"psnr": round(psnr(a, b), 3), "ssim": round(ssim(a, b), 4), "delta_e": round(delta_e(a, b), 3)}
