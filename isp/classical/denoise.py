"""Edge-preserving denoisers: noise estimate, guided filter, bilateral, chroma-only cleanup."""
from __future__ import annotations

import cv2
import numpy as np

_IMMERKAER = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)


def estimate_sigma(img: np.ndarray) -> float:
    """Noise std from the Immerkaer (1996) Laplacian-difference kernel, using the median absolute
    response (robust to edges/texture) instead of the mean. White noise of std s gives response std 6s."""
    h, w = img.shape
    if h < 8 or w < 8:
        return 0.0
    conv = cv2.filter2D(img.astype(np.float32), -1, _IMMERKAER, borderType=cv2.BORDER_REFLECT_101)
    conv = conv[1:-1, 1:-1]
    return float(np.median(np.abs(conv)) / (0.6745 * 6.0))


def estimate_nlf(packed: np.ndarray, bins: int = 16) -> tuple[float, float]:
    """Noise level function var(x) = a*x + b of a packed RAW image (Poisson-Gaussian sensor model),
    fitted from the image itself: per intensity bin, the low percentile of the local high-pass variance
    (flat regions only, so texture does not count as noise), then a non-negative least-squares line."""
    g = packed[..., 1].astype(np.float32)
    hp = cv2.filter2D(g, -1, _IMMERKAER, borderType=cv2.BORDER_REFLECT_101) / 6.0  # white noise -> std sigma
    mu = cv2.blur(g, (7, 7))
    var = cv2.blur(hp * hp, (7, 7))
    sub = (slice(4, -4, 2), slice(4, -4, 2))  # drop borders, decimate for speed
    mu, var = mu[sub].ravel(), var[sub].ravel()
    edges = np.quantile(mu, np.linspace(0.01, 0.99, bins + 1))
    xs, ys = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (mu >= lo) & (mu < hi)
        if m.sum() >= 200 and hi > lo:
            xs.append(float(np.median(mu[m])))
            ys.append(float(np.percentile(var[m], 20)) / 0.8)  # 20th pct of a ~chi2(49)/49 variable is ~0.8x its mean
    if len(xs) < 3:
        s = estimate_sigma(g)
        return 0.0, s * s
    A = np.stack([np.maximum(xs, 0), np.ones(len(xs))], axis=1)
    a, b = np.linalg.lstsq(A, np.asarray(ys), rcond=None)[0]
    if a < 0:  # fall back to signal-independent noise
        a, b = 0.0, float(np.median(ys))
    return float(max(a, 0.0)), float(max(b, 1e-10))


def bound_residual(noisy: np.ndarray, denoised: np.ndarray, k: float, nlf: tuple[float, float] | None = None) -> np.ndarray:
    """Trust region for a learned denoiser: never change a pixel by more than k * sigma(x), the noise std the
    sensor's noise model allows at that brightness. Removes noise; cannot erase real detail or shift highlights."""
    a, b = nlf if nlf is not None else estimate_nlf(noisy)
    sigma = np.sqrt(np.maximum(a * np.maximum(denoised, 0.0) + b, 1e-12)).astype(np.float32)
    return (noisy + np.clip(denoised - noisy, -k * sigma, k * sigma)).astype(np.float32)


def guided_filter(img: np.ndarray, radius: int, eps: float) -> np.ndarray:
    """Self-guided filter (He et al. 2010), per channel. Fast O(N) box filters."""
    def _one(p: np.ndarray) -> np.ndarray:
        k = (2 * radius + 1, 2 * radius + 1)
        mean_p = cv2.boxFilter(p, -1, k)
        mean_pp = cv2.boxFilter(p * p, -1, k)
        var = mean_pp - mean_p * mean_p
        a = var / (var + eps)
        b = mean_p - a * mean_p
        return cv2.boxFilter(a, -1, k) * p + cv2.boxFilter(b, -1, k)

    img = img.astype(np.float32, copy=False)
    if img.ndim == 2:
        return _one(img)
    return np.stack([_one(img[..., c]) for c in range(img.shape[-1])], axis=-1)


def bilateral(rgb: np.ndarray, sigma: float, spatial: float = 3.0) -> np.ndarray:
    """Noise-aware bilateral: the range kernel scales with the noise level."""
    if sigma <= 1e-5:
        return rgb
    d = int(2 * round(spatial) + 1)
    return cv2.bilateralFilter(rgb.astype(np.float32), d, float(3.0 * sigma), float(spatial))


def chroma_denoise(srgb: np.ndarray, strength: float) -> np.ndarray:
    """Smooth only the colour (Cr/Cb) channels; keeps luminance detail intact."""
    if strength <= 0:
        return srgb
    ycc = cv2.cvtColor(np.clip(srgb, 0, 1).astype(np.float32), cv2.COLOR_RGB2YCrCb)
    k = int(2 * round(2 + 3 * strength) + 1)
    for c in (1, 2):
        ycc[..., c] = cv2.bilateralFilter(ycc[..., c], k, 0.08 * strength + 0.02, 2.0 + 3 * strength)
    return np.clip(cv2.cvtColor(ycc, cv2.COLOR_YCrCb2RGB), 0, 1)
