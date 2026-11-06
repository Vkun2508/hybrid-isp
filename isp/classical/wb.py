"""White balance on RGGB Bayer data: gain application and illuminant estimators."""
from __future__ import annotations

import numpy as np

from isp.raw import pack

Gains = tuple[float, float, float]


def apply_gains(bayer: np.ndarray, gains: Gains) -> np.ndarray:
    r, g, b = gains
    out = bayer.copy()
    out[0::2, 0::2] *= r / g
    out[1::2, 1::2] *= b / g
    return out


def _channel_stats(bayer: np.ndarray, p: float) -> np.ndarray:
    """Minkowski p-norm mean of R, G, B over unclipped pixels (p=1 gray-world, p=inf white-patch)."""
    q = pack(bayer)
    rgb = np.stack([q[..., 0], 0.5 * (q[..., 1] + q[..., 2]), q[..., 3]], axis=-1).reshape(-1, 3)
    # Ignore clipped pixels AND the brightest 2% (speculars): a relative cut keeps the estimate the same
    # when the whole exposure changes (a fixed 0.97 cut let under-exposed speculars dominate the p=6 norm).
    peak = rgb.max(axis=1)
    keep = (peak < 0.97) & (peak <= np.percentile(peak, 98))
    if keep.sum() > 100:
        rgb = rgb[keep]
    if np.isinf(p):
        return np.percentile(rgb, 99.5, axis=0) + 1e-6
    return np.power(np.mean(np.power(np.maximum(rgb, 0), p), axis=0), 1.0 / p) + 1e-6


def _gains_from_illuminant(ill: np.ndarray) -> Gains:
    return (float(ill[1] / ill[0]), 1.0, float(ill[1] / ill[2]))


def gray_world(bayer: np.ndarray) -> Gains:
    return _gains_from_illuminant(_channel_stats(bayer, 1.0))


def shades_of_gray(bayer: np.ndarray, p: float = 6.0) -> Gains:
    return _gains_from_illuminant(_channel_stats(bayer, p))


def white_patch(bayer: np.ndarray) -> Gains:
    return _gains_from_illuminant(_channel_stats(bayer, np.inf))


def gray_edge(bayer: np.ndarray, p: float = 6.0, sigma: float = 1.0) -> Gains:
    """Gray-edge (van de Weijer et al. 2007): the Minkowski mean of image *gradients* is achromatic.
    Uses edges instead of surfaces, so it fails differently from gray-world on large coloured areas."""
    import cv2

    q = pack(bayer)
    rgb = np.stack([q[..., 0], 0.5 * (q[..., 1] + q[..., 2]), q[..., 3]], axis=-1).astype(np.float32)
    rgb = cv2.GaussianBlur(rgb, (0, 0), sigma)
    peak = rgb.max(axis=-1)
    keep = (peak < 0.97) & (peak <= np.percentile(peak, 98))
    ill = []
    for c in range(3):
        gx = cv2.Sobel(rgb[..., c], cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(rgb[..., c], cv2.CV_32F, 0, 1)
        mag = np.sqrt(gx * gx + gy * gy)[keep]
        ill.append(float(np.power(np.mean(np.power(mag, p)), 1.0 / p)) + 1e-9)
    return _gains_from_illuminant(np.asarray(ill))


def estimator_spread(bayer: np.ndarray) -> float:
    """Largest angle (degrees) between three independent illuminant estimators. Small = the scene behaves like
    a neutral world and the estimates can be trusted; large = coloured scene, the estimators are guessing."""
    est = [gray_world(bayer), shades_of_gray(bayer), gray_edge(bayer)]
    return max(angular_error_deg(a, b) for i, a in enumerate(est) for b in est[i + 1:])


def angular_error_deg(g1: Gains, g2: Gains) -> float:
    """Angle between the two illuminants implied by two gain triplets."""
    a = 1.0 / np.asarray(g1, np.float64)
    b = 1.0 / np.asarray(g2, np.float64)
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
