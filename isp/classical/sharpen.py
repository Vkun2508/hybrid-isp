"""Unsharp masking on luma only, so sharpening never amplifies colour noise."""
from __future__ import annotations

import cv2
import numpy as np


def unsharp_luma(srgb: np.ndarray, amount: float, radius: float = 1.2) -> np.ndarray:
    if amount <= 0:
        return srgb
    img = np.clip(srgb, 0, 1).astype(np.float32)
    ycc = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    y = ycc[..., 0]
    blur = cv2.GaussianBlur(y, (0, 0), radius)
    ycc[..., 0] = np.clip(y + amount * (y - blur), 0, 1)
    return np.clip(cv2.cvtColor(ycc, cv2.COLOR_YCrCb2RGB), 0, 1)


def noise_aware_amount(sigma: float, k: float = 0.5, sigma0: float = 0.01) -> float:
    """Sharpening strength k / (1 + sigma / sigma0): weaker sharpening on noisy images."""
    return float(k / (1.0 + sigma / sigma0))
