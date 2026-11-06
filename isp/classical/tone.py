"""Tone mapping. Forward tone = sRGB gamma then a smoothstep S-curve (Brooks et al. 2019),
so that unprocessing a JPG and re-processing it is exactly invertible."""
from __future__ import annotations

import cv2
import numpy as np

from isp.classical.color import linear_to_srgb


def smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return (3 * x * x - 2 * x * x * x).astype(np.float32)


def inverse_smoothstep(y: np.ndarray) -> np.ndarray:
    y = np.clip(y, 0.0, 1.0)
    return (0.5 - np.sin(np.arcsin(1.0 - 2.0 * y) / 3.0)).astype(np.float32)


_LUT_N = 65536
_TONE_LUT = smoothstep(linear_to_srgb(np.linspace(0.0, 1.0, _LUT_N, dtype=np.float32)))


def global_tone(linear_rgb: np.ndarray) -> np.ndarray:
    """Linear sRGB -> display sRGB (sRGB gamma then S-curve), via a 16-bit lookup table."""
    idx = (np.clip(linear_rgb, 0.0, 1.0) * (_LUT_N - 1) + 0.5).astype(np.uint16)
    return _TONE_LUT[idx]


def local_tone(srgb: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Gentle local contrast (CLAHE on L*) for high-dynamic-range scenes."""
    lab = cv2.cvtColor(np.clip(srgb, 0, 1).astype(np.float32), cv2.COLOR_RGB2LAB)
    L = np.clip(lab[..., 0] * (255.0 / 100.0), 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=1.5 + strength, tileGridSize=(8, 8))
    L2 = clahe.apply(L).astype(np.float32) * (100.0 / 255.0)
    lab[..., 0] = 0.5 * lab[..., 0] + 0.5 * L2
    return np.clip(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB), 0, 1)
