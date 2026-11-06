"""Colour-space transforms: CCM, sRGB transfer function, camera matrices."""
from __future__ import annotations

import numpy as np

# Linear sRGB (D65) -> CIE XYZ
SRGB_TO_XYZ = np.array(
    [[0.4124564, 0.3575761, 0.1804375],
     [0.2126729, 0.7151522, 0.0721750],
     [0.0193339, 0.1191920, 0.9503041]],
    dtype=np.float64,
)

# A typical DSLR/phone camera->sRGB matrix (rows sum to 1). Used for simulated RAW
# made from JPG/PNG, and as the fallback when a RAW file has no colour matrix.
DEFAULT_CAM2RGB = np.array(
    [[1.88, -0.85, -0.03],
     [-0.20, 1.55, -0.35],
     [0.03, -0.55, 1.52]],
    dtype=np.float32,
)


def cam2rgb_from_xyz2cam(xyz2cam: np.ndarray) -> np.ndarray | None:
    """dcraw-style: camera->sRGB from the Adobe XYZ->camera matrix (3x3)."""
    m = np.asarray(xyz2cam, dtype=np.float64)[:3, :3]
    if not np.isfinite(m).all() or np.abs(m).sum() < 1e-6:
        return None
    rgb2cam = m @ SRGB_TO_XYZ
    rgb2cam = rgb2cam / rgb2cam.sum(axis=1, keepdims=True)
    try:
        cam2rgb = np.linalg.inv(rgb2cam)
    except np.linalg.LinAlgError:
        return None
    if not np.isfinite(cam2rgb).all():
        return None
    return cam2rgb.astype(np.float32)


def apply_ccm(rgb: np.ndarray, ccm: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.transform(np.ascontiguousarray(rgb, np.float32), np.asarray(ccm, np.float32))


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055).astype(np.float32)


def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.04045, x / 12.92, np.power((x + 0.055) / 1.055, 2.4)).astype(np.float32)


def rgb_to_luma(rgb: np.ndarray) -> np.ndarray:
    return (0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]).astype(np.float32)
