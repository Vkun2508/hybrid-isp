"""The single internal RAW representation every pipeline works on.

Convention (everything downstream relies on it):
    bayer      float32 (H, W), RGGB pattern, black level subtracted, scaled so white level = 1.0
    wb_gains   (r, g, b) multipliers with g == 1, from camera metadata (None if unknown)
    cam2rgb    3x3 matrix, camera RGB -> linear sRGB (rows sum to 1)
    gt         optional ground-truth sRGB image (H, W, 3) float32 in [0, 1]
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

IDENTITY = np.eye(3, dtype=np.float32)


@dataclass
class RawImage:
    bayer: np.ndarray
    wb_gains: tuple[float, float, float] | None = None
    cam2rgb: np.ndarray = field(default_factory=lambda: IDENTITY.copy())
    gt: np.ndarray | None = None
    noise_profile: tuple[float, float] | None = None  # variance = a*x + b
    flip: int = 0  # rawpy/libraw orientation code, applied only when displaying
    meta: dict = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int]:
        return self.bayer.shape  # type: ignore[return-value]

    def copy(self, **changes) -> "RawImage":
        new = replace(self, **changes)
        if "bayer" not in changes:
            new.bayer = self.bayer.copy()
        new.meta = dict(self.meta)
        return new


def pack(bayer: np.ndarray) -> np.ndarray:
    """RGGB (H, W) -> (H/2, W/2, 4) in channel order R, Gr, Gb, B."""
    return np.stack(
        [bayer[0::2, 0::2], bayer[0::2, 1::2], bayer[1::2, 0::2], bayer[1::2, 1::2]], axis=-1
    )


def unpack(packed: np.ndarray) -> np.ndarray:
    """(H/2, W/2, 4) -> RGGB (H, W)."""
    h, w, _ = packed.shape
    out = np.empty((h * 2, w * 2), dtype=packed.dtype)
    out[0::2, 0::2] = packed[..., 0]
    out[0::2, 1::2] = packed[..., 1]
    out[1::2, 0::2] = packed[..., 2]
    out[1::2, 1::2] = packed[..., 3]
    return out


def mosaic(rgb: np.ndarray) -> np.ndarray:
    """(H, W, 3) linear camera RGB -> RGGB Bayer (H, W). H and W must be even."""
    h, w, _ = rgb.shape
    out = np.empty((h, w), dtype=np.float32)
    out[0::2, 0::2] = rgb[0::2, 0::2, 0]
    out[0::2, 1::2] = rgb[0::2, 1::2, 1]
    out[1::2, 0::2] = rgb[1::2, 0::2, 1]
    out[1::2, 1::2] = rgb[1::2, 1::2, 2]
    return out


def cfa_masks(h: int, w: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r = np.zeros((h, w), bool)
    g = np.zeros((h, w), bool)
    b = np.zeros((h, w), bool)
    r[0::2, 0::2] = True
    g[0::2, 1::2] = True
    g[1::2, 0::2] = True
    b[1::2, 1::2] = True
    return r, g, b


def even_crop(arr: np.ndarray, multiple: int = 4) -> np.ndarray:
    h, w = arr.shape[:2]
    return arr[: h - h % multiple, : w - w % multiple]
