"""Demosaicing for RGGB Bayer: bilinear and Malvar-He-Cutler (2004)."""
from __future__ import annotations

import cv2
import numpy as np

_B = cv2.BORDER_REFLECT_101

# Malvar-He-Cutler 5x5 kernels (all /8).
_G_AT_RB = np.array(
    [[0, 0, -1, 0, 0], [0, 0, 2, 0, 0], [-1, 2, 4, 2, -1], [0, 0, 2, 0, 0], [0, 0, -1, 0, 0]], np.float32
) / 8
_RB_AT_G_ROW = np.array(  # R at a green pixel in a red row (or B at green in a blue row)
    [[0, 0, 0.5, 0, 0], [0, -1, 0, -1, 0], [-1, 4, 5, 4, -1], [0, -1, 0, -1, 0], [0, 0, 0.5, 0, 0]], np.float32
) / 8
_RB_AT_G_COL = _RB_AT_G_ROW.T.copy()  # R at a green pixel in a red column
_RB_AT_BR = np.array(
    [[0, 0, -1.5, 0, 0], [0, 2, 0, 2, 0], [-1.5, 0, 6, 0, -1.5], [0, 2, 0, 2, 0], [0, 0, -1.5, 0, 0]], np.float32
) / 8


def malvar(bayer: np.ndarray) -> np.ndarray:
    """High-quality linear demosaic. Input RGGB (H, W) float32, output (H, W, 3)."""
    cfa = bayer.astype(np.float32, copy=False)
    g_rb = cv2.filter2D(cfa, -1, _G_AT_RB, borderType=_B)
    rb_row = cv2.filter2D(cfa, -1, _RB_AT_G_ROW, borderType=_B)
    rb_col = cv2.filter2D(cfa, -1, _RB_AT_G_COL, borderType=_B)
    rb_diag = cv2.filter2D(cfa, -1, _RB_AT_BR, borderType=_B)

    h, w = cfa.shape
    R = np.empty((h, w), np.float32)
    G = np.empty((h, w), np.float32)
    B = np.empty((h, w), np.float32)

    # Red sites (even, even)
    R[0::2, 0::2] = cfa[0::2, 0::2]
    G[0::2, 0::2] = g_rb[0::2, 0::2]
    B[0::2, 0::2] = rb_diag[0::2, 0::2]
    # Green in red row (even, odd): red is left/right, blue is up/down
    R[0::2, 1::2] = rb_row[0::2, 1::2]
    G[0::2, 1::2] = cfa[0::2, 1::2]
    B[0::2, 1::2] = rb_col[0::2, 1::2]
    # Green in blue row (odd, even): blue is left/right, red is up/down
    R[1::2, 0::2] = rb_col[1::2, 0::2]
    G[1::2, 0::2] = cfa[1::2, 0::2]
    B[1::2, 0::2] = rb_row[1::2, 0::2]
    # Blue sites (odd, odd)
    R[1::2, 1::2] = rb_diag[1::2, 1::2]
    G[1::2, 1::2] = g_rb[1::2, 1::2]
    B[1::2, 1::2] = cfa[1::2, 1::2]
    return np.clip(np.stack([R, G, B], axis=-1), 0.0, None)


def bilinear(bayer: np.ndarray) -> np.ndarray:
    """Simple bilinear demosaic (the minimal step a learned-only pipeline still needs for RGB models)."""
    cfa = bayer.astype(np.float32, copy=False)
    h, w = cfa.shape
    r = np.zeros_like(cfa)
    g = np.zeros_like(cfa)
    b = np.zeros_like(cfa)
    r[0::2, 0::2] = cfa[0::2, 0::2]
    g[0::2, 1::2] = cfa[0::2, 1::2]
    g[1::2, 0::2] = cfa[1::2, 0::2]
    b[1::2, 1::2] = cfa[1::2, 1::2]
    k_rb = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], np.float32) / 4
    k_g = np.array([[0, 1, 0], [1, 4, 1], [0, 1, 0]], np.float32) / 4
    return np.stack(
        [cv2.filter2D(r, -1, k_rb, borderType=_B), cv2.filter2D(g, -1, k_g, borderType=_B),
         cv2.filter2D(b, -1, k_rb, borderType=_B)],
        axis=-1,
    )
