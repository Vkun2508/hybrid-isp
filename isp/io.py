"""Input loading. Every supported input becomes a RawImage (RGGB, normalised, black level removed).

- Camera RAW / DNG (any Bayer sensor libraw can read) -> real RAW + metadata.
- Demosaiced "linear" DNGs (e.g. Apple ProRAW) -> re-mosaiced to RGGB so one code path serves all.
- JPG / PNG / WebP -> "unprocessed" into simulated RAW (Brooks et al., CVPR 2019); the original
  image is kept as ground truth, so PSNR / SSIM / dE can be shown for any uploaded photo.
"""
from __future__ import annotations

import io as _io
import os

import cv2
import numpy as np

from isp.classical.color import DEFAULT_CAM2RGB, apply_ccm, cam2rgb_from_xyz2cam, srgb_to_linear
from isp.classical.tone import inverse_smoothstep
from isp.raw import RawImage, even_crop, mosaic, pack, unpack

MAX_MP = float(os.environ.get("ISP_MAX_MP", "2.0"))
SIM_WB_GAINS = (2.0, 1.0, 1.7)  # daylight-like as-shot gains used for simulated RAW


class InputError(Exception):
    """A user-facing input problem. `code` maps to an HTTP status in the API."""

    def __init__(self, code: str, message: str, hint: str = "", status: int = 400):
        super().__init__(message)
        self.code, self.message, self.hint, self.status = code, message, hint, status


def sniff(data: bytes) -> str:
    head = data[:16]
    if head.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if head.startswith(b"\x89PNG"):
        return "png"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "webp"
    if head.startswith(b"FUJIFILM"):
        return "raf"
    if head[:2] in (b"II", b"MM") or head[4:8] == b"ftyp":
        return "raw"  # TIFF-based RAWs (DNG, CR2, NEF, ARW, ORF, RW2...) and CR3
    if head[:4] in (b"\x00\x00\x00\x0c", b"MRM\x00"):
        return "raw"
    if head.startswith(b"MATLAB") or head[:8] == b"\x89HDF\r\n\x1a\n":
        return "mat"
    return "unknown"


def load_any(data: bytes, filename: str = "") -> RawImage:
    if not data:
        raise InputError("empty_file", "The file is empty.", "Choose a photo or RAW file and try again.")
    kind = sniff(data)
    ext = os.path.splitext(filename.lower())[1]
    if kind in ("jpeg", "png", "webp"):
        return load_image(data, filename)
    if kind == "mat":
        raise InputError(
            "unsupported_mat", "SIDD .MAT files are not accepted by the web demo.",
            "Use one of the built-in SIDD samples, or upload a DNG/RAW or JPG/PNG.", 415,
        )
    if kind in ("raw", "raf") or ext in (".dng", ".cr2", ".cr3", ".nef", ".arw", ".orf", ".rw2", ".raf", ".pef", ".srw"):
        return load_raw(data, filename)
    raise InputError(
        "unsupported_type", "That file type isn't supported.",
        "Upload a camera RAW (DNG, CR2, NEF, ARW...) or a JPG/PNG photo.", 415,
    )


# --------------------------------------------------------------------------- RAW files
def load_raw(data: bytes, filename: str = "") -> RawImage:
    try:
        import rawpy
    except ImportError as e:  # pragma: no cover
        raise InputError("server_raw", "RAW decoding is unavailable on this server.", "Try a JPG/PNG.", 500) from e
    try:
        rp = rawpy.imread(_io.BytesIO(data))
    except Exception as e:
        raise InputError(
            "corrupt_raw", "This RAW file couldn't be read. It may be corrupt or an unsupported camera.",
            "Try a DNG from your phone, or a JPG/PNG photo.", 422,
        ) from e
    with rp:
        pattern = np.asarray(rp.raw_pattern) if rp.raw_pattern is not None else None
        notes: list[str] = []
        flip = int(getattr(rp.sizes, "flip", 0) or 0)
        cam2rgb = cam2rgb_from_xyz2cam(rp.rgb_xyz_matrix)
        if cam2rgb is None:
            cam2rgb = DEFAULT_CAM2RGB.copy()
            notes.append("No colour matrix in file; used a generic camera matrix.")
        wb = _wb_from(rp.camera_whitebalance)
        white = float(rp.white_level) if rp.white_level else float(np.max(rp.raw_image_visible))

        if pattern is not None and pattern.shape == (6, 6):
            raise InputError(
                "xtrans", "Fujifilm X-Trans sensors aren't supported (this demo handles Bayer sensors).",
                "Try a DNG from a phone or a Bayer camera, or a JPG/PNG photo.", 422,
            )
        is_bayer = pattern is not None and pattern.shape == (2, 2) and rp.num_colors == 3 and rp.raw_image_visible.ndim == 2
        if is_bayer:
            raw = rp.raw_image_visible.astype(np.float32)
            colors = rp.raw_colors_visible
            black = np.asarray(rp.black_level_per_channel, np.float32)
            raw = raw - black[colors]
            # keep the noise floor's negatives (the RAW denoiser was trained on unclipped data)
            raw = np.clip(raw / max(white - float(black.mean()), 1.0), -0.25, 1.0)
            desc = rp.color_desc.decode(errors="ignore")  # e.g. "RGBG"
            pat = "".join(desc[pattern[i, j]] for i in range(2) for j in range(2))
            bayer = _to_rggb(raw, pat)
            pattern_name = pat
        else:
            # Already demosaiced (linear DNG / ProRAW): take linear camera RGB and re-mosaic.
            try:
                rgb16 = rp.postprocess(
                    gamma=(1, 1), no_auto_bright=True, output_bps=16, use_camera_wb=False,
                    user_wb=[1.0, 1.0, 1.0, 1.0], output_color=rawpy.ColorSpace.raw, user_flip=0,
                )
            except Exception as e:
                raise InputError("corrupt_raw", "This RAW file couldn't be decoded.", "Try another file.", 422) from e
            rgb = even_crop(rgb16.astype(np.float32) / 65535.0)
            bayer = mosaic(rgb)
            pattern_name = "linear (re-mosaiced)"
            notes.append("Linear (already demosaiced) DNG: re-mosaiced to RGGB for a like-for-like comparison.")

    bayer = even_crop(bayer)
    orig_h, orig_w = bayer.shape
    bayer, scale = cap_bayer(bayer)
    if scale < 1:
        notes.append(f"Downscaled {orig_w}x{orig_h} -> {bayer.shape[1]}x{bayer.shape[0]} (2x2-aware binning) to stay fast.")
    return RawImage(
        bayer=bayer, wb_gains=wb, cam2rgb=cam2rgb, flip=flip,
        meta={"source": "raw", "filename": filename, "pattern": pattern_name,
              "orig_size": [orig_w, orig_h], "notes": notes, "has_metadata_wb": wb is not None},
    )


def _wb_from(cam_wb) -> tuple[float, float, float] | None:
    try:
        r, g, b = float(cam_wb[0]), float(cam_wb[1]), float(cam_wb[2])
    except Exception:
        return None
    if g <= 0 or r <= 0 or b <= 0:
        return None
    return (r / g, 1.0, b / g)


def _to_rggb(raw: np.ndarray, pattern: str) -> np.ndarray:
    pattern = pattern.replace("G2", "G")
    offsets = {"RGGB": (0, 0), "GRBG": (0, 1), "GBRG": (1, 0), "BGGR": (1, 1)}
    if pattern not in offsets:
        raise InputError("unsupported_cfa", f"Unsupported colour filter pattern '{pattern}'.", "Try a Bayer RAW or a JPG.", 422)
    dy, dx = offsets[pattern]
    return raw[dy:, dx:]


def cap_bayer(bayer: np.ndarray, max_mp: float = MAX_MP) -> tuple[np.ndarray, float]:
    """Downscale a Bayer image per colour plane so it never exceeds max_mp megapixels."""
    h, w = bayer.shape
    mp = h * w / 1e6
    if mp <= max_mp:
        return bayer, 1.0
    s = (max_mp / mp) ** 0.5
    q = pack(bayer)
    nh, nw = max(2, int(q.shape[0] * s) // 2 * 2), max(2, int(q.shape[1] * s) // 2 * 2)
    q2 = np.stack([cv2.resize(q[..., c], (nw, nh), interpolation=cv2.INTER_AREA) for c in range(4)], axis=-1)
    return unpack(q2), s


# --------------------------------------------------------------------------- JPG / PNG
def decode_image(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)  # applies EXIF orientation
    if img is None:
        raise InputError("corrupt_image", "This image couldn't be decoded. It may be corrupt.", "Try another JPG/PNG.", 422)
    scale = 65535.0 if img.dtype == np.uint16 else 255.0
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / scale


def load_image(data: bytes, filename: str = "") -> RawImage:
    srgb = decode_image(data)
    return from_srgb(srgb, filename=filename)


def from_srgb(srgb: np.ndarray, filename: str = "", source: str = "image") -> RawImage:
    notes = ["JPG/PNG input: converted to simulated RAW (Brooks et al. 2019 'unprocessing'); "
             "the original photo is used as ground truth for the metrics."]
    h, w = srgb.shape[:2]
    mp = h * w / 1e6
    if mp > MAX_MP:
        s = (MAX_MP / mp) ** 0.5
        srgb = cv2.resize(srgb, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        notes.append(f"Downscaled {w}x{h} -> {srgb.shape[1]}x{srgb.shape[0]} to stay fast.")
    srgb = np.clip(even_crop(srgb), 0, 1).astype(np.float32)
    bayer = unprocess(srgb, DEFAULT_CAM2RGB, SIM_WB_GAINS)
    return RawImage(
        bayer=bayer, wb_gains=SIM_WB_GAINS, cam2rgb=DEFAULT_CAM2RGB.copy(), gt=srgb,
        meta={"source": source, "filename": filename, "pattern": "RGGB (simulated)",
              "orig_size": [w, h], "notes": notes, "has_metadata_wb": True},
    )


def unprocess(srgb: np.ndarray, cam2rgb: np.ndarray, gains: tuple[float, float, float]) -> np.ndarray:
    """Exact inverse of our forward pipeline (tone -> gamma -> CCM -> WB -> mosaic)."""
    lin = srgb_to_linear(inverse_smoothstep(srgb))
    cam = apply_ccm(lin, np.linalg.inv(cam2rgb).astype(np.float32))
    cam = cam / np.asarray(gains, np.float32)
    return np.clip(mosaic(cam), 0.0, 1.0)
