"""THE MODEL SLOT: the only place a learned model plugs into the app.

To plug in a model, put two files in `models/`:
    models/model.json   - describes the model (see Manifest below and docs/MODEL_HANDOFF.md)
    models/<file>       - .onnx (preferred), .pt/.pth TorchScript (local only), or a .py with predict()

Nothing else in the app (API, UI, pipelines) needs to change. If no model is present, or it fails
to load, a classical stand-in ("stub") is used and the UI shows a "placeholder model" badge and
the load error, so the site keeps working.

The slot handles: padding to a multiple, overlapping tiles (bounded memory, fixed-size models),
NCHW/NHWC layout, float16 models, an optional noise-map input, NaN/shape checks.
"""
from __future__ import annotations

import importlib.util
import json
import os
import threading
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = Path(os.environ.get("ISP_MODELS_DIR", ROOT / "models"))
MANIFEST_PATH = Path(os.environ.get("ISP_MODEL_MANIFEST", MODELS_DIR / "model.json"))

INPUTS = ("raw4", "rgb")
OUTPUTS = ("raw4", "srgb", "linear_srgb", "linear_camera")


@dataclass
class Manifest:
    name: str = "Placeholder (classical stand-in)"
    file: str | None = None          # file name inside models/; null -> stub
    input: str = "raw4"              # "raw4": packed (H/2, W/2, 4) [R, G1, G2, B], black-subtracted, /white,
                                     #         unclipped (negatives kept); "rgb": (H, W, 3)
    input_wb: bool = False           # input already white-balanced with the as-shot gains
    input_gamma: bool = False        # rgb input only: model expects sRGB-encoded (gamma) input, not linear
    output: str = "raw4"             # "raw4" (denoised packed RAW, same size) | "srgb" | "linear_srgb" | "linear_camera"
    upscale: int = 1                 # output H / input H (2 for packed RAW -> full-res RGB, else 1)
    pad_multiple: int = 8            # input H, W padded to a multiple of this (U-Net downsampling)
    needs_noise_map: bool = False    # model takes a per-pixel noise std map as an extra input/channel
    layout: str = "NCHW"             # tensor layout the model expects: "NCHW" or "NHWC"
    tile: int = 1024                 # input-space tile size (memory bound); fixed-shape models override it
    overlap: int = 32                # tile overlap in input pixels (hides seams)
    notes: str = ""

    @classmethod
    def load(cls, path: Path) -> "Manifest":
        raw = json.loads(path.read_text(encoding="utf-8"))
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(raw) - known - {"$schema", "_comment"})
        if unknown:
            raise ValueError(f"model.json has unknown keys {unknown}; allowed: {sorted(known)}")
        m = cls(**{k: v for k, v in raw.items() if k in known})
        m.validate()
        return m

    def validate(self) -> None:
        if self.input not in INPUTS:
            raise ValueError(f"model.json 'input' must be one of {INPUTS}, got {self.input!r}")
        if self.output not in OUTPUTS:
            raise ValueError(f"model.json 'output' must be one of {OUTPUTS}, got {self.output!r}")
        if self.layout not in ("NCHW", "NHWC"):
            raise ValueError("model.json 'layout' must be 'NCHW' or 'NHWC'")
        if self.upscale not in (1, 2, 4):
            raise ValueError("model.json 'upscale' must be 1, 2 or 4")
        if self.output == "raw4" and (self.input != "raw4" or self.upscale != 1):
            raise ValueError("output 'raw4' requires input 'raw4' and upscale 1")
        if self.tile <= 2 * self.overlap:
            raise ValueError("model.json 'tile' must be larger than 2 * 'overlap'")


# --------------------------------------------------------------------------- backends
class _Backend:
    kind = "base"
    fixed_hw: tuple[int, int] | None = None
    params: int | None = None

    def infer(self, x: np.ndarray, nmap: np.ndarray | None) -> np.ndarray:  # HWC -> HWC
        raise NotImplementedError


class OnnxBackend(_Backend):
    kind = "onnx"

    def __init__(self, path: Path, m: Manifest):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = max(1, os.cpu_count() or 1)
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        self.m = m
        ins = self.sess.get_inputs()
        self.in_names = [i.name for i in ins]
        self.out_name = self.sess.get_outputs()[0].name
        self.dtype = np.float16 if "float16" in ins[0].type else np.float32
        shp = ins[0].shape
        hw = (shp[2], shp[3]) if m.layout == "NCHW" else (shp[1], shp[2])
        if all(isinstance(v, int) and v > 0 for v in hw):
            self.fixed_hw = (int(hw[0]), int(hw[1]))
        try:
            import onnx
            from onnx import numpy_helper

            g = onnx.load(str(path), load_external_data=False).graph
            # count weights/biases only (INT8 graphs also store per-channel scales and zero points)
            self.params = int(sum(int(np.prod(t.dims)) for t in g.initializer
                                  if not t.name.endswith(("_scale", "_zero_point"))))
            _ = numpy_helper  # keep import (validates onnx install)
        except Exception:
            self.params = None

    def _to_tensor(self, a: np.ndarray) -> np.ndarray:
        t = a[None]
        if self.m.layout == "NCHW":
            t = t.transpose(0, 3, 1, 2)
        return np.ascontiguousarray(t, dtype=self.dtype)

    def infer(self, x, nmap):
        feeds = {}
        if nmap is not None and len(self.in_names) >= 2:
            feeds[self.in_names[0]] = self._to_tensor(x)
            feeds[self.in_names[1]] = self._to_tensor(nmap[..., None])
        else:
            if nmap is not None:
                x = np.concatenate([x, nmap[..., None]], axis=-1)
            feeds[self.in_names[0]] = self._to_tensor(x)
        y = self.sess.run([self.out_name], feeds)[0]
        y = y[0]
        if self.m.layout == "NCHW":
            y = y.transpose(1, 2, 0)
        return y.astype(np.float32)


class TorchBackend(_Backend):
    kind = "torchscript"

    def __init__(self, path: Path, m: Manifest):
        try:
            import torch
        except ImportError as e:
            raise RuntimeError(
                "A .pt/.pth model needs PyTorch, which is not in the deploy image. "
                "Convert it with scripts/export_onnx.py and point model.json at the .onnx file."
            ) from e
        self.torch = torch
        try:
            self.net = torch.jit.load(str(path), map_location="cpu").eval()
        except Exception as e:
            raise RuntimeError(
                f"{path.name} is not a TorchScript file (a state_dict needs the model class). "
                "Use scripts/export_onnx.py --model-py <file.py> --cls <Class> --weights <file.pth>."
            ) from e
        self.m = m
        self.params = int(sum(p.numel() for p in self.net.parameters()))

    def infer(self, x, nmap):
        if nmap is not None:
            x = np.concatenate([x, nmap[..., None]], axis=-1)
        t = self.torch.from_numpy(np.ascontiguousarray(x[None]))
        if self.m.layout == "NCHW":
            t = t.permute(0, 3, 1, 2)
        with self.torch.no_grad():
            y = self.net(t.float())
        if isinstance(y, (tuple, list)):
            y = y[0]
        y = y[0]
        if self.m.layout == "NCHW":
            y = y.permute(1, 2, 0)
        return y.cpu().numpy().astype(np.float32)


class PythonBackend(_Backend):
    """Escape hatch: a .py file exposing predict(x: HWC float32) -> HWC float32 (and optionally load())."""
    kind = "python"

    def __init__(self, path: Path, m: Manifest):
        spec = importlib.util.spec_from_file_location("isp_user_model", path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot import {path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if not hasattr(mod, "predict"):
            raise RuntimeError(f"{path.name} must define predict(x) -> np.ndarray")
        if hasattr(mod, "load"):
            mod.load()
        self.mod = mod
        self.params = getattr(mod, "PARAMS", None)

    def infer(self, x, nmap):
        if nmap is not None:
            x = np.concatenate([x, nmap[..., None]], axis=-1)
        return np.asarray(self.mod.predict(x), dtype=np.float32)


class StubBackend(_Backend):
    """Classical stand-in used until real weights arrive. Same contract as the team's model
    (raw4 in -> denoised raw4 out): a guided filter on each packed Bayer plane."""
    kind = "stub"
    params = 0

    def infer(self, x, nmap):
        from isp.classical.denoise import estimate_sigma, guided_filter

        packed = x[..., :4]
        sigma = estimate_sigma(packed[..., 1]) if nmap is None else float(np.median(nmap))
        if sigma <= 1e-4:
            return packed.copy()
        return guided_filter(packed, radius=2, eps=(1.5 * sigma) ** 2)


# --------------------------------------------------------------------------- the slot
@dataclass
class ModelInfo:
    name: str
    kind: str
    is_stub: bool
    file: str | None
    input: str
    output: str
    params: int | None
    size_mb: float | None
    load_error: str | None = None
    manifest: dict = field(default_factory=dict)


class LearnedModel:
    def __init__(self, manifest: Manifest, backend: _Backend, info: ModelInfo):
        self.m = manifest
        self.backend = backend
        self.info = info

    @property
    def is_stub(self) -> bool:
        return self.info.is_stub

    def run(self, x: np.ndarray, noise_map: np.ndarray | None = None) -> np.ndarray:
        """x: (h, w, C) float32 in the manifest's input convention. Returns (h*u, w*u, 3) float32."""
        x = np.ascontiguousarray(x, np.float32)
        nmap = noise_map.astype(np.float32) if (self.m.needs_noise_map and noise_map is not None) else None
        if self.m.needs_noise_map and nmap is None:
            nmap = np.zeros(x.shape[:2], np.float32)
        u = self.m.upscale
        h, w = x.shape[:2]
        if isinstance(self.backend, StubBackend):
            y = self.backend.infer(x, nmap)
        elif self.backend.fixed_hw is None and h <= self.m.tile and w <= self.m.tile:
            y = self._run_padded(x, nmap)
        else:
            y = self._run_tiled(x, nmap)
        expected = (h * u, w * u)
        want_c = 4 if self.m.output == "raw4" else 3
        if y.ndim != 3 or y.shape[:2] != expected or y.shape[2] < want_c:
            raise RuntimeError(
                f"Model output shape {tuple(y.shape)} doesn't match the manifest: expected "
                f"({expected[0]}, {expected[1]}, {want_c}) for input {tuple(x.shape)} with upscale={u}."
            )
        y = y[..., :want_c]
        if not np.isfinite(y).all():
            raise RuntimeError("Model produced NaN/Inf values.")
        if self.m.output == "raw4":
            return y.astype(np.float32)  # RAW keeps its sign (noise floor); clipped later, for display
        return np.clip(y, 0.0, None if self.m.output != "srgb" else 1.0).astype(np.float32)

    def _run_padded(self, x, nmap):
        h, w = x.shape[:2]
        pm = max(1, self.m.pad_multiple)
        ph, pw = (-h) % pm, (-w) % pm
        if ph or pw:
            x = np.pad(x, ((0, ph), (0, pw), (0, 0)), mode="reflect")
            if nmap is not None:
                nmap = np.pad(nmap, ((0, ph), (0, pw)), mode="reflect")
        y = self.backend.infer(x, nmap)
        u = self.m.upscale
        return y[: h * u, : w * u]

    def _run_tiled(self, x, nmap):
        h, w = x.shape[:2]
        u, o = self.m.upscale, self.m.overlap
        if self.backend.fixed_hw is not None:
            t = min(self.backend.fixed_hw)
            o = min(o, t // 4)
        else:
            pm = max(1, self.m.pad_multiple)
            t = max(pm, self.m.tile // pm * pm)
        step = t - 2 * o
        hc, wc = -(-h // step) * step, -(-w // step) * step
        pad = ((o, hc - h + o), (o, wc - w + o))
        xp = np.pad(x, pad + ((0, 0),), mode="reflect")
        np_ = np.pad(nmap, pad, mode="reflect") if nmap is not None else None
        out = None
        for i in range(0, hc, step):
            for j in range(0, wc, step):
                win = xp[i:i + t, j:j + t]
                y = self.backend.infer(win, np_[i:i + t, j:j + t] if np_ is not None else None)
                if out is None:
                    out = np.zeros((hc * u, wc * u, y.shape[2]), np.float32)
                out[i * u:(i + step) * u, j * u:(j + step) * u] = y[o * u:(o + step) * u, o * u:(o + step) * u]
        return out[: h * u, : w * u]


_lock = threading.Lock()
_model: LearnedModel | None = None


def _stub(error: str | None = None) -> LearnedModel:
    m = Manifest()
    info = ModelInfo(name=m.name, kind="stub", is_stub=True, file=None, input=m.input, output=m.output,
                     params=0, size_mb=0.0, load_error=error, manifest=asdict(m))
    return LearnedModel(m, StubBackend(), info)


def load_model(manifest_path: Path = MANIFEST_PATH) -> LearnedModel:
    if not manifest_path.exists():
        return _stub()
    try:
        m = Manifest.load(manifest_path)
    except Exception as e:
        return _stub(f"model.json invalid: {e}")
    if not m.file:
        return _stub()
    path = (manifest_path.parent / m.file).resolve()
    if not path.exists():
        return _stub(f"model file not found: {m.file}")
    ext = path.suffix.lower()
    try:
        if ext == ".onnx":
            backend: _Backend = OnnxBackend(path, m)
        elif ext in (".pt", ".pth", ".torchscript"):
            backend = TorchBackend(path, m)
        elif ext == ".py":
            backend = PythonBackend(path, m)
        else:
            raise RuntimeError(f"unsupported model file type '{ext}' (use .onnx, .pt or .py)")
    except Exception as e:
        return _stub(f"{type(e).__name__}: {e}")
    info = ModelInfo(
        name=m.name, kind=backend.kind, is_stub=False, file=m.file, input=m.input, output=m.output,
        params=backend.params, size_mb=round(path.stat().st_size / 1e6, 3), manifest=asdict(m),
    )
    return LearnedModel(m, backend, info)


def get_model() -> LearnedModel:
    global _model
    with _lock:
        if _model is None:
            t0 = time.perf_counter()
            _model = load_model()
            _model.info.manifest["load_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return _model


def reload_model() -> LearnedModel:
    global _model
    with _lock:
        _model = None
    return get_model()
