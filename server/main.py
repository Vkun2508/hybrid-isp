"""Hybrid ISP web server: JSON API + the built React site, in one process.

Run locally:   uvicorn server.main:app --port 7860
Docker / HF:   see Dockerfile (port 7860)

Privacy: uploads are processed in memory only. Nothing is written to disk; results live in a small
in-memory LRU cache and are re-encoded as WebP (no EXIF / GPS survives).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
import traceback
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from isp import io as isp_io
from isp import metrics as isp_metrics
from isp import pipelines
from isp.io import InputError
from isp.model_slot import get_model
from isp.samples import get_entry, load_sample, registry
from isp.simulate import SimParams, iso_label, simulate
from server import schemas

VERSION = "1.0.0"
ROOT = Path(__file__).resolve().parent.parent
WEB_DIST = ROOT / "web" / "dist"
RESULTS_JSON = ROOT / "results" / "benchmarks.json"
MAX_UPLOAD = int(os.environ.get("ISP_MAX_UPLOAD_MB", "60")) * 1024 * 1024
CACHE_RESULTS = int(os.environ.get("ISP_CACHE_RESULTS", "24"))
BUSY_TIMEOUT_S = 45

log = logging.getLogger("hybrid-isp")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

app = FastAPI(title="Hybrid ISP", version=VERSION,
              description="Classical + learned smartphone ISP: RAW -> RGB with traditional / learned / hybrid comparison.")
app.add_middleware(GZipMiddleware, minimum_size=2048)

_jobs = threading.BoundedSemaphore(int(os.environ.get("ISP_CONCURRENCY", "1")))
_encode_pool = ThreadPoolExecutor(max_workers=4)


# --------------------------------------------------------------------------- in-memory result store
class ResultStore:
    """LRU of finished results: response JSON + encoded images. Nothing touches disk."""

    def __init__(self, max_items: int):
        self.max = max_items
        self._items: OrderedDict[str, dict] = OrderedDict()
        self._by_key: dict[str, str] = {}
        self._lock = threading.Lock()

    def put(self, rid: str, key: str | None, response: dict, images: dict[str, bytes]) -> None:
        with self._lock:
            self._items[rid] = {"response": response, "images": images, "key": key}
            if key:
                self._by_key[key] = rid
            while len(self._items) > self.max:
                old_id, old = self._items.popitem(last=False)
                if old["key"] and self._by_key.get(old["key"]) == old_id:
                    del self._by_key[old["key"]]

    def by_key(self, key: str) -> dict | None:
        with self._lock:
            rid = self._by_key.get(key)
            if rid and rid in self._items:
                self._items.move_to_end(rid)
                return self._items[rid]["response"]
            return None

    def image(self, rid: str, name: str) -> bytes | None:
        with self._lock:
            item = self._items.get(rid)
            if item:
                self._items.move_to_end(rid)
                return item["images"].get(name)
            return None


store = ResultStore(CACHE_RESULTS)
_thumbs: dict[str, bytes] = {}


# --------------------------------------------------------------------------- helpers
def _orient(img: np.ndarray, flip: int) -> np.ndarray:
    if flip == 3:
        return cv2.rotate(img, cv2.ROTATE_180)
    if flip == 5:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if flip == 6:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return img


def _webp(img: np.ndarray, quality: int = 90, flip: int = 0) -> bytes:
    u8 = (np.clip(_orient(img, flip), 0, 1) * 255.0 + 0.5).astype(np.uint8)
    ok, buf = cv2.imencode(".webp", cv2.cvtColor(u8, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_WEBP_QUALITY, quality])
    if not ok:
        raise RuntimeError("WebP encoding failed")
    return buf.tobytes()


def _error(status: int, code: str, message: str, hint: str = "") -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message, "hint": hint}})


@app.exception_handler(InputError)
async def _input_error(_: Request, e: InputError):
    return _error(e.status, e.code, e.message, e.hint)


@app.exception_handler(RequestValidationError)
async def _bad_request(_: Request, e: RequestValidationError):
    errs = e.errors()
    field = ".".join(str(p) for p in errs[0].get("loc", [])[1:]) if errs else "request"
    msg = errs[0].get("msg", "invalid value") if errs else "invalid request"
    return _error(422, "invalid_parameter", f"Invalid value for '{field}': {msg}.",
                  "Check the slider values and try again.")


@app.exception_handler(Exception)
async def _unexpected(_: Request, e: Exception):
    log.error("unhandled error: %s\n%s", e, traceback.format_exc())
    return _error(500, "internal", "Something went wrong while processing this image.",
                  "Try again, or pick one of the built-in samples.")


def _model_info() -> dict:
    i = get_model().info
    return {"name": i.name, "kind": i.kind, "is_stub": i.is_stub, "file": i.file, "input": i.input,
            "output": i.output, "params": i.params, "size_mb": i.size_mb, "load_error": i.load_error}


# --------------------------------------------------------------------------- core processing
def process(raw, *, name: str, source: str, sim: SimParams, force_path: str | None, names: list[str],
            key: str | None) -> dict:
    t0 = time.perf_counter()
    sim = sim.clamp()
    raw = simulate(raw, sim)
    t_sim = time.perf_counter()
    out = pipelines.run(raw, names=names, force_path=force_path)
    t_run = time.perf_counter()

    rid = uuid.uuid4().hex[:12]
    flip = raw.flip
    to_encode: dict[str, tuple[np.ndarray, int]] = {}
    naive = pipelines.naive_render(raw)
    to_encode["input"] = (naive, 88)
    if raw.gt is not None:
        to_encode["gt"] = (raw.gt, 90)
    outputs = {}
    for pname, res in out.results.items():
        to_encode[pname] = (res.image, 90)
        stages = []
        for i, s in enumerate(res.stages):
            sname = f"{pname}-stage{i}"
            to_encode[sname] = (s.image, 80)
            stages.append({"name": s.name, "ms": s.ms, "note": s.note, "image_url": f"/api/result/{rid}/{sname}.webp"})
        outputs[pname] = {"image_url": f"/api/result/{rid}/{pname}.webp", "latency_ms": res.latency_ms,
                          "metrics": res.metrics, "stages": stages, "info": res.info}

    futures = {k: _encode_pool.submit(_webp, img, q, flip) for k, (img, q) in to_encode.items()}
    images = {k: f.result() for k, f in futures.items()}
    t_enc = time.perf_counter()

    input_metrics = isp_metrics.compute(naive, raw.gt) if raw.gt is not None and raw.gt.shape == naive.shape else None
    h, w = raw.bayer.shape
    if flip in (5, 6):
        h, w = w, h
    orig = raw.meta.get("orig_size") or [w, h]
    response = {
        "id": rid, "cached": False,
        "input": {"name": name, "kind": raw.meta.get("source", "raw"), "source": source, "width": w, "height": h,
                  "orig_width": int(orig[0]), "orig_height": int(orig[1]),
                  "pattern": str(raw.meta.get("pattern", "RGGB")), "exposure": round(float(raw.exposure), 3),
                  "has_gt": raw.gt is not None, "notes": list(raw.meta.get("notes", [])),
                  "image_url": f"/api/result/{rid}/input.webp",
                  "gt_url": f"/api/result/{rid}/gt.webp" if raw.gt is not None else None,
                  "metrics": input_metrics},
        "simulate": {**sim.as_dict(), "iso": iso_label(sim.noise)},
        "analysis": out.analysis.as_dict(),
        "route": out.route.as_dict(),
        "model": _model_info(),
        "outputs": outputs,
        "timings": {"simulate_ms": round((t_sim - t0) * 1000, 1), "analysis_ms": out.analysis_ms,
                    "pipelines_ms": round((t_run - t_sim) * 1000, 1), "encode_ms": round((t_enc - t_run) * 1000, 1),
                    "total_ms": round((time.perf_counter() - t0) * 1000, 1)},
    }
    store.put(rid, key, response, images)
    return response


def _run_guarded(fn):
    if not _jobs.acquire(timeout=BUSY_TIMEOUT_S):
        raise InputError("busy", "The server is busy with other photos right now.",
                         "Wait a few seconds and try again.", 503)
    try:
        return fn()
    finally:
        _jobs.release()


def _parse_names(pipelines_csv: str) -> list[str]:
    names = [n.strip() for n in pipelines_csv.split(",") if n.strip()]
    bad = [n for n in names if n not in pipelines.PIPELINES]
    if bad or not names:
        raise InputError("bad_pipelines", f"Unknown pipeline(s): {', '.join(bad) or '(none)'}.",
                         f"Use any of: {', '.join(pipelines.PIPELINES)}.", 422)
    return names


# --------------------------------------------------------------------------- routes
@app.get("/api/health", response_model=schemas.Health)
def health():
    return {"status": "ok", "version": VERSION, "model": _model_info(), "max_megapixels": isp_io.MAX_MP,
            "samples": len(registry())}


@app.get("/api/samples", response_model=list[schemas.Sample])
def samples():
    return [{"id": e["id"], "name": e["name"], "description": e["description"], "kind": e["kind"],
             "has_gt": e["has_gt"], "preset": e["preset"], "source": e.get("source", {}),
             "thumb_url": f"/api/samples/{e['id']}/thumb.webp"} for e in registry()]


@app.get("/api/samples/{sample_id}/thumb.webp")
def sample_thumb(sample_id: str):
    if sample_id not in _thumbs:
        raw = load_sample(sample_id)
        img = raw.gt if raw.gt is not None else pipelines.naive_render(raw)
        h, w = img.shape[:2]
        s = 480 / max(h, w)
        small = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
        _thumbs[sample_id] = _webp(small, 82, raw.flip)
    return Response(_thumbs[sample_id], media_type="image/webp", headers={"Cache-Control": "public, max-age=86400"})


@app.post("/api/process", response_model=schemas.ProcessResponse,
          responses={400: {"model": schemas.ErrorResponse}, 413: {"model": schemas.ErrorResponse},
                     415: {"model": schemas.ErrorResponse}, 422: {"model": schemas.ErrorResponse},
                     503: {"model": schemas.ErrorResponse}})
def process_endpoint(
    request: Request,
    file: UploadFile | None = File(None, description="DNG / camera RAW / JPG / PNG"),
    sample_id: str | None = Form(None),
    noise: float = Form(0.0, ge=0, le=1),
    exposure_ev: float = Form(0.0, ge=-5, le=2),
    wb_shift: float = Form(0.0, ge=-1, le=1),
    seed: int = Form(0),
    force_path: str = Form("auto", description="auto | fast | learned | learned_chroma"),
    pipelines_csv: str = Form("traditional,learned,hybrid", alias="pipelines"),
):
    names = _parse_names(pipelines_csv)
    fp = None if force_path in ("", "auto") else force_path
    if fp not in (None, "fast", "learned", "learned_chroma"):
        raise InputError("bad_path", f"Unknown route '{force_path}'.", "Use auto, fast, learned or learned_chroma.", 422)
    sim = SimParams(noise=noise, exposure_ev=exposure_ev, wb_shift=wb_shift, seed=seed).clamp()

    if file is not None and file.filename:
        clen = request.headers.get("content-length")
        if clen and int(clen) > MAX_UPLOAD + 1024 * 1024:
            raise InputError("too_large", f"That file is over {MAX_UPLOAD // (1024 * 1024)} MB.",
                             "Use a smaller photo, or export a JPG from your phone.", 413)
        data = file.file.read(MAX_UPLOAD + 1)
        if len(data) > MAX_UPLOAD:
            raise InputError("too_large", f"That file is over {MAX_UPLOAD // (1024 * 1024)} MB.",
                             "Use a smaller photo, or export a JPG from your phone.", 413)
        digest = hashlib.sha1(data).hexdigest()[:16]
        key = json.dumps([digest, sim.as_dict(), fp, names])
        cached = store.by_key(key)
        if cached:
            return {**cached, "cached": True}
        name = os.path.basename(file.filename)[:120]

        def job():
            t = time.perf_counter()
            raw = isp_io.load_any(data, name)
            r = process(raw, name=name, source="upload", sim=sim, force_path=fp, names=names, key=key)
            r["timings"]["decode_ms"] = round((time.perf_counter() - t) * 1000 - r["timings"]["total_ms"], 1)
            return r

        return _run_guarded(job)

    if sample_id:
        entry = get_entry(sample_id)
        key = json.dumps([f"sample:{sample_id}", sim.as_dict(), fp, names])
        cached = store.by_key(key)
        if cached:
            return {**cached, "cached": True}
        return _run_guarded(lambda: process(load_sample(sample_id), name=entry["name"], source="sample",
                                            sim=sim, force_path=fp, names=names, key=key))

    raise InputError("no_input", "No photo was provided.", "Upload a RAW/DNG or JPG/PNG, or pick a sample.", 400)


@app.get("/api/result/{rid}/{name}.webp")
def result_image(rid: str, name: str):
    data = store.image(rid, name)
    if data is None:
        raise InputError("expired", "This result is no longer in memory.", "Process the photo again.", 404)
    return Response(data, media_type="image/webp", headers={"Cache-Control": "private, max-age=3600"})


@app.get("/api/benchmarks")
def benchmarks():
    if RESULTS_JSON.exists():
        return JSONResponse(json.loads(RESULTS_JSON.read_text(encoding="utf-8")))
    return JSONResponse({"status": "pending", "message": "Evaluation results have not been generated yet."})


# --------------------------------------------------------------------------- warm-up + static site
def _warm():
    """Load the model and precompute every sample at its preset, so first clicks are instant."""
    try:
        get_model()
        for e in registry():
            sim = SimParams(**e["preset"]).clamp()
            key = json.dumps([f"sample:{e['id']}", sim.as_dict(), None, list(pipelines.PIPELINES)])
            if store.by_key(key) is None:
                with _jobs:
                    process(load_sample(e["id"]), name=e["name"], source="sample", sim=sim, force_path=None,
                            names=list(pipelines.PIPELINES), key=key)
            sample_thumb(e["id"])
        log.info("warm-up done: model=%s, %d samples precomputed", get_model().info.name, len(registry()))
    except Exception:  # never block start-up
        log.error("warm-up failed:\n%s", traceback.format_exc())


@app.on_event("startup")
def _startup():
    if os.environ.get("ISP_NO_WARMUP") != "1":
        threading.Thread(target=_warm, daemon=True).start()


if WEB_DIST.exists():
    app.mount("/assets", StaticFiles(directory=WEB_DIST / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        f = (WEB_DIST / path).resolve()
        if path and f.is_file() and WEB_DIST.resolve() in f.parents:
            return FileResponse(f)
        return FileResponse(WEB_DIST / "index.html", headers={"Cache-Control": "no-cache"})
else:
    @app.get("/", include_in_schema=False)
    def root():
        return JSONResponse({"name": "Hybrid ISP API", "docs": "/docs", "note": "web/dist not built yet"})
