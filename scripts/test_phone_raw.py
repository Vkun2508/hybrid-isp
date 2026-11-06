"""Run real phone RAW/DNG files through the app's loader and pipelines; record what was read and check it.

    python scripts/test_phone_raw.py .cache/phone_raw         # writes results/phone_raw.json + preview strips

For each file: the CFA pattern, black/white levels, as-shot white balance and colour matrix LibRaw reports;
how our loader normalised it; automatic sanity checks (blacks at zero, highlights not crushed, no extreme
cast); timings; and a preview strip (sensor input | traditional | hybrid) for a human to judge colour.
"""
from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from isp import pipelines  # noqa: E402
from isp.io import InputError, load_any  # noqa: E402
from isp.raw import pack  # noqa: E402

OUT = ROOT / "results" / "phone_raw"


def libraw_meta(data: bytes) -> dict:
    import rawpy

    with rawpy.imread(io.BytesIO(data)) as r:
        pat = r.raw_pattern
        desc = r.color_desc.decode(errors="ignore")
        cfa = "".join(desc[pat[i, j]] for i in range(2) for j in range(2)) if pat is not None and pat.shape == (2, 2) else (
            f"{pat.shape[0]}x{pat.shape[1]} (non-Bayer)" if pat is not None else "none (linear)")
        m = np.asarray(r.rgb_xyz_matrix)[:3]
        return {"raw_type": str(r.raw_type).split(".")[-1], "cfa": cfa, "colors": int(r.num_colors),
                "black_level_per_channel": [int(v) for v in r.black_level_per_channel], "white_level": int(r.white_level),
                "as_shot_wb": [round(float(v), 4) for v in r.camera_whitebalance],
                "has_color_matrix": bool(np.abs(m).sum() > 1e-6), "sensor_size": [int(r.sizes.raw_width), int(r.sizes.raw_height)],
                "flip": int(r.sizes.flip)}


def checks(raw, out: np.ndarray, camera: np.ndarray) -> dict:
    q = pack(raw.bayer)
    g = 0.5 * (q[..., 1] + q[..., 2])
    p01, p999 = float(np.percentile(g, 0.1)), float(np.percentile(g, 99.9))
    means = out.reshape(-1, 3).mean(0)
    cam = camera.reshape(-1, 3).mean(0)
    cos = float(means @ cam / (np.linalg.norm(means) * np.linalg.norm(cam) + 1e-12))
    vs_camera = float(np.degrees(np.arccos(np.clip(cos, -1, 1))))
    cast = (float(means[0] / max(means[1], 1e-6)), float(means[2] / max(means[1], 1e-6)))
    clipped = float((out >= 0.995).any(axis=-1).mean())
    return {
        "black_at_zero": abs(p01) < 0.03, "p0.1_green": round(p01, 4),
        "highlights_in_range": p999 <= 1.0001, "p99.9_green": round(p999, 4),
        "output_rb_over_g": [round(cast[0], 3), round(cast[1], 3)],
        "colour_vs_camera_wb_deg": round(vs_camera, 2),
        "colour_matches_camera_wb": vs_camera < 6.0,  # hybrid may refine WB, never swing it
        "output_clipped_share": round(clipped, 4), "not_overexposed": clipped < 0.2,
    }


def strip(images: list[np.ndarray], flip: int, path: Path, height: int = 360) -> None:
    from server.main import _orient

    tiles = []
    for im in images:
        im = _orient(np.clip(im, 0, 1), flip)
        h, w = im.shape[:2]
        tiles.append(cv2.resize(im, (int(w * height / h), height), interpolation=cv2.INTER_AREA))
    bar = np.full((height, 6, 3), 0.09, np.float32)
    row = np.concatenate([t for pair in zip(tiles, [bar] * len(tiles)) for t in pair][:-1], axis=1)
    cv2.imwrite(str(path), cv2.cvtColor((row * 255).astype(np.uint8), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])


def main() -> int:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / ".cache" / "phone_raw"
    OUT.mkdir(parents=True, exist_ok=True)
    report = []
    for f in sorted(src.glob("*")):
        if f.suffix.lower() not in (".dng", ".nef", ".cr2", ".cr3", ".arw", ".orf", ".rw2", ".raf", ".pef", ".srw"):
            continue
        data = f.read_bytes()
        entry = {"file": f.name, "size_mb": round(len(data) / 1e6, 1)}
        try:
            entry["libraw"] = libraw_meta(data)
            t0 = time.perf_counter()
            raw = load_any(data, f.name)
            entry["decode_ms"] = round((time.perf_counter() - t0) * 1000)
            out = pipelines.run(raw, names=["traditional", "hybrid"], with_metrics=False)
            entry["loaded"] = {"pattern": raw.meta.get("pattern"), "size": list(raw.bayer.shape[::-1]),
                               "exposure": round(raw.exposure, 2), "wb_gains": [round(v, 3) for v in (raw.wb_gains or ())],
                               "notes": raw.meta.get("notes", [])}
            entry["route"] = {"path": out.route.path, "wb": out.route.wb_method, "sigma": round(out.analysis.sigma, 5)}
            entry["latency_ms"] = {p: r.latency_ms for p, r in out.results.items()}
            entry["checks"] = checks(raw, out.results["hybrid"].image, out.results["traditional"].image)
            entry["passed"] = all(v for k, v in entry["checks"].items() if isinstance(v, bool))
            img = OUT / (f.stem + ".jpg")
            strip([pipelines.naive_render(raw), out.results["traditional"].image, out.results["hybrid"].image], raw.flip, img)
            entry["preview"] = f"results/phone_raw/{img.name}"
        except InputError as e:
            entry.update({"handled_error": e.code, "message": e.message, "passed": e.code in ("xtrans", "unsupported_cfa")})
        except Exception as e:  # a crash is a failure of the test
            entry.update({"crash": f"{type(e).__name__}: {e}", "passed": False})
        report.append(entry)
        lib = entry.get("libraw", {})
        print(f"{f.name:26s} {lib.get('raw_type', '?'):6s} cfa={lib.get('cfa', '?'):22s} black={lib.get('black_level_per_channel')} "
              f"white={lib.get('white_level')} -> {entry.get('loaded', {}).get('size')} exp x{entry.get('loaded', {}).get('exposure')} "
              f"route={entry.get('route', {}).get('path')} {'PASS' if entry.get('passed') else 'FAIL'} "
              f"{entry.get('checks', {}) if not entry.get('passed') else ''}{entry.get('crash', '')}", flush=True)
    (ROOT / "results" / "phone_raw.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n{sum(e.get('passed', False) for e in report)}/{len(report)} files passed -> results/phone_raw.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
