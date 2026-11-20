"""Precompute the Overview hero ("Lights down") from the real pipeline.

    .venv/Scripts/python.exe scripts/make_hero_assets.py            # local server must be running on :7860

For one unseen RAW (Nexus 6, ISO 3200) at 0, -1, -2, -3 and -4 EV, this asks the running API for the
traditional, network-only and hybrid results, then writes them as static files the site serves directly:

    web/public/hero/hero.json          metrics per pipeline per light level, route, sizes
    web/public/hero/d/<pipe>_<k>.webp  desktop: full frame 1536x1024
    web/public/hero/m/<pipe>_<k>.webp  phone: centre crop 768x1024 (phones only show the middle)
    web/public/hero/{d,m}/sensor.webp  small sensor image, only used to draw the RGGB mosaic

Static files mean the hero starts instantly and never waits on a cold server. The numbers are the same
ones the live API returns for these settings (fixed seed), which "Run it live" in the hero lets anyone check.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import urllib.request
import uuid
from datetime import date
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PIPES = ("traditional", "learned", "hybrid")


def post_form(url: str, fields: dict[str, str]) -> dict:
    boundary = uuid.uuid4().hex
    body = b"".join(
        f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode() for k, v in fields.items()
    ) + f"--{boundary}--\r\n".encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.load(r)


def get_image(url: str) -> Image.Image:
    with urllib.request.urlopen(url, timeout=60) as r:
        return Image.open(io.BytesIO(r.read())).convert("RGB")


def centre_crop(im: Image.Image, w: int, h: int, cx: float = 0.5) -> Image.Image:
    """Crop to w:h at full resolution, centred at cx (fraction of the width), then size to w x h."""
    iw, ih = im.size
    if iw / ih > w / h:
        cw = round(ih * w / h)
        x = min(iw - cw, max(0, round(cx * iw - cw / 2)))
        im = im.crop((x, 0, x + cw, ih))
    else:
        ch = round(iw * h / w)
        y = (ih - ch) // 2
        im = im.crop((0, y, iw, y + ch))
    return im if im.size == (w, h) else im.resize((w, h), Image.LANCZOS)


def save(im: Image.Image, path: Path, q: int) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path, "WEBP", quality=q, method=6)
    return path.stat().st_size


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://127.0.0.1:7860")
    ap.add_argument("--sample", default="sidd-n6-iso3200-lowlight")
    ap.add_argument("--levels", default="0,-1,-2,-3,-4")
    ap.add_argument("--out", default=str(ROOT / "web" / "public" / "hero"))
    # phones only see a portrait slice: centre it on the subject (the big chess piece sits at ~30% of the width)
    ap.add_argument("--phone-cx", type=float, default=0.3)
    args = ap.parse_args()

    api = args.api.rstrip("/")
    out = Path(args.out)
    levels = [float(x) for x in args.levels.split(",")]
    health = json.load(urllib.request.urlopen(f"{api}/api/health", timeout=30))
    samples = json.load(urllib.request.urlopen(f"{api}/api/samples", timeout=30))
    samples = samples.get("samples", samples) if isinstance(samples, dict) else samples
    sample = next(s for s in samples if s["id"] == args.sample)

    metrics: dict[str, list[dict]] = {p: [] for p in PIPES}
    routes: list[str] = []
    sizes = {"d": 0, "m": 0}
    dims = {}
    for k, ev in enumerate(levels):
        r = post_form(f"{api}/api/process", {
            "sample_id": args.sample, "noise": "0", "exposure_ev": str(ev), "wb_shift": "0", "force_path": "auto",
        })
        routes.append(r["route"]["path"])
        row = []
        for p in PIPES:
            o = r["outputs"][p]
            m = o["metrics"]
            metrics[p].append({key: round(float(m[key]), 3) for key in ("psnr", "ssim", "delta_e") if key in m})
            im = get_image(api + o["image_url"])
            dims["d"] = list(im.size)
            sizes["d"] += save(im, out / "d" / f"{p}_{k}.webp", 62)
            mob = centre_crop(im, 768, 1024, args.phone_cx)
            dims["m"] = list(mob.size)
            sizes["m"] += save(mob, out / "m" / f"{p}_{k}.webp", 65)
            row.append(f"{p} {m['psnr']:.1f}")
        if k == 0:  # the sensor view, only sampled for the mosaic, so small is enough
            sen = get_image(api + r["input"]["image_url"])
            sizes["d"] += save(sen.resize((768, round(768 * sen.height / sen.width)), Image.LANCZOS), out / "d" / "sensor.webp", 80)
            sizes["m"] += save(centre_crop(sen, 384, 512, args.phone_cx), out / "m" / "sensor.webp", 80)
        print(f"EV {ev:+.0f}: {r['route']['path']:<16} " + " | ".join(row))

    src = sample.get("source", {})
    data = {
        "sample": args.sample,
        "name": sample.get("name"),
        "camera": {"N6": "Nexus 6", "IP": "iPhone 7"}.get(src.get("camera"), src.get("camera")),
        "iso": src.get("iso"),
        "dataset": src.get("dataset"),
        "split": src.get("split"),
        "levels": levels,
        "metrics": metrics,
        "route": routes,
        "size": dims,
        "model": {k: health["model"].get(k) for k in ("name", "params")},
        "generated": date.today().isoformat(),
    }
    (out / "hero.json").write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"desktop set {sizes['d'] / 1e6:.2f} MB · phone set {sizes['m'] / 1e6:.2f} MB -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
