"""Contract test for the model slot: does models/model.json + its file plug into the app?

    python scripts/check_model.py                      # whatever models/model.json points at (or the stub)
    python scripts/check_model.py --manifest other.json

Loads exactly like the server does, runs a synthetic noisy RAW through it at a few sizes
(including odd ones, to exercise padding and tiling), checks shape / finiteness, and prints
params, file size and median latency at 1 MP RAW. Exit code 0 = safe to deploy.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from isp import model_slot  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(model_slot.MANIFEST_PATH))
    args = ap.parse_args()

    m = model_slot.load_model(Path(args.manifest))
    info = m.info
    print(f"model    : {info.name}")
    print(f"backend  : {info.kind}{'  (STUB - no real weights loaded)' if info.is_stub else ''}")
    if info.load_error:
        print(f"LOAD ERROR: {info.load_error}")
        return 1
    print(f"contract : input={m.m.input} (wb={m.m.input_wb}) -> output={m.m.output}, upscale={m.m.upscale}, "
          f"pad={m.m.pad_multiple}, tile={m.m.tile}")
    print(f"params   : {info.params if info.params is not None else 'unknown'} | file: {info.size_mb} MB")

    rng = np.random.default_rng(0)
    cin = 4 if m.m.input == "raw4" else 3
    cout = 4 if m.m.output == "raw4" else 3
    ok = True
    for h, w in [(64, 96), (137, 219), (300, 520)]:
        clean = np.kron(rng.random((h // 8 + 1, w // 8 + 1, cin)), np.ones((8, 8, 1)))[:h, :w].astype(np.float32) * 0.4
        noisy = clean + rng.normal(0, 0.02, clean.shape).astype(np.float32)
        try:
            y = m.run(noisy)
        except Exception as e:
            print(f"  {h}x{w}: FAILED - {e}")
            ok = False
            continue
        good = y.shape == (h * m.m.upscale, w * m.m.upscale, cout) and np.isfinite(y).all()
        err_in = float(np.sqrt(np.mean((noisy - clean) ** 2)))
        err_out = float(np.sqrt(np.mean((y[..., :cin] - clean) ** 2))) if cout == cin and m.m.upscale == 1 else float("nan")
        print(f"  {h}x{w}x{cin} -> {y.shape}  {'OK' if good else 'BAD SHAPE/NaN'}  "
              f"rmse noisy={err_in:.4f} out={err_out:.4f}")
        ok &= good

    x1 = rng.random((512, 512, cin)).astype(np.float32) * 0.3  # 1 MP RAW = 512x512 packed
    m.run(x1)
    t = []
    for _ in range(5):
        t0 = time.perf_counter()
        m.run(x1)
        t.append((time.perf_counter() - t0) * 1000)
    print(f"latency  : {np.median(t):.0f} ms median at 1 MP RAW (this machine)")
    print("RESULT   :", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
