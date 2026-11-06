"""Tune the hybrid router's thresholds on the VALIDATION scene only (SIDD-CC scene 003).

The test scenes (004, 008) are never touched here, so the numbers in eval/run_eval.py stay unbiased.
Coordinate descent over each threshold, scoring the hybrid's mean PSNR (ties broken by dE) on the
validation images under mixed conditions. Writes isp/router_config.json (+ results/router_tuning.json).

    python eval/prepare_testset.py --split val     # once
    python eval/tune_router.py                     # uses models/model.json (re-run when the model changes)
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CACHE_VAL, ROOT, VAL_SCENES, load_pair  # noqa: E402
from run_eval import de2000  # noqa: E402

from isp import analysis, model_slot, pipelines  # noqa: E402
from isp import metrics as M  # noqa: E402
from isp.simulate import SimParams, simulate  # noqa: E402

CONDITIONS = [
    ("as captured", SimParams()),
    ("extra noise", SimParams(noise=0.5)),
    ("under-exposed 2 EV", SimParams(exposure_ev=-2.0)),
    ("WB error", SimParams(wb_shift=0.6)),
    ("noise + under-exposed", SimParams(noise=0.4, exposure_ev=-1.0)),
]
GRID = {
    "wb": [(5.0, 15.0), (8.0, 16.0), (10.0, 20.0), (12.0, 24.0), (15.0, 30.0)],
    "sharpen_k": [0.0, 0.15, 0.3, 0.5],
    "fast_sigma": [0.002, 0.004, 0.008, 0.016],
    "high_sigma": [0.02, 0.03, 0.05],
    "ev_threshold": [0.4, 0.6, 1.0],
}


def evaluate(cfg: analysis.RouterConfig, cases, model) -> tuple[float, float]:
    analysis.CONFIG = cfg
    ps, des = [], []
    for raw, deg in cases:
        out = pipelines.run(deg, names=["hybrid"], model=model, with_metrics=False)
        img = np.clip(out.results["hybrid"].image, 0, 1)
        ps.append(M.psnr(img, raw.gt))
        des.append(de2000(img, raw.gt))
    return float(np.mean(ps)), float(np.mean(des))


def apply(cfg: analysis.RouterConfig, key: str, val) -> analysis.RouterConfig:
    if key == "wb":
        return replace(cfg, wb_angle_lo=val[0], wb_angle_hi=val[1])
    return replace(cfg, **{key: val})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12, help="validation images to use")
    ap.add_argument("--rounds", type=int, default=2)
    args = ap.parse_args()
    pairs = sorted(CACHE_VAL.glob("*.npz"))[: args.n]
    if not pairs:
        print("No validation pairs. Run: python eval/prepare_testset.py --split val")
        return 1
    model = model_slot.load_model()
    print(f"model: {model.info.name} (stub={model.is_stub}); {len(pairs)} validation images x {len(CONDITIONS)} conditions")
    cases = []
    for j, p in enumerate(pairs):
        raw = load_pair(p)[0]
        for _, sp in CONDITIONS:
            cases.append((raw, simulate(raw, replace(sp, seed=300 + j))))

    t0 = time.time()
    best = analysis.RouterConfig()
    best_score = evaluate(best, cases, model)
    start = best_score
    print(f"defaults: PSNR {best_score[0]:.3f} dB, dE {best_score[1]:.3f}")
    history = [{"config": asdict(best), "psnr": best_score[0], "delta_e": best_score[1], "step": "defaults"}]
    for rnd in range(args.rounds):
        changed = False
        for key, values in GRID.items():
            for v in values:
                cand = apply(best, key, v)
                if cand == best:
                    continue
                sc = evaluate(cand, cases, model)
                history.append({"config": asdict(cand), "psnr": sc[0], "delta_e": sc[1], "step": f"round {rnd + 1}: {key}={v}"})
                better = sc[0] > best_score[0] + 1e-3 or (abs(sc[0] - best_score[0]) <= 1e-3 and sc[1] < best_score[1])
                if better:
                    best, best_score, changed = cand, sc, True
                print(f"  r{rnd + 1} {key}={v}: PSNR {sc[0]:.3f}  dE {sc[1]:.3f}{'  <- best' if better else ''}", flush=True)
        if not changed:
            break

    best = replace(best, tuned_on=(f"SIDD-CC validation scene {', '.join(VAL_SCENES)} ({len(pairs)} images x "
                                   f"{len(CONDITIONS)} conditions), model '{model.info.name}', "
                                   f"{dt.datetime.now():%Y-%m-%d %H:%M}"))
    (ROOT / "isp" / "router_config.json").write_text(json.dumps(asdict(best), indent=2), encoding="utf-8")
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "router_tuning.json").write_text(json.dumps({
        "validation_images": [p.stem for p in pairs], "conditions": [c for c, _ in CONDITIONS],
        "start": {"psnr": start[0], "delta_e": start[1]}, "best": {"psnr": best_score[0], "delta_e": best_score[1]},
        "config": asdict(best), "history": history}, indent=2), encoding="utf-8")
    print(f"\nbest: PSNR {best_score[0]:.3f} dB (from {start[0]:.3f}), dE {best_score[1]:.3f} (from {start[1]:.3f}) "
          f"in {time.time() - t0:.0f}s\n{asdict(best)}\nwrote isp/router_config.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
