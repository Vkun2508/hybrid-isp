"""Fill the results sections of README.md and docs/design.md from results/benchmarks.json.

    python scripts/update_docs.py

Replaces the text between <!-- results:start --> and <!-- results:end --> in each file, so the docs
always show exactly the numbers the evaluation produced (and that the website's Results tab shows).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NAMES = {"input": "Input (no ISP)", "traditional": "Traditional ISP", "learned": "Learned only", "hybrid": "**Hybrid (ours)**"}


def table(b: dict) -> str:
    q = {r["pipeline"]: r for r in b["quality"]}
    best = {
        "psnr": max(q[p]["psnr"] for p in ("traditional", "learned", "hybrid")),
        "ssim": max(q[p]["ssim"] for p in ("traditional", "learned", "hybrid")),
        "delta_e": min(q[p]["delta_e"] for p in ("traditional", "learned", "hybrid")),
        "lpips": min((q[p]["lpips"] for p in ("traditional", "learned", "hybrid") if q[p].get("lpips") is not None), default=None),
    }

    def cell(p, k, fmt):
        v = q[p].get(k)
        if v is None:
            return "n/a"
        s = format(v, fmt)
        return f"**{s}**" if p != "input" and v == best[k] else s

    rows = ["| Pipeline | PSNR (dB) ↑ | SSIM ↑ | ΔE2000 ↓ | LPIPS ↓ | Time / image |", "|---|---:|---:|---:|---:|---:|"]
    lat = b.get("extra", {}).get("latency_ms_per_image", {})
    for p in ("input", "traditional", "learned", "hybrid"):
        t = f"{lat[p]:.0f} ms" if p in lat else "n/a"
        rows.append(f"| {NAMES[p]} | {cell(p, 'psnr', '.2f')} | {cell(p, 'ssim', '.4f')} | {cell(p, 'delta_e', '.2f')} | {cell(p, 'lpips', '.4f')} | {t} |")
    return "\n".join(rows)


def efficiency(b: dict) -> str:
    e = b.get("efficiency") or []
    if not e:
        return ""
    rows = ["| Model | Parameters | Size | CPU latency, 1 MP RAW |", "|---|---:|---:|---:|"]
    for r in e:
        rows.append(f"| {r['variant']} | {r['params']:,} | {r['size_mb']:.2f} MB | {r['latency_ms_1mp']:.0f} ms |")
    return "\n".join(rows)


def robustness(b: dict) -> str:
    out = []
    for rb in b.get("robustness", []):
        s = {x["pipeline"]: x["psnr"] for x in rb["series"]}
        last = len(rb["x"]) - 1
        out.append(f"- **{rb['title']}** (worst level, {rb['x'][last]} {rb['x_label']}): hybrid {s['hybrid'][last]:.2f} dB, "
                   f"learned only {s['learned'][last]:.2f} dB, traditional {s['traditional'][last]:.2f} dB.")
    return "\n".join(out)


def block(b: dict) -> str:
    d = b["dataset"]
    status = ("" if b.get("status") == "final" else
              "> **Provisional:** measured with the classical stand-in; the trained network was not yet available.\n\n")
    raw = b.get("extra", {}).get("raw_domain_psnr", {})
    rawline = (f"\nRAW-domain denoising by the network alone (packed RGGB vs clean RAW): **{raw['noisy']:.2f} → {raw['denoised']:.2f} dB**.\n"
               if raw.get("noisy") and raw.get("denoised") else "")
    parts = [status + f"Held-out SIDD-CC test scenes {', '.join(d['test_scenes'])}: {d['n_images']} images from 5 phones, "
             f"{d.get('crop', '')} crops, never seen in training or tuning. Model: {b['model']['name']} (`{b['model']['variant']}`). "
             f"Updated {b['updated']}.", "", table(b), rawline]
    eff = efficiency(b)
    if eff:
        parts += ["**Efficiency** (optimisation: static INT8 post-training quantisation):", "", eff, ""]
    rob = robustness(b)
    if rob:
        parts += ["**Robustness** (mean PSNR over the robustness subset as each degradation grows):", "", rob, ""]
    parts += ["Full tables, curves, the ablation study and failure cases: the website's **Results** tab, "
              "[`results/benchmarks.json`](results/benchmarks.json) and [`docs/failure_cases.md`](docs/failure_cases.md)."]
    return "\n".join(parts).strip()


def fill(path: Path, content: str) -> bool:
    s = path.read_text(encoding="utf-8")
    pat = re.compile(r"(<!-- results:start -->)(.*?)(<!-- results:end -->)", re.S)
    if not pat.search(s):
        print(f"no results markers in {path.name}")
        return False
    path.write_text(pat.sub(lambda m: m.group(1) + "\n" + content + "\n" + m.group(3), s), encoding="utf-8")
    print(f"updated {path.relative_to(ROOT)}")
    return True


def main() -> int:
    b = json.loads((ROOT / "results" / "benchmarks.json").read_text(encoding="utf-8"))
    content = block(b)
    ok = fill(ROOT / "README.md", content)
    ok &= fill(ROOT / "docs" / "design.md", content.replace("](results/", "](../results/").replace("](docs/", "]("))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
