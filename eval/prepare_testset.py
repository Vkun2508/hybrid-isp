"""Download the held-out SIDD-CC test scenes (004, 008): one pair per instance, centre-cropped, cached.

    python eval/prepare_testset.py            # all 26 test instances (~3 GB download, ~150 MB cached)
    python eval/prepare_testset.py --limit 8
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CACHE, CACHE_VAL, ROOT, VAL_SCENES, fetch_crop, test_instances  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--split", choices=["test", "val"], default="test")
    ap.add_argument("--download-dir", default=str(ROOT / ".cache" / "sidd"))
    args = ap.parse_args()
    inst = test_instances(VAL_SCENES) if args.split == "val" else test_instances()
    cache = CACHE_VAL if args.split == "val" else CACHE
    if args.limit:
        inst = inst[: args.limit]
    t0 = time.time()
    for i, name in enumerate(inst, 1):
        p = fetch_crop(name, Path(args.download_dir), cache=cache)
        print(f"[{i}/{len(inst)}] {name} -> {p.name} ({time.time() - t0:.0f}s)", flush=True)
    print(f"done: {len(list(cache.glob('*.npz')))} pairs cached in {cache}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
