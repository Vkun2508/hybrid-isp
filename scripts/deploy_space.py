"""Deploy the app to a Hugging Face Space (Docker SDK). Needs `hf auth login` (write token) first.

    python scripts/deploy_space.py --space <username>/hybrid-isp

Uploads only what the Docker image needs; Hugging Face builds the Dockerfile and serves it at
https://<username>-hybrid-isp.hf.space . Re-run to redeploy (only changed files are uploaded).
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SPACE_HEADER = """---
title: Hybrid ISP
emoji: 📷
colorFrom: gray
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
short_description: Classical + learned smartphone camera ISP, compared live
---

"""

ALLOW = [
    "Dockerfile", ".dockerignore", "requirements.txt",
    "isp/**", "server/**", "models/**", "samples/**", "results/**",
    "web/package.json", "web/package-lock.json", "web/index.html", "web/vite.config.ts", "web/tsconfig.json",
    "web/public/**", "web/src/**",
]
IGNORE = ["**/__pycache__/**", "**/*.pyc", "models/_eval_*", "models/*.pth"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", required=True, help="<username>/<space-name>")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    from huggingface_hub import HfApi

    api = HfApi()
    who = api.whoami()["name"]
    print(f"signed in as {who}")
    api.create_repo(args.space, repo_type="space", space_sdk="docker", private=args.private, exist_ok=True)
    readme = (ROOT / "README.md").read_text(encoding="utf-8") if (ROOT / "README.md").exists() else "# Hybrid ISP\n"
    api.upload_file(path_or_fileobj=io.BytesIO((SPACE_HEADER + readme).encode("utf-8")), path_in_repo="README.md",
                    repo_id=args.space, repo_type="space", commit_message="README with Space metadata")
    api.upload_folder(folder_path=str(ROOT), repo_id=args.space, repo_type="space", allow_patterns=ALLOW,
                      ignore_patterns=IGNORE, commit_message="Deploy hybrid ISP app")
    sub = args.space.replace("/", "-").replace("_", "-").lower()
    print(f"\nSpace:   https://huggingface.co/spaces/{args.space}")
    print(f"Live URL: https://{sub}.hf.space   (first build takes ~5-10 min; watch the Space's 'Logs' tab)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
