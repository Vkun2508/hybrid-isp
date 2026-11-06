# Hybrid ISP: a classical + learned smartphone camera pipeline

**Live demo: https://vkun2508--hybrid-isp-web.us-east.modal.run** · Computer Vision Hackathon 2026, Problem P2

Works on phone and laptop. Hosted on Modal (one always-on CPU container, see `modal_app.py`); the same app runs anywhere with the `Dockerfile`.

Turns noisy, badly exposed, wrongly white-balanced smartphone **RAW** into a clean sRGB photo. Classical
ISP stages do the parts that are exact (black level, white balance, demosaicing, the calibrated colour matrix,
tone curve). A compact learned RAW denoiser does the part classical filters do badly. An adaptive router
decides, photo by photo, how much of the network each image needs, and the app explains why.

![The lab: Traditional vs Hybrid on a simulated night shot, with the pixel loupe](docs/img/lab.jpg)

## What you can do on the live site
- **Pick a real phone RAW** (SIDD Nexus 6 / iPhone 7, from scenes the network never saw), a photo, or **upload your own** DNG/RAW/JPG/PNG.
- **Make it harder** with the sensor simulator: add ISO noise, under-expose by up to 5 stops, or break the white balance.
- **Compare** Traditional | Learned only | Hybrid (and ground truth) with a split view and a **pixel loupe** that shows the same spot in every pipeline at 1:1 to 4×.
- **See the scores** (PSNR, SSIM, ΔE2000, time) and **why the router chose its path** (noise σ, exposure, white-balance trust, dynamic range).
- **Look inside** each pipeline stage by stage, from the Bayer mosaic to the final photo, with the time spent in each stage.
- **Results tab:** full test-set metrics, robustness curves, the ablation study and failure cases.

## Results (held-out SIDD-CC test scenes 004 and 008, 26 images, 5 phones)
_Final numbers are written by `eval/run_eval.py` to [`results/benchmarks.json`](results/benchmarks.json) and `eval_P2_<TeamID>.md`, and shown on the site's Results tab._

## Run it

**One command (Docker):**
```bash
docker build -t hybrid-isp . && docker run -p 7860:7860 hybrid-isp
# open http://localhost:7860
```

**Local development:**
```bash
python -m venv .venv && .venv\Scripts\activate          # Windows (use bin/activate elsewhere)
pip install -r requirements.txt
cd web && npm ci && npm run build && cd ..
uvicorn server.main:app --port 7860                        # API + site on http://localhost:7860
# UI hot reload: cd web && npm run dev  (proxies /api to :7860)
```

No GPU and no PyTorch are needed to run the app: the learned core runs through ONNX Runtime on the CPU.
Without trained weights in `models/`, a classical stand-in with the same interface runs, and the UI says so.

## How it fits together
See **[docs/architecture.md](docs/architecture.md)** (diagram + data contract) and **[docs/design.md](docs/design.md)** (design document).

```
RAW / DNG / JPG ─► isp/io.py ─► simulate.py ─► analysis.py (analyser + router)
                                                   │
          ┌──────────────── isp/pipelines.py ──────┼─────────────────────────────┐
          │ Traditional: WB → Malvar → bilateral → CCM → tone → sharpen          │
          │ Learned:     packed RAW → network → minimal render                    │
          │ Hybrid:      network (RAW denoise) → router WB + gain → Malvar → CCM  │
          │              → tone → chroma cleanup → noise-aware sharpen            │
          └─────────────── isp/model_slot.py ◄── models/model.json + *.onnx ──────┘
FastAPI (server/) serves the API and the React site (web/), in one Docker image.
```

## Reproduce the model and the numbers
1. **Train** on a free GPU: run [`training/SIDD_CC_RAW_Complete_Pipeline.ipynb`](training/SIDD_CC_RAW_Complete_Pipeline.ipynb) on Kaggle or Colab (T4). It downloads SIDD-CC (~7 GB), splits by scene (7 train / 1 val / 2 test), trains `LightweightISPUNet` (1.0 M parameters), then exports ONNX FP32 + static INT8 and `exported_model.zip`. `ISP_NB_QUICK=1` runs a 2-minute CPU dry run of every cell.
2. **Plug it in:** copy `isp_unet*.onnx` and `model.json` into `models/`, then run `python scripts/check_model.py`.
3. **Evaluate** (dev requirements: `pip install -r requirements-dev.txt`):
   ```bash
   python eval/prepare_testset.py              # cache the 26 held-out test pairs
   python eval/prepare_testset.py --split val  # cache the validation scene
   python eval/tune_router.py                  # router thresholds, validation scene only
   python eval/run_eval.py --team-id <ID> --usns "<USN1>,<USN2>,<USN3>,<USN4>"
   ```

## Repository layout
```
isp/            classical stages, analyser + router, model slot, pipelines, metrics, input loading
server/         FastAPI app (API + static site)            web/        React + Vite + TypeScript UI
models/         model.json + ONNX weights (the slot)        samples/    demo inputs (SIDD test scenes, photos)
training/       training notebook, model class, notebook patcher
eval/           test/val set builders, router tuning, full evaluation   results/   benchmarks.json, failure images
scripts/        export_onnx.py, check_model.py, make_samples.py, deploy_space.py
docs/           architecture.md, design.md, failure_cases.md
```

## Limitations
- The network **denoises RAW**; colour is classical and calibrated rather than learned (see the design doc §8).
- sRGB ground truth is the clean SIDD RAW rendered with our own colour and tone stages: it compares the pipelines fairly with each other, but not with numbers from other papers.
- SIDD has 10 physical scenes; the test split holds 2 of them (26 images, 5 phones).
- The web demo bins inputs to ≤ 2 MP for speed; the official metrics are computed offline on full 1536×1024 crops.

## Privacy and ethics
Uploads are processed in memory and never stored, and outputs carry no metadata. The data is public: SIDD (research use) and public-domain test images. See the design doc §9.

## Credits
- **SIDD:** A. Abdelhamed, S. Lin, M. S. Brown, CVPR 2018 (SIDD-CC compact release on Hugging Face, `MhdMohammadi/SIDD-CC`).
- **Unprocessing:** T. Brooks et al., CVPR 2019. **Demosaicing:** Malvar, He, Cutler, ICASSP 2004. **LPIPS:** Zhang et al., CVPR 2018.
- **Photos:** scikit-image test images (public domain / CC0). **Fonts:** Public Sans and IBM Plex Mono (SIL OFL), self-hosted.
- **Model:** `LightweightISPUNet` and the SIDD-CC training pipeline by the team's model lead (`training/`).
