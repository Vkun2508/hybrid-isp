# Design document: Hybrid smartphone ISP (Problem P2)

## 1. Problem
Turn degraded smartphone RAW sensor data (noise, colour distortion, varying illumination, white-balance errors) into a high-quality sRGB image with good colour accuracy, low noise and good detail, at a compute cost that suits a phone. The solution must be **hybrid**: classical ISP stages and learned components working together, compared against a traditional pipeline and against the network alone.

## 2. Goals and success measures
| Goal | Measure (in the brief's order) |
|---|---|
| Fidelity | PSNR, SSIM against the clean ground truth |
| Colour accuracy | mean CIEDE2000 (ΔE) |
| Perceptual quality | LPIPS (AlexNet) |
| Model cost | parameters, model size |
| Speed and memory | CPU inference latency at 1 MP, peak memory |
| Robustness | PSNR as noise, under-exposure and WB error grow |

## 3. Data
- **SIDD-CC** (Abdelhamed et al., CVPR 2018; compact release on Hugging Face `MhdMohammadi/SIDD-CC`): paired noisy/clean RAW from 5 phones (Galaxy S6, iPhone 7, Pixel, Nexus 6, LG G4), 10 physical scenes, ISO 50–10 000, with metadata (CFA pattern, black/white level, AsShotNeutral, ColorMatrix2).
- **Split by physical scene** (seed 42): 7 train / 1 validation (scene 003) / 2 test (scenes 004, 008). No scene appears in two splits, so the test numbers measure generalisation to unseen scenes.
- **Router thresholds are tuned on the validation scene only** (`eval/tune_router.py`); the test scenes are used once, by `eval/run_eval.py`.
- Photos uploaded as JPG/PNG are converted to simulated RAW by **unprocessing** (Brooks et al., CVPR 2019): inverse tone curve, inverse gamma, inverse colour matrix, inverse white balance, Bayer mosaic. The original photo then serves as ground truth.

## 4. Architecture
See [architecture.md](architecture.md) for the diagram and the data contract.

### 4.1 Classical stages (`isp/classical/`)
| Stage | Method | Why |
|---|---|---|
| RAW preprocessing | black level per CFA channel, normalise to the white level, align any Bayer pattern to RGGB, keep negative noise values | exact and cheap; the network needs unclipped noise statistics |
| White balance | as-shot gains (metadata); gray-world and shades-of-gray (p=6) estimators | metadata is usually right; estimators catch the cases where it is not |
| Demosaic | Malvar-He-Cutler 5×5 (2004); bilinear for the learned-only baseline | high quality at linear cost |
| Denoise | noise-aware bilateral (traditional), guided filter (fast path), chroma-only bilateral (high noise) | edge-preserving; chroma-only cleanup removes colour blotches without softening detail |
| Colour | camera→sRGB matrix from `ColorMatrix2` (inverse of the row-normalised `ColorMatrix2 · sRGB→XYZ`) | calibrated colour, per camera |
| Tone | exact sRGB transfer function + smoothstep S-curve (16-bit LUT); CLAHE local tone for very high dynamic range | the forward model is exactly invertible, so unprocessed photos round-trip |
| Sharpen | unsharp mask on luma only, amount `k / (1 + σ/σ0)` | sharpening never amplifies colour noise; it backs off on noisy images |

### 4.2 Learned component
- **LightweightISPUNet** (team, `training/unet_isp.py`): residual U-Net on packed RAW (`[R, G1, G2, B]`, H/2×W/2), 4 levels, depthwise-separable residual blocks, PixelShuffle upsampling, global residual (`clean = noisy + f(noisy)`). **1,004,996 parameters (3.83 MB FP32).**
- Trained on SIDD-CC patches (128×128 packed): Charbonnier + 0.1·(1−SSIM) loss, AdamW, cosine schedule, mixed precision, CFA-safe flips and rotations, early stopping on validation PSNR, resumable checkpoints.
- **Optimisation:** exported to ONNX (parity with PyTorch below 1e-4), then **static INT8 post-training quantisation** (ONNX Runtime QDQ, per-channel weights, calibrated on validation patches). The final residual add stays in FP32 so quantisation noise is not added to the image. FP16 is benchmarked on GPU in the notebook.

### 4.3 Hybrid integration and the adaptive router (`isp/analysis.py`, `isp/pipelines.py`)
The analyser measures the RAW before anything else runs:
- the noise σ (Immerkær Laplacian with a median, so it ignores texture);
- the exposure offset (the same highlight-aware auto-exposure used for the scene, on 4×-binned data);
- trust in the white balance (angle between the as-shot and shades-of-gray illuminants);
- the dynamic range (floored at the noise level).

The router then decides:

| Condition | Action |
|---|---|
| visible noise σ_eff < `fast_sigma` | fast path: skip the network, use a guided filter |
| `fast_sigma` ≤ σ_eff < `high_sigma` | learned core |
| σ_eff ≥ `high_sigma` | learned core + chroma-only cleanup, weaker sharpening |
| under-exposed beyond `ev_threshold` | digital gain (≤ ×16) **after** denoising |
| WB disagreement > `wb_angle_lo` **and** the estimators agree (spread ≤ `wb_spread_max`) | blend towards shades-of-gray; replace beyond `wb_angle_hi` |
| WB estimators disagree with each other (colourful scene) | keep the camera's as-shot WB |
| no WB metadata | shades-of-gray |
| dynamic range > `local_tone_dr` stops | local tone mapping |
| always, after the network | **noise-model trust region:** a correction is limited to `residual_k`·σ(x) |

Every decision is reported as a sentence ("why this route") in the UI. Thresholds live in `isp/router_config.json`, tuned on the validation scene only by `eval/tune_router.py`.

**Two classical safeguards around the network** (both found on real data, see [failure_cases.md](failure_cases.md)):
- **Noise-model trust region:**
  - The sensor's noise-level function σ²(x) = a·x + b is fitted from each image (`estimate_nlf`): the local high-pass variance per brightness bin, flat regions only.
  - The network may change a pixel by at most k·σ(x).
  - So it can remove noise but cannot erase real detail or shift highlights. It fixed the network's over-smoothing of bright regions, which its mostly dark training data never taught it.
- **White-balance consensus:**
  - The camera's as-shot WB is only second-guessed when gray-world, shades-of-gray and gray-edge agree with each other (within 4°).
  - On colourful scenes they disagree, so the camera is trusted. This fixed a sunflower DNG turning blue.
  - Highlights are excluded relative to the image (the brightest 2%), so the estimate doesn't change with exposure.

**Order inside the hybrid** (and why), from [architecture.md](architecture.md):
1. Network on native RAW, its training distribution, then the noise-model trust region.
2. Router white balance and gain.
3. Malvar demosaic.
4. Calibrated colour matrix.
5. Tone.
6. Chroma cleanup.
7. Noise-aware sharpening.

### 4.4 Application
- **FastAPI server:** the API and the built React site in one Docker image (port 7860, non-root user). It holds results in an in-memory LRU cache; uploads are never written to disk, and outputs are re-encoded as WebP, so no EXIF or GPS survives.
- **Inputs:** DNG and most camera RAWs (via LibRaw), plus JPG/PNG/WebP. X-Trans and corrupt files get a clear message.
- **Speed:** inputs are capped at 2 MP by Bayer-aware binning to stay inside the ~10 s budget on 2 shared vCPUs; the built-in samples are precomputed at start-up.
- **Model slot** (`isp/model_slot.py`): the only place a network plugs in. `models/model.json` declares its input and output (`raw4`, `rgb`, `srgb`, `linear_*`), padding, layout and tiling; ONNX, TorchScript or a Python `predict()` file all load. A classical stand-in with the same contract keeps the app working without weights.

## 5. Evaluation protocol
- `eval/run_eval.py`: one noisy/clean pair per test instance (26 instances, scenes 004 and 008, all 5 phones), centre crop 1536×1024.
- Every pipeline renders with the same colour matrix and tone stages, and the ground truth is the clean RAW rendered by those same stages. Differences therefore come from denoising, demosaicing, white balance, exposure and sharpening, not from a different rendering.
- Robustness sweeps add sensor noise (Poisson-Gaussian, Brooks et al. parameters), under-exposure (−1 to −4 EV) and white-balance error (up to ±45% on R/B gain) to the real noisy RAW.
- Ablation removes one hybrid component at a time, averaged over three conditions.
- Results: [`results/benchmarks.json`](../results/benchmarks.json), [`eval_P2_<TeamID>.md`](../), and the website's Results tab.

## 6. Results
<!-- results:start -->
> **Provisional:** measured with the classical stand-in; the trained network was not yet available.

Held-out SIDD-CC test scenes 004, 008: 26 images from 5 phones, centre 1536x1024 crops, never seen in training or tuning. Model: Placeholder (classical stand-in) (`classical stand-in (no trained weights)`). Updated 2026-11-06 23:19.

| Pipeline | PSNR (dB) ↑ | SSIM ↑ | ΔE2000 ↓ | LPIPS ↓ | Time / image |
|---|---:|---:|---:|---:|---:|
| Input (no ISP) | 25.26 | 0.5850 | 11.60 | 0.4005 | n/a |
| Traditional ISP | 25.40 | 0.6493 | 7.73 | 0.5171 | 216 ms |
| Learned only | 28.22 | 0.7161 | 7.73 | **0.2834** | 132 ms |
| **Hybrid (ours)** | **29.13** | **0.7729** | **5.79** | 0.4029 | 223 ms |

RAW-domain denoising by the network alone (packed RGGB vs clean RAW): **40.34 → 42.74 dB**.

**Robustness** (mean PSNR over the robustness subset as each degradation grows):

- **Added sensor noise** (worst level, 12800 approx. ISO): hybrid 19.49 dB, learned only 18.13 dB, traditional 14.78 dB.
- **Under-exposure** (worst level, -4.0 EV): hybrid 27.47 dB, learned only 11.26 dB, traditional 11.29 dB.
- **White-balance error** (worst level, 1.0 error (fraction of R/B gain)): hybrid 24.56 dB, learned only 19.25 dB, traditional 18.68 dB.

Full tables, curves, the ablation study and failure cases: the website's **Results** tab, [`results/benchmarks.json`](../results/benchmarks.json) and [`docs/failure_cases.md`](failure_cases.md).
<!-- results:end -->

## 7. Failure cases
See [failure_cases.md](failure_cases.md), generated from the worst test images.

## 8. Limitations
- The learned component **denoises** RAW; it does not learn the colour transform. Colour is classical and calibrated (camera matrix + white balance). We chose this because the brief ranks colour accuracy (ΔE) second and a calibrated matrix is exact. A learned colour/tone head is the natural next step.
- **Ground truth is rendered:** the SIDD-CC clean RAW passed through our own colour and tone stages. sRGB scores therefore compare pipelines fairly with each other, but not with numbers from other papers.
- **Few scenes:** SIDD has only 10 physical scenes, so 2 test scenes is a small sample. Per-image numbers are in `benchmarks.json`.
- **Simulated degradations:** the robustness sweeps add simulated noise, exposure and WB errors to real RAW. Real low-light captures also have row noise, fixed-pattern noise and black-level drift, which are not modelled.
- **Web-demo compromises:** inputs are binned to ≤ 2 MP, and live scores use a centre crop. The official numbers are full-crop, offline.
- **Training stopped early:** the Kaggle run diverged at epoch 13, so the exported model is the epoch-11 checkpoint. The notebook has a fix (float32 loss, lower learning rate) that was not re-run before the deadline.
- **The network alone over-smooths bright regions:** its training data is mostly dark. The hybrid's trust region contains this, but the learned-only baseline shows it.
- **LPIPS prefers the network alone:** it scores slightly better than the hybrid on LPIPS, while the hybrid wins PSNR, SSIM and ΔE. The hybrid's classical sharpening and chroma cleanup trade some perceptual texture for fidelity.
- **Texture vs noise:** fine periodic texture (fabric weave) is indistinguishable from noise for single-photo noise estimators, so the router can over-denoise it. Camera noise metadata (DNG `NoiseProfile`, ISO) would fix this.
- **WB robustness trade-off:** the consensus check means a genuinely wrong camera WB on a colourful scene is left uncorrected (see failure case 4).

## 9. Ethics and privacy
- Uploads are processed in memory and never stored. Results expire from a small in-memory cache, and outputs carry no metadata.
- All data used is public research data (SIDD: research/non-commercial licence) or public-domain test images (scikit-image).
- No people-centric inference is done. The pipeline treats every pixel the same.
- Denoising can remove real detail, and a hallucinated texture would be a fidelity problem. Comparisons are therefore always shown against ground truth when it exists, and the pixel loupe lets anyone inspect the result at 1:1.

## 10. References
- A. Abdelhamed, S. Lin, M. S. Brown. *A High-Quality Denoising Dataset for Smartphone Cameras* (SIDD). CVPR 2018.
- T. Brooks et al. *Unprocessing Images for Learned Raw Denoising.* CVPR 2019.
- H. S. Malvar, L. He, R. Cutler. *High-quality linear interpolation for demosaicing of Bayer-patterned color images.* ICASSP 2004.
- J. Immerkær. *Fast Noise Variance Estimation.* CVIU 1996.
- K. He, J. Sun, X. Tang. *Guided Image Filtering.* ECCV 2010.
- R. Zhang et al. *The Unreasonable Effectiveness of Deep Features as a Perceptual Metric* (LPIPS). CVPR 2018.
- A. Ignatov et al. *Replacing Mobile Camera ISP with a Single Deep Learning Model* (PyNET). CVPRW 2020.
