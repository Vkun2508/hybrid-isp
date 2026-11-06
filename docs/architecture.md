# System architecture

```mermaid
flowchart LR
    subgraph Client["Browser (React, web/)"]
        UI["Lab: samples · upload · sensor simulator · router mode<br/>Compare: split view + pixel loupe · scores · why-this-route · stage strip<br/>Results: benchmarks.json"]
    end

    subgraph Server["FastAPI (server/main.py) · one Docker image · port 7860"]
        API["/api/process · /api/samples · /api/result/{id}/{name}.webp · /api/benchmarks · /api/health"]
        LRU[("In-memory LRU<br/>results + WebP images<br/>(nothing written to disk)")]
    end

    subgraph ISP["isp/ (NumPy · OpenCV · ONNX Runtime)"]
        IO["io.py<br/>DNG/RAW via rawpy → RGGB, black level, /white<br/>JPG/PNG → unprocessed RAW (Brooks 2019)<br/>≤ 2 MP binning, scene auto-exposure"]
        SIM["simulate.py<br/>Poisson-Gaussian noise · EV · WB error"]
        AN["analysis.py: analyser + router<br/>σ (Immerkær/MAD) · EV · WB confidence · DR<br/>thresholds from router_config.json<br/>(tuned on validation scene 003)"]
        subgraph P["pipelines.py"]
            T["Traditional<br/>as-shot WB → Malvar → bilateral(σ) → CCM → gamma+tone → USM"]
            L["Learned only<br/>packed RAW → network → bilinear + as-shot WB + CCM + tone"]
            H["Hybrid<br/>network denoises native RAW → router WB + gain → Malvar → CCM → tone<br/>→ chroma cleanup (high σ) → noise-aware sharpening"]
        end
        SLOT["model_slot.py<br/>models/model.json + .onnx (FP32 / INT8)<br/>pad to ×8, tiling, NCHW, checks<br/>stub fallback with the same contract"]
        MET["metrics.py<br/>PSNR · SSIM · ΔE2000 vs ground truth"]
    end

    subgraph Train["training/ (Kaggle / Colab GPU)"]
        NB["SIDD_CC_RAW_Complete_Pipeline.ipynb<br/>SIDD-CC, scene split 7/1/2 · LightweightISPUNet (1.0 M)<br/>Charbonnier + SSIM · AMP · early stop"]
        EXP["ONNX export + parity · static INT8 (QDQ)<br/>exported_model.zip"]
    end

    subgraph Eval["eval/"]
        RE["run_eval.py: quality · LPIPS · efficiency<br/>robustness · ablation · failure cases"]
        TU["tune_router.py (validation scene only)"]
    end

    UI -- "multipart: file | sample_id, sliders" --> API
    API --> IO --> SIM --> AN --> P
    L --> SLOT
    H --> SLOT
    P --> MET --> API
    API --> LRU --> UI
    NB --> EXP -- "isp_unet*.onnx, model.json" --> SLOT
    RE -- "results/benchmarks.json" --> API
    TU -- "isp/router_config.json" --> AN
```

## Data contract between the components

| Boundary | Format |
|---|---|
| Any input → pipelines | `RawImage` (`isp/raw.py`): RGGB Bayer float32, black level removed, divided by white level, negatives kept (noise floor); `wb_gains` (as-shot, G=1); `cam2rgb` (from DNG/SIDD `ColorMatrix2`, dcraw convention); `exposure` (scene baseline, applied by every pipeline); optional `gt` |
| Pipelines → network | `raw4`: packed `[R, G1, G2, B]`, (H/2, W/2, 4), no WB, unclipped. Exactly the training notebook's input |
| Network → pipelines | `raw4` denoised (same shape); declared in `models/model.json` |
| Server → browser | JSON (`server/schemas.py`) + WebP images by URL; errors as `{error: {code, message, hint}}` |
| Eval → browser | `results/benchmarks.json` (schema in `web/src/api.ts`) |

## Why the order inside the hybrid is what it is

1. **The network sees the RAW it was trained on.** SIDD-CC pairs are black-level-subtracted, normalised and not white-balanced, so the hybrid runs the learned denoiser on native RAW *before* white balance or digital gain. Gain applied first would push the noise outside the training distribution.
2. **Colour is decided classically, after denoising.** The router checks the as-shot white balance against a shades-of-gray estimate and corrects it only when they disagree strongly. The scene's colour matrix and an exact sRGB transfer function follow, so colour does not depend on what the network happened to learn.
3. **Exposure gain comes after denoising**, so it amplifies a clean signal, not the noise.
4. **Sharpening is noise-aware and measured after denoising**: its strength is set from the noise that is *left*, `k / (1 + σ_residual·gain / σ0)`.
