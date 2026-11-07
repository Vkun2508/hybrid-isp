1RUA24CSE0515, 1RUA24CSE0499, 1RUA24CSE0480, 1RUA24CSE0534

# Evaluation: P2 Hybrid smartphone ISP (2026-11-07 09:20)

Test set: SIDD-CC smartphone RAW, held-out scenes 004, 008, 26 images (scene-level hold-out, 5 phones, one noisy/clean pair per test instance). Model: LightweightISPUNet (SIDD-CC RAW denoiser) (isp_unet.onnx).

## 1. PSNR and SSIM (sRGB, higher is better)

| Pipeline | PSNR (dB) | SSIM |
|---|---:|---:|
| input | 25.26 | 0.5850 |
| traditional | 25.40 | 0.6493 |
| learned | 28.37 | 0.7553 |
| hybrid | 29.84 | 0.8016 |

## 2. Colour accuracy: mean CIEDE2000 (lower is better)

| Pipeline | ΔE2000 |
|---|---:|
| input | 11.599 |
| traditional | 7.734 |
| learned | 6.408 |
| hybrid | 5.336 |

## 3. Perceptual: LPIPS, AlexNet (lower is better)

| Pipeline | LPIPS |
|---|---:|
| input | 0.4005 |
| traditional | 0.5171 |
| learned | 0.3609 |
| hybrid | 0.3651 |

## 4. Model parameters and size

| Variant | Parameters | Size (MB) |
|---|---:|---:|
| ONNX FP32 | 1,004,996 | 4.036 |
| ONNX INT8 | 1,004,996 | 1.156 |

## 5. Inference latency and memory (CPU)

| Variant | Latency, 1 MP RAW (ms) | Memory delta (MB) |
|---|---:|---:|
| ONNX FP32 | 242.1 | 392.2 |
| ONNX INT8 | 183.5 | 38.5 |

End-to-end per image (centre 1536x1024): traditional 176 ms, learned 440 ms, hybrid 576 ms; peak process memory 1436 MB.
Machine: 16 logical CPUs, ONNX Runtime CPU.

RAW-domain denoising (packed RGGB, what the network optimises): noisy 40.34 dB -> denoised 40.96 dB.

## Robustness (PSNR dB as the degradation grows)

**Added sensor noise** (approx. ISO: 100, 336, 1131, 3805, 12800)

- traditional: 24.20, 23.56, 22.27, 19.41, 14.78
- learned: 28.16, 27.92, 27.21, 24.83, 19.50
- hybrid: 28.98, 28.59, 27.77, 25.64, 20.34
- hybrid, router off: 28.42, 28.02, 27.04, 24.18, 18.64

**Under-exposure** (EV: 0.0, -1.0, -2.0, -3.0, -4.0)

- traditional: 24.20, 18.07, 14.18, 12.28, 11.29
- learned: 28.16, 18.34, 14.15, 12.23, 11.25
- hybrid: 28.98, 28.59, 27.81, 26.50, 24.63
- hybrid, router off: 28.42, 27.98, 27.05, 25.54, 23.45

**White-balance error** (error (fraction of R/B gain): 0.0, 0.25, 0.5, 0.75, 1.0)

- traditional: 24.20, 23.50, 21.96, 20.30, 18.68
- learned: 28.16, 26.56, 23.95, 21.54, 19.43
- hybrid: 28.98, 27.11, 24.32, 21.96, 20.14
- hybrid, router off: 28.42, 26.70, 24.05, 21.79, 20.03

## Ablation (hybrid; mean over as-captured, noise + -2 EV, and WB-error conditions)

| Variant | PSNR | SSIM | ΔE |
|---|---:|---:|---:|
| Full hybrid | 25.91 | 0.6971 | 8.40 |
| No learned core (classical denoise) | 24.42 | 0.6298 | 9.68 |
| No router (always network, fixed sharpening) | 25.13 | 0.6542 | 9.19 |
| No white-balance correction | 25.88 | 0.6973 | 8.38 |
| No exposure normalisation | 22.14 | 0.6043 | 11.20 |
| No chroma cleanup | 25.37 | 0.6681 | 9.14 |
| No sharpening | 25.98 | 0.7024 | 8.39 |
| Bilinear instead of Malvar demosaic | 25.95 | 0.7099 | 8.56 |
| No noise-model bound on the network | 25.86 | 0.7034 | 8.28 |
| INT8 learned core (instead of FP32) | 24.74 | 0.6121 | 9.17 |

## Latency and peak memory vs resolution (CPU)

| RAW size | Network FP32 | Network INT8 | Traditional | Learned only | Hybrid | Hybrid peak memory |
|---|---:|---:|---:|---:|---:|---:|
| 0.5 MP (864x576) | 117 ms | 84 ms | 196 ms | 230 ms | 247 ms | 62 MB |
| 1 MP (1224x816) | 212 ms | 182 ms | 276 ms | 475 ms | 463 ms | 109 MB |
| 2 MP (1728x1152) | 464 ms | 399 ms | 500 ms | 813 ms | 1066 ms | 202 MB |
| 4 MP (2448x1632) | 2104 ms | 1563 ms | 1001 ms | 2740 ms | 2889 ms | 388 MB |

## Tiled inference (2448x1632 RAW): seams

_interior = excluding a 32-pixel border, where padding (not tiling) decides the result_

| Variant | Latency | PSNR vs one pass | Interior max abs diff | Interior PSNR |
|---|---:|---:|---:|---:|
| single pass | 939 ms | 99.0 dB | 0.00e+00 | 99.0 dB |
| tiles 512, overlap 32 | 1776 ms | 64.3 dB | 3.58e-07 | 161.0 dB |
| tiles 256, overlap 32 | 2499 ms | 64.3 dB | 5.96e-07 | 157.3 dB |
| tiles 256, overlap 64 | 3314 ms | 64.3 dB | 3.58e-07 | 164.4 dB |

## Adaptive router: quality vs compute (26 images x as captured, extra noise, under-exposed 2 EV, white-balance error)

| Mode | PSNR | ΔE2000 | Mean latency | Images through the network |
|---|---:|---:|---:|---:|
| Adaptive router | 27.97 | 6.56 | 874 ms | 97% |
| Always network | 27.37 | 6.97 | 857 ms | 100% |
| Never network (classical only) | 26.94 | 7.53 | 557 ms | 0% |

Paths chosen by the adaptive router (as captured): ISO 400-800: learned 5, learned_chroma 5; ISO <= 200: learned 8, fast 1, learned_chroma 3; ISO >= 1000: learned_chroma 3, learned 1

## Per phone (PSNR dB / ΔE2000, as captured)

| Group | n | Input | Traditional | Learned only | Hybrid |
|---|---:|---:|---:|---:|---:|
| Galaxy S6 Edge | 1 | 30.15 / 7.69 | 28.83 / 4.73 | 29.91 / 5.29 | 32.39 / 4.22 |
| Google Pixel | 6 | 28.04 / 8.19 | 27.03 / 5.56 | 28.99 / 5.10 | 30.73 / 4.35 |
| LG G4 | 3 | 30.05 / 6.69 | 29.20 / 3.95 | 31.04 / 4.13 | 33.09 / 3.33 |
| Nexus 6 | 5 | 20.39 / 17.27 | 20.92 / 12.33 | 26.24 / 8.99 | 27.20 / 7.35 |
| iPhone 7 | 11 | 24.20 / 12.58 | 25.21 / 8.13 | 28.13 / 6.67 | 29.44 / 5.61 |

## Per ISO band (PSNR dB / ΔE2000, as captured)

| Group | n | Input | Traditional | Learned only | Hybrid |
|---|---:|---:|---:|---:|---:|
| ISO 1600-3200 | 4 | 18.84 / 19.18 | 20.43 / 13.74 | 25.82 / 9.54 | 26.50 / 8.05 |
| ISO 320-800 | 10 | 23.34 / 13.31 | 24.19 / 8.80 | 28.21 / 6.64 | 29.17 / 5.73 |
| ISO 50-200 | 12 | 28.99 / 7.65 | 28.07 / 4.84 | 29.35 / 5.17 | 31.51 / 4.11 |

## Failure cases

- **N6 · ISO 1600 · scene 008**: Input | Hybrid | Ground truth (centre crop). Hybrid 21.22 dB vs traditional 15.73 dB, dE 13.13; router chose 'learned_chroma'. (`results/failures/0170_008_N6_01600_00800_4400_L.jpg`)
- **IP · ISO 800 · scene 008**: Input | Hybrid | Ground truth (centre crop). Hybrid 22.21 dB vs traditional 16.60 dB, dE 12.03; router chose 'learned_chroma'. (`results/failures/0186_008_IP_00800_00800_3200_L.jpg`)
- **GP · ISO 100 · scene 008**: Input | Hybrid | Ground truth (centre crop). Hybrid 24.88 dB vs traditional 22.25 dB, dE 5.88; router chose 'learned_chroma'. (`results/failures/0180_008_GP_00100_00100_5500_N.jpg`)
