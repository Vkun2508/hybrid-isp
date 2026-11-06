USN1, USN2, USN3, USN4

# Evaluation: P2 Hybrid smartphone ISP (2026-11-06 21:59)

Test set: SIDD-CC smartphone RAW, held-out scenes 004, 008, 6 images (scene-level hold-out, 5 phones, one noisy/clean pair per test instance). Model: Placeholder (classical stand-in) (classical stand-in (no trained weights)).

## 1. PSNR and SSIM (sRGB, higher is better)

| Pipeline | PSNR (dB) | SSIM |
|---|---:|---:|
| input | 29.20 | 0.6190 |
| traditional | 29.54 | 0.6789 |
| learned | 32.80 | 0.7921 |
| hybrid | 32.20 | 0.7752 |

## 2. Colour accuracy: mean CIEDE2000 (lower is better)

| Pipeline | ΔE2000 |
|---|---:|
| input | 8.505 |
| traditional | 5.398 |
| learned | 4.848 |
| hybrid | 3.732 |

## 3. Perceptual: LPIPS, AlexNet (lower is better)

| Pipeline | LPIPS |
|---|---:|
| input | 0.4061 |
| traditional | 0.4662 |
| learned | 0.2322 |
| hybrid | 0.3911 |

## 4. Model parameters and size

| Variant | Parameters | Size (MB) |
|---|---:|---:|

## 5. Inference latency and memory (CPU)

| Variant | Latency, 1 MP RAW (ms) | Memory delta (MB) |
|---|---:|---:|

End-to-end per image (centre 1536x1024): traditional 196 ms, learned 119 ms, hybrid 212 ms; peak process memory 910 MB.
Machine: 16 logical CPUs, ONNX Runtime CPU.

RAW-domain denoising (packed RGGB, what the network optimises): noisy 37.48 dB -> denoised 39.25 dB.

## Robustness (PSNR dB as the degradation grows)

**Added sensor noise** (approx. ISO: 100, 336, 1131, 3805, 12800)

- traditional: 31.17, 29.95, 27.89, 24.27, 19.10
- learned: 34.23, 33.31, 31.85, 29.01, 23.86
- hybrid: 33.29, 32.32, 30.02, 25.63, 23.05
- hybrid, router off: 32.85, 31.90, 29.63, 25.30, 21.44

**Under-exposure** (EV: 0.0, -1.0, -2.0, -3.0, -4.0)

- traditional: 31.17, 19.29, 14.95, 12.91, 11.85
- learned: 34.23, 19.40, 14.95, 12.90, 11.84
- hybrid: 33.29, 20.36, 20.36, 20.36, 20.36
- hybrid, router off: 32.85, 20.34, 20.34, 20.34, 20.34

**White-balance error** (error (fraction of R/B gain): 0.0, 0.25, 0.5, 0.75, 1.0)

- traditional: 31.17, 28.67, 25.40, 22.72, 20.46
- learned: 34.23, 30.48, 26.34, 23.19, 20.70
- hybrid: 33.29, 30.34, 27.60, 26.75, 27.58
- hybrid, router off: 32.85, 30.08, 27.45, 26.62, 27.43

## Ablation (hybrid; mean over as-captured, noise + -2 EV, and WB-error conditions)

| Variant | PSNR | SSIM | ΔE |
|---|---:|---:|---:|
| Full hybrid | 26.77 | 0.7257 | 8.17 |
| No learned core (classical denoise) | 26.58 | 0.7268 | 8.43 |
| No router (always network, fixed sharpening) | 26.56 | 0.7126 | 8.19 |
| No white-balance correction | 28.58 | 0.7426 | 5.56 |
| No exposure normalisation | 24.64 | 0.6326 | 8.87 |
| No chroma cleanup | 26.77 | 0.7257 | 8.17 |
| No sharpening | 27.41 | 0.7514 | 8.12 |
| Bilinear instead of Malvar demosaic | 27.30 | 0.7457 | 8.40 |

## Failure cases

- **N6 · ISO 3200 · scene 004**: Input | Hybrid | Ground truth (centre crop). Hybrid 24.50 dB vs traditional 18.42 dB, dE 7.95; router chose 'learned_chroma'. (`results/failures/0076_004_N6_03200_00320_3200_L.jpg`)
- **N6 · ISO 800 · scene 004**: Input | Hybrid | Ground truth (centre crop). Hybrid 29.26 dB vs traditional 25.52 dB, dE 5.41; router chose 'learned'. (`results/failures/0075_004_N6_00800_00080_3200_L.jpg`)
- **G4 · ISO 200 · scene 004**: Input | Hybrid | Ground truth (centre crop). Hybrid 33.46 dB vs traditional 31.74 dB, dE 2.85; router chose 'learned'. (`results/failures/0078_004_G4_00200_00050_3200_N.jpg`)
