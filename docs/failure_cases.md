# Failure cases

All numbers are sRGB PSNR against the SIDD ground truth on held-out test scenes (`eval/run_eval.py`).
Pictures: `results/failures/<instance>.jpg`, each showing input | hybrid | ground truth (centre crop).
They are also on the website's Results tab. Cases 1–3 are the worst hybrid results in the final evaluation;
cases 4–6 are failures found during development, kept here with their fixes.

## 1. Fine fabric texture mistaken for noise (Pixel, ISO 100, normal light)
`0180_008_GP_00100_00100_5500_N`: input 26.01 dB · traditional 22.25 · learned only 22.84 · **hybrid 24.88**

- **What you see:** the input is already fairly clean. The hybrid is *softer than the input*: the fabric's fine weave is partly smoothed, and the dark areas lose some of their purple.
- **Cause:** the weave is a periodic texture at the 1–2 pixel scale, which is exactly what sensor noise looks like to a single-photo noise estimator. Both of our estimators read it as heavy noise: the Laplacian/median estimator (σ ≈ 0.023) and the flat-region noise-level fit (σ ≈ 0.024). So the router sent an ISO 100 photo down the high-noise path, with chroma cleanup and weak sharpening. Every pipeline loses to the unprocessed input here; the hybrid loses least.
- **Mitigation:**
  - Read noise from metadata when it exists (the DNG `NoiseProfile` tag, or ISO) instead of estimating it from the pixels.
  - Or learn a noise estimator that is trained to tell texture from noise.

## 2. Extreme low light: residual colour noise (iPhone 7, ISO 800, dark scene)
`0186_008_IP_00800_00800_3200_L`: input 14.88 dB · traditional 16.60 · learned only 21.12 · **hybrid 22.21**

- **What you see:** the hybrid removes most of the extreme noise (+7.3 dB over the input), but blotchy colour noise remains and the weave texture is lost.
- **Cause:** the RAW is very dark, so the scene needs a ×15.5 exposure lift, and the noise relative to the signal is far beyond most of the training patches. The network removes what it learned; the chroma cleanup removes some of the rest, but at this level it cannot separate the weave from the noise.
- **Mitigation:**
  - Train with more extreme low-light pairs, or with synthetic noise at higher gains.
  - Use multi-frame capture (burst denoising), which is how phones handle this.

## 3. Extreme low light, high ISO (Nexus 6, ISO 1600, dark scene)
`0170_008_N6_01600_00800_4400_L`: input 14.36 dB · traditional 15.73 · learned only 20.36 · **hybrid 21.22**

- **What you see:** the same failure class as case 2 on a different phone. Noise is largely removed, but colour mottling remains and detail is lost.
- **Cause and mitigation:** as in case 2. The Nexus 6 sensor at ISO 1600 is the noisiest phone in SIDD.

## 4. Colourful scenes fooled the white-balance check (fixed)
- **Found on:** a Xiaomi Redmi Note 8T sunflower DNG ([raw_compatibility.md](raw_compatibility.md)). The router saw the camera's WB disagree with a gray-world-style estimate by 17° and overrode it, which turned the photo blue.
- **Fix:** the camera is only second-guessed when three independent estimators (gray-world, shades-of-gray, gray-edge) agree with each other within 4°.
- **Remaining cost:** on non-neutral scenes, a *genuinely* wrong camera WB is now left alone. That's why the robustness gain under injected WB errors on the test scenes is smaller than before the fix. We accept this trade: wrong metadata is rare, while colourful photos are common.

## 5. The network erased detail in bright regions (fixed)
- **Found on:** the full test images. The raw network made several low-ISO, well-lit images worse in the RAW domain, e.g. iPhone ISO 100: 39.4 → 34.8 dB. In regions above about 20% of white its corrections were 3–5× larger than the sensor noise there could explain, so it was removing texture and shifting highlights. (The training data is mostly dark.)
- **Fix: a classical trust region.** The sensor's noise-level function σ²(x) = a·x + b is estimated from each image, and the network may change a pixel by at most k·σ(x), with k tuned on the validation scene.
- **Result:** across the 26 test images, RAW PSNR goes from 40.9 → about 43 dB, and every image the network used to damage now improves on its input.

## 6. Training diverged at epoch 13
- **What happened:** the Kaggle run's loss spiked at epoch 13 and validation PSNR crashed from 52.4 to 29 dB ([training curves](../training/kaggle_run/training_curves.png)). Early stopping ended the run at epoch 21. Because the best checkpoint is only saved on improvement, the exported model is the epoch-11 one (validation 52.4 dB).
- **Likely cause:** the SSIM term computed in float16 under mixed precision, with a learning rate of 2e-4.
- **Fix, ready but not run:** the notebook (v2) computes the loss in float32, skips non-finite batches and halves the learning rate.
