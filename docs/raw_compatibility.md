# RAW compatibility: real phone files

A smartphone ISP has to read each sensor correctly before any processing helps: its Bayer layout, black and
white levels, as-shot white balance and colour matrix. This page records how that was verified.

## Seven phones, public sample DNGs

Files: public samples from [raw.pixls.us](https://raw.pixls.us) (CC0). Run: `python scripts/test_phone_raw.py .cache/phone_raw`,
which writes [`results/phone_raw.json`](../results/phone_raw.json) and the preview strips in `results/phone_raw/`
(sensor input | traditional | hybrid). The same gallery is on the website's Results tab.

| Phone | DNG type | Bayer pattern (LibRaw) | Black level | White level | Sensor | Black at 0 | Hybrid colour vs camera WB |
|---|---|---|---|---|---|---|---|
| Google Pixel 4a | Bayer | RGGB | 1022–1023 (per channel) | 16368 | 4032×3024 | yes | 0.41° |
| Google Pixel 6 Pro | Bayer | **GBRG** | 1023 | 16368 | 4080×3072 | yes | 0.04° |
| Xiaomi Redmi Note 8T | Bayer | **GRBG** | 64 | 1023 (10-bit) | 4000×3000 | yes | 0.03° |
| OnePlus 6 | Bayer | RGGB | 63 | 1023 (10-bit) | 4656×3496 | yes | 0.03° |
| Apple iPhone XS | Bayer | RGGB | 528 | 4095 (12-bit) | 4032×3024 | yes | 0.05° |
| Apple iPhone 12 Pro (ProRAW) | **linear** (demosaiced) | — | 0 | 65535 | 4032×3024 | yes | 0.03° |
| Samsung Galaxy S22 Ultra | **linear** (demosaiced) | — | 0 | 4095 | 3648×2736 | yes | 0.00° |

**Automatic checks, all 7 pass:**
- The darkest 0.1% of green pixels sit at 0 after black-level removal.
- Nothing exceeds the white level.
- Less than 20% of the output is clipped.
- The hybrid's colour stays within 6° of the camera's own white-balance rendering (the hybrid may refine it, never swing it).

The preview strips were checked by eye.

## How each part is handled (`isp/io.py`)

| Concern | Handling |
|---|---|
| Bayer layout | LibRaw's 2×2 pattern is read per file; RGGB/GRBG/GBRG/BGGR are aligned to RGGB by a one-pixel crop (unit-tested for all four in `tests/test_isp.py`). X-Trans (6×6) and other layouts are refused with a clear message. |
| Black level | Subtracted **per CFA channel** (Pixel 4a has 1023/1023/1022/1022). Negative values are kept: the noise floor matters to the denoiser. |
| White level | Normalised by (white − black); 10, 12, 14 and 16-bit files all land in [0, 1]. |
| White balance | The camera's as-shot gains; the router may refine them only under a consensus check (below). |
| Colour | Camera → sRGB from the file's `ColorMatrix` (inverse of the row-normalised `ColorMatrix · sRGB→XYZ`). A generic matrix is used only when a file has none, and the UI says so. |
| Demosaiced (linear) DNGs | Apple ProRAW and Samsung expert RAW are already demosaiced. They are read as linear camera RGB and re-mosaiced, so all three pipelines see the same input. |
| Orientation | The file's orientation flag is applied when images are displayed. |
| Size | Bayer-aware 2×2 binning to ≤ 2 MP keeps the web demo under 10 s. The pipelines themselves handle any size (tiled network inference). |

## What real files caught (and the fixes)

1. **Blue sunflower (Redmi Note 8T).**
   - **What happened:** the router compared the camera's white balance with a gray-world-style estimate, saw a 17° disagreement, and overrode it. A frame full of yellow petals and green leaves is exactly where gray-world assumptions fail, so the hybrid turned the image blue.
   - **Fix: a consensus check.** Three independent estimators (gray-world, shades-of-gray, gray-edge) must agree with each other (within 4°) before the camera is second-guessed.
   - **Evidence:** on validation scenes they agree within 1.4°, so injected WB errors are still corrected. On the colourful phone photos they disagree by 6–12°, so the camera's WB is kept.
2. **White balance that changed with exposure (SIDD validation and test).**
   - **What happened:** highlights were excluded with a fixed threshold. When a photo was under-exposed, shiny specular highlights slipped under it and skewed the estimate.
   - **Fix:** the brightest 2% of pixels are excluded instead, a cut relative to the image. The estimate is now identical at every exposure (unit-tested), and under-exposed photos recover fully (29.6 dB at −1 and −3 EV, from 20.6).
3. **Green-channel order.** Our packing order was checked against the alternative by swapping G1/G2 on all 26 test images. The swap made every phone worse (−0.4 to −0.5 dB), so the order used matches what the network learned.
