// The sensor's view of a photo: one colour per cell in the RGGB Bayer pattern (the same 2x2 tile as the logo).
// Built from the real "sensor, no processing" image, re-mosaicked per cell, so it lines up with the developed
// photo underneath (same cover-crop maths as `object-fit: cover`).

const RGGB = [0, 1, 1, 2]; // channel for (y % 2) * 2 + (x % 2)

function cover(iw: number, ih: number, cw: number, ch: number) {
  const s = Math.max(cw / iw, ch / ih);
  const sw = cw / s, sh = ch / s;
  return { sx: (iw - sw) / 2, sy: (ih - sh) / 2, sw, sh };
}

function sizeCanvas(canvas: HTMLCanvasElement) {
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  const cw = canvas.clientWidth, ch = canvas.clientHeight;
  canvas.width = Math.max(1, Math.round(cw * dpr));
  canvas.height = Math.max(1, Math.round(ch * dpr));
  return { cw, ch, dpr };
}

/** Paint cells (already one-channel RGB) scaled up, crisp, with a thin dark grid between them. */
function paint(canvas: HTMLCanvasElement, small: HTMLCanvasElement, cell: number, dpr: number) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  ctx.imageSmoothingEnabled = false;
  ctx.fillStyle = "#0c0c0e";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(small, 0, 0, small.width * cell * dpr, small.height * cell * dpr);
  // sensor grid
  ctx.fillStyle = "rgba(12, 12, 14, 0.55)";
  const step = cell * dpr, w = Math.max(1, Math.round(dpr));
  for (let x = 0; x < canvas.width; x += step) ctx.fillRect(Math.round(x), 0, w, canvas.height);
  for (let y = 0; y < canvas.height; y += step) ctx.fillRect(0, Math.round(y), canvas.width, w);
}

export function drawMosaic(canvas: HTMLCanvasElement, img: HTMLImageElement, cell: number) {
  const { cw, ch, dpr } = sizeCanvas(canvas);
  const nx = Math.ceil(cw / cell), ny = Math.ceil(ch / cell);
  const small = document.createElement("canvas");
  small.width = nx; small.height = ny;
  const sctx = small.getContext("2d", { willReadFrequently: true });
  if (!sctx) return;
  // sample the same region object-fit: cover shows, one sample per cell
  const c = cover(img.naturalWidth, img.naturalHeight, nx * cell, ny * cell);
  sctx.imageSmoothingEnabled = true;
  sctx.drawImage(img, c.sx, c.sy, c.sw, c.sh, 0, 0, nx, ny);
  const data = sctx.getImageData(0, 0, nx, ny);
  const px = data.data;

  // display gain: a dark RAW should still read as vivid cells (95th percentile → 0.95)
  const hist = new Uint32Array(256);
  for (let i = 0; i < px.length; i += 4) hist[Math.max(px[i], px[i + 1], px[i + 2])]++;
  let acc = 0, p98 = 255;
  const target = (px.length / 4) * 0.95;
  for (let v = 0; v < 256; v++) { acc += hist[v]; if (acc >= target) { p98 = Math.max(1, v); break; } }
  const gain = Math.min(24, Math.max(1, (0.95 * 255) / p98));

  for (let y = 0; y < ny; y++) {
    for (let x = 0; x < nx; x++) {
      const i = (y * nx + x) * 4;
      const chn = RGGB[(y & 1) * 2 + (x & 1)];
      // gamma-lift after gain so mid-tones show, like a camera's preview of RAW
      const v = Math.min(255, 255 * Math.pow(Math.min(1, (px[i + chn] * gain) / 255), 0.62));
      px[i] = chn === 0 ? v : 0;
      px[i + 1] = chn === 1 ? v * 0.85 : 0; // greens are twice as many; tone them down a touch
      px[i + 2] = chn === 2 ? v : 0;
      px[i + 3] = 255;
    }
  }
  sctx.putImageData(data, 0, 0);
  paint(canvas, small, cell, dpr);
}

/** Placeholder while the real data loads: random RGGB cells, so the hero is never blank. */
export function drawShimmer(canvas: HTMLCanvasElement, cell: number) {
  const { cw, ch, dpr } = sizeCanvas(canvas);
  const nx = Math.ceil(cw / cell), ny = Math.ceil(ch / cell);
  const small = document.createElement("canvas");
  small.width = nx; small.height = ny;
  const sctx = small.getContext("2d");
  if (!sctx) return;
  const img = sctx.createImageData(nx, ny);
  const px = img.data;
  for (let y = 0; y < ny; y++) {
    for (let x = 0; x < nx; x++) {
      const i = (y * nx + x) * 4;
      const chn = RGGB[(y & 1) * 2 + (x & 1)];
      // a soft vignette of noise, brighter towards the middle
      const dx = x / nx - 0.5, dy = y / ny - 0.5;
      const v = (40 + Math.random() * 110) * (1 - Math.min(1, (dx * dx + dy * dy) * 1.6));
      px[i] = chn === 0 ? v : 0;
      px[i + 1] = chn === 1 ? v * 0.85 : 0;
      px[i + 2] = chn === 2 ? v : 0;
      px[i + 3] = 255;
    }
  }
  sctx.putImageData(img, 0, 0);
  paint(canvas, small, cell, dpr);
}
