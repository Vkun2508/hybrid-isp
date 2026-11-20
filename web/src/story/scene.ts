// The sensor story's WebGL scene: 24,576 real photosites of one low-light RAW, drawn as square cells in 3D.
// Progress p (0…1, from scroll) drives everything; time only adds a slow drift and the noise's shimmer.
//   0.12–0.36  a gold line sweeps in from the right: behind it the network has denoised the RAW (real values)
//   0.40–0.60  exact maths brightens both sides ×8: noisy spikes on the left, a clean landscape on the right
//   0.64–0.84  the camera tilts top-down, points flatten into cells: classical photo | hybrid photo, same line
import {
  AdditiveBlending, BufferAttribute, BufferGeometry, PerspectiveCamera, Points, Scene, ShaderMaterial, Vector3, WebGLRenderer,
} from "three";

export type Field = { w: number; h: number; data: Uint8Array }; // per photosite: hA hB hC hD r g b r2 g2 b2

export const GRID_W = 3; // world units across the photo (aspect 1.5, like the 1536×1024 RAW)
export const GRID_H = 2;
const FOV = 38;
const HMAX = 1.05;
const SUBJECT_X = 0.3; // the chess piece sits at ~30% of the width; frame it on narrow screens
export const PIECE_X = 0.29; // its stem: where the line settles, cutting it into classical | hybrid
const BYTES = 10;

export async function loadField(url: string): Promise<Field> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`field ${r.status}`);
  const buf = new Uint8Array(await r.arrayBuffer());
  const dv = new DataView(buf.buffer);
  if (String.fromCharCode(...buf.subarray(0, 4)) !== "HISP") throw new Error("field: bad header");
  const w = dv.getUint16(6, true), h = dv.getUint16(8, true);
  const data = buf.subarray(16);
  if (dv.getUint16(4, true) !== 2 || data.length < w * h * BYTES) throw new Error("field: version");
  return { w, h, data };
}

/** Top-down framing that covers the stage exactly like `object-fit: cover`, centred on the subject where it can be. */
export function topView(aspect: number) {
  const vw = aspect > GRID_W / GRID_H ? GRID_W : GRID_H * aspect; // visible world width
  const vh = vw / aspect;
  const d = vh / (2 * Math.tan((FOV * Math.PI) / 360));
  const slack = (GRID_W - vw) / 2;
  const xc = Math.max(-slack, Math.min(slack, (SUBJECT_X - 0.5) * GRID_W));
  // matching CSS object-position for the photo layer (fraction of the overflow on the left)
  const leftFrac = (xc - vw / 2 + GRID_W / 2) / GRID_W;
  const posX = vw >= GRID_W - 1e-6 ? 0.5 : leftFrac / (1 - vw / GRID_W);
  const piece = ((PIECE_X - 0.5) * GRID_W - (xc - vw / 2)) / vw; // the piece's stem, as a fraction of the stage width
  return { d, xc, posX, piece };
}

const VERT = /* glsl */ `
  attribute vec2 aUv;      // 0..1 across / down the photo
  attribute vec4 aH;       // heights 0..1: noisy, denoised, denoised+gain (hybrid), noisy+gain (classical order)
  attribute vec3 aRgb;     // hybrid's final colour
  attribute vec3 aRgb2;    // classical pipeline's colour, brightened the same way
  attribute float aCh;     // 0 red, 1 green, 2 blue
  uniform float uSplit, uLine, uGain, uFlat, uTime, uScale, uCell, uMaxPx;
  varying vec3 vColor;
  varying float vFlat;
  float hash(vec2 p) { return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }
  void main() {
    float ph = hash(aUv * 731.0) * 6.2832;
    // one gold line runs through the whole story: right of it the network has cleaned the RAW (hybrid),
    // left of it the RAW is as captured (classical order). It sweeps in from the right, then holds.
    float hy = smoothstep(uSplit - 0.002, uSplit + 0.002, aUv.x);
    float band = exp(-pow((aUv.x - uSplit) / 0.0045, 2.0)) * uLine;
    float sh = 0.75 + 0.6 * sin(uTime * 3.1 + ph);                       // noise boils frame to frame; signal stays put
    float hClassical = mix(aH.y + (aH.x - aH.y) * sh, aH.z + (aH.w - aH.z) * sh, uGain);
    float hHybrid = mix(aH.y, aH.z, uGain);
    float h = mix(hClassical, hHybrid, hy);
    float y = h * ${HMAX.toFixed(2)} * (1.0 - uFlat);
    vec3 pos = vec3((aUv.x - 0.5) * ${GRID_W.toFixed(1)}, y, (aUv.y - 0.5) * ${GRID_H.toFixed(1)});
    vec4 mv = modelViewMatrix * vec4(pos, 1.0);
    gl_Position = projectionMatrix * mv;
    float depth = -mv.z;
    float size = mix(uCell * 1.2, uCell * 1.001, uFlat);
    gl_PointSize = clamp(size * uScale / depth, 1.0, mix(uMaxPx, 512.0, uFlat));
    vec3 filt = aCh < 0.5 ? vec3(1.0, 0.27, 0.2) : (aCh < 1.5 ? vec3(0.3, 1.0, 0.42) : vec3(0.32, 0.5, 1.0));
    filt = mix(filt, vec3(0.86, 0.88, 0.92), 0.5 * hy);                   // the cleaned side calms down
    float lum = 0.05 + 1.55 * pow(max(h, 0.0), 1.15);
    float tw = 0.9 + 0.1 * sin(uTime * 1.7 + ph * 1.3);
    vec3 c = filt * lum * tw;
    float fog = smoothstep(4.6, 1.6, depth) * mix(smoothstep(0.45, 1.05, depth), 1.0, uFlat);
    vec3 photo = mix(aRgb2, aRgb, hy);
    vColor = mix(c * fog, photo, uFlat);
    vColor = mix(vColor, vec3(1.0, 0.74, 0.38) * (1.4 - 0.4 * uFlat), band); // the gold line
    vFlat = uFlat;
  }
`;

const FRAG = /* glsl */ `
  varying vec3 vColor;
  varying float vFlat;
  void main() {
    vec2 q = gl_PointCoord * 2.0 - 1.0;
    float r = length(q);
    float sq = max(abs(q.x), abs(q.y));
    float cell = 1.0 - smoothstep(0.66, 0.78, sq);                          // a square photosite with a small gap
    float a = mix(cell, 1.0, vFlat);                                        // gaps close as the field flattens
    if (a < 0.01) discard;
    gl_FragColor = vec4(vColor * a, 1.0);
  }
`;

type Key = { p: number; r: number; phi: number; theta: number; tx: number; ty: number; tz: number };
const smooth = (t: number) => t * t * (3 - 2 * t);
const clamp01 = (t: number) => Math.min(1, Math.max(0, t));
const ramp = (p: number, a: number, b: number) => smooth(clamp01((p - a) / (b - a)));

export function createScene(canvas: HTMLCanvasElement, field: Field, opts: { onLost?: () => void } = {}) {
  const renderer = new WebGLRenderer({ canvas, antialias: false, alpha: false, powerPreference: "high-performance" });
  renderer.setClearColor(0x0b0b0d, 1);
  const scene = new Scene();
  const camera = new PerspectiveCamera(FOV, 1, 0.05, 20);

  const { w, h, data } = field;
  const n = w * h;
  const uv = new Float32Array(n * 2), hh = new Float32Array(n * 4), rgb = new Float32Array(n * 3), rgb2 = new Float32Array(n * 3);
  const ch = new Float32Array(n);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x, o = i * BYTES;
      uv[i * 2] = (x + 0.5) / w;
      uv[i * 2 + 1] = (y + 0.5) / h;
      for (let k = 0; k < 4; k++) hh[i * 4 + k] = data[o + k] / 255;
      for (let k = 0; k < 3; k++) { rgb[i * 3 + k] = data[o + 4 + k] / 255; rgb2[i * 3 + k] = data[o + 7 + k] / 255; }
      ch[i] = (y & 1) === 0 ? ((x & 1) === 0 ? 0 : 1) : ((x & 1) === 0 ? 1 : 2); // RGGB
    }
  }
  const geo = new BufferGeometry();
  geo.setAttribute("position", new BufferAttribute(new Float32Array(n * 3), 3)); // positions come from aUv/aH in the shader
  geo.setAttribute("aUv", new BufferAttribute(uv, 2));
  geo.setAttribute("aH", new BufferAttribute(hh, 4));
  geo.setAttribute("aRgb", new BufferAttribute(rgb, 3));
  geo.setAttribute("aRgb2", new BufferAttribute(rgb2, 3));
  geo.setAttribute("aCh", new BufferAttribute(ch, 1));
  const uniforms = {
    uSplit: { value: 1.1 }, uLine: { value: 0 }, uGain: { value: 0 }, uFlat: { value: 0 }, uTime: { value: 0 },
    uScale: { value: 1 }, uCell: { value: GRID_W / w }, uMaxPx: { value: 24 },
  };
  const mat = new ShaderMaterial({ vertexShader: VERT, fragmentShader: FRAG, uniforms, transparent: true, depthWrite: false, blending: AdditiveBlending });
  const pts = new Points(geo, mat);
  pts.frustumCulled = false;
  scene.add(pts);

  let p = 0, aspect = 1, top = topView(1), raf = 0, running = false, t0 = performance.now();
  const keys = (): Key[] => [
    { p: 0.0, r: 1.85, phi: 68, theta: -9, tx: -0.4, ty: 0.06, tz: 0.25 },
    { p: 0.12, r: 1.85, phi: 67, theta: -3, tx: -0.4, ty: 0.06, tz: 0.18 },
    { p: 0.38, r: 1.95, phi: 66, theta: 10, tx: -0.5, ty: 0.06, tz: 0.05 },
    { p: 0.62, r: 3.0, phi: 50, theta: 4, tx: -0.2, ty: 0.12, tz: 0 },
    { p: 0.86, r: top.d, phi: 0.02, theta: 0, tx: top.xc, ty: 0, tz: 0 },
  ];
  const target = new Vector3(), up = new Vector3();

  function place(time: number) {
    const k = keys();
    let a = k[0], b = k[0], t = 0;
    for (let i = 0; i < k.length - 1; i++) {
      if (p >= k[i].p) { a = k[i]; b = k[i + 1]; t = smooth(clamp01((p - a.p) / (b.p - a.p))); }
    }
    if (p >= k[k.length - 1].p) { a = b = k[k.length - 1]; t = 0; }
    const lerp = (x: number, y: number) => x + (y - x) * t;
    const idle = 1 - ramp(p, 0.05, 0.3);
    const r = lerp(a.r, b.r), phi = (lerp(a.phi, b.phi) * Math.PI) / 180;
    const theta = ((lerp(a.theta, b.theta) + idle * 7 * Math.sin(time * 0.16)) * Math.PI) / 180;
    target.set(lerp(a.tx, b.tx), lerp(a.ty, b.ty), lerp(a.tz, b.tz));
    if (aspect < 1) {
      const lineX = (uniforms.uSplit.value - 0.5) * GRID_W;
      const follow = ramp(p, 0.1, 0.16) * (1 - ramp(p, 0.64, 0.8));
      target.x += (Math.max(-1.3, Math.min(1.3, lineX)) - target.x) * follow;
    }
    camera.position.set(target.x + r * Math.sin(phi) * Math.sin(theta), target.y + r * Math.cos(phi), target.z + r * Math.sin(phi) * Math.cos(theta));
    const f = ramp(p, 0.64, 0.84);
    up.set(0, 1 - f, -f).normalize();
    camera.up.copy(up);
    camera.lookAt(target);
  }

  function frame(now: number) {
    const time = (now - t0) / 1000;
    uniforms.uTime.value = time;
    uniforms.uSplit.value = 1.04 + (PIECE_X - 1.04) * ramp(p, 0.12, 0.36);
    uniforms.uLine.value = ramp(p, 0.1, 0.14);
    uniforms.uGain.value = ramp(p, 0.4, 0.6);
    uniforms.uFlat.value = ramp(p, 0.64, 0.84);
    place(time);
    renderer.render(scene, camera);
    if (running) raf = requestAnimationFrame(frame);
  }

  function resize() {
    const cw = canvas.clientWidth || 1, chh = canvas.clientHeight || 1;
    const dpr = Math.min(window.devicePixelRatio || 1, cw < 700 ? 1.75 : 2);
    renderer.setPixelRatio(dpr);
    renderer.setSize(cw, chh, false);
    aspect = cw / chh;
    camera.aspect = aspect;
    camera.updateProjectionMatrix();
    top = topView(aspect);
    uniforms.uScale.value = (chh * dpr) / (2 * Math.tan((FOV * Math.PI) / 360));
    uniforms.uMaxPx.value = 22 * dpr; // no giant out-of-focus discs from points right at the lens
    if (!running) frame(performance.now());
  }

  const lost = (e: Event) => { e.preventDefault(); stop(); opts.onLost?.(); };
  canvas.addEventListener("webglcontextlost", lost);

  function start() { if (running) return; running = true; raf = requestAnimationFrame(frame); }
  function stop() { running = false; cancelAnimationFrame(raf); }

  resize();
  return {
    setProgress(v: number) { p = clamp01(v); if (!running) frame(performance.now()); },
    resize,
    start,
    stop,
    top: () => top,
    dispose() {
      stop();
      canvas.removeEventListener("webglcontextlost", lost);
      geo.dispose(); mat.dispose(); renderer.dispose();
    },
  };
}
export type StoryScene = ReturnType<typeof createScene>;
