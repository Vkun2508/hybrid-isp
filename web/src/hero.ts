// The Overview hero's data: one unseen RAW through the three pipelines at five light levels, precomputed by
// scripts/make_hero_assets.py from the same pipeline the API runs, and served as static files from /hero/.

export type HeroPipe = "traditional" | "learned" | "hybrid";
export type HeroVariant = "d" | "m"; // desktop full frame, phone centre crop
export type HeroMetric = { psnr: number; ssim?: number; delta_e?: number };

export type HeroData = {
  sample: string;
  name: string;
  camera: string;
  iso: number;
  dataset: string;
  split: string;
  levels: number[]; // EV per level: 0, -1, -2, -3, -4
  metrics: Record<HeroPipe, HeroMetric[]>;
  route: string[];
  size: Record<HeroVariant, [number, number]>;
};

// left to right: the hybrid sits between its two halves, so both of its edges are seams
export const HERO_PIPES: HeroPipe[] = ["traditional", "hybrid", "learned"];
const BASE = "/hero"; // web/public/hero, served next to the app

export const heroImage = (v: HeroVariant, p: HeroPipe, k: number) => `${BASE}/${v}/${p}_${k}.webp`;
export const heroSensor = (v: HeroVariant) => `${BASE}/${v}/sensor.webp`;
export const heroVariant = (): HeroVariant => (window.matchMedia("(max-width: 699px)").matches ? "m" : "d");

let cached: Promise<HeroData> | null = null;
export function loadHero(): Promise<HeroData> {
  cached ??= fetch(`${BASE}/hero.json`).then((r) => {
    if (!r.ok) throw new Error(`hero.json ${r.status}`);
    return r.json() as Promise<HeroData>;
  }).catch((e) => { cached = null; throw e; });
  return cached;
}

/** "As shot", "−3 EV" … and how much light that is. */
export const stopText = (k: number) => (k === 0 ? "As shot" : `−${k} EV`);
export const lessLight = (k: number) => (k === 0 ? "full light" : `${2 ** k}× less light`);
