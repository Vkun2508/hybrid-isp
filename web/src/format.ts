import type { Analysis, Metrics, PipelineName, ProcessResponse, Route } from "./api";

export type ImageKey = "input" | "traditional" | "learned" | "hybrid" | "gt";

export const NAMES: Record<PipelineName, string> = {
  traditional: "Traditional",
  learned: "Network only",
  hybrid: "Hybrid",
};

/** Display names for everything the viewer can show. */
export const LABEL: Record<ImageKey, string> = {
  input: "Sensor (no processing)",
  traditional: "Traditional",
  learned: "Network only",
  hybrid: "Hybrid",
  gt: "Ground truth",
};
export const SHORT: Record<ImageKey, string> = {
  input: "Sensor", traditional: "Traditional", learned: "Network only", hybrid: "Hybrid", gt: "Ground truth",
};
/** One line on what each pipeline is, for legends and captions. */
export const WHAT: Record<PipelineName, string> = {
  traditional: "Textbook camera pipeline, no network",
  learned: "The network alone, minimal classical steps",
  hybrid: "Classical stages + the network, chosen per photo",
};
export const cls = (k: string) => `c-${k}`;

export const fixed = (v: number | null | undefined, d: number) =>
  v === null || v === undefined || !Number.isFinite(v) ? "—" : v.toFixed(d);

export const signed = (v: number, d: number) => (v > 0 ? "+" : v < 0 ? "−" : "±") + Math.abs(v).toFixed(d);

export const ms = (v: number) => (v >= 1000 ? `${(v / 1000).toFixed(2)} s` : `${Math.round(v)} ms`);

export const evText = (ev: number, d = 1) => `${ev > 0 ? "+" : ev < 0 ? "−" : "±"}${Math.abs(ev).toFixed(d)} EV`;

export function bestOf(rows: (Metrics | null)[], key: keyof Metrics, lowerIsBetter = false): number | null {
  let best: number | null = null;
  for (const m of rows) {
    if (!m) continue;
    const v = m[key];
    if (best === null || (lowerIsBetter ? v < best : v > best)) best = v;
  }
  return best;
}

/* ------------------------------------------------------------------ verdict (plain words) */
export interface Verdict {
  scored: boolean;
  winner: PipelineName | null;
  headline: string;
  lines: string[];
  hybridWins: boolean;
}

const TIE_DB = 0.3;

export function verdict(r: ProcessResponse): Verdict {
  const h = r.outputs.hybrid?.metrics, t = r.outputs.traditional?.metrics, l = r.outputs.learned?.metrics;
  if (!h || !t) {
    return {
      scored: false, winner: null, hybridWins: false,
      headline: "No reference photo, so judge by eye",
      lines: [
        "Your own photos have no clean ground truth to score against. Drag the slider and use the loupe to compare.",
        `The hybrid took ${ms(r.outputs.hybrid.latency_ms)} on this server's CPU.`,
      ],
    };
  }
  const ranked = (["traditional", "learned", "hybrid"] as PipelineName[])
    .filter((p) => r.outputs[p]?.metrics)
    .sort((a, b) => r.outputs[b].metrics!.psnr - r.outputs[a].metrics!.psnr);
  const top = ranked[0];
  const topPsnr = r.outputs[top].metrics!.psnr;
  const hybridWins = top === "hybrid" || topPsnr - h.psnr < TIE_DB;
  const dT = h.psnr - t.psnr;
  const lines: string[] = [];
  lines.push(
    Math.abs(dT) < TIE_DB
      ? "About level with the traditional pipeline on noise (PSNR)."
      : dT > 0
        ? `${dT.toFixed(1)} dB cleaner than the traditional pipeline.`
        : `${Math.abs(dT).toFixed(1)} dB behind the traditional pipeline here.`,
  );
  if (l) {
    const dL = h.psnr - l.psnr;
    lines.push(
      Math.abs(dL) < TIE_DB
        ? "Level with the network on its own."
        : dL > 0
          ? `${dL.toFixed(1)} dB cleaner than the network on its own.`
          : `${Math.abs(dL).toFixed(1)} dB behind the network on its own.`,
    );
  }
  lines.push(`Colour error ΔE ${t.delta_e.toFixed(1)} (traditional) → ${h.delta_e.toFixed(1)} (hybrid).`);
  let headline: string;
  if (top === "hybrid" && ranked.length > 1 && topPsnr - r.outputs[ranked[1]].metrics!.psnr >= TIE_DB) headline = "Hybrid wins";
  else if (hybridWins) headline = "Hybrid ties for best";
  else headline = `${NAMES[top]} wins this one`;
  return { scored: true, winner: top, headline, lines, hybridWins };
}

/* ------------------------------------------------------------------ router, in plain words */
// Mirrors isp/router_config.json (thresholds tuned on the validation split).
export const ROUTER = { fastSigma: 0.004, highSigma: 0.02, evThreshold: 0.4, localToneDr: 12 };

export const PATH_PLAIN: Record<Route["path"], string> = {
  fast: "Fast classical path: the network is skipped",
  learned: "Sent through the network",
  learned_chroma: "Through the network, plus colour-noise cleanup",
};

export function routeSummary(a: Analysis, rt: Route): string {
  const noisy = rt.sigma_eff >= ROUTER.highSigma ? "very noisy" : rt.sigma_eff >= ROUTER.fastSigma ? "noisy" : "clean";
  let s = `This RAW is ${noisy} (σ ${rt.sigma_eff.toFixed(3)})`;
  if (Math.abs(a.ev) >= ROUTER.evThreshold) s += ` and ${Math.abs(a.ev).toFixed(1)} stops ${a.ev < 0 ? "too dark" : "too bright"}`;
  if (rt.forced) return `${s}. You forced the ${rt.path === "fast" ? "fast path" : "network"} with the override below.`;
  const then = rt.path === "fast"
    ? "the network isn't needed"
    : rt.path === "learned"
      ? "the network denoises it"
      : "the network denoises it and a filter cleans up the colour noise";
  if (rt.gain > 1.05) return `${s}. It brightens the RAW ×${rt.gain.toFixed(1)} first (exact maths), then ${then}.`;
  return `${s}, so ${then}.`;
}

export const prettyReason = (s: string) =>
  s.replace(/->/g, "→").replace(/>=/g, "≥").replace(/\bsigma\b/g, "σ").replace(/ deg\b/g, "°");
