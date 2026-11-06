import type { Metrics, PipelineName, ProcessResponse } from "./api";

export const NAMES: Record<PipelineName, string> = {
  traditional: "Traditional",
  learned: "Learned only",
  hybrid: "Hybrid",
};

export const fixed = (v: number | null | undefined, d: number) =>
  v === null || v === undefined || !Number.isFinite(v) ? "—" : v.toFixed(d);

export const signed = (v: number, d: number) => (v > 0 ? "+" : v < 0 ? "−" : "±") + Math.abs(v).toFixed(d);

export const ms = (v: number) => (v >= 1000 ? `${(v / 1000).toFixed(2)} s` : `${Math.round(v)} ms`);

export function finding(r: ProcessResponse): { lead: string; detail: string; good: boolean } {
  const h = r.outputs.hybrid, t = r.outputs.traditional, l = r.outputs.learned;
  if (h?.metrics && t?.metrics) {
    const dp = h.metrics.psnr - t.metrics.psnr;
    const de = h.metrics.delta_e - t.metrics.delta_e;
    const vsLearned = l?.metrics ? h.metrics.psnr - l.metrics.psnr : null;
    return {
      lead: `Hybrid ${signed(dp, 1)} dB PSNR against traditional`,
      detail:
        `colour error ΔE ${t.metrics.delta_e.toFixed(1)} → ${h.metrics.delta_e.toFixed(1)} (${signed(de, 1)})` +
        (vsLearned !== null ? `, ${signed(vsLearned, 1)} dB against the network alone` : "") +
        `, in ${ms(h.latency_ms)}.`,
      good: dp > 0,
    };
  }
  return {
    lead: `Router chose: ${r.route.label}`,
    detail: "No ground truth for this photo, so there are no scores. Compare the pipelines by eye with the slider and the loupe.",
    good: true,
  };
}

export function bestOf(rows: (Metrics | null)[], key: keyof Metrics, lowerIsBetter = false): number | null {
  let best: number | null = null;
  for (const m of rows) {
    if (!m) continue;
    const v = m[key];
    if (best === null || (lowerIsBetter ? v < best : v > best)) best = v;
  }
  return best;
}
