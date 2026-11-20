import { useLayoutEffect, useRef, useState } from "react";
import { fixed } from "../format";

export const SERIES_NAME: Record<string, string> = { hybrid: "Hybrid", learned: "Network only", traditional: "Traditional", input: "Sensor" };
const ORDER = ["input", "traditional", "learned", "hybrid"];

interface Series { pipeline: string; psnr: number[] }

/**
 * PSNR against a growing difficulty. Lines are labelled directly at their right end (no colour-only legend),
 * the hybrid is drawn last and thickest, and hovering or tapping shows every value at that x.
 */
export function LineChart({ x, xLabel, series, unit = "dB", fmtX }: {
  x: number[]; xLabel: string; series: Series[]; unit?: string; fmtX?: (v: number) => string;
}) {
  // the viewBox follows the real width, so text stays at its true size on a phone and on a projector
  const [W, setW] = useState(420);
  const boxRef = useRef<HTMLDivElement>(null);
  useLayoutEffect(() => {
    const el = boxRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setW(Math.max(280, Math.round(el.clientWidth))));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  const H = Math.round(Math.min(300, Math.max(220, W * 0.56))), L = 34, R = W < 420 ? 96 : 108, T = 12, B = 40;
  const [hover, setHover] = useState<number | null>(null);
  const ref = useRef<SVGSVGElement>(null);
  const all = series.flatMap((s) => s.psnr).filter(Number.isFinite);
  const span = Math.max(...all) - Math.min(...all);
  const step = [1, 2, 2.5, 5, 10].find((s) => span / s <= 5) ?? 10;
  const lo = Math.floor(Math.min(...all) / step) * step, hi = Math.ceil(Math.max(...all) / step) * step;
  const ticks = Array.from({ length: Math.round((hi - lo) / step) + 1 }, (_, i) => lo + i * step);
  const px = (i: number) => L + (i * (W - L - R)) / Math.max(1, x.length - 1);
  const py = (v: number) => T + (1 - (v - lo) / Math.max(1e-6, hi - lo)) * (H - T - B);
  const sorted = [...series].sort((a, b) => ORDER.indexOf(a.pipeline) - ORDER.indexOf(b.pipeline));
  const fx = fmtX ?? ((v: number) => String(v));

  // end labels: nudge apart so they never collide
  const last = x.length - 1;
  const ends = sorted.map((s) => ({ p: s.pipeline, y: py(s.psnr[last]), v: s.psnr[last] })).sort((a, b) => a.y - b.y);
  for (let i = 1; i < ends.length; i++) if (ends[i].y - ends[i - 1].y < 14) ends[i].y = ends[i - 1].y + 14;

  const onMove = (e: React.PointerEvent) => {
    const r = ref.current?.getBoundingClientRect();
    if (!r) return;
    const sx = ((e.clientX - r.left) / r.width) * W;
    let best = 0;
    x.forEach((_, i) => { if (Math.abs(px(i) - sx) < Math.abs(px(best) - sx)) best = i; });
    setHover(best);
  };

  return (
    <div className="chart-box" ref={boxRef}>
      <svg
        ref={ref} className="chart" viewBox={`0 0 ${W} ${H}`} width="100%" role="img"
        aria-label={`PSNR against ${xLabel}: ` + sorted.map((s) => `${SERIES_NAME[s.pipeline] ?? s.pipeline} from ${fixed(s.psnr[0], 1)} to ${fixed(s.psnr[last], 1)} ${unit}`).join("; ")}
        onPointerMove={onMove} onPointerDown={onMove} onPointerLeave={() => setHover(null)}
      >
        {ticks.map((t) => (
          <g key={t}>
            <line className="gridline" x1={L} x2={W - R} y1={py(t)} y2={py(t)} />
            <text x={L - 8} y={py(t) + 4} textAnchor="end">{Number.isInteger(t) ? t : t.toFixed(1)}</text>
          </g>
        ))}
        {x.map((v, i) => <text key={i} x={px(i)} y={H - 22} textAnchor="middle">{fx(v)}</text>)}
        <text x={(L + W - R) / 2} y={H - 4} textAnchor="middle" className="axis-title">{xLabel}</text>
        {hover !== null && <line className="crosshair" x1={px(hover)} x2={px(hover)} y1={T} y2={H - B} />}
        {sorted.map((s) => (
          <g key={s.pipeline} className={`series c-${s.pipeline}`}>
            <polyline fill="none" strokeWidth={s.pipeline === "hybrid" ? 3 : 2} strokeLinejoin="round" strokeLinecap="round"
              points={s.psnr.map((v, i) => `${px(i)},${py(v)}`).join(" ")} />
            {hover !== null && <circle cx={px(hover)} cy={py(s.psnr[hover])} r={4.5} />}
          </g>
        ))}
        {ends.map((e) => (
          <g key={e.p} className={`end-label c-${e.p}`}>
            <circle cx={W - R + 10} cy={e.y - 4} r={3.5} />
            <text x={W - R + 18} y={e.y}>{SERIES_NAME[e.p] ?? e.p}</text>
          </g>
        ))}
      </svg>
      {hover !== null && (
        <div className="chart-tip" style={{ left: `${(px(hover) / W) * 100}%` }} role="status">
          <b>{xLabel}: {fx(x[hover])}</b>
          {[...sorted].reverse().map((s) => (
            <span key={s.pipeline}><i className={`sw c-${s.pipeline}`} />{SERIES_NAME[s.pipeline] ?? s.pipeline}<span className="num">{fixed(s.psnr[hover], 1)} {unit}</span></span>
          ))}
        </div>
      )}
    </div>
  );
}

/** Horizontal bars with the value at the end; the highlighted row is drawn in its own colour, the rest recede. */
export function HBars({ rows, unit, digits = 1, lowerIsBetter, domain }: {
  rows: { key: string; label: string; value: number; colour?: string; note?: string }[];
  unit?: string; digits?: number; lowerIsBetter?: boolean; domain?: [number, number];
}) {
  const vals = rows.map((r) => r.value);
  const [lo, hi] = domain ?? [Math.min(0, ...vals), Math.max(...vals)];
  const best = lowerIsBetter ? Math.min(...vals) : Math.max(...vals);
  return (
    <div className="hbars" role="table">
      {rows.map((r) => (
        <div key={r.key} className={"hbar-row " + (r.colour ? `c-${r.colour}` : "") + (r.value === best ? " best" : "")} role="row">
          <span className="hb-label" role="cell">{r.colour && <i className="sw" />}{r.label}{r.note && <span className="muted"> · {r.note}</span>}</span>
          <span className="hb-track" role="cell">
            <span className="hb-fill" style={{ width: `${Math.max(1.5, ((r.value - lo) / Math.max(1e-9, hi - lo)) * 100)}%` }} />
          </span>
          <span className="hb-val num" role="cell">{fixed(r.value, digits)}{unit && <span className="unit"> {unit}</span>}</span>
        </div>
      ))}
    </div>
  );
}

/** Signed bars around zero (an ablation's change against the full system). */
export function DeltaBars({ rows, unit = "dB" }: { rows: { key: string; label: string; delta: number }[]; unit?: string }) {
  const m = Math.max(0.5, ...rows.map((r) => Math.abs(r.delta)));
  return (
    <div className="dbars" role="table">
      {rows.map((r) => {
        const w = (Math.abs(r.delta) / m) * 50;
        return (
          <div key={r.key} className={"dbar-row" + (r.delta < -0.25 ? " hurt" : r.delta > 0.05 ? " up" : " flat")} role="row">
            <span className="db-label" role="cell">{r.label}</span>
            <span className="db-track" role="cell" aria-hidden="true">
              <span className="db-zero" />
              <span className="db-fill" style={r.delta < 0 ? { right: "50%", width: `${w}%` } : { left: "50%", width: `${w}%` }} />
            </span>
            <span className="db-val num" role="cell">{r.delta > 0 ? "+" : r.delta < 0 ? "−" : "±"}{Math.abs(r.delta).toFixed(2)} {unit}</span>
          </div>
        );
      })}
    </div>
  );
}
