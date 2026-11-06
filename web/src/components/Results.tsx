import { useEffect, useState } from "react";
import { ApiError, getBenchmarks, type Benchmarks } from "../api";
import { fixed } from "../format";

const SERIES_COLOUR: Record<string, string> = {
  hybrid: "var(--accent)", learned: "var(--chart-neutral-1)", traditional: "var(--chart-neutral-2)", input: "#5a5c62",
};
const SERIES_NAME: Record<string, string> = { hybrid: "Hybrid", learned: "Learned only", traditional: "Traditional", input: "Input" };

function LineChart({ x, xLabel, series }: { x: number[]; xLabel: string; series: { pipeline: string; psnr: number[] }[] }) {
  const W = 640, H = 220, L = 40, R = 12, T = 12, B = 34;
  const all = series.flatMap((s) => s.psnr).filter(Number.isFinite);
  const lo = Math.floor(Math.min(...all) / 2) * 2, hi = Math.ceil(Math.max(...all) / 2) * 2;
  const ticks = Array.from({ length: 5 }, (_, i) => lo + ((hi - lo) * i) / 4);
  const px = (i: number) => L + (i * (W - L - R)) / Math.max(1, x.length - 1);
  const py = (v: number) => T + (1 - (v - lo) / Math.max(1e-6, hi - lo)) * (H - T - B);
  return (
    <svg className="chart" viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label={`PSNR against ${xLabel}`}>
      {ticks.map((t) => (
        <g key={t}>
          <line className="gridline" x1={L} x2={W - R} y1={py(t)} y2={py(t)} />
          <text x={L - 8} y={py(t) + 4} textAnchor="end">{t.toFixed(0)}</text>
        </g>
      ))}
      {x.map((v, i) => <text key={i} x={px(i)} y={H - 14} textAnchor="middle">{v}</text>)}
      <text x={(L + W - R) / 2} y={H - 1} textAnchor="middle">{xLabel}</text>
      {[...series].sort((a) => (a.pipeline === "hybrid" ? 1 : -1)).map((s) => (
        <polyline key={s.pipeline} fill="none" stroke={SERIES_COLOUR[s.pipeline] ?? "#888"} strokeWidth={s.pipeline === "hybrid" ? 2.5 : 1.75}
          points={s.psnr.map((v, i) => `${px(i)},${py(v)}`).join(" ")} />
      ))}
    </svg>
  );
}

function Legend({ keys }: { keys: string[] }) {
  return (
    <div className="legend">
      {keys.map((k) => <span key={k}><i style={{ background: SERIES_COLOUR[k] ?? "#888" }} />{SERIES_NAME[k] ?? k}</span>)}
    </div>
  );
}

export function Results() {
  const [b, setB] = useState<Benchmarks | null>(null);
  const [err, setErr] = useState<ApiError | null>(null);
  const load = () => {
    setErr(null);
    getBenchmarks().then(setB).catch((e) => setErr(e as ApiError));
  };
  useEffect(load, []);

  if (err)
    return (
      <main className="results">
        <section className="card" role="alert">
          <h2>Couldn't load the results</h2>
          <p className="secondary" style={{ margin: "8px 0 12px" }}>{err.message} {err.hint}</p>
          <button className="btn" onClick={load}>Try again</button>
        </section>
      </main>
    );

  const pending = !b || b.status === "pending";
  return (
    <main className="results">
      <div>
        <h1>Results on held-out test scenes</h1>
        <p className="secondary" style={{ marginTop: 6 }}>
          {b?.dataset
            ? `${b.dataset.name} · test scenes ${b.dataset.test_scenes.join(", ")} · ${b.dataset.n_images} images${b.updated ? ` · updated ${b.updated}` : ""}`
            : "SIDD-CC smartphone RAW · scenes never seen in training"}
        </p>
      </div>

      {pending && (
        <section className="card pending">
          <span className="pill warning" style={{ justifySelf: "start" }}><span className="dot" /> Evaluation pending</span>
          <p className="secondary">
            {b?.message ?? "The learned core is still training. These numbers are computed on the held-out test scenes as soon as it finishes."}
          </p>
          <ul>
            <li>Quality: PSNR, SSIM, ΔE2000 and LPIPS for traditional, learned-only and hybrid</li>
            <li>Efficiency: parameters, model size and CPU latency for FP32 and INT8</li>
            <li>Robustness: quality as noise, under-exposure and white-balance error grow</li>
            <li>Ablation: the hybrid with each stage switched off</li>
            <li>Failure cases, with pictures</li>
          </ul>
        </section>
      )}

      {b?.status === "provisional" && (
        <section className="card pending" role="note">
          <span className="pill warning" style={{ justifySelf: "start" }}><span className="dot" /> Provisional numbers</span>
          <p className="secondary">
            Measured with the classical stand-in in place of the trained network ({b.model?.variant}). The learned core is
            still training; these tables are recomputed on the same test scenes when it finishes.
          </p>
        </section>
      )}

      {b?.quality && (
        <section className="card">
          <div className="card-head"><h2>Quality</h2><p>Mean over the test images · best in bold</p></div>
          <table className="table">
            <thead><tr><th>Pipeline</th><th className="num">PSNR ↑</th><th className="num">SSIM ↑</th><th className="num">ΔE ↓</th><th className="num">LPIPS ↓</th></tr></thead>
            <tbody>
              {b.quality.map((q) => {
                const sc = b.quality!.filter((x) => x.pipeline !== "input");
                const bp = Math.max(...sc.map((x) => x.psnr)), bs = Math.max(...sc.map((x) => x.ssim)), bd = Math.min(...sc.map((x) => x.delta_e));
                const lp = sc.map((x) => x.lpips).filter((v): v is number => typeof v === "number");
                const bl = lp.length ? Math.min(...lp) : null;
                const scoredRow = q.pipeline !== "input";
                return (
                  <tr key={q.pipeline} className={q.pipeline === "hybrid" ? "hybrid" : undefined}>
                    <td>{SERIES_NAME[q.pipeline] ?? q.pipeline}</td>
                    <td className={"num" + (scoredRow && q.psnr === bp ? " best" : "")}>{fixed(q.psnr, 2)}<span className="unit">dB</span></td>
                    <td className={"num" + (scoredRow && q.ssim === bs ? " best" : "")}>{fixed(q.ssim, 4)}</td>
                    <td className={"num" + (scoredRow && q.delta_e === bd ? " best" : "")}>{fixed(q.delta_e, 2)}</td>
                    <td className={"num" + (scoredRow && q.lpips === bl ? " best" : "")}>{q.lpips != null ? fixed(q.lpips, 4) : <span className="dash">—</span>}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </section>
      )}

      {b?.robustness && b.robustness.length > 0 && (
        <div className="grid2">
          {b.robustness.map((rb) => (
            <section className="card" key={rb.title}>
              <div className="card-head"><h2>{rb.title}</h2><Legend keys={rb.series.map((s) => s.pipeline)} /></div>
              <LineChart x={rb.x} xLabel={rb.x_label} series={rb.series} />
            </section>
          ))}
        </div>
      )}

      {b?.efficiency && (
        <section className="card">
          <div className="card-head"><h2>Efficiency</h2><p>CPU, one 1-megapixel RAW frame</p></div>
          <div className="grid2">
            {(["size_mb", "latency_ms_1mp"] as const).map((k) => {
              const max = Math.max(...b.efficiency!.map((e) => e[k]));
              return (
                <div className="bars" key={k}>
                  <h3>{k === "size_mb" ? "Model size" : "Latency"}</h3>
                  {b.efficiency!.map((e) => (
                    <div className="bar-row" key={e.variant}>
                      <span className="small secondary">{e.variant}</span>
                      <div className="bar-track"><div className={"bar-fill" + (/int8/i.test(e.variant) ? " accent" : "")} style={{ width: `${(e[k] / max) * 100}%` }} /></div>
                      <span className="num small" style={{ textAlign: "right" }}>{k === "size_mb" ? `${fixed(e.size_mb, 2)} MB` : `${Math.round(e.latency_ms_1mp)} ms`}</span>
                    </div>
                  ))}
                </div>
              );
            })}
          </div>
        </section>
      )}

      {b?.ablation && (
        <section className="card">
          <div className="card-head"><h2>Ablation</h2><p>The hybrid with one part removed at a time</p></div>
          <table className="table">
            <thead><tr><th>Variant</th><th className="num">PSNR</th><th className="num">SSIM</th><th className="num">ΔE</th></tr></thead>
            <tbody>
              {b.ablation.map((a) => (
                <tr key={a.variant}>
                  <td>{a.variant}{a.note && <span className="muted small"> · {a.note}</span>}</td>
                  <td className="num">{fixed(a.psnr, 2)}<span className="unit">dB</span></td>
                  <td className="num">{fixed(a.ssim, 4)}</td>
                  <td className="num">{fixed(a.delta_e, 2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}

      {b?.failures && b.failures.length > 0 && (
        <section className="card">
          <div className="card-head"><h2>Where it still fails</h2><p>Honest failure cases</p></div>
          <div className="gallery">
            {b.failures.map((f) => (
              <figure key={f.title}>
                {f.image_url && <img src={f.image_url} alt={f.title} loading="lazy" />}
                <h3 style={{ marginTop: 8 }}>{f.title}</h3>
                <p className="small muted" style={{ marginTop: 2 }}>{f.description}</p>
              </figure>
            ))}
          </div>
        </section>
      )}
    </main>
  );
}
