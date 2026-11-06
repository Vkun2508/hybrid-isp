import { useState } from "react";
import { PIPELINES, type PipelineName, type ProcessResponse } from "../api";
import { NAMES, bestOf, fixed, ms } from "../format";

/* ------------------------------------------------------------------ metrics */
export function MetricsTable({ r }: { r: ProcessResponse | null }) {
  const rows: { key: string; name: string; m: ProcessResponse["input"]["metrics"]; t: number | null }[] = r
    ? [
        { key: "input", name: "Input (no ISP)", m: r.input.metrics, t: null },
        ...PIPELINES.map((p) => ({ key: p, name: NAMES[p], m: r.outputs[p]?.metrics ?? null, t: r.outputs[p]?.latency_ms ?? null })),
      ]
    : [];
  const scored = rows.filter((x) => x.key !== "input").map((x) => x.m);
  const best = {
    psnr: bestOf(scored, "psnr"), ssim: bestOf(scored, "ssim"), de: bestOf(scored, "delta_e", true),
    t: rows.filter((x) => x.t !== null).reduce<number | null>((b, x) => (b === null || (x.t as number) < b ? x.t : b), null),
  };
  const hasGt = !!r?.input.has_gt;
  return (
    <section className="card" aria-labelledby="metrics-h">
      <div className="card-head">
        <h2 id="metrics-h">Scores against ground truth</h2>
        <p>Best of the three in bold</p>
      </div>
      <table className="table">
        <thead>
          <tr>
            <th scope="col">Pipeline</th>
            <th scope="col" className="num" title="Peak signal-to-noise ratio, higher is better">PSNR ↑</th>
            <th scope="col" className="num" title="Structural similarity, higher is better">SSIM ↑</th>
            <th scope="col" className="num" title="Mean CIEDE2000 colour error, lower is better">ΔE ↓</th>
            <th scope="col" className="num" title="Processing time on this server's CPU">Time</th>
          </tr>
        </thead>
        <tbody>
          {r === null
            ? Array.from({ length: 4 }, (_, i) => (
                <tr key={i}><td colSpan={5}><div className="skel" style={{ height: 14, width: `${70 - i * 8}%` }} /></td></tr>
              ))
            : rows.map((x) => (
                <tr key={x.key} className={x.key === "hybrid" ? "hybrid" : undefined}>
                  <td>{x.name}</td>
                  <td className={"num" + (x.m && x.m.psnr === best.psnr && x.key !== "input" ? " best" : "")}>
                    {x.m ? <>{fixed(x.m.psnr, 1)}<span className="unit">dB</span></> : <span className="dash">—</span>}
                  </td>
                  <td className={"num" + (x.m && x.m.ssim === best.ssim && x.key !== "input" ? " best" : "")}>
                    {x.m ? fixed(x.m.ssim, 3) : <span className="dash">—</span>}
                  </td>
                  <td className={"num" + (x.m && x.m.delta_e === best.de && x.key !== "input" ? " best" : "")}>
                    {x.m ? fixed(x.m.delta_e, 1) : <span className="dash">—</span>}
                  </td>
                  <td className={"num" + (x.t !== null && x.t === best.t ? " best" : "")}>
                    {x.t !== null ? ms(x.t) : <span className="dash">—</span>}
                  </td>
                </tr>
              ))}
        </tbody>
      </table>
      <p className="table-note">
        {r && !hasGt
          ? "No ground truth for this photo, so only times are shown."
          : "Live scores use the centre crop (up to 768 px). The full test-set numbers are on the Results tab."}
      </p>
    </section>
  );
}

/* ------------------------------------------------------------------ why this route */
export function RouteCard({ r }: { r: ProcessResponse | null }) {
  const a = r?.analysis;
  const rt = r?.route;
  const figs: [string, string][] = a && rt
    ? [
        ["Noise σ", a.sigma.toFixed(4)],
        ["Exposure", `${a.ev > 0 ? "+" : a.ev < 0 ? "−" : "±"}${Math.abs(a.ev).toFixed(1)} EV`],
        ["WB trust", `${Math.round(a.wb_conf * 100)}%`],
        ["Dyn. range", `${a.dynamic_range.toFixed(1)} st`],
      ]
    : [];
  return (
    <section className="card" aria-labelledby="route-h">
      <div className="card-head">
        <h2 id="route-h">Why this route</h2>
        <p>What the analyser measured on the RAW</p>
      </div>
      <div className="figures">
        {(figs.length ? figs : [["Noise σ", ""], ["Exposure", ""], ["WB trust", ""], ["Dyn. range", ""]]).map(([l, v]) => (
          <div className="figure" key={l}>
            <span className="label">{l}</span>
            {v ? <span className="value num">{v}</span> : <span className="skel" style={{ display: "block", height: 22, width: "70%", marginTop: 4 }} />}
          </div>
        ))}
      </div>
      {rt && (
        <>
          <span className={"pill route-pill" + (rt.path === "fast" ? "" : " positive")}>
            <span className="dot" /> {rt.label}{rt.forced ? " (forced)" : ""}
          </span>
          <ul className="reasons">
            {rt.reasons.map((s, i) => (
              <li key={i}>
                {s.replace(/->/g, "→").replace(/>=/g, "≥").replace(/\bsigma\b/g, "σ").replace(/ deg\b/g, "°")}
              </li>
            ))}
          </ul>
        </>
      )}
    </section>
  );
}

/* ------------------------------------------------------------------ pipeline strip */
const isCore = (name: string) => /learned core|network/i.test(name) && !/skipped|input/i.test(name);

export function PipelineStrip({ r }: { r: ProcessResponse | null }) {
  const [p, setP] = useState<PipelineName>("hybrid");
  const out = r?.outputs[p];
  const total = out ? out.stages.reduce((s, x) => s + x.ms, 0) : 0;
  return (
    <section className="card o-pipe" aria-labelledby="pipe-h">
      <div className="card-head">
        <h2 id="pipe-h">Inside the pipeline</h2>
        <div className="seg" role="group" aria-label="Pipeline">
          {PIPELINES.map((k) => (
            <button key={k} aria-pressed={p === k} onClick={() => setP(k)}>{NAMES[k]}</button>
          ))}
        </div>
      </div>
      <div className="small muted">
        From the sensor's colour-filter mosaic to the final photo{out ? <> · <span className="num">{ms(total)}</span> in {out.stages.length} stages</> : ""}
      </div>
      <div className="timebar" aria-hidden="true">
        {out?.stages.map((s, i) => (
          <span key={i} className={isCore(s.name) ? "core" : undefined} style={{ flexGrow: Math.max(s.ms, total * 0.01) }} title={`${s.name}: ${ms(s.ms)}`} />
        ))}
      </div>
      <div className="stages" tabIndex={0} aria-label="Pipeline stages">
        {out
          ? out.stages.map((s, i) => (
              <figure className="stage" key={`${r?.id}-${p}-${i}`}>
                <img
                  className="shot" src={s.image_url} alt={`${s.name} stage`} loading="lazy"
                  onLoad={(e) => { const im = e.currentTarget; if (im.naturalWidth === im.naturalHeight) im.classList.add("mosaic"); }}
                />
                <div className="stage-top">
                  <b><span className="step num">{i + 1} </span>{s.name}</b>
                  <span className="num">{ms(s.ms)}</span>
                </div>
                <p>{s.note}</p>
              </figure>
            ))
          : Array.from({ length: 5 }, (_, i) => <div key={i} className="stage"><div className="shot skel" /></div>)}
      </div>
    </section>
  );
}
