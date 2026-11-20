import { useState } from "react";
import { PIPELINES, type PipelineName, type ProcessResponse } from "../api";
import { NAMES, PATH_PLAIN, ROUTER, bestOf, cls, evText, fixed, ms, prettyReason, routeSummary, verdict } from "../format";
import { PipelineMap } from "./PipelineMap";
import { Term } from "./Term";

/* ------------------------------------------------------------------ verdict */
export function Verdict({ r }: { r: ProcessResponse | null }) {
  if (!r)
    return (
      <div className="verdict" aria-busy="true">
        <div className="skel" style={{ height: 28, width: "55%" }} />
        <div className="skel" style={{ height: 14, width: "85%", marginTop: 12 }} />
        <div className="skel" style={{ height: 14, width: "70%", marginTop: 8 }} />
      </div>
    );
  const v = verdict(r);
  const rows: { k: PipelineName | "input"; name: string; psnr: number | null; de: number | null; t: number | null }[] = [
    { k: "input", name: "Sensor", psnr: r.input.metrics?.psnr ?? null, de: r.input.metrics?.delta_e ?? null, t: null },
    ...PIPELINES.map((p) => ({ k: p, name: NAMES[p], psnr: r.outputs[p]?.metrics?.psnr ?? null, de: r.outputs[p]?.metrics?.delta_e ?? null, t: r.outputs[p]?.latency_ms ?? null })),
  ];
  const vals = rows.map((x) => x.psnr).filter((x): x is number => x !== null);
  const lo = vals.length ? Math.floor(Math.min(...vals) - 2) : 0;
  const hi = vals.length ? Math.max(...vals) : 1;
  const width = (p: number) => `${Math.max(4, ((p - lo) / Math.max(0.01, hi - lo)) * 100)}%`;
  return (
    <div className="verdict" aria-live="polite">
      <p className={"verdict-head " + (v.winner ? cls(v.winner) : "")}>
        {v.winner && <i className="sw big" />} {v.headline}
      </p>
      <ul className="verdict-lines">
        {v.lines.map((l, i) => <li key={i}>{l}</li>)}
      </ul>
      {v.scored ? (
        <div className="vbars" role="table" aria-label="Noise score per pipeline">
          <div className="vbar-row head" role="row">
            <span role="columnheader" />
            <span role="columnheader"><Term k="psnr">PSNR</Term> <span className="muted">↑</span></span>
            <span role="columnheader" className="r"><Term k="de" align="right">ΔE</Term> <span className="muted">↓</span></span>
          </div>
          {rows.filter((x) => x.psnr !== null).map((x) => (
            <div key={x.k} className={"vbar-row " + cls(x.k) + (x.k === v.winner ? " win" : "")} role="row">
              <span role="cell" className="vname"><i className="sw" />{x.name}</span>
              <span role="cell" className="vtrack"><span className="vfill" style={{ width: width(x.psnr!) }} /><span className="num">{fixed(x.psnr, 1)}</span></span>
              <span role="cell" className="num r">{fixed(x.de, 1)}</span>
            </div>
          ))}
        </div>
      ) : (
        <div className="vbars">
          {rows.filter((x) => x.t !== null).map((x) => (
            <div key={x.k} className={"vbar-row " + cls(x.k)}>
              <span className="vname"><i className="sw" />{x.name}</span>
              <span className="num muted">{ms(x.t!)}</span>
              <span />
            </div>
          ))}
        </div>
      )}
      <details className="more">
        <summary>All the numbers</summary>
        <MetricsTable r={r} />
      </details>
    </div>
  );
}

export function MetricsTable({ r }: { r: ProcessResponse }) {
  const rows = [
    { key: "input", name: "Sensor (no ISP)", m: r.input.metrics, t: null as number | null },
    ...PIPELINES.map((p) => ({ key: p, name: NAMES[p], m: r.outputs[p]?.metrics ?? null, t: r.outputs[p]?.latency_ms ?? null })),
  ];
  const scored = rows.filter((x) => x.key !== "input").map((x) => x.m);
  const best = {
    psnr: bestOf(scored, "psnr"), ssim: bestOf(scored, "ssim"), de: bestOf(scored, "delta_e", true),
    t: rows.filter((x) => x.t !== null).reduce<number | null>((b, x) => (b === null || (x.t as number) < b ? x.t : b), null),
  };
  return (
    <>
      <div className="table-wrap">
        <table className="table">
          <thead>
            <tr>
              <th scope="col">Pipeline</th>
              <th scope="col" className="num">PSNR ↑</th>
              <th scope="col" className="num">SSIM ↑</th>
              <th scope="col" className="num">ΔE ↓</th>
              <th scope="col" className="num">Time</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((x) => (
              <tr key={x.key} className={x.key === "hybrid" ? "hybrid" : undefined}>
                <td><i className={"sw " + cls(x.key)} />{x.name}</td>
                <td className={"num" + (x.m && x.m.psnr === best.psnr && x.key !== "input" ? " best" : "")}>
                  {x.m ? <>{fixed(x.m.psnr, 1)}<span className="unit">dB</span></> : <span className="dash">—</span>}
                </td>
                <td className={"num" + (x.m && x.m.ssim === best.ssim && x.key !== "input" ? " best" : "")}>{x.m ? fixed(x.m.ssim, 3) : <span className="dash">—</span>}</td>
                <td className={"num" + (x.m && x.m.delta_e === best.de && x.key !== "input" ? " best" : "")}>{x.m ? fixed(x.m.delta_e, 1) : <span className="dash">—</span>}</td>
                <td className={"num" + (x.t !== null && x.t === best.t ? " best" : "")}>{x.t !== null ? ms(x.t) : <span className="dash">—</span>}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="table-note">
        {r.input.has_gt
          ? "Scores use the centre crop (up to 768 px) against the clean reference. Best of the three in bold."
          : "No ground truth for this photo, so only times are shown."}
      </p>
    </>
  );
}

/* ------------------------------------------------------------------ why this route */
function Meter({ label, value, display, pos, zones, marks }: {
  label: React.ReactNode; value: number; display: string; pos: (v: number) => number;
  zones: { from: number; to: number; label: string; kind?: "net" | "chroma" | "dead" | "act" }[];
  marks?: number[];
}) {
  const p = Math.min(100, Math.max(0, pos(value)));
  const active = zones.find((z) => p >= z.from && p <= z.to);
  return (
    <div className="meter">
      <div className="meter-top">
        <span className="meter-label">{label}</span>
        <span className="num">{display}</span>
      </div>
      <div className="meter-bar">
        {zones.map((z, i) => (
          <span key={i} className={"zone" + (z.kind ? ` ${z.kind}` : "") + (z === active ? " active" : "")}
            style={{ left: `${z.from}%`, width: `${z.to - z.from}%` }} />
        ))}
        {marks?.map((m) => <span key={m} className="tick" style={{ left: `${m}%` }} />)}
        <span className="needle" style={{ left: `${p}%` }} />
      </div>
      <div className="meter-zones">
        {zones.map((z, i) => (
          <span key={i} className={z === active ? "active" : undefined} style={{ left: `${z.from}%`, width: `${z.to - z.from}%` }}>{z.label}</span>
        ))}
      </div>
    </div>
  );
}

const logPos = (v: number) => ((Math.log10(Math.max(1e-4, v)) + 3) / 2) * 100; // 0.001 … 0.1
const evPos = (v: number) => ((v + 5) / 7) * 100;                                  // −5 … +2 EV
const drPos = (v: number) => (v / 16) * 100;                                      // 0 … 16 stops

export function RouteCard({ r }: { r: ProcessResponse | null }) {
  const a = r?.analysis;
  const rt = r?.route;
  if (!a || !rt)
    return (
      <div className="route" aria-busy="true">
        <div className="skel" style={{ height: 26, width: "60%" }} />
        <div className="skel" style={{ height: 14, width: "90%", marginTop: 12 }} />
        <div className="skel" style={{ height: 120, marginTop: 16 }} />
      </div>
    );
  const f = logPos(ROUTER.fastSigma), hS = logPos(ROUTER.highSigma);
  const d0 = evPos(-ROUTER.evThreshold), d1 = evPos(ROUTER.evThreshold);
  const lt = drPos(ROUTER.localToneDr);
  const wbPlain: Record<string, string> = {
    "as-shot": "Kept the camera's own white balance",
    blend: "Blended the camera's white balance with a grey-world estimate",
    "gray-world": "Estimated it from the scene (grey-world)",
    "shades-of-gray": "Estimated it from the scene (shades-of-grey)",
    "white-patch": "Estimated it from the brightest patch",
  };
  return (
    <div className="route" aria-live="polite">
      <p className={"route-head" + (rt.path === "fast" ? "" : " net")}>
        <span className="pill">{rt.path === "fast" ? <span className="dot" /> : <span className="dot" />}{PATH_PLAIN[rt.path]}{rt.forced ? " (forced)" : ""}</span>
      </p>
      <p className="route-why">{routeSummary(a, rt)}</p>

      <div className="meters">
        <Meter
          label={<><Term k="sigma">Noise</Term> <span className="muted">after gain</span></>}
          value={rt.sigma_eff} display={`σ ${rt.sigma_eff.toFixed(4)}`} pos={logPos}
          zones={[
            { from: 0, to: f, label: "fast path" },
            { from: f, to: hS, label: "network", kind: "net" },
            { from: hS, to: 100, label: "+ colour fix", kind: "chroma" },
          ]}
        />
        <Meter
          label={<Term k="ev">Exposure</Term>} value={a.ev} display={evText(a.ev)} pos={evPos}
          zones={[
            { from: 0, to: d0, label: "brighten", kind: "act" },
            { from: d0, to: d1, label: "OK", kind: "dead" },
            { from: d1, to: 100, label: "darken", kind: "act" },
          ]}
        />
        <Meter
          label={<Term k="dr">Dynamic range</Term>} value={a.dynamic_range} display={`${a.dynamic_range.toFixed(1)} stops`} pos={drPos}
          zones={[
            { from: 0, to: lt, label: "global tone curve" },
            { from: lt, to: 100, label: "local tone", kind: "act" },
          ]}
        />
        <div className="meter wbrow">
          <div className="meter-top">
            <span className="meter-label"><Term k="wb">White balance</Term></span>
            <span className="num">{Math.round(a.wb_conf * 100)}% trust</span>
          </div>
          <p className="small secondary">{wbPlain[rt.wb_method] ?? rt.wb_method}.</p>
        </div>
      </div>

      <PipelineMap route={rt} />

      <details className="more">
        <summary>Router log</summary>
        <ul className="reasons">
          {rt.reasons.map((s, i) => <li key={i}>{prettyReason(s)}</li>)}
        </ul>
      </details>
    </div>
  );
}

/* ------------------------------------------------------------------ inside the pipeline */
const isNet = (name: string) => /learned core|network/i.test(name) && !/skipped|input/i.test(name);

export function PipelineStrip({ r }: { r: ProcessResponse | null }) {
  const [p, setP] = useState<PipelineName>("hybrid");
  const out = r?.outputs[p];
  const total = out ? out.stages.reduce((s, x) => s + x.ms, 0) : 0;
  return (
    <div className="pipe">
      <div className="chips" role="group" aria-label="Pipeline">
        {PIPELINES.map((k) => (
          <button key={k} className="chip" aria-pressed={p === k} onClick={() => setP(k)}>
            <i className={"sw " + cls(k)} />{NAMES[k]}
          </button>
        ))}
      </div>
      <p className="small muted pipe-sum">
        {out ? <>From the sensor mosaic to the final photo: <span className="num">{out.stages.length}</span> stages in <span className="num">{ms(total)}</span>.</> : " "}
      </p>
      <div className="timebar" aria-hidden="true">
        {out?.stages.map((s, i) => (
          <span key={i} className={isNet(s.name) ? "net" : undefined} style={{ flexGrow: Math.max(s.ms, total * 0.012) }} title={`${s.name}: ${ms(s.ms)}`} />
        ))}
      </div>
      <ol className="stages">
        {out
          ? out.stages.map((s, i) => (
              <li className={"stage" + (isNet(s.name) ? " net" : "") + (/skipped/i.test(s.name) ? " skipped" : "")} key={`${r?.id}-${p}-${i}`}>
                <img
                  className="shot" src={s.image_url} alt={`${s.name} stage`} loading="lazy"
                  onLoad={(e) => { const im = e.currentTarget; if (im.naturalWidth === im.naturalHeight) im.classList.add("mosaic"); }}
                />
                <div className="stage-body">
                  <div className="stage-top">
                    <b><span className="step-i num">{i + 1}</span>{s.name}</b>
                    <span className="num">{ms(s.ms)}</span>
                  </div>
                  <p>{s.note}</p>
                  {isNet(s.name) && <span className="pill learned">neural network</span>}
                </div>
              </li>
            ))
          : Array.from({ length: 4 }, (_, i) => <li key={i} className="stage"><div className="shot skel" /><div className="skel" style={{ height: 14, width: "60%" }} /></li>)}
      </ol>
    </div>
  );
}
