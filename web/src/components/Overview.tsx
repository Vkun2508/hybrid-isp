import { useEffect, useRef, useState } from "react";
import { processImage, type Benchmarks, type ProcessResponse } from "../api";
import { fixed } from "../format";
import { preload, type Lab } from "../lab/useLab";
import { headline } from "../useBenchmarks";
import { LineChart } from "./charts";
import { HeroSweep } from "./HeroSweep";
import { PipelineMap } from "./PipelineMap";
import { Term } from "./Term";
import type { Tab } from "./TopBar";

const NIGHT = { noise: 0, exposure_ev: -3, wb_shift: 0 };
const NIGHT_WAIT_MS = 5000;

/** Counts up to a value once `go` turns true (instant with reduced motion). */
function CountUp({ to, digits, go, prefix = "" }: { to: number | null; digits: number; go: boolean; prefix?: string }) {
  const [v, setV] = useState<number | null>(null);
  const raf = useRef(0);
  useEffect(() => {
    if (to === null || !go) return;
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) { setV(to); return; }
    const t0 = performance.now(), D = 900;
    const tick = (now: number) => {
      const t = Math.min(1, (now - t0) / D);
      setV(to * (1 - Math.pow(1 - t, 3)));
      if (t < 1) raf.current = requestAnimationFrame(tick);
    };
    raf.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf.current);
  }, [to, go]);
  if (to === null) return <>—</>;
  return <>{prefix}{(v ?? (go ? 0 : to)).toFixed(digits)}</>;
}

export function Overview({ lab, bench, onTab, onTour }: { lab: Lab; bench: Benchmarks | null; onTab: (t: Tab) => void; onTour: () => void }) {
  const [shot, setShot] = useState<ProcessResponse | null>(null);
  const [counted, setCounted] = useState(false);
  const first = lab.samples?.[0];

  // the hero wants the night shot (3 stops under) — dramatic, and the tour reuses it from the server cache.
  // The as-captured shot is precomputed; if the night one isn't back within 5 s, use that instead.
  useEffect(() => {
    if (!first) return;
    const c = new AbortController();
    let settled = false;
    const use = (r: ProcessResponse) => {
      if (settled) return;
      settled = true;
      preload([r.input.image_url, r.outputs.traditional.image_url, r.outputs.hybrid.image_url])
        .then(() => { if (!c.signal.aborted) setShot(r); });
    };
    const captured = processImage({ sampleId: first.id, sim: first.preset, route: "auto", signal: c.signal });
    captured.catch(() => undefined);
    processImage({ sampleId: first.id, sim: NIGHT, route: "auto", signal: c.signal })
      .then(use)
      .catch(() => captured.then(use).catch(() => undefined));
    const t = setTimeout(() => captured.then(use).catch(() => undefined), NIGHT_WAIT_MS);
    return () => { settled = true; clearTimeout(t); c.abort(); };
  }, [first?.id]); // eslint-disable-line react-hooks/exhaustive-deps

  const h = headline(bench);
  const under = bench?.robustness?.find((r) => /expos/i.test(r.title));
  const camera = first?.name.split(" · ").slice(0, 2).join(" · ") ?? "Real phone RAW";
  const dark = shot ? shot.simulate.exposure_ev < -0.1 : true;
  const caption = `${camera} · ${dark ? `${Math.abs(shot?.simulate.exposure_ev ?? 3)} stops under-exposed` : "low light"} · real RAW, unseen in training`;
  const captionShort = `${camera.split(" · ")[0]} · ${dark ? `${Math.abs(shot?.simulate.exposure_ev ?? 3)} stops under` : "low light"} · real RAW`;

  const stats = (
    <div className="hero-stats" aria-label="Headline results on the test set">
      <div className="hstat">
        <span className="hstat-n num accent"><CountUp to={h.dPsnr} digits={1} go={counted} prefix="+" /><small>dB</small></span>
        <span className="hstat-l">cleaner than a traditional pipeline (<Term k="psnr">PSNR</Term>, {h.n ?? 26} unseen images)</span>
      </div>
      <div className="hstat">
        <span className="hstat-n num">{fixed(h.deFrom, 1)}<i>→</i><CountUp to={h.deTo} digits={1} go={counted} /></span>
        <span className="hstat-l">colour error <Term k="de">ΔE</Term>, lower is better</span>
      </div>
      <div className="hstat">
        <span className="hstat-n num"><CountUp to={h.int8?.size_mb ?? null} digits={2} go={counted} /><small>MB</small></span>
        <span className="hstat-l">network after <Term k="int8">INT8</Term>{h.int8 ? `, ${Math.round(h.int8.latency_ms_1mp)} ms per megapixel on a CPU` : ""}</span>
      </div>
    </div>
  );

  return (
    <main className="overview" id="main">
      <section className="hero" aria-labelledby="ov-h">
        <HeroSweep shot={shot} caption={caption} captionShort={captionShort} onDone={() => setCounted(true)}>
          <div className="hero-copy">
            <p className="eyebrow">Computer Vision Hackathon 2026 · Problem P2</p>
            <h1 id="ov-h">Camera maths where it’s exact. <span>A network where it isn’t.</span></h1>
            <p className="lead hero-lead">
              A hybrid smartphone <Term k="isp">camera pipeline</Term>: exact classical maths, a 1M-parameter network for the noise,
              and a <Term k="router">router</Term> that decides how much network each photo needs.
            </p>
            <div className="ov-ctas">
              <button className="btn primary large" onClick={onTour}>
                <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true"><path d="M4.5 3v10l8-5z" fill="currentColor" /></svg>
                Start the guided demo
              </button>
              <button className="btn large ghost hero-second" onClick={() => onTab("lab")}>Try your own photo</button>
            </div>
          </div>
          <div className="hero-stats-wrap wide-only">{stats}</div>
        </HeroSweep>
        <div className="hero-below">
          <p className="lead">
            A hybrid smartphone <Term k="isp">camera pipeline</Term>: exact classical maths, a 1M-parameter network for the noise,
            and a <Term k="router">router</Term> that decides how much network each photo needs.
          </p>
          <button className="btn large" onClick={() => onTab("lab")}>Try your own photo</button>
          {stats}
        </div>
      </section>

      <div className="ov-body">
      <section className="ov-how" aria-labelledby="how-h">
        <div className="ov-sec-head">
          <h2 id="how-h">How it works</h2>
          <p>Each stage goes to whichever side does it best. Grey is exact classical maths, orange is the neural network.</p>
        </div>
        <PipelineMap wide />
      </section>

      {under && h.dark && (
        <section className="ov-proof" aria-labelledby="proof-h">
          <div className="ov-sec-head">
            <h2 id="proof-h">When the light drops, it holds</h2>
            <p>
              At {Math.abs(h.dark.ev)} stops under-exposed the hybrid still scores <b className="num">{h.dark.hybrid.toFixed(1)}&nbsp;dB</b>; the
              others fall to <span className="num">{h.dark.others.toFixed(1)}&nbsp;dB</span>. The classical stage fixes exposure exactly, so the
              network only has to remove noise.
            </p>
          </div>
          <div className="card ov-chart">
            <LineChart x={under.x} xLabel="exposure (EV)" series={under.series} fmtX={(v) => (v === 0 ? "0" : `−${Math.abs(v)}`)} />
            <p className="chart-note">Mean PSNR over {bench?.meta?.subset_n ?? 8} unseen test images per level. Higher is better.</p>
          </div>
          <button className="btn" onClick={() => onTab("results")}>See all the evidence →</button>
        </section>
      )}

      <section className="ov-end">
        <h2>See it on a photo of your own</h2>
        <p>RAW from your phone (DNG) works best. A JPG works too; it is turned back into simulated RAW first.</p>
        <div className="ov-ctas">
          <button className="btn primary large" onClick={() => onTab("lab")}>Open the lab</button>
          <button className="btn large" onClick={onTour}>Guided demo</button>
        </div>
      </section>
      </div>
    </main>
  );
}
