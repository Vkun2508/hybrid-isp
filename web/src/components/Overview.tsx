import { useEffect, useRef, useState } from "react";
import { processImage, type Benchmarks } from "../api";
import { fixed } from "../format";
import { lessLight, loadHero, type HeroData } from "../hero";
import type { Lab } from "../lab/useLab";
import { headline } from "../useBenchmarks";
import { LineChart } from "./charts";
import { HeroLight } from "./HeroLight";
import { SensorStory } from "./SensorStory";
import { PipelineMap } from "./PipelineMap";
import { Term } from "./Term";
import type { Tab } from "./TopBar";

// the guided tour opens on this photo at −3 EV: ask the server for it early so the tour never waits
const NIGHT = { noise: 0, exposure_ev: -3, wb_shift: 0 };

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
  const [hero, setHero] = useState<HeroData | null>(null);
  const [counted, setCounted] = useState(false);
  // the 3D sensor story, unless motion is reduced (or WebGL/data fails): then the "Lights down" hero
  const [mode, setMode] = useState<"story" | "light">(() =>
    window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "light" : "story");
  const stop = useRef(0); // the dial's stop, read on click only (state here would re-render the whole page per stop)
  const warmed = useRef(false);

  // the hero is static (precomputed by scripts/make_hero_assets.py), so it never waits on a cold server
  useEffect(() => { loadHero().then(setHero).catch(() => undefined); }, []);
  const sample = lab.samples?.find((s) => s.id === hero?.sample) ?? lab.samples?.[0];

  // once the hero has played, have the server compute the tour's night shot in the background
  useEffect(() => {
    if (!counted || !sample || warmed.current) return;
    warmed.current = true;
    processImage({ sampleId: sample.id, sim: NIGHT, route: "auto" }).catch(() => undefined);
  }, [counted, sample]);

  // the same RAW at the dial's light level, run live by the server
  const runLive = (k: number) => {
    if (!sample) { onTab("lab"); return; }
    lab.setRoute("auto");
    lab.pickSample(sample, { noise: 0, exposure_ev: -k, wb_shift: 0 });
    onTab("lab");
  };

  const h = headline(bench);
  const under = bench?.robustness?.find((r) => /expos/i.test(r.title));
  const last = hero ? hero.levels.length - 1 : 4;
  const hyLast = hero?.metrics.hybrid[last]?.psnr;
  const lead = (
    <>
      One real <Term k="raw">RAW</Term>, three pipelines. Take the light away: camera maths alone and a network alone both fall apart,
      while the hybrid still scores <b>{fixed(hyLast, 1)}&nbsp;dB</b> with {lessLight(last)}.
    </>
  );

  const stats = (
    <div className="hero-stats" aria-label="Headline results on the test set">
      <div className="hstat">
        <span className="hstat-n num accent"><CountUp to={h.dPsnr} digits={1} go={counted} prefix="+" /><small>dB</small></span>
        <span className="hstat-l">cleaner than a traditional pipeline: the <span style={{ whiteSpace: "nowrap" }}>average <Term k="psnr">PSNR</Term></span> over {h.n ?? 26} unseen test images</span>
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
      {mode === "story" ? (
        <>
          <SensorStory
            onFallback={() => setMode("light")}
            onTour={onTour}
            onTab={onTab}
            onRunLive={() => runLive(3)}
            onDone={() => setCounted(true)}
          />
          <section className="hero-below story-below" aria-label="Headline results">{stats}</section>
        </>
      ) : (
      <section className="hero" aria-labelledby="ov-h">
        <HeroLight
          hero={hero}
          onDone={() => setCounted(true)}
          onStop={(k) => { stop.current = k; }}
          onRunLive={runLive}
          copy={
            <div className="hero-copy">
              <p className="eyebrow">Computer Vision Hackathon 2026 · Problem P2</p>
              <h1 id="ov-h">Camera maths where it’s exact. <span>A network where it isn’t.</span></h1>
              <p className="lead hero-lead">{lead}</p>
              <div className="ov-ctas">
                <button className="btn primary large" onClick={onTour}>
                  <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true"><path d="M4.5 3v10l8-5z" fill="currentColor" /></svg>
                  Start the guided demo
                </button>
                <button className="btn large ghost hero-second" onClick={() => onTab("lab")}>Try your own photo</button>
              </div>
            </div>
          }
        />
        <div className="hero-below">
          <p className="lead hb-only">{lead}</p>
          <div className="ov-ctas hb-only">
            <button className="btn large" onClick={() => runLive(stop.current)}>Run this photo live</button>
            <button className="btn large" onClick={() => onTab("lab")}>Try your own photo</button>
          </div>
          <p className="hb-cap hb-only">
            {hero ? `${hero.camera} · ISO ${hero.iso} · real RAW from the ${hero.dataset?.split(" (")[0] ?? "test"} test split, never seen in training. ` : ""}
            Dragging uses precomputed results from this same pipeline; “Run this photo live” runs it on the server.
          </p>
          {stats}
        </div>
      </section>
      )}

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
              baselines, which have no auto-exposure of their own, fall to <span className="num">{h.dark.others.toFixed(1)}&nbsp;dB</span>. The classical stage fixes exposure exactly, so the
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
