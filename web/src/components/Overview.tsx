import { useEffect, useState } from "react";
import { processImage, type Benchmarks, type ProcessResponse } from "../api";
import { fixed } from "../format";
import type { Lab } from "../lab/useLab";
import { headline } from "../useBenchmarks";
import { LineChart } from "./charts";
import { Develop } from "./Develop";
import { PipelineMap } from "./PipelineMap";
import { Term } from "./Term";
import type { Tab } from "./TopBar";

export function Overview({ lab, bench, onTab, onTour }: { lab: Lab; bench: Benchmarks | null; onTab: (t: Tab) => void; onTour: () => void }) {
  const [hero, setHero] = useState<ProcessResponse | null>(null);
  const first = lab.samples?.[0];

  // the first sample is precomputed on the server, so the hero lands instantly
  useEffect(() => {
    if (!first) return;
    const c = new AbortController();
    processImage({ sampleId: first.id, sim: first.preset, route: "auto", signal: c.signal })
      .then(setHero)
      .catch(() => undefined);
    return () => c.abort();
  }, [first?.id]); // eslint-disable-line react-hooks/exhaustive-deps

  const h = headline(bench);
  const under = bench?.robustness?.find((r) => /expos/i.test(r.title));

  return (
    <main className="overview" id="main">
      <section className="ov-hero" aria-labelledby="ov-h">
        <div className="ov-copy">
          <p className="eyebrow">Computer Vision Hackathon 2026 · Problem P2</p>
          <h1 id="ov-h">Camera maths where it’s exact. A network where it isn’t.</h1>
          <p className="lead">
            A hybrid smartphone <Term k="isp">camera pipeline</Term>. Textbook maths handles the steps that have exact answers,
            a 1&nbsp;MB neural network handles the noise, and a <Term k="router">router</Term> decides, photo by photo, how much network each one needs.
          </p>
          <div className="ov-ctas">
            <button className="btn primary large" onClick={onTour}>
              <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true"><path d="M4.5 3v10l8-5z" fill="currentColor" /></svg>
              Start the guided demo
            </button>
            <button className="btn large" onClick={() => onTab("lab")}>Try your own photo</button>
          </div>
        </div>
        <div className="ov-visual">
          <Develop result={hero} />
          {first && <p className="ov-visual-note">{first.name} · real phone RAW from a scene the network never saw.</p>}
        </div>
      </section>

      <section className="ov-stats" aria-label="Headline results">
        <div className="stat">
          <span className="stat-n num">{h.dPsnr !== null ? `+${h.dPsnr.toFixed(1)}` : "—"}<small>dB</small></span>
          <span className="stat-l">cleaner than a traditional camera pipeline</span>
          <span className="stat-s"><Term k="psnr">PSNR</Term>, mean of {h.n ?? "the"} unseen test images</span>
        </div>
        <div className="stat">
          <span className="stat-n num">{h.deFrom !== null && h.deTo !== null ? <>{h.deFrom.toFixed(1)}<i>→</i>{h.deTo.toFixed(1)}</> : "—"}</span>
          <span className="stat-l">colour error{h.deFrom && h.deTo ? `, down ${Math.round((1 - h.deTo / h.deFrom) * 100)}%` : ""}</span>
          <span className="stat-s"><Term k="de">ΔE</Term> against the clean reference, lower is better</span>
        </div>
        <div className="stat">
          <span className="stat-n num">{h.int8 ? fixed(h.int8.size_mb, 2) : "—"}<small>MB</small></span>
          <span className="stat-l">network, after <Term k="int8">INT8</Term> quantisation</span>
          <span className="stat-s">{h.int8 ? `${(h.int8.params / 1e6).toFixed(1)}M parameters · ${Math.round(h.int8.latency_ms_1mp)} ms per megapixel on a CPU` : " "}</span>
        </div>
      </section>

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
    </main>
  );
}
