import { useCallback, useEffect, useState, type ReactNode } from "react";
import type { Lab } from "../lab/useLab";
import { CompareViewer, Loupe } from "./CompareViewer";
import { InputPanel } from "./InputPanel";
import { PipelineStrip, RouteCard, Verdict } from "./Panels";
import { RouterControl, Simulator } from "./Simulator";
import { Term } from "./Term";

export function useMedia(q: string) {
  const [m, setM] = useState(() => window.matchMedia(q).matches);
  useEffect(() => {
    const mq = window.matchMedia(q);
    const on = () => setM(mq.matches);
    mq.addEventListener("change", on);
    return () => mq.removeEventListener("change", on);
  }, [q]);
  return m;
}

function Step({ n, id, title, sub, children, stale }: { n: number; id: string; title: ReactNode; sub: ReactNode; children: ReactNode; stale?: boolean }) {
  return (
    <section className={"step card" + (stale ? " stale" : "")} aria-busy={stale || undefined} id={id} aria-labelledby={`${id}-h`}>
      <div className="step-head">
        <span className="step-n" aria-hidden="true">{n}</span>
        <h2 id={`${id}-h`}>{title}</h2>
        <p>{sub}</p>
      </div>
      {children}
    </section>
  );
}

export function LabView({ lab, onTour }: { lab: Lab; onTour: () => void }) {
  const wide = useMedia("(min-width: 1000px)");
  const [tile, setTile] = useState({ w: 240, h: 160 });
  const onTile = useCallback((t: { w: number; h: number }) => setTile((o) => (o.w === t.w && o.h === t.h ? o : t)), []);
  const { result, source } = lab;
  const loupe = <Loupe result={result} lens={lab.lens} zoom={lab.zoom} onZoom={lab.setZoom} onTile={onTile} />;
  const name = source?.kind === "file" ? source.file.name : source?.sample.name;

  return (
    <main className="lab" id="main">
      <header className="lab-intro">
        <h1>Try it yourself</h1>
        <p className="lead">Pick a photo, make it harder, and see which pipeline copes. Every change reruns all three pipelines on this server's CPU.</p>
        <button className="btn" onClick={onTour}>
          <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true"><path d="M5 3.5v9l7-4.5z" fill="currentColor" /></svg>
          Guided demo, 5 steps
        </button>
      </header>

      <div className="lab-stage">
        <CompareViewer
          result={result} busy={lab.busy} elapsed={lab.elapsed} error={lab.error} onRetry={lab.retry} onReset={lab.backToSamples}
          pair={lab.pair} onPair={lab.setPair} split={lab.split} onSplit={lab.setSplit}
          lens={lab.lens} onLens={lab.setLens} tile={tile} zoom={lab.zoom}
        />
        {wide && loupe}
      </div>

      <div className="lab-steps">
        <Step n={1} id="step-photo" title="Choose a photo" sub={<>Real phone <Term k="raw">RAW</Term> files from scenes the network never saw, or your own.</>}>
          <InputPanel
            samples={lab.samples}
            selectedId={source?.kind === "sample" ? source.sample.id : null}
            fileName={source?.kind === "file" ? source.file.name : null}
            onSample={(s) => lab.pickSample(s)}
            onFile={lab.pickFile}
          />
          {name && (
            <div className="now">
              <p><span className="muted">Now showing</span> <b>{name}</b></p>
              {result && !lab.stale && result.input.notes.map((n, i) => <p key={i} className="small muted">{n}</p>)}
            </div>
          )}
        </Step>

        <Step n={2} id="step-harder" title="Make it harder" sub="Simulate a worse sensor or worse light, then watch the photo above.">
          <Simulator sim={lab.sim} preset={lab.preset} onSim={lab.setSim} />
        </Step>

        <Step n={3} id="step-verdict" stale={lab.stale} title="Who wins" sub={result && !result.input.has_gt ? "Your own photo has no clean reference, so there are no scores: compare by eye." : <>Scored against the clean <Term k="gt">ground truth</Term> of the same scene.</>}>
          <Verdict r={result} />
          {!wide && loupe}
        </Step>

        <Step n={4} id="step-route" stale={lab.stale} title="Why this route" sub={<>What the <Term k="router">router</Term> measured on this RAW, and the path it chose.</>}>
          <RouteCard r={result} />
          <RouterControl mode={lab.route} onMode={lab.setRoute} />
        </Step>

        <Step n={5} id="step-inside" stale={lab.stale} title="Inside the pipeline" sub="Every stage's output and its time, from the sensor mosaic to the photo.">
          <PipelineStrip r={result} />
        </Step>
      </div>
    </main>
  );
}
