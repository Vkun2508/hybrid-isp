import { useCallback, useEffect, useRef, useState } from "react";
import {
  ApiError, MAX_UPLOAD_MB, getHealth, getSamples, processImage,
  type ModelInfo, type ProcessResponse, type RouteMode, type Sample, type SimPreset,
} from "./api";
import { finding } from "./format";
import { CompareViewer } from "./components/CompareViewer";
import { InputPanel } from "./components/InputPanel";
import { MetricsTable, PipelineStrip, RouteCard } from "./components/Panels";
import { Results } from "./components/Results";
import { RouterControl, Simulator } from "./components/Simulator";
import { TopBar, type Tab } from "./components/TopBar";

type Source = { kind: "sample"; sample: Sample } | { kind: "file"; file: File };
const ZERO: SimPreset = { noise: 0, exposure_ev: 0, wb_shift: 0 };
const same = (a: SimPreset, b: SimPreset) => a.noise === b.noise && a.exposure_ev === b.exposure_ev && a.wb_shift === b.wb_shift;

export default function App() {
  const [tab, setTab] = useState<Tab>(() => (location.hash === "#results" ? "results" : "lab"));
  const [model, setModel] = useState<ModelInfo | null>(null);
  const [samples, setSamples] = useState<Sample[] | null>(null);
  const [source, setSource] = useState<Source | null>(null);
  const [sim, setSim] = useState<SimPreset>(ZERO);
  const [route, setRoute] = useState<RouteMode>("auto");
  const [result, setResult] = useState<ProcessResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);
  const [startedAt, setStartedAt] = useState(0);
  const [now, setNow] = useState(0);
  const ctrl = useRef<AbortController | null>(null);

  const changeTab = (t: Tab) => {
    setTab(t);
    history.replaceState(null, "", t === "results" ? "#results" : "#lab");
  };

  // boot: model status + samples; the first sample is precomputed on the server, so it lands instantly
  useEffect(() => {
    getHealth().then((h) => setModel(h.model)).catch(() => undefined);
    getSamples()
      .then((s) => {
        setSamples(s);
        if (s.length) {
          setSource({ kind: "sample", sample: s[0] });
          setSim(s[0].preset);
        }
      })
      .catch((e) => setError(e as ApiError));
  }, []);

  const run = useCallback((src: Source, s: SimPreset, rt: RouteMode) => {
    ctrl.current?.abort();
    const c = new AbortController();
    ctrl.current = c;
    setBusy(true);
    setError(null);
    setStartedAt(performance.now());
    processImage({
      sampleId: src.kind === "sample" ? src.sample.id : undefined,
      file: src.kind === "file" ? src.file : undefined,
      sim: s, route: rt, signal: c.signal,
    })
      .then((r) => {
        if (c.signal.aborted) return;
        setResult(r);
        setModel(r.model);
      })
      .catch((e) => {
        if ((e as Error).name === "AbortError") return;
        setError(e instanceof ApiError ? e : new ApiError("unknown", "Something went wrong.", "Try again.", 0));
      })
      .finally(() => {
        if (ctrl.current === c) setBusy(false);
      });
  }, []);

  // any change of input, simulator or router reprocesses (debounced while a slider is being dragged)
  useEffect(() => {
    if (!source) return;
    const t = setTimeout(() => run(source, sim, route), 350);
    return () => clearTimeout(t);
  }, [source, sim, route, run]);

  // elapsed counter while processing
  useEffect(() => {
    if (!busy) return;
    const id = setInterval(() => setNow(performance.now()), 100);
    return () => clearInterval(id);
  }, [busy]);

  const pickSample = (s: Sample) => {
    setSource({ kind: "sample", sample: s });
    setSim(s.preset);
  };
  const pickFile = (f: File) => {
    if (f.size > MAX_UPLOAD_MB * 1024 * 1024) {
      setError(new ApiError("too_large", `That file is over ${MAX_UPLOAD_MB} MB.`, "Use a smaller photo, or export a JPG from your phone.", 413));
      return;
    }
    setSource({ kind: "file", file: f });
    setSim(ZERO);
  };
  const preset = source?.kind === "sample" ? source.sample.preset : ZERO;
  const f = result ? finding(result) : null;
  const title = source?.kind === "file" ? source.file.name : source?.sample.name ?? "Loading samples…";

  return (
    <>
      <a className="sr-only" href="#main">Skip to content</a>
      <TopBar tab={tab} onTab={changeTab} model={model} />
      {tab === "results" ? (
        <Results />
      ) : (
        <main className="lab" id="main">
          <div className="side">
            <InputPanel
              samples={samples}
              selectedId={source?.kind === "sample" ? source.sample.id : null}
              fileName={source?.kind === "file" ? source.file.name : null}
              onSample={pickSample}
              onFile={pickFile}
            />
            <Simulator sim={sim} onSim={setSim} onReset={() => setSim(preset)} dirty={!same(sim, preset)} />
            <RouterControl mode={route} onMode={setRoute} />
          </div>
          <div className="main">
            <div className="title-row o-title">
              <div style={{ minWidth: 0 }}>
                <h1>{title}</h1>
                <p className="finding" aria-live="polite">
                  {f ? (
                    <>
                      <strong className={f.good ? undefined : "secondary"}>{f.lead}</strong>
                      {", "}
                      {f.detail}
                    </>
                  ) : (
                    <span className="skel" style={{ display: "inline-block", height: 14, width: 320, maxWidth: "100%" }} />
                  )}
                </p>
                {result && result.input.notes.length > 0 && (
                  <div className="notes">{result.input.notes.map((n, i) => <span key={i}>{n}</span>)}</div>
                )}
              </div>
            </div>
            <CompareViewer
              result={result}
              busy={busy}
              elapsed={busy ? Math.max(0, now - startedAt) : 0}
              error={error}
              onRetry={() => source && run(source, sim, route)}
            />
            <div className="duo o-duo">
              <MetricsTable r={result} />
              <RouteCard r={result} />
            </div>
            <PipelineStrip r={result} />
          </div>
        </main>
      )}
      <footer className="foot">
        <p>
          Real RAW samples: SIDD-CC (Abdelhamed et al., CVPR 2018), from scenes held out of training. Photos: scikit-image test images (public domain).
        </p>
        <p>Uploads are processed in memory and never stored. Computer Vision Hackathon 2026 · Problem P2.</p>
      </footer>
    </>
  );
}
