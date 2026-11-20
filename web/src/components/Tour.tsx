import { useEffect, useRef, useState, type ReactNode } from "react";
import type { Benchmarks, ProcessResponse } from "../api";
import { fixed, routeSummary } from "../format";
import type { Lab } from "../lab/useLab";
import { headline } from "../useBenchmarks";

interface Ctx { r: ProcessResponse | null; b: Benchmarks | null; waiting: boolean }
interface StepDef {
  title: string;
  target: string | null;
  /** changes app state; returns true when that triggers a re-run we must wait for */
  enter?: (lab: Lab) => boolean;
  body: (c: Ctx) => ReactNode;
}

const DEMO_SAMPLE = "sidd-n6-iso3200-lowlight";
const psnr = (r: ProcessResponse | null, p: "traditional" | "learned" | "hybrid") => fixed(r?.outputs[p]?.metrics?.psnr, 1);

const STEPS: StepDef[] = [
  {
    title: "A real phone photo, never seen in training",
    target: "#viewer",
    enter: (lab) => {
      const s = lab.samples?.find((x) => x.id === DEMO_SAMPLE) ?? lab.samples?.[0];
      if (!s) return false;
      lab.setRoute("auto");
      lab.setPair(["traditional", "hybrid"]);
      lab.setSplit(50);
      lab.setZoom(2);
      lab.pickSample(s);
      return true;
    },
    body: ({ r }) => (
      <>
        A Nexus 6 at ISO 3200, from a test scene the network never saw. <b>Left: a standard camera pipeline. Right: ours.</b>{" "}
        Drag the white handle across.{r?.outputs.hybrid.metrics && <> Already {psnr(r, "traditional")} → <b>{psnr(r, "hybrid")} dB</b>.</>}
      </>
    ),
  },
  {
    title: "Now give it 8× less light",
    target: "#viewer",
    enter: (lab) => { lab.setSim({ noise: 0, exposure_ev: -3, wb_shift: 0 }); return true; },
    body: ({ r, waiting }) =>
      waiting || !r?.outputs.hybrid.metrics ? (
        <>We just under-exposed the RAW by 3 stops. Re-running all three pipelines…</>
      ) : (
        <>
          3 stops under. The traditional pipeline falls to {psnr(r, "traditional")} dB and the network alone to {psnr(r, "learned")} dB.{" "}
          <b>The hybrid keeps {psnr(r, "hybrid")} dB</b>: its classical stage fixes the exposure exactly, so the network only has to remove noise.
        </>
      ),
  },
  {
    title: "The router saw it coming",
    target: "#step-route",
    body: ({ r }) =>
      r ? (
        <>{routeSummary(r.analysis, r.route)} <b>Every photo gets its own path</b>; clean ones skip the network entirely.</>
      ) : null,
  },
  {
    title: "Look closer",
    target: "#loupe .loupe",
    enter: (lab) => { lab.setZoom(4); lab.setLens({ x: 0.62, y: 0.62 }); return false; },
    body: () => (
      <>The same spot in every pipeline, at 4×. <b>Tap anywhere on the photo</b> to move the loupe. Notice the colour speckle the others leave behind.</>
    ),
  },
  {
    title: "Not cherry-picked",
    target: null,
    body: ({ b }) => {
      const h = headline(b);
      return h.dPsnr !== null ? (
        <>
          Across all {h.n} unseen test images the hybrid is <b>+{h.dPsnr.toFixed(1)} dB</b> over the traditional pipeline
          {h.dPsnrLearned !== null && <> and +{h.dPsnrLearned.toFixed(1)} dB over the network alone</>}, with colour error {h.deFrom?.toFixed(1)} → {h.deTo?.toFixed(1)}.
          {h.int8 && <> The network is {fixed(h.int8.size_mb, 2)} MB.</>}
        </>
      ) : <>The full evaluation on the held-out test scenes is on the Evidence page.</>;
    },
  },
];

export function Tour({ lab, bench, onExit, onEvidence }: { lab: Lab; bench: Benchmarks | null; onExit: () => void; onEvidence: () => void }) {
  const [i, setI] = useState(0);
  const [waiting, setWaiting] = useState(false);
  const sawBusy = useRef(false);
  const sheet = useRef<HTMLDivElement>(null);
  const drag = useRef<{ y: number; t: number } | null>(null);
  const [dy, setDy] = useState(0);
  const step = STEPS[i];
  const ready = !!lab.samples;

  // run the step's action once the samples are known
  useEffect(() => {
    if (!ready) return;
    const needsWait = STEPS[i].enter?.(lab) ?? false;
    sawBusy.current = false;
    setWaiting(needsWait);
    if (!needsWait) return;
    const t = setTimeout(() => setWaiting(false), 20000); // never hang the tour on a slow server
    return () => clearTimeout(t);
  }, [i, ready]); // eslint-disable-line react-hooks/exhaustive-deps

  // a re-run has finished once busy has gone up and come back down
  useEffect(() => {
    if (!waiting) return;
    if (lab.busy) sawBusy.current = true;
    else if (sawBusy.current) setWaiting(false);
  }, [lab.busy, waiting]);

  // bring the target into view (below the sticky viewer on a phone) and ring it
  useEffect(() => {
    if (!step.target) { window.scrollTo({ top: 0, behavior: smooth() }); return; }
    const el = document.querySelector<HTMLElement>(step.target);
    if (!el) return;
    const phone = window.matchMedia("(max-width: 999px)").matches;
    const stage = document.querySelector<HTMLElement>(".lab-stage");
    const inStage = !!stage?.contains(el);
    if (inStage) {
      // the viewer column is sticky: just scroll past the intro so it sits at the top
      const intro = document.querySelector<HTMLElement>(".lab-intro");
      const top = phone && intro ? intro.offsetTop + intro.offsetHeight : 0;
      if (phone || window.scrollY > top) window.scrollTo({ top, behavior: smooth() });
    } else {
      const offset = phone ? (stage?.offsetHeight ?? 0) + 12 : 96;
      window.scrollTo({ top: el.getBoundingClientRect().top + window.scrollY - offset, behavior: smooth() });
    }
    el.classList.add("tour-ring");
    return () => el.classList.remove("tour-ring");
  }, [i, step.target, waiting]);

  // keyboard: arrows step, Escape leaves (but never steal arrows from a slider or text field)
  useEffect(() => {
    const on = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement;
      if (t.closest("input, select, textarea, [role=slider]")) return;
      if (e.key === "Escape") onExit();
      else if (e.key === "ArrowRight" && i < STEPS.length - 1) setI(i + 1);
      else if (e.key === "ArrowLeft" && i > 0) setI(i - 1);
    };
    window.addEventListener("keydown", on);
    return () => window.removeEventListener("keydown", on);
  }, [i, onExit]);

  useEffect(() => { sheet.current?.focus({ preventScroll: true }); }, [i]);

  const last = i === STEPS.length - 1;
  return (
    <div
      className={"tour-sheet" + (drag.current ? " dragging" : "")} role="dialog" aria-modal="false" aria-labelledby="tour-title" ref={sheet} tabIndex={-1}
      style={dy ? { transform: `translateY(${dy}px)`, opacity: Math.max(0.3, 1 - dy / 240) } : undefined}
    >
      {/* swipe down on the grabber (phones) to close; the X stays for everyone else */}
      <div
        className="tour-grab" aria-hidden="true"
        onPointerDown={(e) => { drag.current = { y: e.clientY, t: performance.now() }; (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId); }}
        onPointerMove={(e) => { if (drag.current) setDy(Math.max(0, e.clientY - drag.current.y)); }}
        onPointerUp={(e) => {
          const d = drag.current; drag.current = null;
          if (!d) return;
          const dist = e.clientY - d.y, v = dist / Math.max(1, performance.now() - d.t);
          if (dist > 90 || v > 0.6) onExit(); else setDy(0);
        }}
        onPointerCancel={() => { drag.current = null; setDy(0); }}
      ><span /></div>
      <div className="tour-top">
        <span className="tour-count num">{i + 1} / {STEPS.length}</span>
        <span className="tour-dots" aria-hidden="true">
          {STEPS.map((_, k) => <i key={k} className={k === i ? "on" : k < i ? "done" : undefined} />)}
        </span>
        <button className="tour-x" onClick={onExit} aria-label="End the guided demo">
          <svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true"><path d="M2 2l10 10M12 2L2 12" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" /></svg>
        </button>
      </div>
      <h2 id="tour-title">{step.title}</h2>
      <p className="tour-body" aria-live="polite">
        {!ready ? "Loading the samples…" : step.body({ r: lab.result, b: bench, waiting })}
        {waiting && <span className="tour-wait"><span className="spinner" aria-hidden="true" /> processing on the CPU</span>}
      </p>
      <div className="tour-actions">
        {i > 0 ? <button className="btn text" onClick={() => setI(i - 1)}>Back</button> : <span />}
        <span className="tour-keys" aria-hidden="true"><kbd>←</kbd><kbd>→</kbd> to step · <kbd>Esc</kbd> to close</span>
        {last ? (
          <span className="tour-end">
            <button className="btn" onClick={onExit}>Stay in the lab</button>
            <button className="btn primary" onClick={onEvidence}>See the evidence</button>
          </span>
        ) : (
          <button className="btn primary" onClick={() => setI(i + 1)} disabled={waiting}>
            {waiting ? "Processing…" : "Next"}
          </button>
        )}
      </div>
    </div>
  );
}

const smooth = (): ScrollBehavior => (window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth");
